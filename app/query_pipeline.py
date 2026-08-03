"""
Core query pipeline: ties together every phase we've built into one
reusable function - input guardrails -> router -> retrieval -> retrieval
guardrails -> generation. Both the REST endpoint and WebSocket handler
(Phase 13) call this same function, so pipeline logic lives in ONE place.
"""

import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.agent.arxiv_fetcher import fetch_and_index_from_arxiv
from app.agent.confidence_scorer import TAU_HIGH, TAU_LOW, compute_confidence_score
from app.generation.context_assembler import assemble_context, build_user_message
from app.generation.llm_client import generate_response
from app.generation.prompts import GENERATION_SYSTEM_PROMPT
from app.guardrails.input_guardrails import run_input_guardrails
from app.guardrails.retrieval_guardrails import run_retrieval_guardrails
from app.ingestion.embedder import embed_text
from app.retrieval.qdrant_client import hybrid_search
from app.retrieval.reranker import rerank_chunks
from app.utils.cache import get_cached_response, set_cached_response

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Everything the API layer needs to build a response (Section 7.2's schema)."""

    success: bool
    route_taken: str = ""
    confidence_score: float = 0.0
    answer: str = ""
    error_reason: str = ""
    sources: list = field(default_factory=list)


from app.api.schemas import QueryResponse

def run_query_pipeline(query: str, db: Session, session_id: str) -> QueryResponse:
    """
    Runs the full pipeline for one user query, end to end.

    """
    
    is_safe, reason = run_input_guardrails(query, session_id)
    if not is_safe:
        logger.info(f"Query blocked by input guardrails: {reason}")
        return QueryResult(success=False, error_reason=reason)

    # Check cache first
    cached_response = get_cached_response(query)
    if cached_response:
        # Update the session_id to match the current user's request
        cached_response.session_id = session_id
        cached_response.metadata["cached"] = True
        return cached_response

    # Step 2: Local retrieval + confidence scoring 
    query_embedding = embed_text(query)
    candidates = hybrid_search(query_embedding["dense"], query_embedding["sparse"], top_k=30)
    reranked = rerank_chunks(query, candidates, top_n=10)
    confidence = compute_confidence_score(reranked)

    # Step 3: Routing decision 
    if confidence < TAU_LOW:
        route = "arxiv_fetch_path"
    elif confidence >= TAU_HIGH:
        route = "local_only_path"
    else:
        route = "hybrid_path"

    logger.info(f"Route: {route} (confidence={confidence:.3f})")

    # Step 4: If confidence is too low, fetch from arXiv before answering
    if route == "arxiv_fetch_path":
        try:
            fetch_and_index_from_arxiv(query, db, max_results=1)
            # Re-run retrieval now that we may have new data indexed
            candidates = hybrid_search(query_embedding["dense"], query_embedding["sparse"], top_k=30)
            reranked = rerank_chunks(query, candidates, top_n=10)
        except Exception:
            # arXiv fetch failing shouldn't crash the whole query - fall back
            # to whatever local results we already had.
            logger.exception("arXiv fetch failed, continuing with existing local results")

    # Step 5: Retrieval guardrails (Phase 11)
    final_chunks = run_retrieval_guardrails(query, reranked)

    from app.api.schemas import QueryResponse, Citation, Evidence, ArtifactMetadataModel, ConfidenceDetail
    
    if not final_chunks:
        return QueryResponse(
            session_id=session_id,
            success=True,
            route_taken=route,
            query=query,
            answer="I couldn't find sufficient evidence in the retrieved papers to answer this confidently.",
            confidence=ConfidenceDetail(
                overall_confidence=confidence,
                explanation="No relevant chunks passed the retrieval guardrails.",
                factors=["Lack of relevant context"]
            )
        )

    # Step 6: Generation (Phase 12)
    context = assemble_context(final_chunks)
    user_message = build_user_message(query, context)
    
    from pydantic import BaseModel, Field, ValidationError

    class LLMResponse(BaseModel):
        answer: str = Field(default="", description="The primary natural language response")
        summary: list[str] = Field(default_factory=list)
        limitations: list[str] = Field(default_factory=list)
        follow_up_questions: list[str] = Field(default_factory=list)
        relevant_artifacts: list[str] = Field(default_factory=list)
        confidence: dict = Field(default_factory=dict)

    try:
        # Request native structured output
        llm_response_obj = generate_response(GENERATION_SYSTEM_PROMPT, user_message, response_format=LLMResponse)
        llm_data = llm_response_obj.model_dump()
    except Exception as e:
        logger.warning(f"Failed to generate structured LLM response on first attempt: {e}. Retrying with stricter prompt.")
        
        # Retry once with stricter prompt (still using structured output request)
        stricter_prompt = GENERATION_SYSTEM_PROMPT + "\n\nCRITICAL: You failed to output valid JSON previously. You MUST output ONLY raw JSON matching the schema."
        try:
            llm_response_retry = generate_response(stricter_prompt, user_message, response_format=LLMResponse)
            llm_data = llm_response_retry.model_dump()
        except Exception as retry_e:
            logger.exception(f"Failed to generate structured LLM response on retry: {retry_e}")
            
            # Since .parse() failed entirely, attempt one last fallback without response_format to get ANY text
            try:
                fallback_text = generate_response(stricter_prompt, user_message)
            except Exception:
                fallback_text = "The system encountered an error formatting the response, but relevant documents were found."
                
            llm_data = {
                "answer": fallback_text,
                "summary": [],
                "limitations": [],
                "follow_up_questions": [],
                "relevant_artifacts": [],
                "confidence": {
                    "explanation": "Could not calculate confidence due to formatting error.",
                    "factors": []
                }
            }

    from app.utils.minio_client import get_presigned_url
    from app.config import settings
    
    citations = []
    evidence = []
    artifacts = []
    
    relevant_artifact_ids = set(llm_data.get("relevant_artifacts", []))
    
    for point, score in final_chunks:
        payload = point.payload
        arxiv_id = payload.get("arxiv_id")
        paper_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None
        
        # Add Evidence
        evidence.append(Evidence(
            paper=arxiv_id,
            page=payload.get("page_number"),
            chunk_text=payload.get("text", ""),
            similarity_score=score,
            reranker_score=score # Assuming score here is from the reranker in final_chunks
        ))
        
        # Add Citation
        citations.append(Citation(
            title=arxiv_id, # Using arxiv_id as title since we don't store full title in Qdrant currently
            page=payload.get("page_number"),
            section=payload["section_name"],
            trust_tier=payload.get("trust_tier"),
            retrieval_score=score,
            url=paper_url
        ))
        
        # Add Artifact (only if it was selected by the LLM)
        if payload.get("artifact_path") and payload["artifact_path"] in relevant_artifact_ids:
            bucket = settings.minio_bucket_figures
            object_name = payload["artifact_path"]
            if object_name.startswith(f"{bucket}/"):
                object_name = object_name[len(f"{bucket}/"):]
            elif "/" in object_name and not object_name.startswith("figures/") and not object_name.startswith("tables/") and not object_name.startswith("equations/"):
                object_name = object_name.split("/", 1)[1]
                
            artifact_url = get_presigned_url(bucket, object_name)
            
            artifacts.append(ArtifactMetadataModel(
                type=payload.get("chunk_type", "unknown"),
                title=payload["section_name"],
                page=payload.get("page_number"),
                paper=arxiv_id,
                url=artifact_url,
                content=payload.get("text", "")
            ))
            
    confidence_data = llm_data.get("confidence", {})
            
    final_response = QueryResponse(
        session_id=session_id,
        success=True,
        route_taken=route,
        query=query,
        answer=llm_data.get("answer", ""),
        summary=llm_data.get("summary", []),
        limitations=llm_data.get("limitations", []),
        follow_up_questions=llm_data.get("follow_up_questions", []),
        confidence=ConfidenceDetail(
            overall_confidence=confidence,
            explanation=confidence_data.get("explanation", ""),
            factors=confidence_data.get("factors", [])
        ),
        artifacts=artifacts,
        citations=citations,
        evidence=evidence,
        metadata={"cached": False}
    )

    # Store successful response in cache
    set_cached_response(query, final_response)

    return final_response