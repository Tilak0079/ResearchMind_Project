"""
Core query pipeline for Paper Q&A Mode.
Strictly restricted to answering from a single paper using pre-indexed Qdrant vectors.
"""

import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.agent.confidence_scorer import compute_confidence_score
from app.generation.context_assembler import assemble_context, build_user_message
from app.generation.llm_client import generate_response
from app.generation.prompts import PAPER_MODE_SYSTEM_PROMPT
from app.guardrails.input_guardrails import run_input_guardrails
from app.guardrails.retrieval_guardrails import run_retrieval_guardrails
from app.ingestion.embedder import embed_text
from app.retrieval.qdrant_client import hybrid_search
from app.retrieval.reranker import rerank_chunks

logger = logging.getLogger(__name__)

from app.api.schemas_paper import PaperQueryResponse
from app.api.schemas import Citation, Evidence, ArtifactMetadataModel, ConfidenceDetail

def run_paper_query_pipeline(query: str, db: Session, session_id: str, paper_id: str) -> PaperQueryResponse:
    """
    Runs the pipeline for Paper Q&A Mode.
    """
    is_safe, reason = run_input_guardrails(query, session_id)
    if not is_safe:
        logger.info(f"Query blocked by input guardrails: {reason}")
        return PaperQueryResponse(
            session_id=session_id,
            success=False,
            error_reason=reason,
            query=query
        )

    from app.db.models import Message as DBMessage
    from app.config import settings

    # Retrieve conversation history
    history_records = (
        db.query(DBMessage)
        .filter(DBMessage.session_id == session_id)
        .order_by(DBMessage.created_at.desc())
        .limit(settings.chat_history_limit)
        .all()
    )
    # Reverse to chronological order
    history_records.reverse()
    
    # Format history
    history_blocks = []
    for msg in history_records:
        role_label = "User" if msg.role == "user" else "Assistant"
        history_blocks.append(f"{role_label}:\n{msg.content}")
        
    formatted_history = "\n\n".join(history_blocks)

    # Step 2: Retrieval restricted to paper_id
    query_embedding = embed_text(query)
    candidates = hybrid_search(query_embedding["dense"], query_embedding["sparse"], top_k=30, paper_id=paper_id)
    reranked = rerank_chunks(query, candidates, top_n=10)
    confidence = compute_confidence_score(reranked)

    logger.info(f"Paper Mode Retrieval: (confidence={confidence:.3f})")

    # Step 5: Retrieval guardrails (Optional but good to keep for quality)
    final_chunks = run_retrieval_guardrails(query, reranked)

    if not final_chunks:
        return PaperQueryResponse(
            session_id=session_id,
            success=True,
            query=query,
            answer="This paper does not discuss this topic.",
            confidence=ConfidenceDetail(
                overall_confidence=confidence,
                explanation="No relevant chunks passed the retrieval guardrails.",
                factors=["Lack of relevant context"]
            )
        )

    # Step 6: Generation
    context = assemble_context(final_chunks)
    user_message = build_user_message(query, context, formatted_history)
    
    from app.agent.diagram_intent import detect_diagram_intent
    import re
    
    needs_diagram = detect_diagram_intent(query)
    has_native_artifact = any(chunk.payload.get("artifact_path") for chunk, score in final_chunks)
    request_multiple_artifacts = bool(re.search(r'\b(all|multiple|every)\s+(figures?|tables?|diagrams?|images?)\b', query.lower()))
    
    system_prompt = PAPER_MODE_SYSTEM_PROMPT
    if needs_diagram and not has_native_artifact:
        system_prompt += "\n\nCRITICAL INSTRUCTION: The user requires a visual explanation, but no native figures were retrieved. You MUST generate a structured Mermaid diagram (enclosed in ```mermaid) to represent this concept visually. Embed the Mermaid diagram strictly inside your natural language `answer` field using \\n for newlines so the JSON remains valid. Ensure the diagram is strictly conceptual and 100% grounded in the retrieved context. Do not invent details."
    
    from pydantic import BaseModel, Field

    class LLMResponse(BaseModel):
        answer: str = Field(default="", description="The primary natural language response")
        summary: list[str] = Field(default_factory=list)
        limitations: list[str] = Field(default_factory=list)
        follow_up_questions: list[str] = Field(default_factory=list)
        relevant_artifacts: list[str] = Field(default_factory=list)
        confidence: dict = Field(default_factory=dict)

    try:
        llm_response_obj = generate_response(system_prompt, user_message, response_format=LLMResponse)
        llm_data = llm_response_obj.model_dump()
    except Exception as e:
        logger.warning(f"Failed to generate structured LLM response on first attempt: {e}. Retrying with stricter prompt.")
        stricter_prompt = PAPER_MODE_SYSTEM_PROMPT + "\n\nCRITICAL: You failed to output valid JSON previously. You MUST output ONLY raw JSON matching the schema."
        try:
            llm_response_retry = generate_response(stricter_prompt, user_message, response_format=LLMResponse)
            llm_data = llm_response_retry.model_dump()
        except Exception as retry_e:
            logger.exception(f"Failed to generate structured LLM response on retry: {retry_e}")
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
    
    citations = []
    evidence = []
    artifacts = []
    
    relevant_artifact_ids = set(llm_data.get("relevant_artifacts", []))
    
    for point, score in final_chunks:
        payload = point.payload
        arxiv_id = payload.get("arxiv_id")
        paper_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None
        
        evidence.append(Evidence(
            paper=arxiv_id or paper_id,
            page=payload.get("page_number"),
            chunk_text=payload.get("text", ""),
            similarity_score=score,
            reranker_score=score
        ))
        
        citations.append(Citation(
            title=arxiv_id or "Selected Paper",
            page=payload.get("page_number"),
            section=payload["section_name"],
            trust_tier=payload.get("trust_tier"),
            retrieval_score=score,
            url=paper_url
        ))
        
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
                paper=arxiv_id or paper_id,
                url=artifact_url,
                content=payload.get("text", "")
            ))
            
    # Limit artifacts dynamically based on intent
    if not request_multiple_artifacts:
        artifacts = artifacts[:1]
            
    confidence_data = llm_data.get("confidence", {})
            
    final_response = PaperQueryResponse(
        session_id=session_id,
        success=True,
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
        metadata={}
    )

    try:
        user_msg = DBMessage(session_id=session_id, role="user", content=query)
        asst_msg = DBMessage(session_id=session_id, role="assistant", content=final_response.answer)
        db.add(user_msg)
        db.add(asst_msg)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to persist messages for session {session_id}: {e}")

    return final_response
