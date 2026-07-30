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


def run_query_pipeline(query: str, db: Session) -> QueryResult:
    """
    Runs the full pipeline for one user query, end to end.
    This is the same sequence proven working across Phases 8-12's test
    scripts, now consolidated into one function for the API layer to call.
    """
    # Step 1: Input guardrails (Phase 8)
    is_safe, reason = run_input_guardrails(query)
    if not is_safe:
        logger.info(f"Query blocked by input guardrails: {reason}")
        return QueryResult(success=False, error_reason=reason)

    # Step 2: Local retrieval + confidence scoring (Phase 7, 9)
    query_embedding = embed_text(query)
    candidates = hybrid_search(query_embedding["dense"], query_embedding["sparse"], top_k=30)
    reranked = rerank_chunks(query, candidates, top_n=10)
    confidence = compute_confidence_score(reranked)

    # Step 3: Routing decision (Phase 9)
    if confidence < TAU_LOW:
        route = "arxiv_fetch_path"
    elif confidence >= TAU_HIGH:
        route = "local_only_path"
    else:
        route = "hybrid_path"

    logger.info(f"Route: {route} (confidence={confidence:.3f})")

    # Step 4: If confidence is too low, fetch from arXiv before answering (Phase 10)
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

    if not final_chunks:
        return QueryResult(
            success=True,
            route_taken=route,
            confidence_score=confidence,
            answer="Information not available in the retrieved context.",
        )

    # Step 6: Generation (Phase 12)
    context = assemble_context(final_chunks)
    user_message = build_user_message(query, context)
    answer = generate_response(GENERATION_SYSTEM_PROMPT, user_message)

    sources = []
    for point, _score in final_chunks:
        arxiv_id = point.payload.get("arxiv_id")
        sources.append({
            "section": point.payload["section_name"],
            "page": point.payload.get("page_number"),
            "trust_flag": point.payload.get("trust_flag", False),
            "link": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
        })
    return QueryResult(
        success=True,
        route_taken=route,
        confidence_score=confidence,
        answer=answer,
        sources=sources,
    )