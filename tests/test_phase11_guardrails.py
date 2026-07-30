"""Manual test: run reranked results through retrieval guardrails, inspect changes."""

import logging

from app.guardrails.retrieval_guardrails import run_retrieval_guardrails
from app.ingestion.embedder import embed_text
from app.retrieval.qdrant_client import hybrid_search
from app.retrieval.reranker import rerank_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    query = "How does the attention mechanism work?"

    query_embedding = embed_text(query)
    candidates = hybrid_search(query_embedding["dense"], query_embedding["sparse"], top_k=30)
    reranked = rerank_chunks(query, candidates, top_n=10)

    logger.info(f"Before guardrails: {len(reranked)} chunks")

    final_results = run_retrieval_guardrails(query, reranked)

    logger.info(f"After guardrails: {len(final_results)} chunks")
    for point, score in final_results:
        trust_flag = point.payload.get("trust_flag", False)
        logger.info(f"- {point.payload['section_name']} (score={score:.3f}, trust_flag={trust_flag})")