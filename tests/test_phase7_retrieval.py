"""Manual test: embed a query, run hybrid search, rerank, inspect final results."""

import logging

from app.ingestion.embedder import embed_text
from app.retrieval.qdrant_client import hybrid_search
from app.retrieval.reranker import rerank_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    query = "How does the attention mechanism work?"

    # Step 1: embed the query the same way we embedded chunks (Phase 6)
    query_embedding = embed_text(query)

    # Step 2: hybrid search in Qdrant (dense + sparse + RRF)
    candidates = hybrid_search(query_embedding["dense"], query_embedding["sparse"], top_k=30)
    logger.info(f"Hybrid search returned {len(candidates)} candidates")

    # Step 3: rerank the top candidates
    final_results = rerank_chunks(query, candidates, top_n=5)

    logger.info(f"Top {len(final_results)} results after reranking:")
    for i, (point, score) in enumerate(final_results):
        logger.info(f"--- Result {i+1} (rerank score: {score:.4f}) ---")
        logger.info(f"Section: {point.payload['section_name']}")
        logger.info(f"Text: {point.payload['text'][:200]}")