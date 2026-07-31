"""
Reranker: BGE-reranker-v2-m3, applied to top-15 candidates only.

"""

import logging

from FlagEmbedding import FlagReranker

logger = logging.getLogger(__name__)

RERANK_TOP_N = 15  # only rerank the top 15 fused candidates (locked decision)

_reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)


def rerank_chunks(query: str, candidates: list, top_n: int = 8) -> list:
    """
    Reranks the top RERANK_TOP_N candidates against the query, returns the best top_n after reranking.
    """
    if not candidates:
        return []

    narrowed = candidates[:RERANK_TOP_N]

    # The reranker needs pairs: [query, chunk_text]
    pairs = [[query, point.payload["text"]] for point in narrowed]

    scores = _reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):  # compute_score returns a single float if only 1 pair
        scores = [scores]

    scored = list(zip(narrowed, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return scored[:top_n]