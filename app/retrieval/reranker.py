"""
Reranker (Section 1.2, 2.3): BGE-reranker-v2-m3, applied to top-15 candidates only.

WHAT A RERANKER DOES: Hybrid search (Qdrant) is fast but approximate across
millions of vectors. The reranker is a slower, more accurate model that
looks ONLY at the top candidates and carefully re-scores each one against
the actual query text — like double-checking the top results by hand.
"""

import logging

from FlagEmbedding import FlagReranker

logger = logging.getLogger(__name__)

RERANK_TOP_N = 15  # only rerank the top 15 fused candidates (locked decision)

# Loaded once — loading per-call would be very slow.
# use_fp16=True speeds up inference with minimal accuracy loss.
# NOTE: the architecture doc calls for an 8-bit QUANTIZED version specifically.
# FlagEmbedding's fp16 mode is the readily-available speed/memory optimization
# in this library; true 8-bit quantization would need an additional library
# (e.g. bitsandbytes or an ONNX-quantized checkpoint) - flagging this as a
# gap to revisit, not silently claiming full parity with the doc's spec.
_reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)


def rerank_chunks(query: str, candidates: list, top_n: int = 8) -> list:
    """
    Reranks the top RERANK_TOP_N candidates against the query, returns the
    best top_n after reranking.

    Args:
        query: the user's original question (plain text).
        candidates: list of Qdrant scored points (from hybrid_search()).
        top_n: how many final chunks to return after reranking (default 8, per doc's N=5-8).

    Returns:
        List of (candidate, rerank_score) tuples, sorted best-first.
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