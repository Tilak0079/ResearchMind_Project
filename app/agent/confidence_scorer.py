"""
Confidence Scoring 
"""

import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# Weights from Section 3.2 - must sum to 1.0
WEIGHT_TOP1_SCORE = 0.40
WEIGHT_MARGIN = 0.20
WEIGHT_COVERAGE = 0.20
WEIGHT_RECENCY = 0.10
WEIGHT_DISTINCT_SOURCES = 0.10

TAU_HIGH = 0.78  # per Section 3.2
TAU_LOW = 0.45


def score_top1(reranked_results: list) -> float:
  
    if not reranked_results:
        return 0.0
    _, top_score = reranked_results[0]
    return float(top_score)


def score_margin(reranked_results: list) -> float:
    
    if len(reranked_results) < 2:
        return 0.5  # not enough results to judge margin - neutral score

    scores = [score for _, score in reranked_results[:5]]
    top_score = scores[0]
    mean_of_top5 = sum(scores) / len(scores)

    margin = top_score - mean_of_top5
    # Normalize: a margin of 0.3+ is a strong signal, cap at 1.0
    return min(margin / 0.3, 1.0)


def score_coverage(reranked_results: list, min_relevance: float = 0.5) -> float:
    
    if not reranked_results:
        return 0.0

    relevant_count = sum(1 for _, score in reranked_results if score >= min_relevance)
    return relevant_count / len(reranked_results)


def score_recency(reranked_results: list, query_wants_recent: bool, recency_months: int = 18) -> float:
    
    if not query_wants_recent:
        return 1.0  # recency wasn't requested, don't penalize

    if not reranked_results:
        return 0.0

    cutoff = date.today() - timedelta(days=recency_months * 30)
    recent_count = 0

    for point, _ in reranked_results:
        pub_date_str = point.payload.get("publication_date")
        if pub_date_str:
            pub_date = date.fromisoformat(pub_date_str)
            if pub_date >= cutoff:
                recent_count += 1

    return recent_count / len(reranked_results)


def score_distinct_sources(reranked_results: list) -> float:
    
    if not reranked_results:
        return 0.0

    distinct_papers = {point.payload["paper_id"] for point, _ in reranked_results}
    # Normalize: 3+ distinct papers = full score, fewer = partial
    return min(len(distinct_papers) / 3, 1.0)


def compute_confidence_score(
    reranked_results: list,
    query_wants_recent: bool = False,
) -> float:
    """
    Combines all 5 signals into one weighted confidence score (0.0 - 1.0)
    This score decides which path the router takes
    """
    s1 = score_top1(reranked_results)
    s2 = score_margin(reranked_results)
    s3 = score_coverage(reranked_results)
    s4 = score_recency(reranked_results, query_wants_recent)
    s5 = score_distinct_sources(reranked_results)

    confidence = (
        s1 * WEIGHT_TOP1_SCORE
        + s2 * WEIGHT_MARGIN
        + s3 * WEIGHT_COVERAGE
        + s4 * WEIGHT_RECENCY
        + s5 * WEIGHT_DISTINCT_SOURCES
    )

    logger.info(
        f"Confidence signals: top1={s1:.2f}, margin={s2:.2f}, coverage={s3:.2f}, "
        f"recency={s4:.2f}, distinct_sources={s5:.2f} -> final={confidence:.3f}"
    )

    return confidence