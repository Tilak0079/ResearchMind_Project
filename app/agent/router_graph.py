"""
Agentic Router
"""

import logging
import re
from typing import TypedDict

from app.agent.confidence_scorer import TAU_HIGH, TAU_LOW, compute_confidence_score
from app.ingestion.embedder import embed_text
from app.retrieval.qdrant_client import hybrid_search
from app.retrieval.reranker import rerank_chunks

logger = logging.getLogger(__name__)

# Simple keyword check for "please fetch from arXiv directly" style requests.
EXPLICIT_ARXIV_PATTERNS = [
    r"fetch (this |the )?(paper|from arxiv)",
    r"search arxiv",
    r"get (this |the )?(latest |newest )?paper from arxiv",
]


class RouterState(TypedDict):
    """Shared state that flows through the LangGraph nodes."""

    query: str
    explicit_arxiv_request: bool
    reranked_results: list
    confidence_score: float
    route: str
    route_reasoning: str


def detect_explicit_fetch_intent(query: str) -> bool:
    """Checks if the user explicitly asked to fetch from arXiv."""
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in EXPLICIT_ARXIV_PATTERNS)


def node_check_explicit_intent(state: RouterState) -> RouterState:
    """Node 1: check if the user explicitly wants an arXiv fetch."""
    state["explicit_arxiv_request"] = detect_explicit_fetch_intent(state["query"])
    return state


def node_local_retrieval(state: RouterState) -> RouterState:
    """Node 2: run hybrid search + rerank against our local corpus."""
    query_embedding = embed_text(state["query"])
    candidates = hybrid_search(query_embedding["dense"], query_embedding["sparse"], top_k=30)
    state["reranked_results"] = rerank_chunks(state["query"], candidates, top_n=10)
    return state


def node_score_confidence(state: RouterState) -> RouterState:
    """Node 3: compute the confidence score from local retrieval results."""
    state["confidence_score"] = compute_confidence_score(state["reranked_results"])
    return state


def node_decide_route(state: RouterState) -> RouterState:
    """
    Node 4: the actual routing decision (Section 3.1's pseudocode, as a node).
    """
    if state["explicit_arxiv_request"]:
        state["route"] = "arxiv_fetch_path"
        state["route_reasoning"] = "User explicitly requested an arXiv fetch."
    elif state["confidence_score"] >= TAU_HIGH:
        state["route"] = "local_only_path"
        state["route_reasoning"] = f"Confidence {state['confidence_score']:.3f} >= TAU_HIGH ({TAU_HIGH})"
    elif state["confidence_score"] < TAU_LOW:
        state["route"] = "arxiv_fetch_path"
        state["route_reasoning"] = f"Confidence {state['confidence_score']:.3f} < TAU_LOW ({TAU_LOW})"
    else:
        state["route"] = "hybrid_path"
        state["route_reasoning"] = (
            f"Confidence {state['confidence_score']:.3f} is between TAU_LOW and TAU_HIGH"
        )

    logger.info(f"Route decision: {state['route']} - {state['route_reasoning']}")
    return state


def run_router(query: str) -> RouterState:
    """
    Runs the router pipeline for a single query.
    """
    state: RouterState = {
        "query": query,
        "explicit_arxiv_request": False,
        "reranked_results": [],
        "confidence_score": 0.0,
        "route": "",
        "route_reasoning": "",
    }

    state = node_check_explicit_intent(state)

    if not state["explicit_arxiv_request"]:
        state = node_local_retrieval(state)
        state = node_score_confidence(state)

    state = node_decide_route(state)

    return state