"""Manual test: run different queries through the router, inspect routing decisions."""

import logging

from app.agent.router_graph import run_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_QUERIES = [
    "How does the attention mechanism work?",             # should find good local matches
    "What is the latest research on quantum computing?",  # not in our 1-paper corpus - low confidence
    "Please fetch this paper from arxiv about diffusion models",  # explicit arxiv request
]

if __name__ == "__main__":
    for query in TEST_QUERIES:
        logger.info(f"\n=== Query: '{query}' ===")
        result = run_router(query)
        logger.info(f"Route: {result['route']}")
        logger.info(f"Reasoning: {result['route_reasoning']}")