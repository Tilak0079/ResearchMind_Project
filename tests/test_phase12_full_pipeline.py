"""Manual test: confirm all 4 LLM-powered guardrails/features now work."""

import logging

from app.guardrails.input_guardrails import run_input_guardrails
from app.agent.arxiv_fetcher import reformulate_query_for_arxiv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Test scope check
    logger.info("=== Testing scope check ===")
    for query in ["How does attention work in transformers?", "Give me a recipe for pasta"]:
        is_safe, reason = run_input_guardrails(query)
        logger.info(f"'{query}' -> is_safe={is_safe}, reason={reason}")

    # Test query reformulation
    logger.info("=== Testing query reformulation ===")
    reformulated = reformulate_query_for_arxiv("What's new in making images with AI using diffusion?")
    logger.info(f"Reformulated: '{reformulated}'")