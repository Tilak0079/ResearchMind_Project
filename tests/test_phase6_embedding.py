"""Manual test: confirm BGE-m3 dense + sparse embedding generation works via FlagEmbedding."""

import logging

from app.ingestion.embedder import embed_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    sample_text = "Attention mechanisms allow models to focus on relevant parts of the input sequence."

    result = embed_text(sample_text)

    logger.info(f"Dense vector length: {len(result['dense'])} (expect 1024)")
    logger.info(f"First 5 dense values: {result['dense'][:5]}")
    logger.info(f"Sparse non-zero terms: {len(result['sparse']['indices'])}")
    logger.info(f"Sample sparse indices: {result['sparse']['indices'][:5]}")