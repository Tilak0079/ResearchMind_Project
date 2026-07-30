"""Manual test: search arXiv for a real topic, download and index one paper."""

import logging

from app.agent.arxiv_fetcher import fetch_and_index_from_arxiv
from app.db.postgres import SessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    db = SessionLocal()

    try:
        indexed_ids = fetch_and_index_from_arxiv("diffusion models image generation", db, max_results=1)
        logger.info(f"Successfully indexed {len(indexed_ids)} paper(s): {indexed_ids}")
    finally:
        db.close()