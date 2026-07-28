"""
End-to-end test: parse -> chunk -> embed -> store in Qdrant + Postgres.
Also inserts a dummy paper_registry row first, since chunk_registry
has a foreign key to it.
"""

import logging
import uuid

from app.db.models import PaperRegistry
from app.db.postgres import SessionLocal
from app.ingestion.indexer import index_paper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    db = SessionLocal()

    try:
        paper_id = str(uuid.uuid4())

        # Create a minimal paper_registry row first (FK requirement)
        paper = PaperRegistry(
            paper_id=paper_id,
            title="Attention Is All You Need (test ingest)",
            source_type="local_corpus",
            trust_tier="local_corpus",
            ingestion_status="indexing",
        )
        db.add(paper)
        db.commit()

        chunk_count = index_paper("tests/sample_pdfs/sample.pdf", paper_id, db)
        logger.info(f"Indexing complete: {chunk_count} chunks stored")

    finally:
        db.close()