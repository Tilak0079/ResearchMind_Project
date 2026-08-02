"""
One-time bulk ingestion: walks the PDF folder, finds each PDF's matching
JSON metadata file (same filename, different extension), and indexes
every pair through our existing pipeline with REAL metadata instead of
placeholder-guessed title/authors.
"""

import logging
from pathlib import Path
from app.utils.minio_client import ensure_bucket_exists, upload_pdf_to_minio
from app.config import settings
from app.db.models import PaperRegistry
from app.db.postgres import SessionLocal
from app.ingestion.indexer import index_paper
from app.ingestion.metadata_loader import load_paper_metadata
from app.utils.minio_client import ensure_bucket_exists, upload_pdf_to_minio
from app.config import settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# EDIT THESE TWO PATHS to match your actual folders:
PDF_FOLDER = "local_dataset/pdfs"
JSON_FOLDER = "local_dataset/metadata"

def bulk_ingest():
    db = SessionLocal()
    pdf_dir = Path(PDF_FOLDER)
    json_dir = Path(JSON_FOLDER)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files to process")

    success_count = 0
    skip_count = 0
    fail_count = 0

    try:
        for pdf_path in pdf_files:
            json_path = json_dir / f"{pdf_path.stem}.json"

            if not json_path.exists():
                logger.warning(f"No matching JSON for {pdf_path.name}, skipping")
                fail_count += 1
                continue

            metadata = load_paper_metadata(str(json_path))

            # Skip if this arxiv_id is already in our database (avoids duplicates,
            # same pattern as Phase 10's filter_new_candidates)
            if metadata["arxiv_id"]:
                existing = db.query(PaperRegistry).filter(
                    PaperRegistry.arxiv_id == metadata["arxiv_id"]
                ).first()
                if existing:
                    logger.info(f"Already indexed: {metadata['title']}, skipping")
                    skip_count += 1
                    continue

            import uuid
            paper_id = str(uuid.uuid4())

            ensure_bucket_exists(settings.minio_bucket_raw_pdfs)
            minio_path = upload_pdf_to_minio(str(pdf_path), pdf_path.name)

            paper = PaperRegistry(
                paper_id=paper_id,
                arxiv_id=metadata["arxiv_id"],
                doi=metadata["doi"],
                title=metadata["title"],
                authors=metadata["authors"],
                publication_date=metadata["publication_date"],
                source_type=metadata["source_type"],
                trust_tier="local_corpus",  # override JSON's value - manually curated papers are trusted,
                abstract=metadata["abstract"],
                ingestion_status="indexing",
                raw_pdf_s3_path=minio_path,
            )
            db.add(paper)
            db.commit()

            try:
                chunk_count = index_paper(
                    str(pdf_path),
                    paper_id,
                    db,
                    trust_tier=metadata["trust_tier"],
                    arxiv_id=metadata["arxiv_id"],
                )
                logger.info(f"Indexed '{metadata['title']}': {chunk_count} chunks")
                paper.ingestion_status = "complete"
                db.commit()
                success_count += 1
            except (FileNotFoundError, ValueError):
                logger.exception(f"Failed to index {pdf_path.name}")
                paper.ingestion_status = "failed"
                db.commit()
                fail_count += 1

    finally:
        db.close()

    logger.info(f"Bulk ingest complete: {success_count} indexed, {skip_count} skipped, {fail_count} failed")


if __name__ == "__main__":
    bulk_ingest()