"""
Indexer: ties the full ingestion pipeline together.
parse_pdf() -> chunk_document() -> embed_text() -> Qdrant + Postgres.

This is what runs once per paper, whether from local corpus ingestion
or the arXiv Fetcher Agent's background indexing step.
"""

import logging
import uuid

from qdrant_client.models import PointStruct
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import ChunkRegistry
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embed_text
from app.ingestion.parser_docling import parse_pdf
from app.retrieval.qdrant_client import ensure_collection_exists, get_qdrant_client

logger = logging.getLogger(__name__)


def index_paper(
    file_path: str,
    paper_id: str,
    db: Session,
    trust_tier: str = "local_corpus",
    arxiv_id: str | None = None,
) -> int:
    """
    Runs the full ingestion pipeline for one PDF and indexes it.

    Args:
        file_path: path to the PDF on disk.
        paper_id: UUID string of the paper's row in paper_registry (must already exist).
        db: active SQLAlchemy session, used to write chunk_registry rows.

    Returns:
        Number of chunks successfully indexed

    """
    ensure_collection_exists()
    qdrant = get_qdrant_client()

    logger.info(f"Starting indexing for paper_id={paper_id}, file={file_path}")

    parsed_doc = parse_pdf(file_path, paper_id)
    chunks = chunk_document(parsed_doc, paper_id)

    if not chunks:
        logger.warning(f"No chunks produced for paper_id={paper_id} — nothing to index")
        return 0

    points: list[PointStruct] = []

    for chunk in chunks:
        try:
            embedding = embed_text(chunk.content)
        except Exception:
            # A single bad chunk shouldn't kill the whole paper's indexing —
            # log it and skip, rather than crashing the entire pipeline.
            logger.exception(f"Failed to embed chunk {chunk.chunk_id}, skipping")
            continue

        vector_point_id = str(uuid.uuid4())

        payload = {
            "chunk_id": chunk.chunk_id,
            "paper_id": paper_id,
            "section_name": chunk.section_name,
            "chunk_type": chunk.chunk_type,
            "part_index": chunk.part_index,
            "page_number": chunk.page_number,
            "text": chunk.content,
            "token_count": chunk.token_count,
            "trust_tier": trust_tier,
            "arxiv_id": arxiv_id,
        }
        
        if chunk.artifact_path:
            payload["artifact_path"] = chunk.artifact_path

        points.append(
            PointStruct(
                id=vector_point_id,
                vector={
                    "dense": embedding["dense"],
                    "sparse": embedding["sparse"],
                },
                payload=payload,
            )
        )

        # Mirror the pointer in Postgres so chunk_registry knows which
        # Qdrant point corresponds to which chunk (Section 5.1 schema).
        db_chunk = ChunkRegistry(
            chunk_id=chunk.chunk_id,
            paper_id=paper_id,
            chunk_type=chunk.chunk_type,
            section_name=chunk.section_name,
            page_number=chunk.page_number,
            part_index=chunk.part_index,
            token_count=chunk.token_count,
            vector_db_id=vector_point_id,
            artifact_path=chunk.artifact_path,
        )
        db.add(db_chunk)

    if points:
        qdrant.upsert(collection_name=settings.qdrant_collection_name, points=points)
        db.commit()
        logger.info(f"Indexed {len(points)} chunks for paper_id={paper_id}")

    return len(points)