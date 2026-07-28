"""Manual test: parse a PDF, chunk it, and inspect chunk sizes/boundaries."""

import logging
import uuid

from app.ingestion.parser_docling import parse_pdf
from app.ingestion.chunker import chunk_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    doc = parse_pdf("tests/sample_pdfs/sample.pdf")
    fake_paper_id = str(uuid.uuid4())

    chunks = chunk_document(doc, fake_paper_id)

    logger.info(f"Total chunks: {len(chunks)}")

    token_counts = [c.token_count for c in chunks]
    logger.info(f"Min tokens: {min(token_counts)}, Max tokens: {max(token_counts)}, Avg: {sum(token_counts)//len(token_counts)}")

    # Show a few chunks that got split into multiple parts (part_index > 0)
    split_chunks = [c for c in chunks if c.part_index > 0]
    logger.info(f"Chunks that came from a size-split section: {len(split_chunks)}")

    for c in chunks[:3]:
        logger.info(f"--- {c.section_name} (part {c.part_index}, {c.token_count} tokens) ---")
        logger.info(c.content[:150])