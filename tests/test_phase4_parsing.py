"""Manual test: parse a sample PDF and print structure so we can eyeball correctness."""

import logging

from app.ingestion.parser_docling import parse_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    doc = parse_pdf("tests/sample_pdfs/sample.pdf")

    logger.info(f"Title (guessed): {doc.title}")
    logger.info(f"Number of sections found: {len(doc.sections)}")

    for i, section in enumerate(doc.sections[:5]):  # just first 5 for a quick look
        logger.info(f"--- Section {i} (level {section.level}) ---")
        logger.info(f"Header: {section.header}")
        logger.info(f"Content preview: {section.content_markdown[:200]}")