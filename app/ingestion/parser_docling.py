"""
PDF parsing using Docling

"""

import logging
from pathlib import Path

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

from app.ingestion.schemas import ParsedDocument, ParsedSection

logger = logging.getLogger(__name__)

# OCR is disabled
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# Labels we treat as real section headers (starts a new section)
HEADER_LABELS = {"section_header", "title"}

# Labels we skip entirely - not real body content
SKIP_LABELS = {"page_header", "page_footer", "footnote"}


def parse_pdf(file_path: str) -> ParsedDocument:
    """
    Parses a PDF file into a ParsedDocument, reading Docling's structured text items directly (not the flattened markdown export) so real page numbers are preserved per section.

    """
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info(f"Parsing PDF: {file_path}")
    result = converter.convert(str(pdf_path))
    doc = result.document

    if not doc.texts:
        raise ValueError(f"Docling produced empty output for: {file_path}")

    sections = _build_sections_from_items(doc.texts)
    raw_markdown = doc.export_to_markdown()  # kept for reference/fallback use

    title = pdf_path.stem
    authors: list[str] = []

    return ParsedDocument(
        title=title,
        authors=authors,
        sections=sections,
        raw_markdown=raw_markdown,
        parsing_confidence=1.0,
    )


def _build_sections_from_items(text_items: list) -> list[ParsedSection]:
    """
    Groups Docling's labeled text items into sections
    """
    sections: list[ParsedSection] = []
    current_header = "Untitled"
    current_level = 1
    current_lines: list[str] = []
    current_page: int | None = None

    for item in text_items:
        if item.label in SKIP_LABELS:
            continue

        page_no = item.prov[0].page_no if item.prov else None

        if item.label in HEADER_LABELS:
            # Close out the previous section before starting a new one
            if current_lines:
                sections.append(
                    ParsedSection(
                        header=current_header,
                        level=current_level,
                        content_markdown="\n\n".join(current_lines).strip(),
                        page_number=current_page,
                    )
                )
            current_header = item.text.strip()
            current_level = 1 if item.label == "title" else 2
            current_lines = []
            current_page = page_no
        else:
            if current_page is None:
                current_page = page_no
            current_lines.append(item.text)

    # Don't forget the last section
    if current_lines:
        sections.append(
            ParsedSection(
                header=current_header,
                level=current_level,
                content_markdown="\n\n".join(current_lines).strip(),
                page_number=current_page,
            )
        )

    return sections