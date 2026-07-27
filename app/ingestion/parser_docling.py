"""
PDF parsing using Docling .
Converts a raw PDF into structured markdown: headers, tables, equations, figures.
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


def parse_pdf(file_path: str) -> ParsedDocument:
    """
    Parses a PDF file into a ParsedDocument.

    Raises:
        FileNotFoundError: if the PDF path doesn't exist.
        ValueError: if Docling fails to produce any content (corrupt/unsupported PDF).
    """
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info(f"Parsing PDF: {file_path}")
    result = converter.convert(str(pdf_path))
    doc = result.document

    markdown_output = doc.export_to_markdown()
    if not markdown_output.strip():
        raise ValueError(f"Docling produced empty output for: {file_path}")

    # Docling exposes document metadata via doc.name / doc.texts etc.
   
    title = pdf_path.stem
    authors: list[str] = []

    sections = _split_markdown_into_sections(markdown_output)

    return ParsedDocument(
        title=title,
        authors=authors,
        sections=sections,
        raw_markdown=markdown_output,
        parsing_confidence=1.0,
    )


def _split_markdown_into_sections(markdown_text: str) -> list[ParsedSection]:
    """
    Splits raw markdown into sections based on '#' and '##' headers.
    
    """
    sections: list[ParsedSection] = []
    current_header = "Untitled"
    current_level = 1
    current_lines: list[str] = []

    for line in markdown_text.split("\n"):
        if line.startswith("# ") or line.startswith("## "):
            if current_lines:
                sections.append(
                    ParsedSection(
                        header=current_header,
                        level=current_level,
                        content_markdown="\n".join(current_lines).strip(),
                    )
                )
            current_level = 1 if line.startswith("# ") else 2
            current_header = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            ParsedSection(
                header=current_header,
                level=current_level,
                content_markdown="\n".join(current_lines).strip(),
            )
        )

    return sections