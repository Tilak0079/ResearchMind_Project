"""
PDF parsing using Docling

"""

import logging
from pathlib import Path

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

from app.ingestion.schemas import ParsedDocument, ParsedSection, ParsedArtifact
from app.utils.minio_client import upload_bytes_to_minio
from app.config import settings
import io

logger = logging.getLogger(__name__)

# OCR is disabled
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.generate_picture_images = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# Labels we treat as real section headers (starts a new section)
HEADER_LABELS = {"section_header", "title"}

# Labels we skip entirely - not real body content
SKIP_LABELS = {"page_header", "page_footer", "footnote"}


def parse_pdf(file_path: str, paper_id: str) -> ParsedDocument:
    """
    Parses a PDF file into a ParsedDocument, extracting artifacts (figures, tables, equations).
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
    artifacts = []

    # Process pictures
    for idx, pic in enumerate(doc.pictures):
        if pic.image:
            pil_img = pic.get_image(doc)
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            page_no = pic.prov[0].page_no if pic.prov else None
            page_str = f"{page_no:02d}" if page_no else "00"
            object_name = f"figures/{paper_id}/page_{page_str}_figure_{idx:02d}.png"
            
            artifact_path = upload_bytes_to_minio(
                img_bytes, 
                object_name, 
                settings.minio_bucket_figures,
                "image/png"
            )
            
            try:
                caption_text = pic.captions[0].resolve(doc).text if pic.captions else f"Figure on page {page_no}"
            except Exception:
                caption_text = f"Figure on page {page_no}"
            
            artifacts.append(ParsedArtifact(
                artifact_type="figure",
                content=caption_text,
                page_number=page_no,
                artifact_path=artifact_path
            ))

    # Process tables
    for idx, table in enumerate(doc.tables):
        page_no = table.prov[0].page_no if table.prov else None
        page_str = f"{page_no:02d}" if page_no else "00"
        object_name = f"tables/{paper_id}/page_{page_str}_table_{idx:02d}.md"
        
        md_text = table.export_to_markdown()
        
        artifact_path = upload_bytes_to_minio(
            md_text.encode("utf-8"), 
            object_name, 
            settings.minio_bucket_figures,
            "text/markdown"
        )
        
        artifacts.append(ParsedArtifact(
            artifact_type="table",
            content=md_text,
            page_number=page_no,
            artifact_path=artifact_path
        ))

    # Process equations
    eq_idx = 0
    for item in doc.texts:
        if item.label == "formula":
            page_no = item.prov[0].page_no if item.prov else None
            page_str = f"{page_no:02d}" if page_no else "00"
            object_name = f"equations/{paper_id}/page_{page_str}_eq_{eq_idx:02d}.tex"
            
            eq_text = item.text
            artifact_path = upload_bytes_to_minio(
                eq_text.encode("utf-8"), 
                object_name, 
                settings.minio_bucket_figures,
                "text/plain"
            )
            artifacts.append(ParsedArtifact(
                artifact_type="equation",
                content=eq_text,
                page_number=page_no,
                artifact_path=artifact_path
            ))
            eq_idx += 1

    raw_markdown = doc.export_to_markdown()

    title = pdf_path.stem
    authors: list[str] = []

    return ParsedDocument(
        title=title,
        authors=authors,
        sections=sections,
        artifacts=artifacts,
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