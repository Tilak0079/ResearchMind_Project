"""Temporary diagnostic script - checks Docling's text item labels and page numbers."""

from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("tests/sample_pdfs/sample.pdf")
doc = result.document

for i, item in enumerate(doc.texts[:15]):
    page = item.prov[0].page_no if item.prov else None
    print(f"{i}: label={item.label}, page={page}, text={item.text[:50]}")