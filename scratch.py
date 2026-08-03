from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.generate_picture_images = True
pipeline_options.generate_table_images = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
import sys
if len(sys.argv) > 1:
    res = converter.convert(sys.argv[1])
    doc = res.document
    print("Pictures:", len(list(doc.pictures)))
    print("Tables:", len(list(doc.tables)))
    for p in doc.pictures:
        print("Pic caption:", p.captions[0].text if p.captions else "None")
    for t in doc.tables:
        print("Table md:", t.export_to_markdown()[:50])
