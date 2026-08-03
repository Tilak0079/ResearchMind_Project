from app.ingestion.parser_docling import converter
doc = converter.convert("local_dataset/pdfs/1706.03762v7.pdf").document
for pic in doc.pictures:
    if pic.captions:
        print("Caption RefItem attributes:", dir(pic.captions[0]))
        break
