from app.ingestion.parser_docling import converter
import sys
doc = converter.convert("local_dataset/pdfs/1706.03762v7.pdf").document
for pic in doc.pictures:
    if pic.image:
        print(dir(pic.image))
        if hasattr(pic, 'get_image'):
            print("pic has get_image")
        break
