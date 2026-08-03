from docling.document_converter import DocumentConverter
converter = DocumentConverter()
res = converter.convert("test.pdf") if __import__("os").path.exists("test.pdf") else None
if res:
    labels = set(item.label for item in res.document.texts)
    print("Labels found in texts:", labels)
