from app.ingestion.parser_docling import parse_pdf
from app.config import settings

# Force offline mode for BGE-m3 if we end up importing embedder accidentally
import os
os.environ["HF_HUB_OFFLINE"] = "1"

try:
    doc = parse_pdf("local_dataset/pdfs/1706.03762v7.pdf", "test_paper_id")
    print(f"Success! Found {len(doc.artifacts)} artifacts.")
except Exception as e:
    import traceback
    traceback.print_exc()
