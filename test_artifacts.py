import os
import sys

from app.retrieval.qdrant_client import get_qdrant_client
from app.config import settings

qdrant = get_qdrant_client()

# Fetch chunks with artifacts
res = qdrant.scroll(
    collection_name=settings.qdrant_collection_name,
    scroll_filter=None,
    limit=1000,
    with_payload=True
)

found_artifacts = 0
for point in res[0]:
    payload = point.payload
    if payload.get("artifact_path"):
        print(f"Type: {payload.get('chunk_type')}")
        print(f"Artifact Path: {payload.get('artifact_path')}")
        print(f"Content preview: {payload.get('text')[:100]}")
        print("-" * 50)
        found_artifacts += 1

if found_artifacts == 0:
    print("NO ARTIFACTS FOUND IN QDRANT.")
else:
    print(f"Total Artifacts Found: {found_artifacts}")
