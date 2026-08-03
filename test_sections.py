from app.retrieval.qdrant_client import get_qdrant_client
from app.config import settings

qdrant = get_qdrant_client()
res = qdrant.scroll(
    collection_name=settings.qdrant_collection_name,
    limit=5,
    with_payload=True
)
for point in res[0]:
    print("Section Name:", point.payload.get("section_name"))
