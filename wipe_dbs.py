from app.db.postgres import SessionLocal
from app.db.models import PaperRegistry, ChunkRegistry
from app.retrieval.qdrant_client import get_qdrant_client
from app.config import settings

# 1. Clear Postgres
db = SessionLocal()
db.query(ChunkRegistry).delete()
db.query(PaperRegistry).delete()
db.commit()
db.close()
print("Postgres cleared.")

# 2. Clear Qdrant
qclient = get_qdrant_client()
try:
    qclient.delete_collection(settings.qdrant_collection_name)
    print("Qdrant collection deleted.")
except Exception as e:
    print(f"Qdrant collection delete error (maybe already gone): {e}")

