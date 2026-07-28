"""
Qdrant client setup and collection management.
Collection uses BGE-m3's native dense (1024-dim) + sparse vectors,
matching the payload structure in Section 5.2 of the architecture doc.
"""

import logging
from qdrant_client.models import FusionQuery, Fusion, Prefetch, SparseVector

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseVectorParams,
    VectorParams,
)

from app.config import settings

logger = logging.getLogger(__name__)

BGE_M3_DENSE_DIM = 1024  # BAAI/bge-m3 dense embedding dimension


def get_qdrant_client() -> QdrantClient:
    """Returns a Qdrant client using settings from .env."""
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection_exists() -> None:
    """
    Creates the Qdrant collection if it doesn't exist yet, with named
    vectors for both dense and sparse (BGE-m3 native hybrid search, Section 2.3).
    Safe to call every time the app starts — no-op if already created.
    """
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        logger.info(f"Qdrant collection '{collection_name}' already exists.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=BGE_M3_DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )
    logger.info(f"Created Qdrant collection '{collection_name}'.")



def hybrid_search(dense_vector: list[float], sparse_vector: dict, top_k: int = 30) -> list:
    """
    Runs a hybrid search: dense + sparse vectors combined via RRF fusion,
    all done inside Qdrant in one call (Section 2.3).

    Args:
        dense_vector: 1024-dim dense embedding of the query.
        sparse_vector: {"indices": [...], "values": [...]} sparse embedding of the query.
        top_k: how many fused candidates to return (before reranking).

    Returns:
        List of Qdrant scored points, each with .payload (chunk text, metadata) and .score.
    """
    client = get_qdrant_client()

    results = client.query_points(
        collection_name=settings.qdrant_collection_name,
        prefetch=[
            Prefetch(query=dense_vector, using="dense", limit=top_k),
            Prefetch(
                query=SparseVector(indices=sparse_vector["indices"], values=sparse_vector["values"]),
                using="sparse",
                limit=top_k,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    )

    return results.points