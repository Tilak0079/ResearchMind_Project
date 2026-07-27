"""
Quick manual check: confirms MinIO and Redis are reachable and working.
Not a pytest unit test yet (that starts later) — just a sanity script.
"""

import logging

from app.utils.minio_client import get_minio_client
from app.utils.redis_client import get_redis_client
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_minio() -> None:
    client = get_minio_client()
    response = client.list_buckets()
    bucket_names = [b["Name"] for b in response["Buckets"]]
    logger.info(f"MinIO buckets found: {bucket_names}")
    assert settings.minio_bucket_raw_pdfs in bucket_names, "raw-pdfs bucket missing!"


def test_redis() -> None:
    client = get_redis_client()
    client.set("phase3_test_key", "hello")
    value = client.get("phase3_test_key")
    logger.info(f"Redis test value: {value}")
    assert value == "hello", "Redis read/write failed!"


if __name__ == "__main__":
    test_minio()
    test_redis()
    logger.info("Phase 3: MinIO and Redis both working ✅")