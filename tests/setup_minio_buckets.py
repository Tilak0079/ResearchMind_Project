#creates all buckets the app needs

import logging

from app.config import settings
from app.utils.minio_client import ensure_bucket_exists

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    ensure_bucket_exists(settings.minio_bucket_raw_pdfs)
    ensure_bucket_exists(settings.minio_bucket_parsed)
    ensure_bucket_exists(settings.minio_bucket_figures)