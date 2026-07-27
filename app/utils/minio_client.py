
#MinIO (S3-compatible) client wrapper.
#Single place that creates the boto3 client so every module uses the same config instead of repeating connection setup.


import logging

import boto3
from botocore.exceptions import ClientError

from app.config import settings

logger = logging.getLogger(__name__)


def get_minio_client():
    """Returns a boto3 S3 client pointed at our local MinIO instance."""
    return boto3.client(
        "s3",
        endpoint_url=f"http://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
    )


def ensure_bucket_exists(bucket_name: str) -> None:
    """
    Creates the bucket if it doesn't already exist.
  
    """
    client = get_minio_client()
    try:
        client.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' already exists.")
    except ClientError:
        client.create_bucket(Bucket=bucket_name)
        logger.info(f"Created bucket '{bucket_name}'.")