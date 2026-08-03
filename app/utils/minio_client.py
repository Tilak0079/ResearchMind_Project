
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

        
def upload_pdf_to_minio(local_file_path: str, object_name: str) -> str:
    """
    Uploads a local PDF file to MinIO's raw-pdfs bucket.

    Args:
        local_file_path: path to the PDF currently on local disk.
        object_name: the filename to use in MinIO (e.g. "1706.03762v7.pdf").

    Returns:
        The MinIO object path in "bucket/object_name" format - this is
        what gets stored in paper_registry.raw_pdf_s3_path.

    Raises:
        ClientError: if the upload fails (bubbled up to caller).
    """
    client = get_minio_client()
    bucket = settings.minio_bucket_raw_pdfs

    client.upload_file(local_file_path, bucket, object_name)
    logger.info(f"Uploaded {local_file_path} to MinIO: {bucket}/{object_name}")

    return f"{bucket}/{object_name}"

def upload_pdf(local_file_path: str, object_name: str) -> str:
    """
    Uploads a local PDF file to MinIO's raw-pdfs bucket.

    Args:
        local_file_path: path to the PDF currently sitting on local disk.
        object_name: the filename to store it under in MinIO (e.g. "1706.03762v7.pdf").

    Returns:
        A MinIO-style path string (bucket/object_name) - this is what we'll
        store in paper_registry.raw_pdf_s3_path, replacing local disk paths.

    Raises:
        FileNotFoundError: if the local file doesn't exist.
        ClientError: if the upload fails (bubbled up from boto3).
    """
    from pathlib import Path
    if not Path(local_file_path).exists():
        raise FileNotFoundError(f"Cannot upload, file not found: {local_file_path}")

    client = get_minio_client()
    bucket = settings.minio_bucket_raw_pdfs

    client.upload_file(local_file_path, bucket, object_name)
    logger.info(f"Uploaded {local_file_path} -> {bucket}/{object_name}")

    return f"{bucket}/{object_name}"

def upload_bytes_to_minio(data: bytes, object_name: str, bucket_name: str, content_type: str = "application/octet-stream") -> str:
    """
    Uploads raw bytes to MinIO.
    """
    import io
    ensure_bucket_exists(bucket_name)
    client = get_minio_client()
    
    client.upload_fileobj(
        io.BytesIO(data), 
        bucket_name, 
        object_name,
        ExtraArgs={"ContentType": content_type}
    )
    logger.info(f"Uploaded bytes -> {bucket_name}/{object_name}")
    return f"{bucket_name}/{object_name}"

def get_presigned_url(bucket_name: str, object_name: str, expires_in: int = 3600) -> str:
    """
    Generates a pre-signed URL for an object.
    """
    client = get_minio_client()
    try:
        url = client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_name},
            ExpiresIn=expires_in
        )
        return url
    except ClientError as e:
        logger.error(f"Failed to generate presigned URL for {bucket_name}/{object_name}: {e}")
        return ""        