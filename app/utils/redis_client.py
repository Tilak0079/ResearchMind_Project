"""Redis client wrapper — used later for rate limiting, caching, and Celery queue."""

import redis

from app.config import settings


def get_redis_client() -> redis.Redis:
    """Returns a Redis client using settings from .env."""
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=True,  # returns str instead of bytes — easier to work with
    )