"""
Cache service for LLM Responses.
"""

import hashlib
import logging
import re
from typing import Optional

import redis

from app.api.schemas import QueryResponse
from app.config import settings
from app.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)

CACHE_VERSION = "v1"

def _normalize_query(query: str) -> str:
    """
    Normalizes the user query to ensure trivial formatting differences
    (like extra whitespace or capitalization) hit the same cache key.
    """
    normalized = re.sub(r'\s+', ' ', query).strip().lower()
    return normalized

def _generate_cache_key(query: str) -> str:
    """
    Generates a deterministic Redis cache key from a normalized user query.
    """
    normalized_query = _normalize_query(query)
    query_hash = hashlib.sha256(normalized_query.encode('utf-8')).hexdigest()
    return f"llm_response:{CACHE_VERSION}:{query_hash}"

def get_cached_response(query: str) -> Optional[QueryResponse]:
    """
    Retrieves the cached QueryResponse for a given query, if it exists.
    Fails gracefully if Redis is unavailable.
    """
    try:
        client = get_redis_client()
        key = _generate_cache_key(query)
        cached_data = client.get(key)
        
        if cached_data:
            logger.info(f"Cache HIT for query: '{query[:50]}...'")
            return QueryResponse.model_validate_json(cached_data)
            
        logger.info(f"Cache MISS for query: '{query[:50]}...'")
        return None
    except redis.RedisError as e:
        logger.error(f"Cache ERROR (get): Failed to connect to Redis: {e}")
        return None
    except Exception as e:
        logger.exception(f"Cache ERROR (get): Unexpected error during deserialization: {e}")
        return None

def set_cached_response(query: str, response: QueryResponse) -> None:
    """
    Caches a successful QueryResponse.
    Fails gracefully if Redis is unavailable.
    """
    if not response.success:
        return

    try:
        client = get_redis_client()
        key = _generate_cache_key(query)
        
        # Serialize response
        json_data = response.model_dump_json()
        
        # Store in Redis with TTL
        client.set(key, json_data, ex=settings.redis_cache_ttl)
        logger.info(f"Cache STORE for query: '{query[:50]}...' (TTL: {settings.redis_cache_ttl}s)")
    except redis.RedisError as e:
        logger.error(f"Cache ERROR (set): Failed to connect to Redis: {e}")
    except Exception as e:
        logger.exception(f"Cache ERROR (set): Unexpected error during serialization: {e}")
