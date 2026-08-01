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

SESSION_TOKEN_QUOTA = 10000  # max tokens a session can use before being rate-limited
QUOTA_WINDOW_SECONDS = 3600  # quota resets after 1 hour (rolling window, simple TTL-based)


def check_and_increment_token_quota(session_id: str, tokens_to_add: int) -> tuple[bool, int]:
    """
    Checks whether a session is within its token quota, and if so, adds
    this request's tokens to their running total.

    Uses Redis's TTL (time-to-live) feature: the counter key auto-expires
    after QUOTA_WINDOW_SECONDS, giving us a simple rolling quota without
    needing to manually track/reset timestamps ourselves.

    Returns:
        (is_within_quota, current_total) - is_within_quota=False means
        this request should be rejected.
    """
    client = get_redis_client()
    key = f"token_quota:{session_id}"

    current_total = client.get(key)
    current_total = int(current_total) if current_total else 0

    if current_total + tokens_to_add > SESSION_TOKEN_QUOTA:
        return False, current_total

    new_total = client.incrby(key, tokens_to_add)
    # Only set expiry on first write, so the window doesn't keep resetting
    # every request (which would let a session keep quota forever).
    if current_total == 0:
        client.expire(key, QUOTA_WINDOW_SECONDS)

    return True, new_total