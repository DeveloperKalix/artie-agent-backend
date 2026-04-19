"""Async Redis client singleton.

Used for hot state: conversation context windows, news dedup sets,
scheduler distributed locks, and free-form memory caches.
"""
from __future__ import annotations

import os
from functools import lru_cache

import redis.asyncio as aioredis


@lru_cache(maxsize=1)
def get_redis() -> aioredis.Redis:
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return aioredis.Redis.from_url(url, decode_responses=True)


__all__ = ["get_redis"]
