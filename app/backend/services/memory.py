"""
MemoryService — user profile (experience level) + free-form memory notes.

Two stores:
- Supabase is the durable source of truth.
    * ``public.user_profiles`` — one row per user with ``experience_level``.
    * ``public.user_memory`` — append-only notes captured via ``/skill``.
- Redis is a prompt-context cache keyed by ``mem:{user_id}`` that holds a
  pre-serialized bullet list of the user's most recent notes. The cache is
  invalidated on every write so recommendation prompts never miss fresh
  context.

The RecommendationAgent depends on this service for two things only:

1. ``get_profile_with_notes(user_id)`` — returns ``(experience_level, notes[])``
   in a single call so the agent can enrich its Groq system prompt.
2. ``append_note(user_id, content, ...)`` — persists a new note and invalidates
   the cache.

Everything else on the class is exposed for the routes layer (list/delete
notes, update profile) and kept intentionally small so the service stays a
single, well-scoped dependency.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import redis.asyncio as aioredis

from app.backend.core.redis import get_redis
from app.backend.core.supabase import (
    delete_rows,
    insert_row,
    select_maybe,
    select_rows,
    upsert_row,
)
from app.schemas.memory import ExperienceLevel, MemoryNote, UserProfile

logger = logging.getLogger(__name__)

_PROFILE_TABLE = "user_profiles"
_MEMORY_TABLE = "user_memory"

# Cache keys and limits.
_MEM_KEY_PREFIX = "mem:"
_MEM_CACHE_TTL_SECONDS = 3600  # 1 hour — cheap to re-hydrate from Supabase.
_MAX_NOTES_FOR_PROMPT = 20  # most recent notes injected into prompts.
_MAX_NOTE_LEN = 2_000  # sanity bound on single-note size (chars).


def _mem_key(user_id: str) -> str:
    return f"{_MEM_KEY_PREFIX}{user_id}"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class MemoryService:
    """Profile + notes CRUD with a Redis read-through cache."""

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis

    # ------------------------------------------------------------- profile API

    async def get_profile(self, user_id: str) -> UserProfile:
        """Return the user's profile, creating a ``novice`` default if absent.

        This is a get-or-create so callers can always assume a row exists and
        never have to branch on ``None``.
        """
        row = await asyncio.to_thread(
            select_maybe, _PROFILE_TABLE, filters={"user_id": user_id}
        )
        if row:
            return UserProfile.model_validate(row)

        default = {
            "user_id": user_id,
            "experience_level": ExperienceLevel.novice.value,
        }
        rows = await asyncio.to_thread(
            upsert_row, _PROFILE_TABLE, default, on_conflict="user_id"
        )
        return UserProfile.model_validate(rows[0] if rows else default)

    async def set_experience_level(
        self, user_id: str, level: ExperienceLevel
    ) -> UserProfile:
        payload = {
            "user_id": user_id,
            "experience_level": level.value,
        }
        rows = await asyncio.to_thread(
            upsert_row, _PROFILE_TABLE, payload, on_conflict="user_id"
        )
        # Experience level feeds the agent prompt, so bust the prompt cache too.
        await self._invalidate(user_id)
        return UserProfile.model_validate(rows[0] if rows else payload)

    # -------------------------------------------------------------- memory API

    async def list_notes(
        self, user_id: str, limit: int = _MAX_NOTES_FOR_PROMPT
    ) -> list[MemoryNote]:
        rows = await asyncio.to_thread(
            select_rows,
            _MEMORY_TABLE,
            filters={"user_id": user_id},
            order_column="created_at",
            descending=True,
            limit=limit,
        )
        return _rows_to_notes(rows)

    async def append_note(
        self,
        user_id: str,
        content: str,
        *,
        source: str = "skill",
    ) -> MemoryNote:
        trimmed = (content or "").strip()
        if not trimmed:
            raise ValueError("memory note content must not be empty")
        if len(trimmed) > _MAX_NOTE_LEN:
            trimmed = trimmed[:_MAX_NOTE_LEN]

        payload = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "content": trimmed,
            "source": source,
            "created_at": _now_iso(),
        }
        rows = await asyncio.to_thread(insert_row, _MEMORY_TABLE, payload)
        note = MemoryNote.model_validate(rows[0] if rows else payload)
        await self._invalidate(user_id)
        return note

    async def delete_note(self, user_id: str, note_id: str) -> bool:
        """Delete a single note. Scoped by user_id so a client can't delete
        someone else's note by guessing the id."""
        deleted = await asyncio.to_thread(
            delete_rows,
            _MEMORY_TABLE,
            filters={"id": note_id, "user_id": user_id},
        )
        if deleted:
            await self._invalidate(user_id)
        return bool(deleted)

    # --------------------------------------------------------- prompt helpers

    async def get_profile_with_notes(
        self, user_id: str, *, limit: int = _MAX_NOTES_FOR_PROMPT
    ) -> tuple[UserProfile, list[MemoryNote]]:
        """One-shot read used by RecommendationAgent before building a prompt."""
        profile_task = asyncio.create_task(self.get_profile(user_id))
        notes_task = asyncio.create_task(self.list_notes(user_id, limit=limit))
        profile, notes = await asyncio.gather(profile_task, notes_task)
        return profile, notes

    async def get_memory_prompt(self, user_id: str) -> str:
        """Cached, pre-formatted bullet list for prompt injection.

        Returns an empty string if the user has no notes. The cache eats the
        O(n) query cost on every agent invocation; invalidation on write keeps
        it consistent.
        """
        key = _mem_key(user_id)
        try:
            cached = await self._redis.get(key)
        except Exception:
            logger.debug("[memory] redis GET failed, falling through", exc_info=True)
            cached = None

        if cached is not None:
            try:
                return json.loads(cached)
            except Exception:
                logger.debug(
                    "[memory] stale cache value at %s, rebuilding", key, exc_info=True
                )

        notes = await self.list_notes(user_id)
        formatted = _format_notes_for_prompt(notes)
        try:
            await self._redis.set(key, json.dumps(formatted), ex=_MEM_CACHE_TTL_SECONDS)
        except Exception:
            logger.debug("[memory] redis SET failed", exc_info=True)
        return formatted

    # ------------------------------------------------------------ cache utils

    async def _invalidate(self, user_id: str) -> None:
        try:
            await self._redis.delete(_mem_key(user_id))
        except Exception:
            logger.debug(
                "[memory] cache invalidation failed for %s", user_id, exc_info=True
            )


# --------------------------------------------------------------------- helpers


def _rows_to_notes(rows: list[dict[str, Any]]) -> list[MemoryNote]:
    out: list[MemoryNote] = []
    for row in rows:
        try:
            out.append(MemoryNote.model_validate(row))
        except Exception:
            logger.debug("[memory] skipped malformed note row: %r", row)
    return out


def _format_notes_for_prompt(notes: list[MemoryNote]) -> str:
    """Newest-first bullet list that slots directly into a system prompt."""
    if not notes:
        return ""
    bullets = [f"- {n.content}" for n in notes]
    return "\n".join(bullets)


@lru_cache(maxsize=1)
def get_memory_service() -> MemoryService:
    return MemoryService(redis=get_redis())


__all__ = ["MemoryService", "get_memory_service"]
