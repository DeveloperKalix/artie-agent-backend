"""
ConversationService — CRUD for conversations + messages and the Redis rolling
context window that agents consult for in-conversation memory.

Storage split:
- Supabase (``public.conversations`` / ``public.messages``) — durable log.
- Redis (``ctx:{conversation_id}``) — last-50-turn hot cache, populated as
  messages are appended and rehydrated from Supabase on cache miss.
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
    insert_row,
    select_maybe,
    select_rows,
    update_rows,
)
from app.schemas.conversations import Conversation, Message, MessageRole

logger = logging.getLogger(__name__)

_CTX_KEY_PREFIX = "ctx:"
_CTX_WINDOW = 50  # turns retained in Redis
_CTX_TTL_SECONDS = 86400  # 24 hours


def _ctx_key(conversation_id: str) -> str:
    return f"{_CTX_KEY_PREFIX}{conversation_id}"


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class ConversationService:
    """All conversation + message state changes funnel through this class."""

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis

    # ---------------------------------------------------------- conversations

    async def list_conversations(self, user_id: str) -> list[Conversation]:
        rows = await asyncio.to_thread(
            select_rows,
            "conversations",
            filters={"user_id": user_id},
            order_column="last_message_at",
            descending=True,
            limit=100,
        )
        return _rows_to_conversations(rows)

    async def create_conversation(
        self, user_id: str, title: str | None = None
    ) -> Conversation:
        now = _now()
        payload = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "title": title,
            "created_at": _iso(now),
            "updated_at": _iso(now),
            "last_message_at": None,
        }
        rows = await asyncio.to_thread(insert_row, "conversations", payload)
        row = rows[0] if rows else payload
        return Conversation.model_validate(row)

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        row = await asyncio.to_thread(
            select_maybe,
            "conversations",
            filters={"id": conversation_id},
        )
        if not row:
            return None
        return Conversation.model_validate(row)

    async def ensure_conversation_belongs_to_user(
        self, conversation_id: str, user_id: str
    ) -> Conversation:
        conv = await self.get_conversation(conversation_id)
        if conv is None:
            raise LookupError("conversation not found")
        if conv.user_id != user_id:
            raise PermissionError("conversation does not belong to user")
        return conv

    # --------------------------------------------------------------- messages

    async def list_messages(
        self, conversation_id: str, limit: int = 50
    ) -> list[Message]:
        rows = await asyncio.to_thread(
            select_rows,
            "messages",
            filters={"conversation_id": conversation_id},
            order_column="created_at",
            descending=False,
            limit=limit,
        )
        return _rows_to_messages(rows)

    async def append_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        *,
        audio_url: str | None = None,
        transcript: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        now = _now()
        payload = {
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "role": role.value if isinstance(role, MessageRole) else str(role),
            "content": content,
            "audio_url": audio_url,
            "transcript": transcript,
            "metadata": metadata,
            "created_at": _iso(now),
        }
        rows = await asyncio.to_thread(insert_row, "messages", payload)
        row = rows[0] if rows else payload
        message = Message.model_validate(row)

        # Bump conversation's last_message_at so list ordering stays fresh.
        try:
            await asyncio.to_thread(
                update_rows,
                "conversations",
                {"last_message_at": _iso(now)},
                filters={"id": conversation_id},
            )
        except Exception:
            logger.warning(
                "[conversations] failed to bump last_message_at for %s",
                conversation_id,
                exc_info=True,
            )

        # Push to Redis rolling window; failure is non-fatal.
        try:
            await self.push_context(conversation_id, message.role.value, message.content)
        except Exception:
            logger.warning(
                "[conversations] failed to push context for %s",
                conversation_id,
                exc_info=True,
            )

        return message

    # ----------------------------------------------------- Redis context API

    async def push_context(
        self, conversation_id: str, role: str, content: str
    ) -> None:
        """LPUSH a turn onto the rolling window and trim to the last N."""
        key = _ctx_key(conversation_id)
        turn = json.dumps({"role": role, "content": content})
        pipe = self._redis.pipeline()
        pipe.lpush(key, turn)
        pipe.ltrim(key, 0, _CTX_WINDOW - 1)
        pipe.expire(key, _CTX_TTL_SECONDS)
        await pipe.execute()

    async def get_context(self, conversation_id: str) -> list[dict[str, Any]]:
        """Return the cached window oldest-first. Rehydrate from Supabase on miss."""
        key = _ctx_key(conversation_id)
        raw = await self._redis.lrange(key, 0, _CTX_WINDOW - 1)
        if raw:
            turns: list[dict[str, Any]] = []
            for encoded in reversed(raw):  # stored newest-first via LPUSH
                try:
                    turns.append(json.loads(encoded))
                except Exception:
                    continue
            return turns

        # Cold cache — rehydrate from Supabase.
        messages = await self.list_messages(conversation_id, limit=_CTX_WINDOW)
        if not messages:
            return []

        turns = [
            {"role": m.role.value, "content": m.content}
            for m in messages
        ]
        # Repopulate Redis so subsequent requests are fast.
        try:
            pipe = self._redis.pipeline()
            pipe.delete(key)
            for t in reversed(turns):  # LPUSH newest-last to preserve order
                pipe.lpush(key, json.dumps(t))
            pipe.ltrim(key, 0, _CTX_WINDOW - 1)
            pipe.expire(key, _CTX_TTL_SECONDS)
            await pipe.execute()
        except Exception:
            logger.debug(
                "[conversations] context rehydrate-write skipped", exc_info=True
            )
        return turns


def _rows_to_conversations(rows: list[dict[str, Any]]) -> list[Conversation]:
    out: list[Conversation] = []
    for row in rows:
        try:
            out.append(Conversation.model_validate(row))
        except Exception:
            logger.debug("[conversations] skipped malformed row: %r", row)
    return out


def _rows_to_messages(rows: list[dict[str, Any]]) -> list[Message]:
    out: list[Message] = []
    for row in rows:
        try:
            out.append(Message.model_validate(row))
        except Exception:
            logger.debug("[conversations] skipped malformed message row: %r", row)
    return out


@lru_cache(maxsize=1)
def get_conversation_service() -> ConversationService:
    return ConversationService(redis=get_redis())


__all__ = [
    "ConversationService",
    "get_conversation_service",
]
