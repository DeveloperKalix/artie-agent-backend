"""
AgentOrchestrator — single entry point for an incoming user message.

Responsibilities
----------------
1. **Route**: if the message opens with ``/skill``, hand off to
   :class:`SkillSystem`. Otherwise ask :class:`RecommendationAgent` for a
   conversational reply.
2. **Persist**: always append the user message first (so it's durable even if
   the agent layer crashes), then append the assistant reply on success, or a
   graceful fallback message on failure.
3. **Carry metadata**: each turn is tagged with ``{"source": ...}`` so the
   frontend can render skill acknowledgements, recommendation stubs, or plain
   chat replies with distinct UI affordances.

The orchestrator is the *only* place the conversation routes touch higher-
level agents — this keeps the HTTP layer thin and lets us swap agent
implementations without rewriting the routes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.backend.services.conversations import (
    ConversationService,
    get_conversation_service,
)
from app.backend.services.recommendations import (
    RecommendationAgent,
    get_recommendation_agent,
)
from app.backend.services.skills import (
    SkillKind,
    SkillResult,
    SkillSystem,
    get_skill_system,
)
from app.schemas.conversations import Message, MessageRole

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OrchestratorOutput:
    """What the HTTP layer returns to the client."""

    user_message: Message
    assistant_message: Message


class AgentOrchestrator:
    def __init__(
        self,
        *,
        conversations: ConversationService,
        skills: SkillSystem,
        recommender: RecommendationAgent,
    ) -> None:
        self._conversations = conversations
        self._skills = skills
        self._recommender = recommender

    async def handle_user_message(
        self,
        *,
        user_id: str,
        conversation_id: str,
        content: str,
        user_metadata: dict[str, Any] | None = None,
    ) -> OrchestratorOutput:
        """Persist a user turn, run the right agent, persist the assistant turn."""
        user_message = await self._conversations.append_message(
            conversation_id=conversation_id,
            role=MessageRole.user,
            content=content,
            metadata=user_metadata,
            transcript=(user_metadata or {}).get("transcript") if user_metadata else None,
        )

        try:
            if SkillSystem.detect(content):
                assistant_message = await self._handle_skill(
                    user_id, conversation_id, content
                )
            else:
                assistant_message = await self._handle_chat(
                    user_id, conversation_id, content
                )
        except Exception:
            logger.exception(
                "[orchestrator] agent pipeline failed for conversation=%s",
                conversation_id,
            )
            assistant_message = await self._conversations.append_message(
                conversation_id=conversation_id,
                role=MessageRole.assistant,
                content=(
                    "Sorry — I ran into an error generating a reply. "
                    "Please try again in a moment."
                ),
                metadata={"source": "error"},
            )

        return OrchestratorOutput(
            user_message=user_message,
            assistant_message=assistant_message,
        )

    # --------------------------------------------------------- route: /skill

    async def _handle_skill(
        self, user_id: str, conversation_id: str, content: str
    ) -> Message:
        result: SkillResult = await self._skills.handle(user_id, content)
        metadata: dict[str, Any] = {"source": "skill", "kind": result.kind.value}
        if result.kind == SkillKind.note_appended and result.note is not None:
            metadata["note_id"] = result.note.id
        if result.kind == SkillKind.level_updated and result.profile is not None:
            metadata["experience_level"] = result.profile.experience_level.value
        return await self._conversations.append_message(
            conversation_id=conversation_id,
            role=MessageRole.assistant,
            content=result.reply,
            metadata=metadata,
        )

    # ---------------------------------------------------------- route: chat

    async def _handle_chat(
        self, user_id: str, conversation_id: str, content: str
    ) -> Message:
        # Pull the rolling context *excluding* the turn we just persisted — we
        # pass ``content`` as the final user turn separately, so we want the
        # prior turns here. Redis has the fresh turn on top, so drop that.
        history = await self._conversations.get_context(conversation_id)
        if history and history[-1].get("content") == content:
            history = history[:-1]

        reply_text = await self._recommender.chat(
            user_id=user_id,
            message=content,
            history=history,
        )
        if not reply_text:
            reply_text = (
                "I wasn't able to come up with a useful reply for that. "
                "Try asking me about a specific position or news item."
            )
        return await self._conversations.append_message(
            conversation_id=conversation_id,
            role=MessageRole.assistant,
            content=reply_text,
            metadata={"source": "chat"},
        )


_SINGLETON: AgentOrchestrator | None = None


def get_agent_orchestrator() -> AgentOrchestrator:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = AgentOrchestrator(
            conversations=get_conversation_service(),
            skills=get_skill_system(),
            recommender=get_recommendation_agent(),
        )
    return _SINGLETON


__all__ = [
    "AgentOrchestrator",
    "OrchestratorOutput",
    "get_agent_orchestrator",
]
