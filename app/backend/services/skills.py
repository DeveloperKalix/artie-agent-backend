"""
SkillSystem — parses ``/skill`` commands from user messages.

A "skill" is a free-form note the user wants Artie to remember. The syntax is
intentionally tiny so users can speak it naturally:

    /skill I'm worried about my TSLA position
    /skill prefer dividend stocks over growth
    /skill set-level veteran           # also bumps experience level

Design notes
------------
- The system is *intent-aware*: ``/skill set-level <novice|intermediate|veteran>``
  is recognized as a structured subcommand that updates the profile in addition
  to persisting the note. Every other body is stored verbatim as a memory note.
- Detection is cheap and deterministic (a regex on the first token). The
  recommendation agent is the expensive path; routing here is pure string
  handling so the orchestrator can short-circuit without invoking the LLM.
- A ``SkillResult`` is returned rather than raw side effects so the caller
  (AgentOrchestrator) decides what to say back to the user and how to label
  the assistant message.

This module is intentionally dependency-light — it holds a reference to
:class:`MemoryService` and does nothing else. That makes it trivial to unit
test without Redis or Supabase.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from app.backend.services.memory import MemoryService, get_memory_service
from app.schemas.memory import ExperienceLevel, MemoryNote, UserProfile

logger = logging.getLogger(__name__)

_SKILL_PREFIX_RE = re.compile(r"^\s*/skill\b\s*(.*)$", re.IGNORECASE | re.DOTALL)
_SET_LEVEL_RE = re.compile(
    r"^\s*set[-_ ]level\s+(novice|intermediate|veteran)\s*$",
    re.IGNORECASE,
)


class SkillKind(str, Enum):
    """What the system actually did with the message."""

    note_appended = "note_appended"
    level_updated = "level_updated"
    invalid = "invalid"


@dataclass(slots=True)
class SkillResult:
    """Structured response from ``SkillSystem.handle``.

    ``reply`` is a user-visible confirmation string the orchestrator will
    persist as the assistant's reply for this turn. ``note`` / ``profile`` are
    the concrete objects that were written, returned so downstream consumers
    (e.g. websocket push, analytics) don't need a second round trip.
    """

    kind: SkillKind
    reply: str
    note: MemoryNote | None = None
    profile: UserProfile | None = None


class SkillSystem:
    def __init__(self, memory: MemoryService) -> None:
        self._memory = memory

    @staticmethod
    def detect(content: str) -> bool:
        """True if the message opens with ``/skill``."""
        return bool(_SKILL_PREFIX_RE.match(content or ""))

    @staticmethod
    def extract_body(content: str) -> str | None:
        """Return everything after ``/skill``, or ``None`` if not a skill command."""
        match = _SKILL_PREFIX_RE.match(content or "")
        if not match:
            return None
        return match.group(1).strip()

    async def handle(self, user_id: str, content: str) -> SkillResult:
        """Dispatch a ``/skill ...`` message to the right subhandler.

        Raises nothing — malformed input yields a ``SkillKind.invalid`` result
        so the conversation layer can gracefully echo an error message back
        to the user instead of 500-ing.
        """
        body = self.extract_body(content)
        if body is None:
            return SkillResult(
                kind=SkillKind.invalid,
                reply="That wasn't a /skill command.",
            )

        if not body:
            return SkillResult(
                kind=SkillKind.invalid,
                reply=(
                    "Usage: `/skill <note>` to remember something, or "
                    "`/skill set-level novice|intermediate|veteran` to set your "
                    "experience level."
                ),
            )

        # Structured subcommand: set-level.
        level_match = _SET_LEVEL_RE.match(body)
        if level_match:
            level = ExperienceLevel(level_match.group(1).lower())
            profile = await self._memory.set_experience_level(user_id, level)
            logger.info(
                "[skills] user=%s set experience_level=%s", user_id, level.value
            )
            return SkillResult(
                kind=SkillKind.level_updated,
                reply=f"Got it — I'll tailor my advice to a {level.value} investor.",
                profile=profile,
            )

        # Default: append a free-form memory note.
        try:
            note = await self._memory.append_note(user_id, body, source="skill")
        except ValueError as e:
            return SkillResult(kind=SkillKind.invalid, reply=str(e))

        logger.info(
            "[skills] user=%s appended note id=%s len=%d",
            user_id,
            note.id,
            len(note.content),
        )
        return SkillResult(
            kind=SkillKind.note_appended,
            reply=f"Got it — I'll remember: \u201c{note.content}\u201d",
            note=note,
        )


_SINGLETON: SkillSystem | None = None


def get_skill_system() -> SkillSystem:
    """Singleton; safe because MemoryService is a singleton as well."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = SkillSystem(memory=get_memory_service())
    return _SINGLETON


__all__ = ["SkillKind", "SkillResult", "SkillSystem", "get_skill_system"]
