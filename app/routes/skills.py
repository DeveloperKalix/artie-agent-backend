"""
Memory-notes endpoints — list / create / delete the free-form notes backing
the ``/skill`` command.

Notes appended in-band via ``/conversations/{id}/messages`` (when a message
opens with ``/skill``) and out-of-band via ``POST /skills`` share the same
persistence path (``MemoryService.append_note``), so the two entry points are
always consistent.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.backend.services.memory import MemoryService, get_memory_service
from app.backend.services.skills import SkillKind, SkillSystem, get_skill_system
from app.routes._deps import require_user_id
from app.schemas.memory import MemoryListResponse, MemoryNote

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/skills", tags=["skills"])


class AppendNoteBody(BaseModel):
    """Request body for ``POST /skills``.

    ``content`` may include the leading ``/skill`` prefix (so clients can POST
    the raw voice transcript without pre-parsing) — the SkillSystem strips it.
    Leaving the prefix off is also accepted and stored verbatim as a note.
    """

    content: str


class AppendNoteResponse(BaseModel):
    note: MemoryNote | None = None
    reply: str
    kind: str


@router.get("", response_model=MemoryListResponse)
async def list_notes(
    user_id: str = Depends(require_user_id),
    limit: int = Query(50, ge=1, le=200),
    memory: MemoryService = Depends(get_memory_service),
) -> MemoryListResponse:
    notes = await memory.list_notes(user_id, limit=limit)
    return MemoryListResponse(notes=notes)


@router.post("", response_model=AppendNoteResponse)
async def append_note(
    body: AppendNoteBody,
    user_id: str = Depends(require_user_id),
    skills: SkillSystem = Depends(get_skill_system),
    memory: MemoryService = Depends(get_memory_service),
) -> AppendNoteResponse:
    """Append a free-form memory note. Accepts ``/skill ...`` or raw text."""
    content = (body.content or "").strip()
    if not content:
        raise HTTPException(status_code=422, detail="content must not be empty")

    # Route through SkillSystem when the user opts into the ``/skill`` syntax
    # so subcommands (``set-level ...``) are honored. Otherwise just append.
    if SkillSystem.detect(content):
        result = await skills.handle(user_id, content)
        return AppendNoteResponse(
            note=result.note,
            reply=result.reply,
            kind=result.kind.value,
        )

    note = await memory.append_note(user_id, content, source="skill")
    return AppendNoteResponse(
        note=note,
        reply=f"Remembered: \u201c{note.content}\u201d",
        kind=SkillKind.note_appended.value,
    )


@router.delete("/{note_id}")
async def delete_note(
    note_id: str,
    user_id: str = Depends(require_user_id),
    memory: MemoryService = Depends(get_memory_service),
) -> dict:
    ok = await memory.delete_note(user_id, note_id)
    if not ok:
        raise HTTPException(status_code=404, detail="note not found")
    return {"deleted": True}
