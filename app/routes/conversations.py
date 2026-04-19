"""Conversations + messages endpoints.

Text and voice messages share the same processing pipeline:

1. Validate the conversation ownership (``X-User-Id`` must match).
2. For voice: save the upload to a temp file and transcribe via Groq Whisper.
3. Hand the user text off to :class:`AgentOrchestrator`, which persists both
   the user turn and the assistant reply and returns the assistant message.
4. Return the assistant message — the frontend can render it immediately.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)

from app.backend.services.agent_orchestrator import (
    AgentOrchestrator,
    get_agent_orchestrator,
)
from app.backend.services.conversations import (
    ConversationService,
    get_conversation_service,
)
from app.routes._deps import require_user_id
from app.schemas.conversations import (
    Conversation,
    ConversationListResponse,
    CreateConversationBody,
    Message,
    MessageListResponse,
    PostMessageBody,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])


async def _ensure_owner(
    svc: ConversationService, conversation_id: str, user_id: str
) -> None:
    """Translate service-level errors into the right HTTP status codes."""
    try:
        await svc.ensure_conversation_belongs_to_user(conversation_id, user_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    user_id: str = Depends(require_user_id),
    svc: ConversationService = Depends(get_conversation_service),
) -> ConversationListResponse:
    conversations = await svc.list_conversations(user_id)
    return ConversationListResponse(conversations=conversations)


@router.post("", response_model=Conversation)
async def create_conversation(
    body: CreateConversationBody,
    user_id: str = Depends(require_user_id),
    svc: ConversationService = Depends(get_conversation_service),
) -> Conversation:
    return await svc.create_conversation(user_id=user_id, title=body.title)


@router.get("/{conversation_id}/messages", response_model=MessageListResponse)
async def list_messages(
    conversation_id: str,
    user_id: str = Depends(require_user_id),
    limit: int = Query(50, ge=1, le=500),
    svc: ConversationService = Depends(get_conversation_service),
) -> MessageListResponse:
    await _ensure_owner(svc, conversation_id, user_id)
    messages = await svc.list_messages(conversation_id, limit=limit)
    return MessageListResponse(messages=messages)


@router.post("/{conversation_id}/messages", response_model=Message)
async def post_text_message(
    conversation_id: str,
    body: PostMessageBody,
    user_id: str = Depends(require_user_id),
    svc: ConversationService = Depends(get_conversation_service),
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
) -> Message:
    """Append a user text message and return the assistant reply."""
    content = (body.content or "").strip()
    if not content:
        raise HTTPException(status_code=422, detail="content must not be empty")

    await _ensure_owner(svc, conversation_id, user_id)

    output = await orchestrator.handle_user_message(
        user_id=user_id,
        conversation_id=conversation_id,
        content=content,
        user_metadata={"input": "text"},
    )
    return output.assistant_message


@router.post("/{conversation_id}/messages/voice", response_model=Message)
async def post_voice_message(
    conversation_id: str,
    file: UploadFile = File(...),
    user_id: str = Depends(require_user_id),
    svc: ConversationService = Depends(get_conversation_service),
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
) -> Message:
    """Accept an audio upload, transcribe via Groq Whisper, then run the
    orchestrator on the transcript just like a text message."""
    await _ensure_owner(svc, conversation_id, user_id)

    # Lazy import keeps this module free of a hard dependency on app.main.
    from app.main import groq_client

    suffix = os.path.splitext(file.filename or "")[1] or ".m4a"
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        with open(tmp_path, "rb") as audio:
            transcription = groq_client.audio.transcriptions.create(
                file=(os.path.basename(tmp_path), audio.read()),
                model="whisper-large-v3-turbo",
            )
    except Exception as e:
        logger.exception("[conversations] voice transcription failed")
        raise HTTPException(status_code=502, detail=f"transcription failed: {e}") from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                logger.debug(
                    "[conversations] could not remove temp file %s", tmp_path
                )

    transcript_text = (getattr(transcription, "text", None) or "").strip()
    if not transcript_text:
        raise HTTPException(
            status_code=422, detail="transcription returned empty text"
        )

    output = await orchestrator.handle_user_message(
        user_id=user_id,
        conversation_id=conversation_id,
        content=transcript_text,
        user_metadata={"input": "voice", "transcript": transcript_text},
    )
    return output.assistant_message
