"""Pydantic schemas for conversations and messages.

Conversations group messages and carry a user_id; each message has a role
matching the Groq chat-completions convention plus optional voice metadata.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    tool = "tool"


class Message(BaseModel):
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    audio_url: Optional[str] = None
    transcript: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: datetime


class Conversation(BaseModel):
    id: str
    user_id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime] = None


class CreateConversationBody(BaseModel):
    title: Optional[str] = None


class PostMessageBody(BaseModel):
    content: str


class ConversationListResponse(BaseModel):
    conversations: list[Conversation]


class MessageListResponse(BaseModel):
    messages: list[Message]


__all__ = [
    "MessageRole",
    "Message",
    "Conversation",
    "CreateConversationBody",
    "PostMessageBody",
    "ConversationListResponse",
    "MessageListResponse",
]
