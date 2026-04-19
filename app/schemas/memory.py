"""Pydantic schemas for user profile (experience level) and free-form memory notes.

Memory notes are appended by the /skill command and injected into the
recommendation agent's system prompt at generation time.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ExperienceLevel(str, Enum):
    novice = "novice"
    intermediate = "intermediate"
    veteran = "veteran"


class UserProfile(BaseModel):
    user_id: str
    experience_level: ExperienceLevel = ExperienceLevel.novice
    onboarded_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class MemoryNote(BaseModel):
    id: str
    user_id: str
    content: str
    source: str
    created_at: datetime


class MemoryListResponse(BaseModel):
    notes: list[MemoryNote]


class UpdateProfileBody(BaseModel):
    experience_level: ExperienceLevel


__all__ = [
    "ExperienceLevel",
    "UserProfile",
    "MemoryNote",
    "MemoryListResponse",
    "UpdateProfileBody",
]
