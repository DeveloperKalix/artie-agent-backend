"""Pydantic schemas for user profile (experience level) and free-form memory notes.

Memory notes are appended by the /skill command and injected into the
recommendation agent's system prompt at generation time.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ExperienceLevel(str, Enum):
    novice = "novice"
    intermediate = "intermediate"
    veteran = "veteran"


class TradingConfig(BaseModel):
    """Per-user trading risk + UX config, stored as ``user_profiles.trading``.

    Hard-enforced by ``OrderIntentService`` on every confirm. UI exposes the
    same fields in settings so users can tighten/loosen their own caps.
    """

    enabled: bool = False
    max_order_usd: float = Field(1000.0, gt=0)
    max_daily_usd: float = Field(5000.0, gt=0)
    llm_proposals_enabled: bool = True
    disclaimer_acknowledged_at: Optional[datetime] = None


class UserProfile(BaseModel):
    user_id: str
    experience_level: ExperienceLevel = ExperienceLevel.novice
    onboarded_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    trading: TradingConfig = Field(default_factory=TradingConfig)


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
    "TradingConfig",
    "UserProfile",
    "MemoryNote",
    "MemoryListResponse",
    "UpdateProfileBody",
]
