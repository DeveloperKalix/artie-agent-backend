"""
Profile endpoints — inspect and update the caller's experience level.

The experience level is read by :class:`RecommendationAgent` to tailor tone
and depth of advice (novice → plain language, veteran → advanced terminology).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.backend.services.memory import MemoryService, get_memory_service
from app.routes._deps import require_user_id
from app.schemas.memory import UpdateProfileBody, UserProfile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("", response_model=UserProfile)
async def get_profile(
    user_id: str = Depends(require_user_id),
    memory: MemoryService = Depends(get_memory_service),
) -> UserProfile:
    return await memory.get_profile(user_id)


@router.patch("", response_model=UserProfile)
async def patch_profile(
    body: UpdateProfileBody,
    user_id: str = Depends(require_user_id),
    memory: MemoryService = Depends(get_memory_service),
) -> UserProfile:
    return await memory.set_experience_level(user_id, body.experience_level)
