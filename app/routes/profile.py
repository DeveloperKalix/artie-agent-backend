"""
Profile endpoints — inspect and update the caller's experience level
and per-user trading configuration.

The experience level is read by :class:`RecommendationAgent` to tailor
tone and depth of advice (novice → plain language, veteran → advanced
terminology). The trading block gates all ``/trade/*`` calls — see
``app/backend/services/order_intents.py`` for the enforcement logic.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.backend.services.memory import MemoryService, get_memory_service
from app.routes._deps import require_user_id
from app.schemas.memory import TradingConfig, UpdateProfileBody, UserProfile

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


class UpdateTradingBody(BaseModel):
    """Partial update for ``user_profiles.trading``. All fields optional.

    Server-side ``patch_trading_config`` validates the merged result
    against :class:`TradingConfig` — e.g. a negative ``max_order_usd`` is
    rejected before persistence.
    """

    enabled: bool | None = None
    max_order_usd: float | None = Field(None, gt=0)
    max_daily_usd: float | None = Field(None, gt=0)
    llm_proposals_enabled: bool | None = None


@router.get("/trading", response_model=TradingConfig)
async def get_trading_config(
    user_id: str = Depends(require_user_id),
    memory: MemoryService = Depends(get_memory_service),
) -> TradingConfig:
    profile = await memory.get_profile(user_id)
    return profile.trading


@router.patch("/trading", response_model=UserProfile)
async def patch_trading_config(
    body: UpdateTradingBody,
    user_id: str = Depends(require_user_id),
    memory: MemoryService = Depends(get_memory_service),
) -> UserProfile:
    patch = body.model_dump(exclude_none=True)
    return await memory.patch_trading_config(user_id, patch)


@router.post("/trading/disclaimer", response_model=UserProfile)
async def acknowledge_disclaimer(
    user_id: str = Depends(require_user_id),
    memory: MemoryService = Depends(get_memory_service),
) -> UserProfile:
    """Mark the trading disclaimer as acknowledged for this user.

    Until this is called, every ``/trade/*`` state-change endpoint returns
    403 regardless of the ``enabled`` flag.
    """
    return await memory.acknowledge_trading_disclaimer(user_id)
