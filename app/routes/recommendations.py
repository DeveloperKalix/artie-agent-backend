"""
Recommendations endpoints.

``GET  /recommendations``            — list ``new`` + ``viewed`` foresights.
``GET  /recommendations/status``     — cheap badge check (no LLM, no cursor advance).
``POST /recommendations``            — generate + persist foresights (calls Groq, advances cursor).
``POST /recommendations/{id}/viewed`` — mark a foresight as seen.

The status endpoint is designed to be called on every app-foreground event so
the frontend can show a notification badge without burning Groq tokens. Only
call the POST endpoint when the user explicitly requests new foresights.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.backend.services.recommendations import (
    RecommendationAgent,
    get_recommendation_agent,
)
from app.routes._deps import require_user_id
from app.schemas.news import (
    RecommendationResponse,
    RecommendationsListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class RecommendationStatusResponse(BaseModel):
    """Signal shape for the frontend badge.

    ``new_count`` is the primary badge number (count of stored, unviewed
    foresights). ``pending_news_count`` tells the frontend whether a fresh
    ``POST /recommendations`` call would likely yield anything — useful for
    a secondary "refresh" affordance that doesn't mislead the user when the
    LLM already produced every foresight the current news supports.
    """

    has_new: bool
    new_count: int
    pending_news_count: int
    last_seen_at: str | None
    tickers: list[str]


class MarkViewedResponse(BaseModel):
    marked: bool


@router.get("/status", response_model=RecommendationStatusResponse)
async def get_recommendation_status(
    user_id: str = Depends(require_user_id),
    agent: RecommendationAgent = Depends(get_recommendation_agent),
) -> RecommendationStatusResponse:
    """Return the per-user foresight badge state."""
    try:
        result: dict[str, Any] = await agent.check_for_new_news(user_id)
        return RecommendationStatusResponse(**result)
    except Exception as e:
        logger.exception("[recommendations] status check failed for user=%s", user_id)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("", response_model=RecommendationsListResponse)
async def list_recommendations(
    user_id: str = Depends(require_user_id),
    agent: RecommendationAgent = Depends(get_recommendation_agent),
) -> RecommendationsListResponse:
    """Fetch stored foresights split into ``new`` (unviewed) and ``viewed``."""
    try:
        new, viewed = await agent.list_recommendations(user_id)
        return RecommendationsListResponse(new=new, viewed=viewed)
    except Exception as e:
        logger.exception("[recommendations] list failed for user=%s", user_id)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("", response_model=RecommendationResponse)
async def generate_recommendations(
    user_id: str = Depends(require_user_id),
    agent: RecommendationAgent = Depends(get_recommendation_agent),
) -> RecommendationResponse:
    """Generate fresh foresights, persist them (fingerprint dedup), and
    return the complete unviewed feed for the user."""
    try:
        return await agent.generate(user_id)
    except Exception as e:
        logger.exception("[recommendations] generate failed for user=%s", user_id)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/{recommendation_id}/viewed", response_model=MarkViewedResponse)
async def mark_recommendation_viewed(
    recommendation_id: str,
    user_id: str = Depends(require_user_id),
    agent: RecommendationAgent = Depends(get_recommendation_agent),
) -> MarkViewedResponse:
    """Flip a single foresight's ``viewed_at`` to now.

    Returns ``404`` if the id doesn't belong to the caller.
    """
    try:
        ok = await agent.mark_viewed(user_id, recommendation_id)
    except Exception as e:
        logger.exception(
            "[recommendations] mark_viewed failed for user=%s id=%s",
            user_id,
            recommendation_id,
        )
        raise HTTPException(status_code=500, detail=str(e)) from e
    if not ok:
        raise HTTPException(status_code=404, detail="recommendation not found")
    return MarkViewedResponse(marked=True)
