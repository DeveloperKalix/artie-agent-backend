"""
Recommendations endpoints.

``GET  /recommendations/status`` — cheap badge check (no LLM, no cursor advance).
``POST /recommendations``        — full on-demand recommendations (calls Groq, advances cursor).

The status endpoint is designed to be called on every app-foreground event so
the frontend can show a notification badge without burning Groq tokens. Only
call the POST endpoint when the user explicitly requests recommendations.
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
from app.schemas.news import RecommendationResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class RecommendationStatusResponse(BaseModel):
    has_new: bool
    new_count: int
    last_seen_at: str | None
    tickers: list[str]


@router.get("/status", response_model=RecommendationStatusResponse)
async def get_recommendation_status(
    user_id: str = Depends(require_user_id),
    agent: RecommendationAgent = Depends(get_recommendation_agent),
) -> RecommendationStatusResponse:
    """Check whether there is new news since the user's last recommendation.

    Cheap: no LLM call, no cursor advancement. Returns a count so the
    frontend can show a badge (e.g. "3 new updates") without calling
    ``POST /recommendations``.
    """
    try:
        result: dict[str, Any] = await agent.check_for_new_news(user_id)
        return RecommendationStatusResponse(**result)
    except Exception as e:
        logger.exception("[recommendations] status check failed for user=%s", user_id)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("", response_model=RecommendationResponse)
async def generate_recommendations(
    user_id: str = Depends(require_user_id),
    agent: RecommendationAgent = Depends(get_recommendation_agent),
) -> RecommendationResponse:
    """Generate full structured recommendations and advance the news cursor."""
    try:
        return await agent.generate(user_id)
    except Exception as e:
        logger.exception("[recommendations] generate failed for user=%s", user_id)
        raise HTTPException(status_code=500, detail=str(e)) from e
