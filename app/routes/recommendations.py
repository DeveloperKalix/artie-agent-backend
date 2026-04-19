"""
``POST /recommendations`` — on-demand structured portfolio recommendations.

Separate from the conversation endpoints because:
- The response shape is structured (``RecommendationResponse``) rather than a
  free-form message.
- It's safe for the frontend to call from a dashboard/home screen on tab focus
  without creating a conversation row.
- It advances the ``user_news_cursor`` on success, which we don't want every
  idle chat turn to do.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.backend.services.recommendations import (
    RecommendationAgent,
    get_recommendation_agent,
)
from app.routes._deps import require_user_id
from app.schemas.news import RecommendationResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("", response_model=RecommendationResponse)
async def generate_recommendations(
    user_id: str = Depends(require_user_id),
    agent: RecommendationAgent = Depends(get_recommendation_agent),
) -> RecommendationResponse:
    try:
        return await agent.generate(user_id)
    except Exception as e:
        logger.exception("[recommendations] generate failed for user=%s", user_id)
        raise HTTPException(status_code=500, detail=str(e)) from e
