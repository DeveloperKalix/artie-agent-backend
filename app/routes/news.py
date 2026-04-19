"""GET /news — read-only view over the news_items table populated by the
scheduled NewsAggregatorAgent ingest. No live source fetching happens here."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.backend.services.news import NewsAggregatorAgent, get_news_agent
from app.schemas.news import NewsResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/news", tags=["news"])

_MAX_LIMIT = 100
_DEFAULT_LIMIT = 20


def _parse_tickers(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


@router.get("", response_model=NewsResponse)
async def get_news(
    tickers: Optional[str] = Query(
        None,
        description="Comma-separated ticker filter (e.g. AAPL,BTC-USD).",
    ),
    query: Optional[str] = Query(
        None,
        description="Free-text substring match over title and summary.",
    ),
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
    agent: NewsAggregatorAgent = Depends(get_news_agent),
) -> NewsResponse:
    """Return news items filtered by tickers, query, or neither (most recent first).

    Priority: ``tickers`` > ``query`` > recent. Tickers and query are mutually
    exclusive — if both are supplied, tickers wins so the response stays focused.
    """
    try:
        ticker_list = _parse_tickers(tickers)
        if ticker_list:
            items = await agent.get_recent_news_for_user(
                user_id="",  # not used when tickers are supplied explicitly
                tickers=ticker_list,
                since=None,
                limit=limit,
            )
        elif query:
            items = await agent.search(query=query.strip(), limit=limit)
        else:
            items = await agent.recent(limit=limit)
        return NewsResponse(items=items, total=len(items))
    except Exception as e:
        logger.exception("[news] GET /news failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
