"""Pydantic schemas for news items and portfolio recommendations.

NewsItem mirrors the news_items table one-to-one. PortfolioRecommendation +
RecommendationResponse are the contract the RecommendationAgent returns to
the client (and the JSON shape Groq is instructed to emit).

PortfolioRecommendation carries persistence metadata (``id``,
``generated_at``, ``viewed_at``) so the frontend can split "new" vs
"previously-viewed" foresights and mark individual cards as viewed.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class NewsItem(BaseModel):
    id: Optional[str] = None
    url: str
    title: str
    summary: Optional[str] = None
    source: str
    tickers: list[str] = []
    published_at: Optional[datetime] = None
    sentiment: Optional[str] = None


class NewsResponse(BaseModel):
    items: list[NewsItem]
    total: int


class PortfolioRecommendation(BaseModel):
    id: Optional[str] = None
    ticker: str
    action: str  # "buy" | "sell" | "hold" | "increase" | "reduce"
    confidence: str  # "high" | "medium" | "low"
    explanation: str
    supporting_articles: list[NewsItem] = []
    generated_at: Optional[str] = None
    viewed_at: Optional[str] = None


class RecommendationResponse(BaseModel):
    """POST /recommendations response.

    ``recommendations`` is the merged list of everything *unviewed* for the
    user, ordered newest-first. ``new_count`` reports how many rows were
    inserted by this call (i.e. brand-new foresights the user has not seen
    yet under any previous generation).
    """

    user_id: str
    recommendations: list[PortfolioRecommendation]
    generated_at: str
    new_count: int = 0
    disclaimer: str = (
        "This is not financial advice. Artie recommendations are AI-generated "
        "from public news and do not account for your full financial situation."
    )


class RecommendationsListResponse(BaseModel):
    """GET /recommendations — new (unviewed) and viewed buckets."""

    new: list[PortfolioRecommendation]
    viewed: list[PortfolioRecommendation]


__all__ = [
    "NewsItem",
    "NewsResponse",
    "PortfolioRecommendation",
    "RecommendationResponse",
    "RecommendationsListResponse",
]
