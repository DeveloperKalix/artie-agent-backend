"""Pydantic schemas for news items and portfolio recommendations.

NewsItem mirrors the news_items table one-to-one. PortfolioRecommendation +
RecommendationResponse are the contract the RecommendationAgent returns to
the client (and the JSON shape Groq is instructed to emit).
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
    ticker: str
    action: str  # "buy" | "sell" | "hold" | "increase" | "reduce"
    confidence: str  # "high" | "medium" | "low"
    explanation: str
    supporting_articles: list[NewsItem]


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[PortfolioRecommendation]
    generated_at: str
    disclaimer: str = (
        "This is not financial advice. Artie recommendations are AI-generated "
        "from public news and do not account for your full financial situation."
    )


__all__ = [
    "NewsItem",
    "NewsResponse",
    "PortfolioRecommendation",
    "RecommendationResponse",
]
