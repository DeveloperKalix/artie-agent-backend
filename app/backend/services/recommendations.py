"""
RecommendationAgent — the user-facing agent.

Pulls everything the LLM needs to reason about a user's portfolio:

- SnapTrade positions (tickers, quantity, cost basis, current price).
- Recent ``news_items`` for those tickers since the user's news cursor.
- The user's experience level (from ``user_profiles``).
- Free-form memory notes (from ``user_memory``, captured via ``/skill``).
- The rolling conversation context (when invoked from a conversation).

Ships two surfaces:

1. ``generate(user_id)`` — full structured recommendations. JSON-mode Groq
   response validated against :class:`RecommendationResponse`. Used by
   ``POST /recommendations``.
2. ``chat(user_id, message, history)`` — conversational reply grounded in the
   same context. Used by the orchestrator when the incoming message isn't a
   ``/skill`` command. Returns plain text and optional structured recs.

Both paths share a single ``_build_system_prompt`` so tone, disclaimer, and
experience-level tailoring stay consistent.

All external I/O (SnapTrade REST, Supabase, Groq) runs through
``asyncio.to_thread`` since the underlying SDKs are synchronous. The agent
itself is otherwise fully async so the FastAPI event loop never blocks on a
downstream API call.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from app.backend.core.supabase import (
    select_maybe,
    upsert_row,
)
from app.backend.services.memory import MemoryService, get_memory_service
from app.backend.services.news import NewsAggregatorAgent, get_news_agent
from app.schemas.memory import ExperienceLevel, MemoryNote, UserProfile
from app.schemas.news import (
    NewsItem,
    PortfolioRecommendation,
    RecommendationResponse,
)

logger = logging.getLogger(__name__)

_CURSOR_TABLE = "user_news_cursor"
_RECOMMENDATION_MODEL = "llama-3.3-70b-versatile"
_CHAT_MODEL = "llama-3.3-70b-versatile"
_MAX_CONTEXT_TURNS = 12  # how many prior turns we pass to the chat model
_MAX_NEWS_FOR_PROMPT = 20

_TONE_BY_LEVEL: dict[ExperienceLevel, str] = {
    ExperienceLevel.novice: (
        "The user is a novice investor. Use plain language, avoid jargon, "
        "define every financial term the first time you use it, keep "
        "recommendations simple, and remind them to verify with a licensed "
        "advisor before acting."
    ),
    ExperienceLevel.intermediate: (
        "The user is an intermediate investor. You may use standard financial "
        "terminology (P/E, yield, beta, duration) without defining it, and "
        "give reasoning with quantitative context where helpful."
    ),
    ExperienceLevel.veteran: (
        "The user is a veteran investor. You may use advanced terminology "
        "freely (Greeks, spread conventions, factor tilts, correlation), be "
        "concise, skip basic context, and focus on edge cases and risk "
        "factors they might not already have considered."
    ),
}

_DISCLAIMER = (
    "This is not financial advice. Artie's recommendations are AI-generated "
    "from public news and do not account for your full financial situation."
)


# ---------------------------------------------------------- helper dataclasses


class _Ctx:
    """Bag of inputs collected before calling the LLM. Mutable for clarity."""

    __slots__ = (
        "positions",
        "news",
        "profile",
        "notes",
        "cursor",
    )

    def __init__(self) -> None:
        self.positions: list[dict[str, Any]] = []
        self.news: list[NewsItem] = []
        self.profile: UserProfile | None = None
        self.notes: list[MemoryNote] = []
        self.cursor: datetime | None = None


# ----------------------------------------------------------- RecommendationAgent


class RecommendationAgent:
    def __init__(
        self,
        *,
        news: NewsAggregatorAgent,
        memory: MemoryService,
    ) -> None:
        self._news = news
        self._memory = memory

    # ------------------------------------------------------------ public API

    async def generate(self, user_id: str) -> RecommendationResponse:
        """Produce structured portfolio recommendations for the user."""
        ctx = await self._collect_context(user_id)
        system_prompt = self._build_system_prompt(ctx, mode="recommendations")
        user_prompt = self._build_user_prompt_for_recommendations(ctx)

        raw = await self._call_groq_json(system_prompt, user_prompt)
        recommendations = self._parse_recommendations(raw, ctx.news)

        generated_at = datetime.now(tz=timezone.utc).isoformat()
        response = RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            generated_at=generated_at,
        )

        # Advance the user's news cursor to "now" only after a successful
        # generation so a failed LLM call doesn't silently drop news.
        await self._update_cursor(user_id, datetime.now(tz=timezone.utc))
        return response

    async def chat(
        self,
        user_id: str,
        message: str,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Return a conversational reply grounded in the user's portfolio."""
        ctx = await self._collect_context(user_id)
        system_prompt = self._build_system_prompt(ctx, mode="chat")

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for turn in (history or [])[-_MAX_CONTEXT_TURNS:]:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})

        return await self._call_groq_chat(messages)

    # ----------------------------------------------------- context collection

    async def _collect_context(self, user_id: str) -> _Ctx:
        """Fan out all context gathering concurrently."""
        positions_task = asyncio.create_task(_load_positions(user_id))
        profile_notes_task = asyncio.create_task(
            self._memory.get_profile_with_notes(user_id)
        )
        cursor_task = asyncio.create_task(_load_cursor(user_id))

        positions, (profile, notes), cursor = await asyncio.gather(
            positions_task, profile_notes_task, cursor_task
        )

        tickers = sorted({p["symbol"] for p in positions if p.get("symbol")})
        news = await self._news.get_recent_news_for_user(
            user_id=user_id,
            tickers=tickers or None,
            since=cursor,
            limit=_MAX_NEWS_FOR_PROMPT,
        )

        ctx = _Ctx()
        ctx.positions = positions
        ctx.news = news
        ctx.profile = profile
        ctx.notes = notes
        ctx.cursor = cursor
        return ctx

    # ------------------------------------------------------- prompt builders

    def _build_system_prompt(self, ctx: _Ctx, *, mode: str) -> str:
        level = (
            ctx.profile.experience_level if ctx.profile else ExperienceLevel.novice
        )
        tone = _TONE_BY_LEVEL[level]

        notes_block = _format_notes_block(ctx.notes)
        positions_block = _format_positions_block(ctx.positions)
        news_block = _format_news_block(ctx.news)

        base = (
            "You are Artie, an agentic investment co-pilot. You help the user "
            "reason about their portfolio using live market news and their "
            "own stated preferences. You must never invent news articles, "
            "prices, or positions that aren't in the context below.\n"
            f"\n{tone}"
            "\n\n=== User preferences and memory notes ==="
            f"\n{notes_block or '(no notes yet)'}"
            "\n\n=== Current portfolio positions ==="
            f"\n{positions_block or '(no connected positions)'}"
            "\n\n=== Recent news the user has not yet seen ==="
            f"\n{news_block or '(no new news)'}"
        )

        if mode == "recommendations":
            base += (
                "\n\nYou MUST reply with a single JSON object with this schema:\n"
                "{\n"
                '  "recommendations": [\n'
                "    {\n"
                '      "ticker": string,\n'
                '      "action": "buy"|"sell"|"hold"|"increase"|"reduce",\n'
                '      "confidence": "high"|"medium"|"low",\n'
                '      "explanation": string,\n'
                '      "article_urls": [string, ...]\n'
                "    }\n"
                "  ]\n"
                "}\n"
                "Each ``article_urls`` entry MUST be copied verbatim from the "
                "news URLs above — do not fabricate links. If you can't "
                "recommend anything responsibly, return "
                '{"recommendations": []} rather than guessing.'
            )
        else:
            base += (
                "\n\nAnswer the user conversationally. Recommend actions only "
                "when the user's question warrants them, and always cite the "
                "news URL when your reasoning relies on a specific article."
                f"\n\nAlways end with: \u201c{_DISCLAIMER}\u201d"
            )
        return base

    @staticmethod
    def _build_user_prompt_for_recommendations(ctx: _Ctx) -> str:
        if not ctx.positions:
            return (
                "I have no connected brokerage positions yet. Given the news "
                "above, suggest a couple of general tickers worth researching, "
                "but keep confidence low."
            )
        return (
            "Given my positions and the news since my last check, what "
            "actions should I consider? Only include recommendations that are "
            "clearly supported by one or more of the news items above."
        )

    # ------------------------------------------------------------- Groq calls

    async def _call_groq_json(self, system: str, user: str) -> dict[str, Any]:
        from app.main import groq_client

        def _sync() -> dict[str, Any]:
            completion = groq_client.chat.completions.create(
                model=_RECOMMENDATION_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            text = completion.choices[0].message.content or "{}"
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.warning(
                    "[recommendations] Groq returned non-JSON despite json_object "
                    "mode: %r",
                    text[:200],
                )
                return {}

        return await asyncio.to_thread(_sync)

    async def _call_groq_chat(self, messages: list[dict[str, Any]]) -> str:
        from app.main import groq_client

        def _sync() -> str:
            completion = groq_client.chat.completions.create(
                model=_CHAT_MODEL,
                messages=messages,
                temperature=0.4,
            )
            return (completion.choices[0].message.content or "").strip()

        return await asyncio.to_thread(_sync)

    # ----------------------------------------------------------- parse utils

    @staticmethod
    def _parse_recommendations(
        raw: dict[str, Any], news: list[NewsItem]
    ) -> list[PortfolioRecommendation]:
        """Map the LLM's JSON into validated recommendations, resolving URLs.

        The LLM emits ``article_urls`` (a list of strings); we match them
        against the news we actually showed it so the response always contains
        real :class:`NewsItem` objects the frontend can link to.
        """
        url_index: dict[str, NewsItem] = {n.url: n for n in news}
        out: list[PortfolioRecommendation] = []
        for entry in (raw.get("recommendations") or []):
            if not isinstance(entry, dict):
                continue
            try:
                supporting_urls = entry.get("article_urls") or []
                supporting = [
                    url_index[url]
                    for url in supporting_urls
                    if isinstance(url, str) and url in url_index
                ]
                rec = PortfolioRecommendation(
                    ticker=str(entry.get("ticker", "")).upper() or "?",
                    action=str(entry.get("action", "hold")).lower(),
                    confidence=str(entry.get("confidence", "low")).lower(),
                    explanation=str(entry.get("explanation", "")).strip(),
                    supporting_articles=supporting,
                )
            except Exception:
                logger.debug(
                    "[recommendations] skipped malformed entry %r", entry, exc_info=True
                )
                continue
            if not rec.explanation:
                continue
            out.append(rec)
        return out

    # ---------------------------------------------------------- cursor utils

    @staticmethod
    async def _update_cursor(user_id: str, ts: datetime) -> None:
        payload = {"user_id": user_id, "last_seen_at": ts.isoformat()}
        try:
            await asyncio.to_thread(
                upsert_row, _CURSOR_TABLE, payload, on_conflict="user_id"
            )
        except Exception:
            logger.warning(
                "[recommendations] failed to update news cursor for %s",
                user_id,
                exc_info=True,
            )


# ------------------------------------------------------------ module helpers


async def _load_positions(user_id: str) -> list[dict[str, Any]]:
    """SnapTrade is sync; isolate it in a thread and swallow lookup errors.

    The agent should degrade gracefully when a user has no brokerage linked —
    a fresh user can still get generic news-driven suggestions.
    """
    try:
        from app.backend.services.snaptrade import SnapTradeService

        svc = SnapTradeService()
    except Exception:
        logger.exception("[recommendations] SnapTradeService unavailable")
        return []

    def _sync() -> list[dict[str, Any]]:
        try:
            return svc.get_positions(user_id)
        except Exception:
            logger.warning(
                "[recommendations] SnapTrade get_positions failed", exc_info=True
            )
            return []

    return await asyncio.to_thread(_sync)


async def _load_cursor(user_id: str) -> datetime | None:
    row = await asyncio.to_thread(
        select_maybe, _CURSOR_TABLE, filters={"user_id": user_id}
    )
    if not row or not row.get("last_seen_at"):
        return None
    try:
        return datetime.fromisoformat(str(row["last_seen_at"]).replace("Z", "+00:00"))
    except Exception:
        return None


def _format_notes_block(notes: list[MemoryNote]) -> str:
    if not notes:
        return ""
    return "\n".join(f"- {n.content}" for n in notes)


def _format_positions_block(positions: list[dict[str, Any]]) -> str:
    if not positions:
        return ""
    lines: list[str] = []
    for pos in positions:
        symbol = pos.get("symbol") or "?"
        qty = pos.get("quantity")
        price = pos.get("current_price")
        cost = pos.get("average_purchase_price")
        mv = pos.get("market_value")
        unreal = pos.get("unrealized_gain")
        lines.append(
            f"- {symbol}: qty={qty} cost_basis={cost} price={price} "
            f"market_value={mv} unrealized={unreal}"
        )
    return "\n".join(lines)


def _format_news_block(news: list[NewsItem]) -> str:
    if not news:
        return ""
    lines: list[str] = []
    for item in news:
        published = item.published_at.isoformat() if item.published_at else "?"
        tickers = ", ".join(item.tickers) if item.tickers else "-"
        summary = (item.summary or "").strip().replace("\n", " ")[:280]
        lines.append(
            f"- [{tickers}] {item.title} ({published}) {item.url}\n  {summary}"
        )
    return "\n".join(lines)


@lru_cache(maxsize=1)
def get_recommendation_agent() -> RecommendationAgent:
    return RecommendationAgent(
        news=get_news_agent(),
        memory=get_memory_service(),
    )


__all__ = [
    "RecommendationAgent",
    "get_recommendation_agent",
]
