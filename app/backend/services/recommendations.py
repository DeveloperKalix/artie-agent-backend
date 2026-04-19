"""
RecommendationAgent — the user-facing agent.

Pulls everything the LLM needs to reason about a user's portfolio:

- SnapTrade positions (tickers, quantity, cost basis, current price).
- Recent ``news_items`` for those tickers since the user's news cursor.
- The user's experience level (from ``user_profiles``).
- Free-form memory notes (from ``user_memory``, captured via ``/skill``).
- The rolling conversation context (when invoked from a conversation).

Ships three surfaces:

1. ``generate(user_id)`` — full structured recommendations. JSON-mode Groq
   response validated against :class:`RecommendationResponse`, then persisted
   to ``public.user_recommendations`` with a stable fingerprint so repeated
   POSTs never duplicate the same foresight. Used by ``POST /recommendations``.
2. ``list_recommendations(user_id)`` — returns ``(new, viewed)`` pulled from
   the persistent store. Used by ``GET /recommendations``.
3. ``chat(user_id, message, history)`` — conversational reply grounded in the
   same context. Used by the orchestrator when the incoming message isn't a
   ``/skill`` command. Returns plain text.

All external I/O (SnapTrade REST, Supabase, Groq) runs through
``asyncio.to_thread`` since the underlying SDKs are synchronous. The agent
itself is otherwise fully async so the FastAPI event loop never blocks on a
downstream API call.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from app.backend.core.redis import get_redis
from app.backend.core.supabase import (
    count_news_for_tickers_since,
    select_maybe,
    update_rows,
    upsert_row,
)
from app.backend.services.memory import MemoryService, get_memory_service
from app.backend.services.snaptrade import unwrap_snaptrade_ticker
from app.backend.services.news import NewsAggregatorAgent, get_news_agent
from app.schemas.memory import ExperienceLevel, MemoryNote, UserProfile
from app.schemas.news import (
    NewsItem,
    PortfolioRecommendation,
    RecommendationResponse,
)

logger = logging.getLogger(__name__)

_CURSOR_TABLE = "user_news_cursor"
_RECOMMENDATIONS_TABLE = "user_recommendations"
_RECOMMENDATION_MODEL = "llama-3.3-70b-versatile"
_CHAT_MODEL = "llama-3.3-70b-versatile"
_MAX_CONTEXT_TURNS = 12  # how many prior turns we pass to the chat model
_MAX_NEWS_FOR_PROMPT = 40
_MAX_VIEWED_RETURN = 50  # cap on "previously viewed" rows returned to client
# If the user's cursor has advanced past most news, the LLM ends up with a
# starvation-thin context and keeps producing the same single foresight.
# When ``len(news) < _MIN_NEWS_BEFORE_WIDEN`` we re-query ignoring the cursor
# so the model always sees a usable pool — dedup is handled by fingerprinting.
_MIN_NEWS_BEFORE_WIDEN = 8
# Auto-generation cooldown (seconds). ``GET /recommendations`` will silently
# trigger a fresh generation when the unviewed feed is empty and there is
# pending news, but only once per user per cooldown window. This keeps Groq
# usage bounded no matter how aggressively the frontend polls.
_AUTOGEN_COOLDOWN_SECONDS = 180
_AUTOGEN_LOCK_PREFIX = "recs:autogen:"

_CASH_EQUIVALENTS = frozenset({"SPAXX", "FDRXX", "VMMXX", "VMFXX", "SWVXX"})

# Groq/OpenAI-style tool schema for the LLM's ``propose_order`` function.
# Keep this minimal — Groq's llama-3.3 is more reliable with shallow schemas
# than deep JSON-schema trees. The orchestrator validates the returned
# arguments against :class:`LLMProposedOrder` before touching SnapTrade.
_PROPOSE_ORDER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "propose_order",
            "description": (
                "Propose a buy or sell order for the user's review. NEVER "
                "executes directly — the user must tap Confirm in the UI. "
                "Use only when the user clearly asks to trade a specific "
                "ticker or crypto pair."
            ),
            "parameters": {
                "type": "object",
                "required": ["asset_class", "symbol", "action"],
                "properties": {
                    "asset_class": {"type": "string", "enum": ["equity", "crypto"]},
                    "account_id": {
                        "type": "string",
                        "description": (
                            "SnapTrade account id shown in the portfolio context above "
                            "(e.g. 'account_id=abc-123'). Omit if unsure — the server "
                            "will pick the best account automatically."
                        ),
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Ticker (equity) or pair 'BTC/USD' (crypto).",
                    },
                    "action": {"type": "string", "enum": ["buy", "sell"]},
                    "order_type": {
                        "type": "string",
                        "enum": ["market", "limit", "stop", "stop_limit"],
                        "default": "market",
                    },
                    "time_in_force": {
                        "type": "string",
                        "enum": ["day", "gtc", "fok", "ioc"],
                        "default": "day",
                    },
                    "units": {"type": "number", "minimum": 0},
                    "notional_value": {"type": "number", "minimum": 0},
                    "price": {"type": "number", "minimum": 0},
                    "stop": {"type": "number", "minimum": 0},
                    "rationale": {
                        "type": "string",
                        "description": "One-sentence explanation for the user.",
                    },
                },
            },
        },
    }
]

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
        "has_equity_positions",
    )

    def __init__(self) -> None:
        self.positions: list[dict[str, Any]] = []
        self.news: list[NewsItem] = []
        self.profile: UserProfile | None = None
        self.notes: list[MemoryNote] = []
        self.cursor: datetime | None = None
        self.has_equity_positions: bool = False


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
        """Produce structured foresights for the user.

        Persists new (by fingerprint) rows to ``user_recommendations`` and
        returns the complete unviewed feed so the frontend can render the
        "new foresights" section on one round trip.
        """
        ctx = await self._collect_context(user_id)
        system_prompt = self._build_system_prompt(ctx, mode="recommendations")
        user_prompt = self._build_user_prompt_for_recommendations(ctx)

        raw = await self._call_groq_json(system_prompt, user_prompt)
        fresh = self._parse_recommendations(raw, ctx.news)
        logger.info(
            "[recommendations] user=%s llm_raw=%d parsed=%d",
            user_id,
            len(raw.get("recommendations") or []) if isinstance(raw, dict) else 0,
            len(fresh),
        )

        inserted = await self._persist_recommendations(user_id, fresh)
        unviewed = await self._list_unviewed(user_id)

        generated_at = datetime.now(tz=timezone.utc).isoformat()

        # Only advance the cursor for users with real equity positions.
        # Users with no investable holdings (new users / cash-only) always
        # receive general market news so they can pick their first positions —
        # advancing the cursor would filter out all news on the next call.
        if ctx.has_equity_positions:
            await self._update_cursor(user_id, datetime.now(tz=timezone.utc))

        return RecommendationResponse(
            user_id=user_id,
            recommendations=unviewed,
            generated_at=generated_at,
            new_count=len(inserted),
        )

    async def list_recommendations(
        self, user_id: str
    ) -> tuple[list[PortfolioRecommendation], list[PortfolioRecommendation]]:
        """Return ``(new, viewed)`` foresights straight from the store.

        When the ``new`` list is empty and there is actually news to reason
        about, transparently kick off a generation — rate-limited per user
        via a Redis cooldown so repeated GETs never stampede Groq. The
        frontend therefore doesn't need to explicitly POST; polling this
        endpoint is enough to keep foresights flowing.
        """
        new_rows = await asyncio.to_thread(_select_unviewed_rows, user_id, 100)
        viewed_rows = await asyncio.to_thread(
            _select_viewed_rows, user_id, _MAX_VIEWED_RETURN
        )

        if not new_rows and await self._should_autogen(user_id):
            try:
                logger.info(
                    "[recommendations] user=%s auto-generating (empty feed)",
                    user_id,
                )
                await self.generate(user_id)
            except Exception:
                logger.exception(
                    "[recommendations] auto-generate failed for user=%s", user_id
                )
            else:
                new_rows = await asyncio.to_thread(
                    _select_unviewed_rows, user_id, 100
                )

        return (
            [_row_to_recommendation(r) for r in new_rows],
            [_row_to_recommendation(r) for r in viewed_rows],
        )

    async def _should_autogen(self, user_id: str) -> bool:
        """Return True iff we should auto-generate right now.

        Gates on two things: (1) a Redis SETNX lock so only one concurrent
        caller per user triggers Groq, and (2) either pending news exists or
        the user has no investable positions (cold-start first-picks).
        """
        redis = get_redis()
        lock_key = f"{_AUTOGEN_LOCK_PREFIX}{user_id}"
        try:
            acquired = await redis.set(
                lock_key, "1", ex=_AUTOGEN_COOLDOWN_SECONDS, nx=True
            )
        except Exception:
            logger.warning(
                "[recommendations] redis unavailable — skipping autogen lock",
                exc_info=True,
            )
            return False
        if not acquired:
            return False

        status = await self.check_for_new_news(user_id)
        has_tickers = bool(status.get("tickers"))
        pending = int(status.get("pending_news_count") or 0)
        # Cold-start users (no tickers) always have "pending" general news,
        # so rely on the lock alone to rate-limit them.
        return (not has_tickers) or pending > 0

    async def mark_viewed(self, user_id: str, recommendation_id: str) -> bool:
        """Flip ``viewed_at`` to now for this rec if it belongs to the user.

        Returns ``True`` if a row was updated, ``False`` otherwise.
        """
        now = datetime.now(tz=timezone.utc).isoformat()

        def _apply() -> list[Any]:
            return update_rows(
                _RECOMMENDATIONS_TABLE,
                {"viewed_at": now},
                filters={"id": recommendation_id, "user_id": user_id},
            )

        updated = await asyncio.to_thread(_apply)
        return bool(updated)

    async def check_for_new_news(self, user_id: str) -> dict[str, Any]:
        """Cheap status check — no LLM call, no cursor advancement.

        Returns both the number of unviewed stored foresights (the primary
        badge the frontend should render) and the number of news items since
        the user's cursor (secondary — signals whether a fresh POST would
        yield anything new).
        """
        positions_task = asyncio.create_task(_load_positions(user_id))
        cursor_task = asyncio.create_task(_load_cursor(user_id))
        unviewed_task = asyncio.create_task(
            asyncio.to_thread(_count_unviewed, user_id)
        )
        positions, cursor, unviewed_count = await asyncio.gather(
            positions_task, cursor_task, unviewed_task
        )

        tickers = sorted(
            {
                t.upper()
                for p in positions
                if (t := unwrap_snaptrade_ticker(p.get("symbol")))
                and t.upper() not in _CASH_EQUIVALENTS
            }
        )

        if not tickers:
            # No investable positions — general market news is always available.
            pending_news = await asyncio.to_thread(
                count_news_for_tickers_since, [], None
            )
        else:
            pending_news = await asyncio.to_thread(
                count_news_for_tickers_since, tickers, cursor
            )

        return {
            "has_new": unviewed_count > 0 or pending_news > 0,
            "new_count": unviewed_count,
            "pending_news_count": pending_news,
            "last_seen_at": cursor.isoformat() if cursor else None,
            "tickers": tickers,
        }

    async def chat(
        self,
        user_id: str,
        message: str,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Return a conversational reply grounded in the user's portfolio."""
        outcome = await self.chat_with_tools(
            user_id=user_id,
            message=message,
            history=history,
            allow_propose_order=False,
        )
        return outcome["text"]

    async def chat_with_tools(
        self,
        *,
        user_id: str,
        message: str,
        history: list[dict[str, Any]] | None = None,
        allow_propose_order: bool = False,
    ) -> dict[str, Any]:
        """Chat turn with optional ``propose_order`` tool-calling.

        Returns a dict with:
          * ``text`` — the assistant's reply (non-empty even when a tool was called).
          * ``proposal`` — raw dict payload from the LLM's ``propose_order``
            tool call, or ``None`` if the model didn't call a tool. The
            orchestrator validates this against :class:`LLMProposedOrder`
            before creating an intent.

        The LLM is told it MUST NOT execute orders — tool calls always
        become user-facing confirmations. This is enforced at the
        ``OrderIntentService`` layer too: :meth:`create_from_llm` creates an
        intent in ``awaiting_confirmation`` and never auto-submits.
        """
        ctx = await self._collect_context(user_id)
        system_prompt = self._build_system_prompt(ctx, mode="chat")
        if allow_propose_order:
            system_prompt += (
                "\n\nYou MAY call the ``propose_order`` tool when the user "
                "clearly asks to buy or sell a specific ticker (equities or "
                "crypto). NEVER use the tool on vague messages. Tool calls "
                "are ALWAYS surfaced to the user for tap-to-confirm — you "
                "are never placing an order directly. Always include a "
                "short rationale alongside the tool call."
            )

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for turn in (history or [])[-_MAX_CONTEXT_TURNS:]:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})

        tools = _PROPOSE_ORDER_TOOLS if allow_propose_order else None
        return await self._call_groq_chat_with_tools(messages, tools=tools)

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

        # Filter money-market / cash-equivalent tickers that never appear in
        # news feeds so we don't constrain the query to meaningless tickers.
        tickers = sorted(
            {
                t.upper()
                for p in positions
                if (t := unwrap_snaptrade_ticker(p.get("symbol")))
                and t.upper() not in _CASH_EQUIVALENTS
            }
        )

        if tickers:
            news = await self._news.get_recent_news_for_user(
                user_id=user_id,
                tickers=tickers,
                since=cursor,
                limit=_MAX_NEWS_FOR_PROMPT,
            )
            # Starvation guard: if the cursor has advanced past most news, the
            # LLM has nothing to work with and keeps repeating the same
            # foresight. Widen the window to the full available pool for
            # these tickers — fingerprinting still prevents duplicates.
            if len(news) < _MIN_NEWS_BEFORE_WIDEN:
                logger.info(
                    "[recommendations] user=%s widening news window "
                    "(cursor=%s yielded only %d items)",
                    user_id,
                    cursor.isoformat() if cursor else None,
                    len(news),
                )
                news = await self._news.get_recent_news_for_user(
                    user_id=user_id,
                    tickers=tickers,
                    since=None,
                    limit=_MAX_NEWS_FOR_PROMPT,
                )
        else:
            # No investable positions yet (new user or only cash equivalents).
            # Ignore the cursor so we always supply fresh general market news
            # to the LLM — this enables first-pick / discovery recommendations.
            news = await self._news.recent(limit=_MAX_NEWS_FOR_PROMPT)

        logger.info(
            "[recommendations] user=%s ctx tickers=%d news=%d notes=%d positions=%d",
            user_id,
            len(tickers),
            len(news),
            len(notes),
            len(positions),
        )

        ctx = _Ctx()
        ctx.positions = positions
        ctx.news = news
        ctx.profile = profile
        ctx.notes = notes
        ctx.cursor = cursor
        ctx.has_equity_positions = bool(tickers)
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
        if not ctx.has_equity_positions:
            prefs = (
                "\n".join(f"- {n.content}" for n in ctx.notes)
                if ctx.notes
                else "(none stated yet)"
            )
            return (
                "I have no invested positions yet and I'm looking for my first "
                "stock or ETF picks based on the news above.\n\n"
                f"My stated investment preferences:\n{prefs}\n\n"
                "Using only the news articles provided above, suggest 2–4 "
                "specific tickers I should research. Use action='buy' with "
                "confidence='low' since I haven't invested yet. "
                "Tailor your suggestions to my preferences if any are stated — "
                "otherwise pick the most newsworthy opportunities from the "
                "articles above."
            )
        return (
            "Given my positions and the news above, produce between 3 and 6 "
            "distinct foresights. Cover different tickers whenever the news "
            "supports it — do not return only a single recommendation unless "
            "the news truly only concerns one ticker. For every foresight, "
            "cite at least one article URL from the list above in "
            "``article_urls`` (verbatim). It is fine to include a hold or "
            "reduce call on a position that appears fragile in the news."
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
                temperature=0.45,
                response_format={"type": "json_object"},
            )
            text = completion.choices[0].message.content or "{}"
            # Truncated preview at INFO so we can see what Groq actually
            # emitted without leaking full prompts into logs.
            logger.info(
                "[recommendations] Groq JSON (%d chars): %s",
                len(text),
                text[:500].replace("\n", " "),
            )
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

    async def _call_groq_chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Groq completion that may return a ``propose_order`` tool call.

        Returns ``{"text": str, "proposal": dict | None}``. The model can
        both call the tool and produce text alongside it — we preserve
        both so the orchestrator can echo the rationale into chat while
        rendering the confirm card.
        """
        from app.main import groq_client

        def _sync() -> dict[str, Any]:
            kwargs: dict[str, Any] = {
                "model": _CHAT_MODEL,
                "messages": messages,
                "temperature": 0.4,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            completion = groq_client.chat.completions.create(**kwargs)
            choice = completion.choices[0].message
            text = (getattr(choice, "content", None) or "").strip()
            proposal: dict[str, Any] | None = None
            tool_calls = getattr(choice, "tool_calls", None) or []
            for call in tool_calls:
                fn = getattr(call, "function", None) or {}
                name = getattr(fn, "name", None) or (
                    fn.get("name") if isinstance(fn, dict) else None
                )
                if name != "propose_order":
                    continue
                raw_args = getattr(fn, "arguments", None) or (
                    fn.get("arguments") if isinstance(fn, dict) else None
                )
                if isinstance(raw_args, str):
                    try:
                        proposal = json.loads(raw_args)
                    except json.JSONDecodeError:
                        logger.warning(
                            "[recommendations] propose_order args not JSON: %r",
                            raw_args[:200],
                        )
                        proposal = None
                elif isinstance(raw_args, dict):
                    proposal = raw_args
                break
            return {"text": text, "proposal": proposal}

        return await asyncio.to_thread(_sync)

    # ----------------------------------------------------------- parse utils

    @staticmethod
    def _parse_recommendations(
        raw: dict[str, Any], news: list[NewsItem]
    ) -> list[PortfolioRecommendation]:
        """Map the LLM's JSON into validated recommendations, resolving URLs."""
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
                if supporting_urls and not supporting:
                    logger.warning(
                        "[recommendations] LLM cited %d URL(s) for %s but none "
                        "matched the news pool — likely hallucinated: %r",
                        len(supporting_urls),
                        entry.get("ticker"),
                        supporting_urls[:3],
                    )
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

    # ------------------------------------------------------ persistence utils

    async def _persist_recommendations(
        self,
        user_id: str,
        fresh: list[PortfolioRecommendation],
    ) -> list[PortfolioRecommendation]:
        """Upsert-ignore by fingerprint; return only rows newly inserted."""
        if not fresh:
            return []

        rows: list[dict[str, Any]] = []
        for rec in fresh:
            fp = _fingerprint(rec)
            rows.append(
                {
                    "user_id": user_id,
                    "ticker": rec.ticker,
                    "action": rec.action,
                    "confidence": rec.confidence,
                    "explanation": rec.explanation,
                    "supporting_articles": [
                        a.model_dump(mode="json") for a in rec.supporting_articles
                    ],
                    "fingerprint": fp,
                }
            )

        def _apply() -> list[Any]:
            # on_conflict + ignore_duplicates means PostgREST returns only
            # rows that were actually inserted — repeat foresights are skipped.
            from app.backend.core.supabase import upsert_rows as _upsert_rows

            return _upsert_rows(
                _RECOMMENDATIONS_TABLE,
                rows,
                on_conflict="user_id,fingerprint",
                ignore_duplicates=True,
            )

        try:
            inserted_rows = await asyncio.to_thread(_apply)
        except Exception:
            logger.exception(
                "[recommendations] failed to persist %d rows for user=%s",
                len(rows),
                user_id,
            )
            return []

        inserted = [_row_to_recommendation(r) for r in inserted_rows or []]
        logger.info(
            "[recommendations] user=%s persisted=%d/%d",
            user_id,
            len(inserted),
            len(rows),
        )
        return inserted

    async def _list_unviewed(
        self, user_id: str
    ) -> list[PortfolioRecommendation]:
        rows = await asyncio.to_thread(_select_unviewed_rows, user_id, 100)
        return [_row_to_recommendation(r) for r in rows]

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


def _fingerprint(rec: PortfolioRecommendation) -> str:
    """Stable identity for dedup: ticker + action + sorted article URLs.

    Confidence is intentionally excluded so two near-identical foresights
    generated moments apart on the same news don't duplicate just because
    the LLM nudged the confidence label.
    """
    urls = sorted({a.url for a in rec.supporting_articles if a.url})
    payload = "|".join(
        [rec.ticker.upper(), rec.action.lower(), *urls]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_to_recommendation(row: dict[str, Any]) -> PortfolioRecommendation:
    articles_raw = row.get("supporting_articles") or []
    articles: list[NewsItem] = []
    for a in articles_raw:
        if not isinstance(a, dict):
            continue
        try:
            articles.append(NewsItem(**a))
        except Exception:
            logger.debug("[recommendations] skipped malformed article %r", a)
    return PortfolioRecommendation(
        id=row.get("id"),
        ticker=str(row.get("ticker", "?")),
        action=str(row.get("action", "hold")),
        confidence=str(row.get("confidence", "low")),
        explanation=str(row.get("explanation", "")),
        supporting_articles=articles,
        generated_at=_as_iso(row.get("generated_at")),
        viewed_at=_as_iso(row.get("viewed_at")),
    )


def _as_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


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


def _count_unviewed(user_id: str) -> int:
    """Cheap count of ``user_recommendations`` rows with ``viewed_at IS NULL``."""
    from app.backend.core.supabase import get_supabase

    res = (
        get_supabase()
        .table(_RECOMMENDATIONS_TABLE)
        .select("id", count="exact")
        .eq("user_id", user_id)
        .is_("viewed_at", "null")
        .execute()
    )
    return res.count if res.count is not None else len(res.data or [])


def _select_unviewed_rows(user_id: str, limit: int) -> list[dict[str, Any]]:
    """PostgREST ``is.null`` + order-by generated_at DESC."""
    from app.backend.core.supabase import get_supabase

    res = (
        get_supabase()
        .table(_RECOMMENDATIONS_TABLE)
        .select("*")
        .eq("user_id", user_id)
        .is_("viewed_at", "null")
        .order("generated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


def _select_viewed_rows(user_id: str, limit: int) -> list[dict[str, Any]]:
    """PostgREST ``not.is=null`` + order-by viewed_at DESC."""
    from app.backend.core.supabase import get_supabase

    res = (
        get_supabase()
        .table(_RECOMMENDATIONS_TABLE)
        .select("*")
        .eq("user_id", user_id)
        .not_.is_("viewed_at", "null")
        .order("viewed_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


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
        account_id = pos.get("account_id") or ""
        # Include account_id so the LLM can pass it to propose_order.
        account_hint = f" account_id={account_id}" if account_id else ""
        lines.append(
            f"- {symbol}:{account_hint} qty={qty} cost_basis={cost} price={price} "
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
