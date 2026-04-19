"""
NewsAggregatorAgent — owns all news I/O.

Runs on a scheduler (see app/main.py lifespan), fans out to multiple sources
concurrently, deduplicates by URL, and persists normalized NewsItem rows
into Supabase public.news_items. The user-facing RecommendationAgent (Phase 3)
only reads from Supabase via ``get_recent_news_for_user``; it never talks
directly to the outside world.

Resilience: each fetcher is independent. A source failure logs + returns [],
so the overall ingest degrades gracefully. A Redis lock (``agent:ingest_lock``)
protects against concurrent ingests if we ever scale horizontally.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from time import mktime
from typing import Any, Iterable

import feedparser
import httpx
import redis.asyncio as aioredis
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.backend.core.redis import get_redis
from app.backend.core.supabase import (
    select_news_by_query,
    select_news_for_tickers,
    select_recent_news,
    upsert_rows,
)
from app.schemas.news import NewsItem

logger = logging.getLogger(__name__)

# The DB's ``news_items.url UNIQUE`` constraint is the single source of truth
# for dedup. Redis is only used to prevent concurrent ingest runs and to
# expose a ``news:last_ingest`` heartbeat for observability.
_INGEST_LOCK_KEY = "agent:ingest_lock"
_LAST_INGEST_KEY = "news:last_ingest"
# An ingest run normally finishes in a few seconds; a 5-minute TTL means
# a crashed run clears up quickly without us having to manually intervene.
_INGEST_LOCK_TTL = 300

_DEFAULT_WATCHLIST: tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN",
    "BTC-USD", "ETH-USD",
)

_YAHOO_RSS_URL = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline"
    "?s=^GSPC,^DJI,BTC-USD&region=US&lang=en-US"
)
_COINBASE_RSS_URL = "https://blog.coinbase.com/feed"
_MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"

_TINYFISH_QUERIES: tuple[str, ...] = (
    "market news today",
    "crypto news today",
    "stock market update",
)


def _parse_feed_time(entry: Any) -> datetime | None:
    """feedparser returns ``published_parsed`` as a time.struct_time in UTC."""
    parsed = getattr(entry, "published_parsed", None) or entry.get("published_parsed") \
        if hasattr(entry, "get") else None
    if not parsed:
        return None
    try:
        return datetime.fromtimestamp(mktime(parsed), tz=timezone.utc)
    except Exception:
        return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        s = str(value).strip()
    except Exception:
        return None
    return s or None


class NewsAggregatorAgent:
    """Ingests news from multiple sources and exposes a read API for agents."""

    def __init__(self, redis: aioredis.Redis, tf_client: Any | None = None) -> None:
        self._redis = redis
        self._tf_client = tf_client
        self._watchlist: tuple[str, ...] = _DEFAULT_WATCHLIST

    # ---------------------------------------------------------------- ingest

    async def ingest(self) -> int:
        """Fetch, dedup, upsert. Returns count of newly-inserted items.

        Dedup is the DB's responsibility via ``news_items.url UNIQUE`` —
        ``upsert_rows(..., on_conflict='url', ignore_duplicates=True)`` makes
        PostgREST skip already-present URLs and return only freshly inserted
        rows. That keeps us correct even if Redis is wiped or out of sync.
        """
        lock_acquired = await self._acquire_lock()
        if not lock_acquired:
            logger.info("[news] ingest skipped — another run holds the lock")
            return 0
        try:
            logger.info("[news] ingest starting")
            source_names = (
                "yahoo_rss",
                "coinbase_rss",
                "yfinance",
                "tinyfish_search",
                "marketaux",
            )
            results = await asyncio.gather(
                self._fetch_yahoo_rss(),
                self._fetch_coinbase_rss(),
                self._fetch_yfinance(list(self._watchlist)),
                self._fetch_tinyfish_search(),
                self._fetch_marketaux(),
                return_exceptions=True,
            )

            collected: list[NewsItem] = []
            for name, result in zip(source_names, results):
                if isinstance(result, Exception):
                    logger.warning("[news] source %s raised: %s", name, result)
                    continue
                items = result or []
                logger.info("[news] source %s returned %d items", name, len(items))
                collected.extend(items)

            if not collected:
                logger.info("[news] ingest produced 0 items")
                return 0

            deduped = self._dedup_by_url(collected)
            inserted_count = await asyncio.to_thread(self._upsert_news, deduped)
            await self._redis.set(
                _LAST_INGEST_KEY,
                datetime.now(tz=timezone.utc).isoformat(),
            )
            logger.info(
                "[news] ingest done — collected=%d dedup=%d newly_inserted=%d",
                len(collected),
                len(deduped),
                inserted_count,
            )
            return inserted_count
        except Exception:
            logger.exception("[news] ingest crashed")
            return 0
        finally:
            await self._release_lock()

    # -------------------------------------------------------------- read API

    async def get_recent_news_for_user(
        self,
        user_id: str,  # noqa: ARG002 — reserved for future per-user filtering
        tickers: list[str] | None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[NewsItem]:
        """Read-only query against news_items, filtered by tickers/since."""
        if tickers:
            rows = await asyncio.to_thread(
                select_news_for_tickers,
                tickers,
                since=since,
                limit=limit,
            )
        else:
            rows = await asyncio.to_thread(select_recent_news, limit=limit)
        return _rows_to_news_items(rows)

    async def search(self, query: str, limit: int = 50) -> list[NewsItem]:
        rows = await asyncio.to_thread(select_news_by_query, query, limit=limit)
        return _rows_to_news_items(rows)

    async def recent(self, limit: int = 50) -> list[NewsItem]:
        rows = await asyncio.to_thread(select_recent_news, limit=limit)
        return _rows_to_news_items(rows)

    # -------------------------------------------------------- private: locks

    async def _acquire_lock(self) -> bool:
        token = datetime.now(tz=timezone.utc).isoformat()
        got = await self._redis.set(_INGEST_LOCK_KEY, token, nx=True, ex=_INGEST_LOCK_TTL)
        return bool(got)

    async def _release_lock(self) -> None:
        try:
            await self._redis.delete(_INGEST_LOCK_KEY)
        except Exception:
            logger.warning("[news] failed to release ingest lock", exc_info=True)

    # ---------------------------------------------------- private: dedup/db

    @staticmethod
    def _dedup_by_url(items: Iterable[NewsItem]) -> list[NewsItem]:
        by_url: dict[str, NewsItem] = {}
        for item in items:
            if not item.url:
                continue
            by_url.setdefault(item.url, item)
        return list(by_url.values())

    def _upsert_news(self, items: list[NewsItem]) -> int:
        rows = []
        for item in items:
            rows.append(
                {
                    "url": item.url,
                    "title": item.title,
                    "summary": item.summary,
                    "source": item.source,
                    "tickers": item.tickers or [],
                    "published_at": item.published_at.isoformat() if item.published_at else None,
                    "sentiment": item.sentiment,
                }
            )
        try:
            inserted = upsert_rows(
                "news_items",
                rows,
                on_conflict="url",
                ignore_duplicates=True,
            )
            return len(inserted)
        except Exception:
            logger.exception("[news] upsert_rows failed")
            return 0

    # ------------------------------------------------------ private: fetchers

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        retry_error_callback=lambda _: [],
    )
    async def _fetch_yahoo_rss(self) -> list[NewsItem]:
        try:
            parsed = await asyncio.to_thread(feedparser.parse, _YAHOO_RSS_URL)
        except Exception:
            logger.warning("[news] yahoo RSS parse failed", exc_info=True)
            return []
        items: list[NewsItem] = []
        for entry in getattr(parsed, "entries", []) or []:
            url = _coerce_str(getattr(entry, "link", None))
            title = _coerce_str(getattr(entry, "title", None))
            if not url or not title:
                continue
            items.append(
                NewsItem(
                    url=url,
                    title=title,
                    summary=_coerce_str(getattr(entry, "summary", None)),
                    source="yahoo_rss",
                    tickers=[],
                    published_at=_parse_feed_time(entry),
                )
            )
        return items

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        retry_error_callback=lambda _: [],
    )
    async def _fetch_coinbase_rss(self) -> list[NewsItem]:
        try:
            parsed = await asyncio.to_thread(feedparser.parse, _COINBASE_RSS_URL)
        except Exception:
            logger.warning("[news] coinbase RSS parse failed", exc_info=True)
            return []
        items: list[NewsItem] = []
        for entry in getattr(parsed, "entries", []) or []:
            url = _coerce_str(getattr(entry, "link", None))
            title = _coerce_str(getattr(entry, "title", None))
            if not url or not title:
                continue
            items.append(
                NewsItem(
                    url=url,
                    title=title,
                    summary=_coerce_str(getattr(entry, "summary", None)),
                    source="coinbase_rss",
                    tickers=["BTC-USD", "ETH-USD"],
                    published_at=_parse_feed_time(entry),
                )
            )
        return items

    async def _fetch_yfinance(self, tickers: list[str]) -> list[NewsItem]:
        """yfinance is sync + flaky; isolate each ticker in its own thread call."""
        coroutines = [self._fetch_yfinance_one(t) for t in tickers]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        out: list[NewsItem] = []
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                logger.warning("[news] yfinance fetch failed for %s: %s", ticker, result)
                continue
            out.extend(result or [])
        return out

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(Exception),
        retry_error_callback=lambda _: [],
    )
    async def _fetch_yfinance_one(self, ticker: str) -> list[NewsItem]:
        def _sync() -> list[dict[str, Any]]:
            import yfinance as yf

            return yf.Ticker(ticker).news or []

        raw = await asyncio.to_thread(_sync)
        items: list[NewsItem] = []
        for entry in raw:
            # Recent yfinance returns nested ``content`` dicts; old shape is flat.
            content = entry.get("content") if isinstance(entry, dict) else None
            if content and isinstance(content, dict):
                url = _coerce_str(
                    content.get("canonicalUrl", {}).get("url")
                    or content.get("clickThroughUrl", {}).get("url")
                )
                title = _coerce_str(content.get("title"))
                summary = _coerce_str(content.get("summary") or content.get("description"))
                pub_raw = content.get("pubDate")
                published_at: datetime | None = None
                if pub_raw:
                    try:
                        published_at = datetime.fromisoformat(str(pub_raw).replace("Z", "+00:00"))
                    except Exception:
                        published_at = None
            else:
                url = _coerce_str(entry.get("link"))
                title = _coerce_str(entry.get("title"))
                summary = _coerce_str(entry.get("summary") or entry.get("publisher"))
                ts = entry.get("providerPublishTime")
                published_at = (
                    datetime.fromtimestamp(int(ts), tz=timezone.utc) if ts else None
                )

            if not url or not title:
                continue
            items.append(
                NewsItem(
                    url=url,
                    title=title,
                    summary=summary,
                    source="yfinance",
                    tickers=[ticker],
                    published_at=published_at,
                )
            )
        return items

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(Exception),
        retry_error_callback=lambda _: [],
    )
    async def _fetch_tinyfish_search(self) -> list[NewsItem]:
        if self._tf_client is None:
            return []
        items: list[NewsItem] = []
        for query in _TINYFISH_QUERIES:
            try:
                results = await asyncio.to_thread(_tinyfish_search_call, self._tf_client, query)
            except Exception:
                logger.warning("[news] tinyfish search failed for %r", query, exc_info=True)
                continue
            for entry in results:
                url = _coerce_str(entry.get("url") or entry.get("link"))
                title = _coerce_str(entry.get("title"))
                if not url or not title:
                    continue
                items.append(
                    NewsItem(
                        url=url,
                        title=title,
                        summary=_coerce_str(entry.get("snippet") or entry.get("summary")),
                        source="tinyfish_search",
                        tickers=[],
                        published_at=None,
                    )
                )
        return items

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(Exception),
        retry_error_callback=lambda _: [],
    )
    async def _fetch_marketaux(self) -> list[NewsItem]:
        api_key = os.getenv("MARKETAUX_API_KEY", "").strip()
        if not api_key:
            return []
        params = {
            "api_token": api_key,
            "limit": 50,
            "language": "en",
            "filter_entities": "true",
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(_MARKETAUX_URL, params=params)
            resp.raise_for_status()
        except Exception:
            logger.warning("[news] marketaux fetch failed", exc_info=True)
            return []

        data = resp.json() if resp.content else {}
        items: list[NewsItem] = []
        for entry in data.get("data") or []:
            url = _coerce_str(entry.get("url"))
            title = _coerce_str(entry.get("title"))
            if not url or not title:
                continue
            pub_raw = entry.get("published_at")
            published_at: datetime | None = None
            if pub_raw:
                try:
                    published_at = datetime.fromisoformat(str(pub_raw).replace("Z", "+00:00"))
                except Exception:
                    published_at = None

            tickers: list[str] = []
            for ent in entry.get("entities") or []:
                sym = _coerce_str(ent.get("symbol"))
                if sym:
                    tickers.append(sym)

            items.append(
                NewsItem(
                    url=url,
                    title=title,
                    summary=_coerce_str(entry.get("description") or entry.get("snippet")),
                    source="marketaux",
                    tickers=tickers,
                    published_at=published_at,
                )
            )
        return items


def _tinyfish_search_call(tf_client: Any, query: str) -> list[dict[str, Any]]:
    """Adapt the TinyFish SDK's varying response shapes into a list of dicts."""
    search_api = getattr(tf_client, "search", None)
    if search_api is None:
        return []
    query_fn = getattr(search_api, "query", None) or getattr(search_api, "run", None)
    if query_fn is None:
        return []
    raw = query_fn(query)
    if raw is None:
        return []
    # Accept a few possible shapes.
    for key in ("results", "items", "data"):
        value = getattr(raw, key, None)
        if value is None and isinstance(raw, dict):
            value = raw.get(key)
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
    if isinstance(raw, list):
        return [v for v in raw if isinstance(v, dict)]
    return []


def _rows_to_news_items(rows: list[dict[str, Any]]) -> list[NewsItem]:
    out: list[NewsItem] = []
    for row in rows:
        try:
            out.append(NewsItem.model_validate(row))
        except Exception:
            logger.debug("[news] skipped row that failed validation: %r", row)
    return out


@lru_cache(maxsize=1)
def get_news_agent() -> NewsAggregatorAgent:
    """Singleton; tf_client injected lazily from app.main to avoid circular imports."""
    tf_client: Any | None = None
    try:
        from app.main import tf_client as _tf_client  # type: ignore

        tf_client = _tf_client
    except Exception:
        logger.warning("[news] TinyFish client unavailable at init — continuing without it")
    return NewsAggregatorAgent(redis=get_redis(), tf_client=tf_client)


__all__ = ["NewsAggregatorAgent", "get_news_agent"]
