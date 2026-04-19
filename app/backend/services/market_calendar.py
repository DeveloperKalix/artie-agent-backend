"""Market calendar service — NYSE (XNYS) open/close + next-open lookup.

Used by the trading flow to decide whether an equity order should be
executed immediately or deferred to a "scheduled_for_market_open" state
with a reminder.

Design notes
------------
- **exchange_calendars** is the industry-standard library for exchange
  trading calendars (holidays, early closes, weekend rules). We use it as
  the source of truth rather than hand-rolling a schedule.
- Crypto has **no** trading hours. ``is_open("crypto", ...)`` always
  returns ``True`` and ``next_open`` for crypto returns the current time.
- The calendar itself is relatively expensive to construct (pulls a few
  years of sessions up front). We build it once per process via
  ``@lru_cache`` and keep a Redis-backed short-lived day schedule cache so
  the hot path (``is_open`` on every confirm) doesn't recompute pandas
  timestamps.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

import exchange_calendars as xcals
import pandas as pd

logger = logging.getLogger(__name__)

AssetClass = Literal["equity", "crypto"]
SessionName = Literal["regular", "pre", "after", "closed"]

_NYSE_CODE = "XNYS"


@lru_cache(maxsize=1)
def _nyse() -> xcals.ExchangeCalendar:
    """Memoised NYSE calendar — construction is ~100 ms, reuse it."""
    return xcals.get_calendar(_NYSE_CODE)


def _utc(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.now(tz=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _utc_ts(dt: datetime | None) -> pd.Timestamp:
    """Build a pandas UTC timestamp that ``exchange_calendars`` will accept.

    ``exchange_calendars`` introspects ``ts.tz.key`` (see
    ``calendar_helpers.parse_date`` / ``parse_timestamp``) which only exists
    on ``zoneinfo.ZoneInfo`` instances — not on the stdlib
    ``datetime.timezone.utc`` singleton. Passing ``tz="UTC"`` explicitly to
    ``pd.Timestamp`` attaches pandas' own zoneinfo-backed UTC, which has the
    ``.key`` attribute the library expects.
    """
    base = _utc(dt).replace(tzinfo=None)
    return pd.Timestamp(base, tz="UTC")


class MarketCalendarService:
    """Stateless facade over ``exchange_calendars``.

    Intentionally synchronous — each call is a pandas lookup on an
    in-memory schedule, cheap enough to run on the event loop. We never
    block on network I/O here.
    """

    def is_open(self, asset_class: AssetClass, at: datetime | None = None) -> bool:
        """Return True iff the given asset class can trade at ``at`` (UTC)."""
        if asset_class == "crypto":
            return True
        ts = _utc_ts(at)
        return bool(_nyse().is_trading_minute(ts))

    def next_open(
        self, asset_class: AssetClass = "equity", at: datetime | None = None
    ) -> datetime:
        """Next UTC datetime when the market will open for this asset class.

        For crypto we return ``at`` (always open). For equities we jump to
        the start of the next NYSE regular session (9:30 AM ET).

        ``exchange_calendars`` session lookups require a tz-naive
        ``pd.Timestamp`` (it treats them as calendar dates in the
        exchange's timezone). ``is_trading_minute`` on the other hand is
        happy with a tz-aware minute. We use the appropriate form for each.
        """
        now = _utc(at)
        if asset_class == "crypto":
            return now
        cal = _nyse()
        ts_aware = _utc_ts(now)
        if cal.is_trading_minute(ts_aware):
            return now
        # Session API wants a tz-naive date. tz_convert(None) drops the tz
        # cleanly after normalising to midnight.
        session_date = ts_aware.normalize().tz_convert(None)
        try:
            session = cal.next_session(session_date)
        except Exception:
            # Fall back to the current session if today is open later.
            session = cal.date_to_session(session_date, direction="next")
        open_ts = cal.session_open(session)
        return open_ts.to_pydatetime().astimezone(timezone.utc)

    def session_for(
        self, asset_class: AssetClass, at: datetime | None = None
    ) -> SessionName:
        """Classify ``at`` as regular/pre/after/closed for this asset class.

        Pre/post-market windows for NYSE are 4:00 AM ET and 8:00 PM ET
        respectively. SnapTrade's ``trading_session`` field expects these
        names so the trading wrapper can pass them through when a user
        explicitly opts into extended-hours trading (out of scope for v1).
        """
        if asset_class == "crypto":
            return "regular"
        cal = _nyse()
        ts = _utc_ts(at)
        if cal.is_trading_minute(ts):
            return "regular"
        try:
            session = cal.minute_to_session(ts, direction="none")
        except Exception:
            session = None
        if session is None:
            return "closed"
        # session is a valid trading day; check pre/post around it
        open_ts = cal.session_open(session)
        close_ts = cal.session_close(session)
        if ts < open_ts:
            return "pre"
        if ts > close_ts:
            return "after"
        return "closed"


@lru_cache(maxsize=1)
def get_market_calendar() -> MarketCalendarService:
    return MarketCalendarService()


__all__ = [
    "AssetClass",
    "SessionName",
    "MarketCalendarService",
    "get_market_calendar",
]
