"""
Reusable Supabase REST helpers (PostgREST via supabase-py).

Prefer these functions over calling ``client.table(...)`` directly so filtering
and return shapes stay consistent across services.
"""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Mapping, Sequence, TypeVar

from postgrest.types import ReturnMethod
from supabase import Client, create_client

JSON = dict[str, Any]
T = TypeVar("T")

__all__ = [
    "get_supabase",
    "insert_row",
    "insert_rows",
    "upsert_row",
    "upsert_rows",
    "update_rows",
    "delete_rows",
    "select_rows",
    "select_one",
    "select_maybe",
    "select_news_for_tickers",
    "select_news_by_query",
    "select_recent_news",
    "rpc",
]


@lru_cache
def get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    # Prefer the service_role key for server-side writes (bypasses RLS).
    # Falls back to SUPABASE_KEY (anon/publishable) if not set.
    key = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def _eq_filters(builder: T, filters: Mapping[str, Any] | None) -> T:
    if not filters:
        return builder
    b = builder
    for key, value in filters.items():
        b = b.eq(key, value)
    return b


def insert_row(
    table: str,
    row: JSON,
    *,
    returning: ReturnMethod = ReturnMethod.representation,
) -> list[Any]:
    res = get_supabase().table(table).insert(row, returning=returning).execute()
    return res.data or []


def insert_rows(
    table: str,
    rows: Sequence[JSON],
    *,
    returning: ReturnMethod = ReturnMethod.representation,
    default_to_null: bool = True,
) -> list[Any]:
    res = (
        get_supabase()
        .table(table)
        .insert(
            list(rows),
            returning=returning,
            default_to_null=default_to_null,
        )
        .execute()
    )
    return res.data or []


def upsert_row(
    table: str,
    row: JSON,
    *,
    on_conflict: str | None = None,
    ignore_duplicates: bool = False,
    returning: ReturnMethod = ReturnMethod.representation,
) -> list[Any]:
    kwargs: dict[str, Any] = {
        "returning": returning,
        "ignore_duplicates": ignore_duplicates,
        "default_to_null": True,
    }
    if on_conflict:
        kwargs["on_conflict"] = on_conflict
    res = get_supabase().table(table).upsert(row, **kwargs).execute()
    return res.data or []


def upsert_rows(
    table: str,
    rows: Sequence[JSON],
    *,
    on_conflict: str | None = None,
    ignore_duplicates: bool = False,
    returning: ReturnMethod = ReturnMethod.representation,
) -> list[Any]:
    kwargs: dict[str, Any] = {
        "returning": returning,
        "ignore_duplicates": ignore_duplicates,
        "default_to_null": True,
    }
    if on_conflict:
        kwargs["on_conflict"] = on_conflict
    res = get_supabase().table(table).upsert(list(rows), **kwargs).execute()
    return res.data or []


def update_rows(
    table: str,
    patch: JSON,
    *,
    filters: Mapping[str, Any],
    returning: ReturnMethod = ReturnMethod.representation,
) -> list[Any]:
    if not filters:
        raise ValueError("update_rows requires at least one eq filter to avoid full-table updates")
    q = get_supabase().table(table).update(patch, returning=returning)
    q = _eq_filters(q, filters)
    res = q.execute()
    return res.data or []


def delete_rows(
    table: str,
    *,
    filters: Mapping[str, Any],
    returning: ReturnMethod = ReturnMethod.representation,
) -> list[Any]:
    if not filters:
        raise ValueError("delete_rows requires at least one eq filter to avoid full-table deletes")
    q = get_supabase().table(table).delete(returning=returning)
    q = _eq_filters(q, filters)
    res = q.execute()
    return res.data or []


def select_rows(
    table: str,
    *columns: str,
    filters: Mapping[str, Any] | None = None,
    order_column: str | None = None,
    descending: bool = False,
    limit: int | None = None,
    offset: int | None = None,
) -> list[Any]:
    cols: tuple[str, ...] = columns if columns else ("*",)
    q = get_supabase().table(table).select(*cols)
    q = _eq_filters(q, filters)
    if order_column is not None:
        q = q.order(order_column, desc=descending)
    if limit is not None:
        q = q.limit(limit)
    if offset is not None:
        q = q.offset(offset)
    res = q.execute()
    return res.data or []


def select_one(
    table: str,
    *columns: str,
    filters: Mapping[str, Any],
) -> JSON:
    rows = select_rows(table, *columns, filters=filters, limit=2)
    if not rows:
        raise LookupError(f"No row in {table!r} matching filters")
    if len(rows) > 1:
        raise LookupError(f"Expected one row in {table!r}, got {len(rows)}")
    return rows[0]


def select_maybe(
    table: str,
    *columns: str,
    filters: Mapping[str, Any],
) -> JSON | None:
    rows = select_rows(table, *columns, filters=filters, limit=1)
    return rows[0] if rows else None


def rpc(fn_name: str, params: JSON | None = None) -> Any:
    res = get_supabase().rpc(fn_name, params or {}).execute()
    return res.data


def select_news_for_tickers(
    tickers: Sequence[str],
    *,
    since: datetime | None = None,
    limit: int = 50,
) -> list[JSON]:
    """Return news_items whose ``tickers[]`` overlaps the provided list.

    Uses the PostgREST ``overlaps`` filter (``&&`` in raw SQL) — the generic
    ``select_rows`` helper can't express that since it only emits ``eq``.
    Ordered by ``published_at DESC`` so freshly ingested rows surface first.
    """
    q = (
        get_supabase()
        .table("news_items")
        .select("*")
        .overlaps("tickers", list(tickers))
    )
    if since is not None:
        q = q.gte("published_at", since.isoformat())
    q = q.order("published_at", desc=True).limit(limit)
    res = q.execute()
    return res.data or []


def select_news_by_query(query: str, *, limit: int = 50) -> list[JSON]:
    """Free-text search over title + summary via PostgREST ``or`` + ``ilike``."""
    pattern = f"%{query}%"
    res = (
        get_supabase()
        .table("news_items")
        .select("*")
        .or_(f"title.ilike.{pattern},summary.ilike.{pattern}")
        .order("published_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


def select_recent_news(*, limit: int = 50) -> list[JSON]:
    """Most recent news items ordered by ``published_at DESC``."""
    res = (
        get_supabase()
        .table("news_items")
        .select("*")
        .order("published_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []
