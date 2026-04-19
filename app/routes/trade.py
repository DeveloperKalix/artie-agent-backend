"""Trading endpoints (equities + crypto) — mounted at ``/trade``.

All endpoints are multi-tenant via the ``X-User-Id`` header. State changes
go through ``OrderIntentService`` so the state machine and audit log are
never bypassed.

See ``app/backend/services/order_intents.py`` for the invariants enforced
on each transition (risk checks, market-closed fallback, etc.) and
``docs/frontend-phase3-integration.md`` (Phase 4) for the UI integration
guide.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from snaptrade_client import ApiException

from app.backend.services.market_calendar import (
    MarketCalendarService,
    get_market_calendar,
)
from app.backend.services.order_intents import (
    IntentNotFound,
    InvalidStateTransition,
    OrderIntentService,
    RiskCheckFailed,
    TradingDisabledError,
    get_order_intent_service,
)
from app.backend.services.snaptrade import (
    SnapTradeService,
    get_snaptrade_service,
)
from app.routes._deps import require_user_id
from app.schemas.trade import (
    CreateOrderIntent,
    MarkAcknowledgedResponse,
    OrderIntent,
    OrderIntentResponse,
    OrderIntentsListResponse,
    QuoteResponse,
    SymbolSearchResponse,
    SymbolSearchResult,
    TradeRemindersResponse,
    TradeStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trade", tags=["trade"])


# ---------------------------------------------------------------------------
# Status / reminders / symbol search / quote
# ---------------------------------------------------------------------------


@router.get("/status", response_model=TradeStatusResponse)
async def get_trade_status(
    user_id: str = Depends(require_user_id),
    intents: OrderIntentService = Depends(get_order_intent_service),
) -> TradeStatusResponse:
    """Cheap badge poll — no SnapTrade calls. Safe to hit on app foreground."""
    data = await intents.status(user_id)
    return TradeStatusResponse(**data)


@router.get("/reminders", response_model=TradeRemindersResponse)
async def list_reminders(
    user_id: str = Depends(require_user_id),
    intents: OrderIntentService = Depends(get_order_intent_service),
) -> TradeRemindersResponse:
    """Intents whose reminders fired but which the user hasn't acknowledged."""
    items = await intents.reminder_queue(user_id)
    return TradeRemindersResponse(reminders=items)


class SymbolSearchBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=64)
    account_id: str | None = Field(
        None,
        description=(
            "When provided, the search is scoped to symbols that broker "
            "supports — preferred for accurate order routing."
        ),
    )


@router.post("/symbols/search", response_model=SymbolSearchResponse)
async def search_symbols(
    body: SymbolSearchBody,
    user_id: str = Depends(require_user_id),
    snap: SnapTradeService = Depends(get_snaptrade_service),
) -> SymbolSearchResponse:
    try:
        raw = await _to_thread(
            snap.search_symbol, user_id, account_id=body.account_id, query=body.query
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ApiException as e:
        raise HTTPException(
            status_code=502, detail=f"SnapTrade symbol search failed: {e}"
        ) from e

    results: list[SymbolSearchResult] = []
    for item in raw if isinstance(raw, list) else []:
        symbol = _pick_symbol(item)
        if not symbol:
            continue
        results.append(
            SymbolSearchResult(
                symbol=symbol,
                description=item.get("description") or item.get("raw_symbol_description"),
                universal_symbol_id=item.get("id") or item.get("universal_symbol_id"),
                exchange=_pick_exchange(item),
                currency=_pick_currency(item),
                raw=item,
            )
        )
    return SymbolSearchResponse(results=results)


@router.get("/quote", response_model=QuoteResponse)
async def get_quote(
    account_id: str = Query(...),
    symbol: str = Query(..., min_length=1, max_length=32),
    user_id: str = Depends(require_user_id),
    snap: SnapTradeService = Depends(get_snaptrade_service),
) -> QuoteResponse:
    try:
        quotes = await _to_thread(
            snap.account_quote,
            user_id,
            account_id=account_id,
            symbols=symbol.upper(),
            use_ticker=True,
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ApiException as e:
        raise HTTPException(status_code=502, detail=f"SnapTrade quote failed: {e}") from e

    quote = quotes[0] if isinstance(quotes, list) and quotes else {}
    return QuoteResponse(
        symbol=symbol.upper(),
        bid=_to_float(quote.get("bid_price") or quote.get("bid")),
        ask=_to_float(quote.get("ask_price") or quote.get("ask")),
        last=_to_float(quote.get("last_trade_price") or quote.get("last")),
        currency=quote.get("currency"),
        raw=quote if isinstance(quote, dict) else {},
    )


# ---------------------------------------------------------------------------
# Intent lifecycle
# ---------------------------------------------------------------------------


@router.post("/intents", response_model=OrderIntentResponse)
async def create_intent(
    body: CreateOrderIntent,
    user_id: str = Depends(require_user_id),
    intents: OrderIntentService = Depends(get_order_intent_service),
) -> OrderIntentResponse:
    try:
        intent = await intents.create_from_ui(user_id, body)
    except TradingDisabledError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    return OrderIntentResponse(intent=intent)


@router.post("/intents/{intent_id}/preview", response_model=OrderIntentResponse)
async def preview_intent(
    intent_id: str,
    user_id: str = Depends(require_user_id),
    intents: OrderIntentService = Depends(get_order_intent_service),
) -> OrderIntentResponse:
    try:
        intent = await intents.preview(user_id, intent_id)
    except IntentNotFound as e:
        raise HTTPException(status_code=404, detail="intent not found") from e
    except InvalidStateTransition as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ApiException as e:
        raise HTTPException(status_code=502, detail=f"SnapTrade preview failed: {e}") from e
    return OrderIntentResponse(intent=intent)


@router.post("/intents/{intent_id}/confirm", response_model=OrderIntentResponse)
async def confirm_intent(
    intent_id: str,
    user_id: str = Depends(require_user_id),
    intents: OrderIntentService = Depends(get_order_intent_service),
) -> OrderIntentResponse:
    try:
        intent = await intents.confirm(user_id, intent_id)
    except IntentNotFound as e:
        raise HTTPException(status_code=404, detail="intent not found") from e
    except TradingDisabledError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except RiskCheckFailed as e:
        raise HTTPException(
            status_code=422,
            detail={"message": str(e), "risk_checks": e.checks},
        ) from e
    except InvalidStateTransition as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ApiException as e:
        raise HTTPException(status_code=502, detail=f"SnapTrade submit failed: {e}") from e
    return OrderIntentResponse(intent=intent)


@router.post("/intents/{intent_id}/cancel", response_model=OrderIntentResponse)
async def cancel_intent(
    intent_id: str,
    user_id: str = Depends(require_user_id),
    intents: OrderIntentService = Depends(get_order_intent_service),
) -> OrderIntentResponse:
    try:
        intent = await intents.cancel(user_id, intent_id)
    except IntentNotFound as e:
        raise HTTPException(status_code=404, detail="intent not found") from e
    return OrderIntentResponse(intent=intent)


@router.post(
    "/intents/{intent_id}/acknowledge", response_model=MarkAcknowledgedResponse
)
async def acknowledge_reminder(
    intent_id: str,
    user_id: str = Depends(require_user_id),
    intents: OrderIntentService = Depends(get_order_intent_service),
) -> MarkAcknowledgedResponse:
    try:
        await intents.acknowledge_reminder(user_id, intent_id)
    except IntentNotFound as e:
        raise HTTPException(status_code=404, detail="intent not found") from e
    return MarkAcknowledgedResponse(acknowledged=True)


@router.get("/intents/{intent_id}", response_model=OrderIntentResponse)
async def get_intent(
    intent_id: str,
    user_id: str = Depends(require_user_id),
    intents: OrderIntentService = Depends(get_order_intent_service),
) -> OrderIntentResponse:
    try:
        intent = await intents.get(user_id, intent_id)
    except IntentNotFound as e:
        raise HTTPException(status_code=404, detail="intent not found") from e
    return OrderIntentResponse(intent=intent)


@router.get("/intents", response_model=OrderIntentsListResponse)
async def list_intents(
    status_filter: str | None = Query(
        None,
        alias="status",
        description=(
            "Filter: a single status (e.g. 'submitted') or a comma-"
            "separated list (e.g. 'previewed,confirmed,scheduled_for_market_open')."
        ),
    ),
    limit: int = Query(50, ge=1, le=200),
    user_id: str = Depends(require_user_id),
    intents: OrderIntentService = Depends(get_order_intent_service),
) -> OrderIntentsListResponse:
    statuses: str | list[str] | None = None
    if status_filter:
        parts = [s.strip() for s in status_filter.split(",") if s.strip()]
        statuses = parts[0] if len(parts) == 1 else parts
    rows = await intents.list_for_user(user_id, status=statuses, limit=limit)
    return OrderIntentsListResponse(intents=rows)


@router.get("/market/open")
async def market_open(
    asset_class: str = Query("equity"),
    calendar: MarketCalendarService = Depends(get_market_calendar),
) -> dict[str, Any]:
    """Convenience endpoint — returns whether the exchange is currently open
    and when the next open is. Used by the preview sheet to show
    "Markets closed — I'll remind you at …" alongside Confirm.
    """
    if asset_class not in ("equity", "crypto"):
        raise HTTPException(status_code=422, detail="asset_class must be equity or crypto")
    now_open = calendar.is_open(asset_class)
    next_open = None if now_open else calendar.next_open(asset_class)
    return {
        "asset_class": asset_class,
        "open": now_open,
        "next_open": next_open.isoformat() if next_open else None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _to_thread(func, *args, **kwargs):
    import asyncio

    return await asyncio.to_thread(func, *args, **kwargs)


def _pick_symbol(item: dict[str, Any]) -> str | None:
    sym = item.get("symbol")
    if isinstance(sym, dict):
        sym = sym.get("symbol") or sym.get("raw_symbol")
    if isinstance(sym, str) and sym.strip():
        return sym.strip().upper()
    for key in ("raw_symbol", "ticker"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip().upper()
    return None


def _pick_exchange(item: dict[str, Any]) -> str | None:
    ex = item.get("exchange") or (
        item.get("symbol", {}).get("exchange") if isinstance(item.get("symbol"), dict) else None
    )
    if isinstance(ex, dict):
        return ex.get("code") or ex.get("name")
    if isinstance(ex, str):
        return ex
    return None


def _pick_currency(item: dict[str, Any]) -> str | None:
    cur = item.get("currency")
    if isinstance(cur, dict):
        return cur.get("code")
    if isinstance(cur, str):
        return cur
    return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["router"]


async def _assert_intent_ownership(intent: OrderIntent, user_id: str) -> None:
    """Defense-in-depth — the service already filters by user_id."""
    if intent.user_id != user_id:
        raise HTTPException(status_code=404, detail="intent not found")
