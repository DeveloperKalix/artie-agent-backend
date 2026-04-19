"""Pydantic schemas for the trading flow.

Shared between the HTTP layer (``app/routes/trade.py``), the LLM
tool-calling layer (``app/backend/services/agent_orchestrator.py``), and
the core state machine (``app/backend/services/order_intents.py``).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

AssetClass = Literal["equity", "crypto"]
OrderAction = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop", "stop_limit"]
TimeInForce = Literal["day", "gtc", "fok", "ioc"]
IntentSource = Literal["ui", "llm"]

IntentStatus = Literal[
    "awaiting_confirmation",
    "previewed",
    "confirmed",
    "scheduled_for_market_open",
    "submitted",
    "filled",
    "cancelled",
    "failed",
    "rejected",
]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class CreateOrderIntent(BaseModel):
    """Body for ``POST /trade/intents``.

    Either ``units`` OR ``notional_value`` must be provided, not both.
    ``limit``/``stop_limit`` require ``price``; ``stop``/``stop_limit``
    require ``stop``. The service layer enforces these shape rules before
    calling SnapTrade.
    """

    asset_class: AssetClass
    account_id: str = Field(..., min_length=1)
    symbol: str = Field(..., min_length=1, description="Ticker (equity) or pair (e.g. 'BTC/USD').")
    universal_symbol_id: Optional[str] = Field(
        None,
        description=(
            "Preferred for equities — SnapTrade's unambiguous symbol id. "
            "If omitted we fall back to the ticker, which may fail for "
            "symbols listed on multiple exchanges."
        ),
    )
    action: OrderAction
    order_type: OrderType = "market"
    time_in_force: TimeInForce = "day"
    units: Optional[float] = Field(None, gt=0)
    notional_value: Optional[float] = Field(None, gt=0)
    price: Optional[float] = Field(None, gt=0)
    stop: Optional[float] = Field(None, gt=0)
    conversation_id: Optional[str] = Field(
        None,
        description=(
            "When the order is initiated from a chat turn, passing the "
            "conversation id lets the market-open reminder post back "
            "into the same thread."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one_quantity(self) -> "CreateOrderIntent":
        if (self.units is None) == (self.notional_value is None):
            raise ValueError("Exactly one of 'units' or 'notional_value' must be set")
        if self.order_type in ("limit", "stop_limit") and self.price is None:
            raise ValueError(f"order_type={self.order_type!r} requires 'price'")
        if self.order_type in ("stop", "stop_limit") and self.stop is None:
            raise ValueError(f"order_type={self.order_type!r} requires 'stop'")
        return self


class LLMProposedOrder(BaseModel):
    """Shape emitted by the LLM via the ``propose_order`` tool call.

    This mirrors :class:`CreateOrderIntent` but is *more lenient* — we let
    the model omit ``universal_symbol_id`` entirely and resolve it server-
    side. The orchestrator validates the resolved shape before persisting.
    """

    asset_class: AssetClass
    account_id: Optional[str] = Field(
        None,
        description=(
            "SnapTrade account id. Optional — when absent the server picks "
            "the best available account for the user automatically."
        ),
    )
    symbol: str = Field(..., description="Ticker (equity) or pair (crypto)")
    action: OrderAction
    order_type: OrderType = "market"
    time_in_force: TimeInForce = "day"
    units: Optional[float] = None
    notional_value: Optional[float] = None
    price: Optional[float] = None
    stop: Optional[float] = None
    rationale: Optional[str] = Field(
        None,
        description="Free-text explanation the LLM gives for why this order was proposed",
    )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


class ImpactPreview(BaseModel):
    """Normalised view of SnapTrade's ``get_order_impact`` response.

    SnapTrade returns a mix of camelCase/snake_case depending on the
    endpoint — we smooth that out here so the frontend has a stable
    contract.
    """

    trade_id: str | None = Field(None, description="Ephemeral SnapTrade trade id — valid ~5 min")
    symbol: str | None = None
    action: str | None = None
    order_type: str | None = None
    time_in_force: str | None = None
    units: float | None = None
    price: float | None = None
    estimated_value: float | None = Field(None, description="units × price or notional_value")
    commission: float | None = None
    currency: str | None = None
    buying_power_effect: float | None = None
    remaining_buying_power: float | None = None
    expires_at: str | None = Field(None, description="ISO-8601 — when the preview stops being valid")
    warnings: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class OrderIntent(BaseModel):
    """Durable order intent visible to the UI. Lives in ``order_intents``."""

    id: str
    user_id: str
    conversation_id: str | None = None
    source: IntentSource
    asset_class: AssetClass

    account_id: str
    universal_symbol_id: str | None = None
    symbol: str
    action: OrderAction
    order_type: OrderType
    time_in_force: TimeInForce
    units: float | None = None
    notional_value: float | None = None
    price: float | None = None
    stop: float | None = None

    status: IntentStatus
    snaptrade_trade_id: str | None = None
    snaptrade_order_id: str | None = None
    impact: ImpactPreview | None = None
    impact_expires_at: str | None = None
    estimated_value: float | None = None

    scheduled_for: str | None = Field(
        None,
        description="When trading resumes for this asset class (e.g. next NYSE open).",
    )
    reminder_fires_at: str | None = None
    reminded_at: str | None = None
    acknowledged_at: str | None = None
    confirmed_at: str | None = None
    submitted_at: str | None = None
    filled_at: str | None = None

    error: dict[str, Any] | None = None
    risk_checks: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class OrderIntentResponse(BaseModel):
    """Single-object wrapper so the route can add status-specific extras later."""

    intent: OrderIntent


class OrderIntentsListResponse(BaseModel):
    intents: list[OrderIntent]


class TradeStatusResponse(BaseModel):
    """Cheap badge poll for the UI (no SnapTrade calls)."""

    pending_reminder_count: int
    active_intent_count: int
    market_open: bool
    next_market_open: str | None
    trading_enabled: bool
    disclaimer_acknowledged: bool


class TradeRemindersResponse(BaseModel):
    """Intents that have fired a reminder and are waiting for the user to re-confirm."""

    reminders: list[OrderIntent]


class SymbolSearchResult(BaseModel):
    symbol: str
    description: str | None = None
    universal_symbol_id: str | None = None
    exchange: str | None = None
    currency: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class SymbolSearchResponse(BaseModel):
    results: list[SymbolSearchResult]


class QuoteResponse(BaseModel):
    symbol: str
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    currency: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class MarkAcknowledgedResponse(BaseModel):
    acknowledged: bool


__all__ = [
    "AssetClass",
    "OrderAction",
    "OrderType",
    "TimeInForce",
    "IntentSource",
    "IntentStatus",
    "CreateOrderIntent",
    "LLMProposedOrder",
    "ImpactPreview",
    "OrderIntent",
    "OrderIntentResponse",
    "OrderIntentsListResponse",
    "TradeStatusResponse",
    "TradeRemindersResponse",
    "SymbolSearchResult",
    "SymbolSearchResponse",
    "QuoteResponse",
    "MarkAcknowledgedResponse",
]
