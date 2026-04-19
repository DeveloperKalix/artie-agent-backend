from typing import Any

from pydantic import BaseModel, Field


class SnapTradeAccountBalance(BaseModel):
    amount: float | None = None
    currency: str | None = None


class SnapTradeAccount(BaseModel):
    """Rich brokerage account object returned by GET /snaptrade/accounts."""

    id: str
    name: str | None = None
    number: str | None = None
    institution_name: str | None = None
    account_type: str | None = Field(None, description="e.g. 'Margin', 'Cash', 'TFSA'")
    status: str | None = None
    is_paper: bool | None = None
    balance_total: float | None = Field(None, description="Total account value in account currency")
    balance_currency: str | None = None
    # Pass through the raw SnapTrade object for anything the frontend needs
    # that we haven't explicitly typed above.
    raw: dict[str, Any] = Field(default_factory=dict)


class SnapTradeConnectBody(BaseModel):
    """Start a SnapTrade connection session for an end user."""

    user_id: str = Field(..., min_length=1, description="Your app user id (e.g. Supabase auth user id)")
    custom_redirect: str | None = Field(
        None,
        description=(
            "Deep-link URI SnapTrade redirects to after the user connects "
            "(e.g. 'artie://snaptrade-complete'). "
            "If omitted the portal stays open after connection."
        ),
    )
    connection_type: str = Field(
        "read",
        description="'read' (default) or 'trade' to enable order placement",
    )
    broker: str | None = Field(
        None,
        description="Brokerage slug to pre-select (e.g. 'ALPACA'). Omit for user to choose.",
    )


class SnapTradeConnectResponse(BaseModel):
    """Returned after ensuring the user exists and generating a Connection Portal URL."""

    redirect_uri: str = Field(..., description="SnapTrade-hosted Connection Portal URL (open in WebBrowser).")
    url: str = Field(
        ...,
        description="Same as redirect_uri — provided for clients that expect a field named `url`.",
    )
    snaptrade_user_id: str
    session_id: str | None = Field(None, description="SnapTrade connection portal session id, when present.")


class SnapTradeAddBrokerBody(BaseModel):
    """Add another brokerage to an already-registered SnapTrade user."""

    user_id: str = Field(..., min_length=1)
    custom_redirect: str | None = Field(None, description="Deep-link URI to redirect back to after connecting.")
    connection_type: str = Field("read", description="'read' or 'trade'")
    broker: str | None = Field(None, description="Brokerage slug to pre-select, e.g. 'ALPACA'.")


class SnapTradeAccountsResponse(BaseModel):
    """Brokerage accounts linked via SnapTrade for a given app user."""

    accounts: list[SnapTradeAccount]


class ExchangeAccountsResponse(BaseModel):
    """GET /exchanges/accounts — brokerage data via SnapTrade only (not Plaid bank accounts)."""

    registered: bool = Field(
        ...,
        description="True if this user has a SnapTrade registration (after POST /snaptrade/connect).",
    )
    accounts: list[SnapTradeAccount] = Field(
        default_factory=list,
        description="Brokerage accounts from SnapTrade when linked; empty if not registered or no broker yet.",
    )


class SnapTradeHoldingsResponse(BaseModel):
    """Equity + option positions across all connected brokerage accounts."""

    holdings: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Positions (per-account cost basis, quantity, current price)
# ---------------------------------------------------------------------------

class SnapTradePosition(BaseModel):
    """A single equity/ETF/crypto position inside one brokerage account."""

    account_id: str
    symbol: str | None = Field(None, description="Ticker, e.g. 'AAPL'")
    description: str | None = Field(None, description="Company/fund name")
    quantity: float | None = None
    average_purchase_price: float | None = Field(None, description="Cost basis per share")
    current_price: float | None = Field(None, description="Last known market price")
    market_value: float | None = Field(None, description="quantity × current_price")
    unrealized_gain: float | None = Field(None, description="market_value − (quantity × cost_basis)")
    currency: str | None = None
    asset_class: str | None = Field(None, description="e.g. 'equity', 'crypto'")
    raw: dict[str, Any] = Field(default_factory=dict)


class SnapTradePositionsResponse(BaseModel):
    positions: list[SnapTradePosition]


# ---------------------------------------------------------------------------
# Transactions / Activity history
# ---------------------------------------------------------------------------

class SnapTradeTransaction(BaseModel):
    """A single historical transaction (BUY, SELL, DIVIDEND, etc.)."""

    account_id: str | None = None
    transaction_id: str | None = None
    type: str | None = Field(None, description="BUY | SELL | DIVIDEND | INTEREST | CONTRIBUTION | WITHDRAWAL")
    symbol: str | None = None
    description: str | None = None
    quantity: float | None = None
    price: float | None = None
    amount: float | None = Field(None, description="Total dollar amount of the transaction")
    currency: str | None = None
    trade_date: str | None = None
    settlement_date: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class SnapTradeTransactionsResponse(BaseModel):
    transactions: list[SnapTradeTransaction]


# ---------------------------------------------------------------------------
# Orders (live status: pending, filled, canceled)
# ---------------------------------------------------------------------------

class SnapTradeOrder(BaseModel):
    """A brokerage order with live status."""

    account_id: str | None = None
    brokerage_order_id: str | None = None
    symbol: str | None = None
    action: str | None = Field(None, description="BUY or SELL")
    order_type: str | None = Field(None, description="Market, Limit, Stop, etc.")
    status: str | None = Field(None, description="pending, filled, canceled, etc.")
    quantity: float | None = None
    limit_price: float | None = None
    filled_quantity: float | None = None
    execution_price: float | None = None
    time_placed: str | None = None
    time_filled: str | None = None
    time_in_force: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class SnapTradeOrdersResponse(BaseModel):
    orders: list[SnapTradeOrder]
