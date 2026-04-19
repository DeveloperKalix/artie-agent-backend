import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from snaptrade_client import ApiException

from app.backend.services.snaptrade import SnapTradeService, get_snaptrade_service
from app.schemas.snaptrade import (
    SnapTradeAccountsResponse,
    SnapTradeAddBrokerBody,
    SnapTradeConnectBody,
    SnapTradeConnectResponse,
    SnapTradeHoldingsResponse,
    SnapTradeOrdersResponse,
    SnapTradePositionsResponse,
    SnapTradeTransactionsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/snaptrade", tags=["snaptrade"])


def _snaptrade_http(exc: ApiException) -> HTTPException:
    try:
        detail = exc.body if isinstance(exc.body, str) else str(exc.body)
    except Exception:
        detail = str(exc)
    code = getattr(exc, "status", None) or 400
    if not (400 <= code <= 599):
        code = 400
    return HTTPException(status_code=code, detail=detail)


def _resolve_user_id(
    user_id: str | None = Query(None),
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> str:
    uid = (user_id or x_user_id or "").strip()
    if not uid:
        raise HTTPException(
            status_code=422,
            detail="Missing user_id: use ?user_id=<id> or header X-User-Id: <id>",
        )
    return uid


@router.post("/connect", response_model=SnapTradeConnectResponse)
async def connect(
    body: SnapTradeConnectBody,
    svc: SnapTradeService = Depends(get_snaptrade_service),
) -> SnapTradeConnectResponse:
    """Register (or look up) a SnapTrade user and return a Connection Portal URL.

    The frontend opens ``redirect_uri`` in an in-app browser. SnapTrade redirects
    to ``custom_redirect`` (e.g. ``artie://snaptrade-complete``) when the user
    finishes connecting their brokerage.
    """
    logger.info(
        "[snaptrade] POST /snaptrade/connect user_id=%s broker=%s connection_type=%s",
        body.user_id,
        body.broker,
        body.connection_type,
    )
    try:
        result = svc.get_connection_url(
            body.user_id,
            custom_redirect=body.custom_redirect,
            connection_type=body.connection_type,
            broker=body.broker,
        )
        return SnapTradeConnectResponse(**result)
    except ApiException as e:
        raise _snaptrade_http(e) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.post("/add-broker", response_model=SnapTradeConnectResponse)
async def add_broker(
    body: SnapTradeAddBrokerBody,
    svc: SnapTradeService = Depends(get_snaptrade_service),
) -> SnapTradeConnectResponse:
    """Generate a Connection Portal URL to add *another* brokerage for an existing user.

    Unlike ``/connect``, this never calls ``register_snap_trade_user`` — safe to
    call repeatedly without hitting the personal-key "one user" limit.
    The frontend opens ``redirect_uri`` in WebBrowser the same way as ``/connect``.
    """
    logger.info("[snaptrade] POST /snaptrade/add-broker user_id=%s broker=%s", body.user_id, body.broker)
    try:
        result = svc.add_broker(
            body.user_id,
            custom_redirect=body.custom_redirect,
            connection_type=body.connection_type,
            broker=body.broker,
        )
        return SnapTradeConnectResponse(**result)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ApiException as e:
        raise _snaptrade_http(e) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.get("/accounts", response_model=SnapTradeAccountsResponse)
async def get_accounts(
    user_id: str = Depends(_resolve_user_id),
    svc: SnapTradeService = Depends(get_snaptrade_service),
) -> SnapTradeAccountsResponse:
    """List all connected brokerage accounts for the user."""
    logger.info("[snaptrade] GET /snaptrade/accounts user_id=%s", user_id)
    try:
        accounts = svc.get_accounts(user_id)
        return SnapTradeAccountsResponse(accounts=accounts)
    except ApiException as e:
        raise _snaptrade_http(e) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.get("/holdings", response_model=SnapTradeHoldingsResponse)
async def get_holdings(
    user_id: str = Depends(_resolve_user_id),
    svc: SnapTradeService = Depends(get_snaptrade_service),
) -> SnapTradeHoldingsResponse:
    """Full SnapTrade holdings bundle (accounts + positions + orders) per connection."""
    logger.info("[snaptrade] GET /snaptrade/holdings user_id=%s", user_id)
    try:
        holdings = svc.get_holdings(user_id)
        return SnapTradeHoldingsResponse(holdings=holdings)
    except ApiException as e:
        raise _snaptrade_http(e) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.get("/positions", response_model=SnapTradePositionsResponse)
async def get_positions(
    user_id: str = Depends(_resolve_user_id),
    svc: SnapTradeService = Depends(get_snaptrade_service),
) -> SnapTradePositionsResponse:
    """Equity/ETF/crypto positions across all accounts with cost basis and current price.

    Key fields for AI suggestions: ``symbol``, ``quantity``, ``average_purchase_price``
    (cost basis), ``current_price``, ``market_value``, ``unrealized_gain``.
    """
    logger.info("[snaptrade] GET /snaptrade/positions user_id=%s", user_id)
    try:
        positions = svc.get_positions(user_id)
        return SnapTradePositionsResponse(positions=positions)
    except ApiException as e:
        raise _snaptrade_http(e) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.get("/transactions", response_model=SnapTradeTransactionsResponse)
async def get_transactions(
    user_id: str = Depends(_resolve_user_id),
    start_date: str | None = Query(None, description="ISO-8601 start date, e.g. '2025-01-01'"),
    end_date: str | None = Query(None, description="ISO-8601 end date, e.g. '2026-04-19'"),
    svc: SnapTradeService = Depends(get_snaptrade_service),
) -> SnapTradeTransactionsResponse:
    """Historical transactions (BUY, SELL, DIVIDEND, INTEREST, etc.) across all accounts.

    Supports optional ``start_date`` / ``end_date`` query params for date filtering.
    """
    logger.info("[snaptrade] GET /snaptrade/transactions user_id=%s start=%s end=%s", user_id, start_date, end_date)
    try:
        txs = svc.get_transactions(user_id, start_date=start_date, end_date=end_date)
        return SnapTradeTransactionsResponse(transactions=txs)
    except ApiException as e:
        raise _snaptrade_http(e) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.get("/orders", response_model=SnapTradeOrdersResponse)
async def get_orders(
    user_id: str = Depends(_resolve_user_id),
    svc: SnapTradeService = Depends(get_snaptrade_service),
) -> SnapTradeOrdersResponse:
    """Live order status (pending, filled, canceled) across all accounts."""
    logger.info("[snaptrade] GET /snaptrade/orders user_id=%s", user_id)
    try:
        orders = svc.get_orders(user_id)
        return SnapTradeOrdersResponse(orders=orders)
    except ApiException as e:
        raise _snaptrade_http(e) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
