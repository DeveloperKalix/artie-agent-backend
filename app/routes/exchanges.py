"""Exchange (brokerage) routes — SnapTrade only. Bank accounts use GET /plaid/accounts."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from snaptrade_client import ApiException

from app.backend.services.snaptrade import SnapTradeService, get_snaptrade_service
from app.routes.plaid import _resolve_accounts_user_id
from app.schemas.snaptrade import ExchangeAccountsResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/exchanges", tags=["exchanges"])


@router.get("/accounts", response_model=ExchangeAccountsResponse)
async def get_exchange_accounts(
    user_id: str = Depends(_resolve_accounts_user_id),
    svc: SnapTradeService = Depends(get_snaptrade_service),
) -> ExchangeAccountsResponse:
    """Connected **brokerage** accounts (SnapTrade). Not Plaid depository accounts.

    Returns ``registered: false`` and empty ``accounts`` until the user has started
    SnapTrade (POST ``/snaptrade/connect`` creates the registration row).
    """
    logger.info("[exchanges] GET /exchanges/accounts user_id=%s", user_id)
    if not svc.is_registered(user_id):
        return ExchangeAccountsResponse(registered=False, accounts=[])
    try:
        accounts = svc.get_accounts(user_id)
        return ExchangeAccountsResponse(registered=True, accounts=accounts)
    except ApiException as e:
        try:
            detail = e.body if isinstance(e.body, str) else str(e.body)
        except Exception:
            detail = str(e)
        code = getattr(e, "status", None) or 400
        if not (400 <= code <= 599):
            code = 400
        raise HTTPException(status_code=code, detail=detail) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
