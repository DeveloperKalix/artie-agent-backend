import json
import logging

import plaid
from fastapi import APIRouter, Depends, Header, HTTPException, Query

from app.backend.services.plaid import PlaidService, get_plaid_service
from app.schemas.plaid import (
    CompleteHostedLinkBody,
    CompleteHostedLinkResponse,
    CreateLinkTokenBody,
    ExchangePublicTokenBody,
    LinkTokenResponse,
    PlaidAccountsResponse,
    PlaidItemStored,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/plaid", tags=["plaid"])


def _plaid_http(exc: plaid.ApiException) -> HTTPException:
    try:
        body = json.loads(exc.body) if exc.body else {}
    except (json.JSONDecodeError, TypeError):
        body = {}
    code = getattr(exc, "status", None) or 400
    if code < 400 or code > 599:
        code = 400
    return HTTPException(
        status_code=code,
        detail=body.get("error_message") or body.get("description") or str(exc),
    )


@router.post("/link_token", response_model=LinkTokenResponse)
async def create_link_token(
    body: CreateLinkTokenBody,
    svc: PlaidService = Depends(get_plaid_service),
) -> LinkTokenResponse:
    logger.info(
        "[plaid] POST /plaid/link_token user_id=%s hosted=%s",
        body.user_id,
        bool(body.completion_redirect_uri),
    )
    try:
        data = svc.create_link_token(body.user_id, completion_redirect_uri=body.completion_redirect_uri)
        return LinkTokenResponse(**data)
    except plaid.ApiException as e:
        raise _plaid_http(e) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


def _get_accounts_core(
    user_id: str,
    item_id: str | None,
    svc: PlaidService,
) -> PlaidAccountsResponse:
    data = svc.get_accounts(user_id, item_id=item_id)
    return PlaidAccountsResponse(**data)


def _resolve_accounts_user_id(
    user_id: str | None = Query(
        None,
        description="Same id as Link / exchange (Supabase auth user id, etc.)",
    ),
    x_user_id: str | None = Header(
        None,
        alias="X-User-Id",
        description="Alternative to user_id query param (recommended for GET from mobile clients)",
    ),
) -> str:
    uid = (user_id or x_user_id or "").strip()
    if not uid:
        raise HTTPException(
            status_code=422,
            detail=(
                "Missing user_id: use ?user_id=<id> or send header X-User-Id: <id> "
                "(must match the user_id used for link_token and exchange)."
            ),
        )
    return uid


@router.get("/accounts", response_model=PlaidAccountsResponse)
async def get_accounts(
    user_id: str = Depends(_resolve_accounts_user_id),
    item_id: str | None = Query(
        None,
        description="Limit to a single linked Item; omit to load accounts for all Items for this user",
    ),
    svc: PlaidService = Depends(get_plaid_service),
) -> PlaidAccountsResponse:
    logger.info(
        "[plaid] GET /plaid/accounts user_id=%s item_id=%s",
        user_id,
        item_id,
    )
    try:
        return _get_accounts_core(user_id, item_id, svc)
    except plaid.ApiException as e:
        raise _plaid_http(e) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.post("/exchange", response_model=PlaidItemStored)
async def exchange_public_token(
    body: ExchangePublicTokenBody,
    svc: PlaidService = Depends(get_plaid_service),
) -> PlaidItemStored:
    logger.info(
        "[plaid] POST /plaid/exchange user_id=%s public_token_len=%s",
        body.user_id,
        len(body.public_token),
    )
    try:
        data = svc.exchange_public_token(body.public_token, body.user_id)
        out = PlaidItemStored(**data)
        logger.info(
            "[plaid] POST /plaid/exchange ok user_id=%s item_id=%s",
            body.user_id,
            out.item_id,
        )
        return out
    except plaid.ApiException as e:
        raise _plaid_http(e) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except KeyError as e:
        logger.exception("Missing configuration")
        raise HTTPException(status_code=503, detail=f"Server configuration error: {e}") from e


@router.post("/complete_hosted_link", response_model=CompleteHostedLinkResponse)
async def complete_hosted_link(
    body: CompleteHostedLinkBody,
    svc: PlaidService = Depends(get_plaid_service),
) -> CompleteHostedLinkResponse:
    """Poll a Hosted Link session and exchange the public_token when complete.

    Call this after the frontend is redirected back to ``completion_redirect_uri``
    (the deep-link passed in the link_token request). Returns ``success=False`` with
    ``status="incomplete"`` if Plaid hasn't finished processing yet — retry after a
    short delay (1–2 s) until ``success=True``.
    """
    logger.info("[plaid] POST /plaid/complete_hosted_link user_id=%s", body.user_id)
    try:
        result = svc.complete_hosted_link(body.link_token, body.user_id)
        if result.get("success"):
            logger.info(
                "[plaid] POST /plaid/complete_hosted_link ok user_id=%s item_id=%s",
                body.user_id,
                result.get("item_id"),
            )
        return CompleteHostedLinkResponse(**result)
    except plaid.ApiException as e:
        raise _plaid_http(e) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
