"""
Plaid Link + persistence.

Run this in Supabase SQL editor before using exchange:

    create table if not exists public.plaid_items (
      id uuid primary key default gen_random_uuid(),
      user_id text not null,
      item_id text not null unique,
      access_token text not null,
      institution_id text,
      institution_name text,
      created_at timestamptz not null default now(),
      updated_at timestamptz not null default now()
    );

Server inserts use the configured SUPABASE_KEY; use the service_role key if RLS blocks inserts.
Encrypt or vault access_token before production — never expose it to the frontend.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from functools import lru_cache
from typing import Any

import plaid
from plaid.api import plaid_api
from postgrest.exceptions import APIError
from plaid.model.accounts_get_request import AccountsGetRequest
from plaid.model.country_code import CountryCode
from plaid.model.depository_account_subtype import DepositoryAccountSubtype
from plaid.model.depository_account_subtypes import DepositoryAccountSubtypes
from plaid.model.depository_filter import DepositoryFilter
from plaid.model.link_token_account_filters import LinkTokenAccountFilters
from plaid.model.item_get_request import ItemGetRequest
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.institutions_get_by_id_request import InstitutionsGetByIdRequest
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_hosted_link import LinkTokenCreateHostedLink
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.link_token_get_request import LinkTokenGetRequest
from plaid.model.products import Products

from app.backend.core.supabase import select_rows, upsert_row

logger = logging.getLogger(__name__)

# plaid-python exposes Sandbox and Production; "development" maps to Sandbox (Plaid hosted dev).
_PLAID_HOSTS = {
    "sandbox": plaid.Environment.Sandbox,
    "development": plaid.Environment.Sandbox,
    "production": plaid.Environment.Production,
}


def _plaid_host() -> str:
    env = os.getenv("PLAID_ENV", "sandbox").lower()
    return _PLAID_HOSTS.get(env, plaid.Environment.Sandbox)


_PLAID_ITEMS_MISSING_MSG = (
    "Supabase table public.plaid_items is missing. "
    "In Supabase Dashboard → SQL Editor, run the file "
    "supabase/migrations/20260418120000_plaid_items.sql (then retry)."
)


def _reraise_supabase_table_error(exc: APIError) -> None:
    if exc.code == "PGRST205":
        raise RuntimeError(_PLAID_ITEMS_MISSING_MSG) from exc
    raise exc


def _json_safe_expiration(value: Any) -> str | None:
    """Plaid may return ``expiration`` as ``datetime``; API clients expect a string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class PlaidService:
    """Reusable Plaid API + Supabase persistence (no HTTP concerns)."""

    def __init__(self) -> None:
        client_id = os.getenv("PLAID_CLIENT_ID")
        secret = os.getenv("PLAID_SECRET")
        if not client_id or not secret:
            raise RuntimeError("PLAID_CLIENT_ID and PLAID_SECRET must be set")

        configuration = plaid.Configuration(
            host=_plaid_host(),
            api_key={
                "clientId": client_id,
                "secret": secret,
                "plaidVersion": "2020-09-14",
            },
        )
        api_client = plaid.ApiClient(configuration)
        self._client = plaid_api.PlaidApi(api_client)

    def create_link_token(self, user_id: str, completion_redirect_uri: str | None = None) -> dict:
        """Returns dict with link_token, expiration, and (when using Hosted Link) hosted_link_url.

        Pass ``completion_redirect_uri`` to activate Plaid Hosted Link — Plaid serves the entire
        Link UI at a ``https://secure.plaid.com/link/…`` URL and redirects back to
        ``completion_redirect_uri`` (e.g. ``artie://plaid-complete``) when done.
        The frontend opens ``hosted_link_url`` directly; no WebBrowser redirect URI needed.

        Without ``completion_redirect_uri`` the classic redirect-URI flow is used
        (``PLAID_REDIRECT_URI`` env var must be set and registered in Plaid Dashboard).
        """
        account_filters = LinkTokenAccountFilters(
            depository=DepositoryFilter(
                account_subtypes=DepositoryAccountSubtypes(
                    [
                        DepositoryAccountSubtype("checking"),
                        DepositoryAccountSubtype("cash management"),
                    ]
                )
            )
        )
        payload: dict = {
            "products": [Products("transactions")],
            "client_name": os.getenv("PLAID_CLIENT_NAME", "Artie"),
            "country_codes": [CountryCode("US")],
            "language": "en",
            "user": LinkTokenCreateRequestUser(client_user_id=user_id),
            "account_filters": account_filters,
        }

        # redirect_uri is always required when is_mobile_app=True (Plaid rule).
        # It's also needed for OAuth institutions (Chase, etc.) in both flows.
        redirect_uri = os.getenv("PLAID_REDIRECT_URI", "").strip()
        if redirect_uri:
            payload["redirect_uri"] = redirect_uri

        if completion_redirect_uri:
            # Hosted Link flow — Plaid serves the UI at hosted_link_url.
            # redirect_uri (HTTPS, registered in Plaid Dashboard) must also be present
            # when is_mobile_app=True — it handles OAuth bank handoffs.
            payload["hosted_link"] = LinkTokenCreateHostedLink(
                completion_redirect_uri=completion_redirect_uri,
                is_mobile_app=True,
            )

        webhook = os.getenv("PLAID_WEBHOOK_URL")
        if webhook:
            payload["webhook"] = webhook

        request = LinkTokenCreateRequest(**payload)
        response = self._client.link_token_create(request)
        result: dict = {
            "link_token": response["link_token"],
            "expiration": _json_safe_expiration(response.get("expiration")),
        }
        if completion_redirect_uri and response.get("hosted_link_url"):
            result["hosted_link_url"] = response["hosted_link_url"]
        return result

    def complete_hosted_link(self, link_token: str, user_id: str) -> dict:
        """Poll a Hosted Link session and exchange the public_token when Link is complete.

        Plaid stores the result of the Link session on the link_token; call this after
        the frontend is redirected back to ``completion_redirect_uri``.

        Returns ``{"success": True, ...item data...}`` when complete, or
        ``{"success": False, "status": "incomplete"|"no_session"}`` when not yet done.
        """
        sessions_resp = self._client.link_token_get(LinkTokenGetRequest(link_token=link_token))
        link_sessions = sessions_resp.get("link_sessions") or []
        if not link_sessions:
            logger.info("[plaid] complete_hosted_link: no link sessions yet for user=%s", user_id)
            return {"success": False, "status": "no_session"}

        latest = link_sessions[-1]
        results = latest.get("results") or {}
        items = results.get("item_add_results") or []
        if not items:
            logger.info("[plaid] complete_hosted_link: session incomplete for user=%s", user_id)
            return {"success": False, "status": "incomplete"}

        public_token = items[0]["public_token"]
        logger.info("[plaid] complete_hosted_link: exchanging public_token for user=%s", user_id)

        # Reuse the same exchange + persist path as the classic /exchange endpoint.
        stored = self.exchange_public_token(public_token, user_id)
        return {"success": True, "status": "complete", **stored}

    def exchange_public_token(self, public_token: str, user_id: str) -> dict:
        """Exchange Link public_token, fetch institution metadata, upsert row in Supabase."""
        exchange_request = ItemPublicTokenExchangeRequest(public_token=public_token)
        exchange = self._client.item_public_token_exchange(exchange_request)
        access_token = exchange["access_token"]
        item_id = exchange["item_id"]

        item_resp = self._client.item_get(ItemGetRequest(access_token=access_token))
        item = item_resp["item"]
        institution_id = item.get("institution_id")

        institution_name: str | None = None
        if institution_id:
            try:
                inst_req = InstitutionsGetByIdRequest(
                    institution_id=institution_id,
                    country_codes=[CountryCode("US")],
                )
                inst = self._client.institutions_get_by_id(inst_req)
                institution_name = inst.get("institution", {}).get("name")
            except plaid.ApiException as e:
                logger.warning("institutions_get_by_id failed: %s", e)

        row = {
            "user_id": user_id,
            "item_id": item_id,
            "access_token": access_token,
            "institution_id": institution_id,
            "institution_name": institution_name,
        }
        try:
            upsert_row("plaid_items", row, on_conflict="item_id")
        except APIError as e:
            _reraise_supabase_table_error(e)

        return {
            "item_id": item_id,
            "institution_id": institution_id,
            "institution_name": institution_name,
        }

    def get_accounts(self, user_id: str, item_id: str | None = None) -> dict:
        """Load stored Items for user, call Plaid ``/accounts/get`` per access_token."""
        filters: dict[str, str] = {"user_id": user_id}
        if item_id:
            filters["item_id"] = item_id

        try:
            rows = select_rows(
                "plaid_items",
                "access_token",
                "item_id",
                "institution_name",
                filters=filters,
            )
        except APIError as e:
            _reraise_supabase_table_error(e)
            rows = []  # unreachable but keeps type-checker happy
        accounts_out: list[dict] = []
        for row in rows:
            access_token = row["access_token"]
            iid = row["item_id"]
            resp = self._client.accounts_get(AccountsGetRequest(access_token=access_token))
            for acc in resp.get("accounts") or []:
                # Deep-serialize via JSON round-trip so nested Plaid objects
                # (e.g. AccountBalance) become plain dicts with numeric values.
                raw = acc if isinstance(acc, dict) else acc.to_dict()
                payload = json.loads(json.dumps(raw, default=str))
                payload["item_id"] = iid
                if row.get("institution_name"):
                    payload.setdefault("institution_name", row["institution_name"])
                accounts_out.append(payload)
        return {"accounts": accounts_out}


@lru_cache
def get_plaid_service() -> PlaidService:
    return PlaidService()
