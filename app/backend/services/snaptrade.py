"""
SnapTrade brokerage connection + account/holdings retrieval.

SnapTrade requires a per-user registration step that produces a ``user_secret``.
We store the mapping ``app_user_id → (snaptrade_user_id, user_secret)`` in Supabase
table ``public.snaptrade_users`` so we don't need to re-register on every request.

Run the migration in supabase/migrations/20260419000000_snaptrade_users.sql before use.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from functools import lru_cache
from typing import Any

from snaptrade_client import ApiException, SnapTrade

from app.backend.core.supabase import select_maybe, upsert_row

logger = logging.getLogger(__name__)

_TABLE = "snaptrade_users"


def _extract_snaptrade_login_portal(resp: Any) -> tuple[str, str | None]:
    """Parse login response across SDK shapes (dict, DictSchema, raw JSON on urllib3 response)."""
    body = getattr(resp, "body", None)
    uri = ""
    session_id: str | None = None

    def apply_mapping(m: dict[str, Any]) -> None:
        nonlocal uri, session_id
        u = m.get("redirectURI") or m.get("redirect_uri") or m.get("redirectUri")
        if u:
            uri = str(u).strip()
        s = m.get("sessionId") or m.get("session_id")
        if s is not None and str(s).strip():
            session_id = str(s).strip()

    if body is not None:
        if isinstance(body, dict):
            apply_mapping(body)
        else:
            for key in ("redirectURI", "redirect_uri", "redirectUri"):
                try:
                    v = body.get(key) if hasattr(body, "get") else body[key]  # type: ignore[index]
                except Exception:
                    v = None
                if v:
                    uri = str(v).strip()
                    break
            for key in ("sessionId", "session_id"):
                try:
                    v = body.get(key) if hasattr(body, "get") else body[key]  # type: ignore[index]
                except Exception:
                    v = None
                if v:
                    session_id = str(v).strip()
                    break

    if not uri:
        raw = getattr(resp, "response", None)
        data = getattr(raw, "data", None) if raw is not None else None
        if data:
            try:
                if isinstance(data, (bytes, bytearray, memoryview)):
                    data = bytes(data).decode("utf-8", errors="replace")
                parsed = json.loads(data) if isinstance(data, str) else {}
                if isinstance(parsed, dict):
                    apply_mapping(parsed)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.debug("[snaptrade] login raw body parse skipped: %s", e)

    return uri, session_id


def _safe(obj: Any) -> Any:
    """Recursively convert SnapTrade response objects to JSON-safe primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(i) for i in obj]
    # SnapTrade SDK model objects expose __dict__ or to_dict()
    if hasattr(obj, "to_dict"):
        return _safe(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return _safe(vars(obj))
    return str(obj)


def unwrap_snaptrade_ticker(sym_obj: Any) -> str | None:
    """Resolve SnapTrade nested ``symbol`` payloads to a plain ticker string.

    The API may return ``symbol`` as a dict whose ``symbol`` field is another
    dict (e.g. UniversalSymbol). Recurse until a string is found, with
    fallbacks for ``raw_symbol`` / ``local_id`` / ``ticker`` / ``code``.
    """
    if sym_obj is None:
        return None
    if isinstance(sym_obj, str):
        t = sym_obj.strip()
        return t if t else None

    cur: Any = sym_obj
    for _ in range(8):
        if cur is None:
            return None
        if isinstance(cur, str):
            t = cur.strip()
            return t if t else None
        if isinstance(cur, dict):
            nxt = cur.get("symbol")
            if nxt is not None:
                cur = nxt
                continue
            for key in ("raw_symbol", "local_id", "ticker", "code"):
                v = cur.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return None
        return str(cur).strip() or None
    return None


class SnapTradeService:
    """Wraps the SnapTrade SDK with Supabase-backed user registration."""

    def __init__(self) -> None:
        client_id = os.getenv("SNAPTRADE_CLIENT_ID")
        consumer_key = os.getenv("SNAPTRADE_CONSUMER_KEY")
        if not client_id or not consumer_key:
            raise RuntimeError("SNAPTRADE_CLIENT_ID and SNAPTRADE_CONSUMER_KEY must be set")
        self._client = SnapTrade(client_id=client_id, consumer_key=consumer_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_register_user(self, app_user_id: str) -> tuple[str, str]:
        """Return (snaptrade_user_id, user_secret), registering with SnapTrade if needed."""
        row = select_maybe(_TABLE, "snaptrade_user_id", "user_secret", filters={"app_user_id": app_user_id})
        if row:
            logger.info("[snaptrade] existing user for app_user_id=%s", app_user_id)
            return row["snaptrade_user_id"], row["user_secret"]

        # First time — register a new SnapTrade user.
        snaptrade_user_id = str(uuid.uuid4())
        logger.info("[snaptrade] registering new user snaptrade_user_id=%s for app_user_id=%s", snaptrade_user_id, app_user_id)
        resp = self._client.authentication.register_snap_trade_user(body={"userId": snaptrade_user_id})
        user_secret: str = resp.body["userSecret"]

        upsert_row(
            _TABLE,
            {
                "app_user_id": app_user_id,
                "snaptrade_user_id": snaptrade_user_id,
                "user_secret": user_secret,
            },
            on_conflict="app_user_id",
        )
        return snaptrade_user_id, user_secret

    def is_registered(self, app_user_id: str) -> bool:
        """True if we have stored SnapTrade credentials for this app user."""
        return (
            select_maybe(_TABLE, "snaptrade_user_id", filters={"app_user_id": app_user_id}) is not None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_connection_url(
        self,
        app_user_id: str,
        *,
        custom_redirect: str | None = None,
        connection_type: str = "read",
        broker: str | None = None,
    ) -> dict[str, str]:
        """Register (or look up) the user and return a Connection Portal URL.

        The frontend opens ``redirect_uri`` in an in-app browser / WebBrowser.
        SnapTrade closes the portal and sends the user to ``custom_redirect`` when done.
        """
        snaptrade_user_id, user_secret = self._get_or_register_user(app_user_id)

        # Only pass kwargs supported by the installed snaptrade-python-sdk (varies by version).
        kwargs: dict[str, Any] = {
            "user_id": snaptrade_user_id,
            "user_secret": user_secret,
            "connection_type": connection_type,
        }
        if custom_redirect:
            kwargs["custom_redirect"] = custom_redirect
        if broker:
            kwargs["broker"] = broker

        resp = self._client.authentication.login_snap_trade_user(**kwargs)
        redirect_uri, session_id = _extract_snaptrade_login_portal(resp)
        if not redirect_uri:
            logger.error(
                "[snaptrade] login response missing redirectURI body_type=%s",
                type(getattr(resp, "body", None)),
            )
            raise RuntimeError(
                "SnapTrade did not return a connection portal URL. "
                "Verify SNAPTRADE_CLIENT_ID and SNAPTRADE_CONSUMER_KEY in the dashboard."
            )
        logger.info("[snaptrade] generated connection URL for app_user_id=%s", app_user_id)
        return {
            "redirect_uri": redirect_uri,
            "url": redirect_uri,
            "snaptrade_user_id": snaptrade_user_id,
            "session_id": session_id,
        }

    def add_broker(
        self,
        app_user_id: str,
        *,
        custom_redirect: str | None = None,
        connection_type: str = "read",
        broker: str | None = None,
    ) -> dict[str, str]:
        """Generate a new Connection Portal URL for an *existing* SnapTrade user.

        Unlike ``get_connection_url``, this raises ``LookupError`` when the user
        has not been registered yet (they must call ``connect`` first).  It never
        calls ``register_snap_trade_user``, so personal-key "one user" limits are
        unaffected.
        """
        row = select_maybe(_TABLE, "snaptrade_user_id", "user_secret", filters={"app_user_id": app_user_id})
        if not row:
            raise LookupError(
                "No SnapTrade registration found for this user. "
                "Call POST /snaptrade/connect first."
            )
        kwargs: dict[str, Any] = {
            "user_id": row["snaptrade_user_id"],
            "user_secret": row["user_secret"],
            "connection_type": connection_type,
        }
        if custom_redirect:
            kwargs["custom_redirect"] = custom_redirect
        if broker:
            kwargs["broker"] = broker

        resp = self._client.authentication.login_snap_trade_user(**kwargs)
        redirect_uri, session_id = _extract_snaptrade_login_portal(resp)
        if not redirect_uri:
            raise RuntimeError("SnapTrade did not return a connection portal URL.")
        logger.info("[snaptrade] add_broker portal URL for app_user_id=%s", app_user_id)
        return {
            "redirect_uri": redirect_uri,
            "url": redirect_uri,
            "snaptrade_user_id": row["snaptrade_user_id"],
            "session_id": session_id,
        }

    def get_accounts(self, app_user_id: str) -> list[dict]:
        """List all connected brokerage accounts for the user.

        Returns dicts matching the ``SnapTradeAccount`` schema (typed fields
        + ``raw`` for the full SnapTrade payload).
        """
        row = select_maybe(_TABLE, "snaptrade_user_id", "user_secret", filters={"app_user_id": app_user_id})
        if not row:
            return []

        resp = self._client.account_information.list_user_accounts(
            user_id=row["snaptrade_user_id"],
            user_secret=row["user_secret"],
        )
        raw_list: list[Any] = resp.body if isinstance(resp.body, list) else []
        safe_list: list[dict] = json.loads(json.dumps(_safe(raw_list)))

        out: list[dict] = []
        for acc in safe_list:
            balance = acc.get("balance") or {}
            total_obj = balance.get("total") or {}
            cash_obj = balance.get("cash") or {}
            meta = acc.get("meta") or {}
            sync_status = acc.get("sync_status") or {}
            holdings_sync = sync_status.get("holdings") or {}
            tx_sync = sync_status.get("transactions") or {}

            balance_total = total_obj.get("amount")
            balance_cash = cash_obj.get("amount")
            balance_currency = total_obj.get("currency") or cash_obj.get("currency")

            # Fidelity (and some other brokers) leave balance.total null until
            # SnapTrade's first holdings sync finishes. Fall back to cash so we
            # at least show *something* instead of "$0.00".
            effective_total = balance_total if balance_total is not None else balance_cash

            if balance_total is None:
                logger.info(
                    "[snaptrade] account=%s institution=%s has null balance.total "
                    "(holdings_sync_completed=%s) — surfacing cash=%s",
                    acc.get("id"),
                    acc.get("institution_name") or meta.get("institution_name"),
                    holdings_sync.get("initial_sync_completed"),
                    balance_cash,
                )

            out.append({
                "id": acc.get("id", ""),
                "name": acc.get("name"),
                "number": acc.get("number"),
                "institution_name": acc.get("institution_name") or meta.get("institution_name"),
                "account_type": meta.get("type") or acc.get("account_type"),
                "status": meta.get("status") or acc.get("status"),
                "is_paper": acc.get("is_paper"),
                "balance_total": effective_total,
                "balance_cash": balance_cash,
                "balance_currency": balance_currency,
                "sync": {
                    "holdings_initial_sync_completed": holdings_sync.get("initial_sync_completed"),
                    "transactions_initial_sync_completed": tx_sync.get("initial_sync_completed"),
                    "holdings_last_successful_sync": holdings_sync.get("last_successful_sync"),
                    "transactions_last_successful_sync": tx_sync.get("last_successful_sync"),
                },
                "raw": acc,
            })
        return out

    def get_holdings(self, app_user_id: str) -> list[dict]:
        """Return the full SnapTrade holdings bundle (accounts + positions + orders) per connection."""
        row = select_maybe(_TABLE, "snaptrade_user_id", "user_secret", filters={"app_user_id": app_user_id})
        if not row:
            return []

        resp = self._client.account_information.get_all_user_holdings(
            query_params={
                "userId": row["snaptrade_user_id"],
                "userSecret": row["user_secret"],
            }
        )
        raw: list[Any] = resp.body if isinstance(resp.body, list) else []
        return json.loads(json.dumps(_safe(raw)))

    def get_positions(self, app_user_id: str) -> list[dict]:
        """Return typed position dicts (cost basis, quantity, current price) across all accounts.

        Each dict matches the ``SnapTradePosition`` schema.
        """
        row = select_maybe(_TABLE, "snaptrade_user_id", "user_secret", filters={"app_user_id": app_user_id})
        if not row:
            return []

        # First get all account ids for this user.
        accounts_resp = self._client.account_information.list_user_accounts(
            user_id=row["snaptrade_user_id"],
            user_secret=row["user_secret"],
        )
        accounts_raw: list[dict] = json.loads(json.dumps(_safe(
            accounts_resp.body if isinstance(accounts_resp.body, list) else []
        )))

        positions_out: list[dict] = []
        for acc in accounts_raw:
            account_id: str = acc.get("id", "")
            try:
                pos_resp = self._client.account_information.get_user_account_positions(
                    account_id=account_id,
                    user_id=row["snaptrade_user_id"],
                    user_secret=row["user_secret"],
                )
                pos_list: list[dict] = json.loads(json.dumps(_safe(
                    pos_resp.body if isinstance(pos_resp.body, list) else []
                )))
            except ApiException as e:
                logger.warning("[snaptrade] get_positions failed for account=%s: %s", account_id, e)
                continue

            for pos in pos_list:
                sym_obj = pos.get("symbol") or {}
                sym_ticker = unwrap_snaptrade_ticker(sym_obj) or ""
                description = (sym_obj.get("description") or sym_obj.get("name")) if isinstance(sym_obj, dict) else None
                currency = (sym_obj.get("currency") or {}).get("code") if isinstance(sym_obj, dict) else None
                asset_class = (sym_obj.get("type") or {}).get("code") if isinstance(sym_obj, dict) else None

                qty = pos.get("units")
                cost_basis = pos.get("average_purchase_price")
                current_price = pos.get("price")
                market_val = (qty * current_price) if (qty is not None and current_price is not None) else None
                unrealized = (
                    (qty * current_price) - (qty * cost_basis)
                    if (qty is not None and current_price is not None and cost_basis is not None)
                    else None
                )

                positions_out.append({
                    "account_id": account_id,
                    "symbol": sym_ticker,
                    "description": description,
                    "quantity": qty,
                    "average_purchase_price": cost_basis,
                    "current_price": current_price,
                    "market_value": market_val,
                    "unrealized_gain": unrealized,
                    "currency": currency,
                    "asset_class": asset_class,
                    "raw": pos,
                })

        return positions_out

    def get_transactions(
        self,
        app_user_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        """Return typed transaction dicts (BUY/SELL/DIVIDEND/etc.) across all accounts.

        Dates should be ISO-8601 strings (e.g. '2025-01-01'). Omit to get all history.
        Each dict matches the ``SnapTradeTransaction`` schema.
        """
        row = select_maybe(_TABLE, "snaptrade_user_id", "user_secret", filters={"app_user_id": app_user_id})
        if not row:
            return []

        kwargs: dict[str, Any] = {
            "user_id": row["snaptrade_user_id"],
            "user_secret": row["user_secret"],
        }
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date

        try:
            resp = self._client.account_information.get_account_activities(**kwargs)
        except ApiException as e:
            logger.warning("[snaptrade] get_transactions failed: %s", e)
            return []

        raw_list: list[dict] = json.loads(json.dumps(_safe(
            resp.body if isinstance(resp.body, list) else []
        )))

        out: list[dict] = []
        for tx in raw_list:
            sym_obj = tx.get("symbol") or {}
            symbol = unwrap_snaptrade_ticker(sym_obj)
            desc = sym_obj.get("description") if isinstance(sym_obj, dict) else None
            currency_obj = tx.get("currency") or {}
            currency = currency_obj.get("code") if isinstance(currency_obj, dict) else str(currency_obj) if currency_obj else None
            account_obj = tx.get("account") or {}
            account_id = account_obj.get("id") if isinstance(account_obj, dict) else str(account_obj) if account_obj else None

            out.append({
                "account_id": account_id,
                "transaction_id": tx.get("id"),
                "type": tx.get("type"),
                "symbol": symbol,
                "description": desc or tx.get("description"),
                "quantity": tx.get("units"),
                "price": tx.get("price"),
                "amount": tx.get("amount"),
                "currency": currency,
                "trade_date": tx.get("trade_date"),
                "settlement_date": tx.get("settlement_date"),
                "raw": tx,
            })
        return out

    def get_orders(self, app_user_id: str) -> list[dict]:
        """Return typed order dicts (status: pending/filled/canceled) across all accounts.

        Each dict matches the ``SnapTradeOrder`` schema.
        """
        row = select_maybe(_TABLE, "snaptrade_user_id", "user_secret", filters={"app_user_id": app_user_id})
        if not row:
            return []

        accounts_resp = self._client.account_information.list_user_accounts(
            user_id=row["snaptrade_user_id"],
            user_secret=row["user_secret"],
        )
        accounts_raw: list[dict] = json.loads(json.dumps(_safe(
            accounts_resp.body if isinstance(accounts_resp.body, list) else []
        )))

        orders_out: list[dict] = []
        for acc in accounts_raw:
            account_id: str = acc.get("id", "")
            try:
                ord_resp = self._client.account_information.get_user_account_orders(
                    account_id=account_id,
                    user_id=row["snaptrade_user_id"],
                    user_secret=row["user_secret"],
                )
                ord_list: list[dict] = json.loads(json.dumps(_safe(
                    ord_resp.body if isinstance(ord_resp.body, list) else []
                )))
            except ApiException as e:
                logger.warning("[snaptrade] get_orders failed for account=%s: %s", account_id, e)
                continue

            for order in ord_list:
                sym_obj = order.get("symbol") or {}
                symbol = unwrap_snaptrade_ticker(sym_obj)

                orders_out.append({
                    "account_id": account_id,
                    "brokerage_order_id": order.get("brokerage_order_id"),
                    "symbol": symbol,
                    "action": order.get("action"),
                    "order_type": order.get("order_type"),
                    "status": order.get("status"),
                    "quantity": order.get("units"),
                    "limit_price": order.get("price"),
                    "filled_quantity": order.get("filled_quantity") or order.get("units_filled"),
                    "execution_price": order.get("execution_price"),
                    "time_placed": order.get("time_placed"),
                    "time_filled": order.get("time_filled") or order.get("time_executed"),
                    "time_in_force": order.get("time_in_force"),
                    "raw": order,
                })
        return orders_out

    def list_connections(self, app_user_id: str) -> list[dict]:
        """Return all brokerage authorizations for the user.

        Each dict contains an ``id`` (authorization_id) that the frontend
        passes to ``remove_connection`` to disconnect a broker.
        """
        row = select_maybe(_TABLE, "snaptrade_user_id", "user_secret", filters={"app_user_id": app_user_id})
        if not row:
            return []

        resp = self._client.connections.list_brokerage_authorizations(
            user_id=row["snaptrade_user_id"],
            user_secret=row["user_secret"],
        )
        raw_list: list[Any] = resp.body if isinstance(resp.body, list) else []
        safe_list: list[dict] = json.loads(json.dumps(_safe(raw_list)))

        out: list[dict] = []
        for auth in safe_list:
            brokerage = auth.get("brokerage") or {}
            out.append({
                "id": auth.get("id", ""),
                "brokerage_id": brokerage.get("id") if isinstance(brokerage, dict) else str(brokerage),
                "brokerage_name": brokerage.get("name") if isinstance(brokerage, dict) else None,
                "type": auth.get("type"),
                "created_date": auth.get("created_date"),
                "disabled": auth.get("disabled"),
                "disabled_date": auth.get("disabled_date"),
                "raw": auth,
            })
        return out

    def refresh_connection(self, app_user_id: str, authorization_id: str) -> dict:
        """Ask SnapTrade to re-pull holdings for a single authorization.

        Useful immediately after a user connects Fidelity (or any broker
        whose initial sync is slow) so balances and positions appear without
        waiting for the next polling cycle. Returns the raw SnapTrade
        response so the route can surface ``sync_status`` back to the UI.
        """
        row = select_maybe(_TABLE, "snaptrade_user_id", "user_secret", filters={"app_user_id": app_user_id})
        if not row:
            raise LookupError(
                "No SnapTrade registration found for this user. "
                "Call POST /snaptrade/connect first."
            )
        resp = self._client.connections.refresh_brokerage_authorization(
            authorization_id=authorization_id,
            user_id=row["snaptrade_user_id"],
            user_secret=row["user_secret"],
        )
        logger.info(
            "[snaptrade] refreshed authorization_id=%s for app_user_id=%s",
            authorization_id,
            app_user_id,
        )
        body = getattr(resp, "body", None)
        return json.loads(json.dumps(_safe(body))) if body is not None else {}

    def remove_connection(self, app_user_id: str, authorization_id: str) -> None:
        """Revoke a single brokerage authorization.

        Raises ``LookupError`` if the user isn't registered, and
        ``ApiException`` (re-raised) for SnapTrade-level errors (e.g. unknown
        authorization_id or permission denied).
        """
        row = select_maybe(_TABLE, "snaptrade_user_id", "user_secret", filters={"app_user_id": app_user_id})
        if not row:
            raise LookupError(
                "No SnapTrade registration found for this user. "
                "Call POST /snaptrade/connect first."
            )
        self._client.connections.remove_brokerage_authorization(
            authorization_id=authorization_id,
            user_id=row["snaptrade_user_id"],
            user_secret=row["user_secret"],
        )
        logger.info(
            "[snaptrade] removed authorization_id=%s for app_user_id=%s",
            authorization_id,
            app_user_id,
        )

    # ------------------------------------------------------------------
    # Trading — symbols, quotes, order preview/place/cancel
    # ------------------------------------------------------------------
    #
    # All trading methods delegate to the ``trading`` and ``reference_data``
    # namespaces on ``SnapTrade``. Previews (``get_order_impact``) return a
    # ``trade_id`` that is valid for ~5 minutes. The caller (``OrderIntentService``)
    # persists that id and calls ``place_equity_order(trade_id=...)`` to
    # execute. We never forward the raw SDK model — responses are always
    # normalised via ``_safe`` + ``json.loads(json.dumps(...))`` so downstream
    # code can treat them as plain JSON dicts.
    #
    # Errors: ``ApiException`` is re-raised so routes can translate it to the
    # correct HTTP status. ``LookupError`` is raised only when the user has
    # not called ``POST /snaptrade/connect`` yet.

    def _require_auth(self, app_user_id: str) -> dict[str, str]:
        row = select_maybe(
            _TABLE,
            "snaptrade_user_id",
            "user_secret",
            filters={"app_user_id": app_user_id},
        )
        if not row:
            raise LookupError(
                "No SnapTrade registration found for this user. "
                "Call POST /snaptrade/connect first."
            )
        return row

    def search_symbol(
        self, app_user_id: str, *, account_id: str | None, query: str
    ) -> list[dict]:
        """Search for tradable symbols.

        When ``account_id`` is provided we use the account-scoped search so
        results are limited to symbols that broker actually supports (e.g.
        Robinhood won't return the same OTC tickers Fidelity does). Without
        an account we fall back to the global reference-data search.
        """
        auth = self._require_auth(app_user_id)
        try:
            if account_id:
                # Account-scoped symbol search lives on ReferenceDataApi in the
                # snaptrade_client SDK (NOT TradingApi — that only exposes the
                # crypto-pair instrument search). Results are filtered to the
                # symbols this broker actually supports for the account.
                resp = self._client.reference_data.symbol_search_user_account(
                    user_id=auth["snaptrade_user_id"],
                    user_secret=auth["user_secret"],
                    account_id=account_id,
                    substring=query,
                )
            else:
                resp = self._client.reference_data.get_symbols(
                    body={"substring": query},
                )
        except ApiException as e:
            logger.warning("[snaptrade] search_symbol failed q=%r: %s", query, e)
            raise
        body = getattr(resp, "body", None) or []
        return json.loads(json.dumps(_safe(body)))

    def account_quote(
        self, app_user_id: str, *, account_id: str, symbols: str, use_ticker: bool = True
    ) -> list[dict]:
        """Return account-scoped quotes for one or more symbols.

        SnapTrade discourages using this endpoint as a primary market-data
        feed (it's broker-dependent and can be throttled), but it's the
        correct way to get a pre-trade price estimate for display in the
        order composer. ``symbols`` is a comma-separated list of tickers
        when ``use_ticker=True``, otherwise universal symbol IDs.
        """
        auth = self._require_auth(app_user_id)
        try:
            resp = self._client.trading.get_user_account_quotes(
                user_id=auth["snaptrade_user_id"],
                user_secret=auth["user_secret"],
                account_id=account_id,
                symbols=symbols,
                use_ticker=use_ticker,
            )
        except ApiException as e:
            logger.warning(
                "[snaptrade] account_quote failed account=%s symbols=%s: %s",
                account_id,
                symbols,
                e,
            )
            raise
        body = getattr(resp, "body", None) or []
        return json.loads(json.dumps(_safe(body)))

    def preview_equity_order(
        self,
        app_user_id: str,
        *,
        account_id: str,
        action: str,
        order_type: str,
        time_in_force: str,
        universal_symbol_id: str | None = None,
        symbol: str | None = None,
        units: float | None = None,
        notional_value: float | None = None,
        price: float | None = None,
        stop: float | None = None,
    ) -> dict:
        """Run SnapTrade's ``/trade/impact`` to preview an equity order.

        Returns a dict containing ``trade`` (with a ``id`` field — the
        ephemeral ``trade_id`` valid for 5 minutes) and ``trade_impact``
        (commission, buying power, estimated value, etc.). The caller
        stores these and later calls :meth:`place_equity_order` with the
        ``trade_id`` inside that window.

        Either ``universal_symbol_id`` OR ``symbol`` should be provided —
        the former is preferred since it's unambiguous across exchanges.
        """
        auth = self._require_auth(app_user_id)
        kwargs: dict[str, Any] = {
            "user_id": auth["snaptrade_user_id"],
            "user_secret": auth["user_secret"],
            "account_id": account_id,
            "action": action.upper(),
            "order_type": _map_order_type(order_type),
            "time_in_force": _map_time_in_force(time_in_force),
        }
        if universal_symbol_id:
            kwargs["universal_symbol_id"] = universal_symbol_id
        if units is not None:
            kwargs["units"] = float(units)
        if notional_value is not None:
            kwargs["notional_value"] = float(notional_value)
        if price is not None:
            kwargs["price"] = float(price)
        if stop is not None:
            kwargs["stop"] = float(stop)

        try:
            resp = self._client.trading.get_order_impact(**kwargs)
        except ApiException as e:
            logger.warning(
                "[snaptrade] preview_equity_order failed account=%s sym=%s: %s",
                account_id,
                universal_symbol_id or symbol,
                e,
            )
            raise
        body = getattr(resp, "body", None) or {}
        return json.loads(json.dumps(_safe(body)))

    def place_equity_order(
        self,
        app_user_id: str,
        *,
        trade_id: str,
        wait_to_confirm: bool = False,
    ) -> dict:
        """Execute a previously-previewed equity order within its 5-minute window."""
        auth = self._require_auth(app_user_id)
        try:
            resp = self._client.trading.place_order(
                user_id=auth["snaptrade_user_id"],
                user_secret=auth["user_secret"],
                trade_id=trade_id,
                wait_to_confirm=wait_to_confirm,
            )
        except ApiException as e:
            logger.warning(
                "[snaptrade] place_equity_order failed trade_id=%s: %s",
                trade_id,
                e,
            )
            raise
        body = getattr(resp, "body", None) or {}
        return json.loads(json.dumps(_safe(body)))

    def cancel_equity_order(
        self,
        app_user_id: str,
        *,
        account_id: str,
        brokerage_order_id: str,
    ) -> dict:
        """Cancel an open equity order at the brokerage."""
        auth = self._require_auth(app_user_id)
        try:
            resp = self._client.trading.cancel_order(
                user_id=auth["snaptrade_user_id"],
                user_secret=auth["user_secret"],
                account_id=account_id,
                brokerage_order_id=brokerage_order_id,
            )
        except ApiException as e:
            logger.warning(
                "[snaptrade] cancel_equity_order failed account=%s order=%s: %s",
                account_id,
                brokerage_order_id,
                e,
            )
            raise
        body = getattr(resp, "body", None) or {}
        return json.loads(json.dumps(_safe(body)))

    def preview_crypto_order(
        self,
        app_user_id: str,
        *,
        account_id: str,
        instrument: str,
        side: str,
        order_type: str,
        time_in_force: str,
        amount: float,
        price: float | None = None,
    ) -> dict:
        """Preview a crypto pair order. Crypto uses a different endpoint and
        body shape than equities (``instrument`` is a pair like ``BTC/USD``).
        """
        auth = self._require_auth(app_user_id)
        body: dict[str, Any] = {
            "account_id": account_id,
            "instrument": instrument,
            "side": side.upper(),
            "type": _map_order_type(order_type),
            "time_in_force": _map_time_in_force(time_in_force),
            "amount": float(amount),
        }
        if price is not None:
            body["price"] = float(price)
        try:
            resp = self._client.trading.preview_crypto_order(
                body=body,
                user_id=auth["snaptrade_user_id"],
                user_secret=auth["user_secret"],
            )
        except ApiException as e:
            logger.warning(
                "[snaptrade] preview_crypto_order failed account=%s pair=%s: %s",
                account_id,
                instrument,
                e,
            )
            raise
        out = getattr(resp, "body", None) or {}
        return json.loads(json.dumps(_safe(out)))

    def place_crypto_order(
        self,
        app_user_id: str,
        *,
        account_id: str,
        instrument: str,
        side: str,
        order_type: str,
        time_in_force: str,
        amount: float,
        price: float | None = None,
    ) -> dict:
        """Place a crypto pair order directly (no two-phase preview in the SDK)."""
        auth = self._require_auth(app_user_id)
        body: dict[str, Any] = {
            "account_id": account_id,
            "instrument": instrument,
            "side": side.upper(),
            "type": _map_order_type(order_type),
            "time_in_force": _map_time_in_force(time_in_force),
            "amount": float(amount),
        }
        if price is not None:
            body["price"] = float(price)
        try:
            resp = self._client.trading.place_crypto_order(
                body=body,
                user_id=auth["snaptrade_user_id"],
                user_secret=auth["user_secret"],
            )
        except ApiException as e:
            logger.warning(
                "[snaptrade] place_crypto_order failed account=%s pair=%s: %s",
                account_id,
                instrument,
                e,
            )
            raise
        out = getattr(resp, "body", None) or {}
        return json.loads(json.dumps(_safe(out)))


# ----------------------------------------------------------------------
# Small helpers for mapping our canonical enum values → SnapTrade SDK
# strings. Keeping them at module level makes them trivially unit-testable.
# ----------------------------------------------------------------------


_ORDER_TYPE_MAP = {
    "market": "Market",
    "limit": "Limit",
    "stop": "Stop",
    "stop_limit": "StopLimit",
}

_TIF_MAP = {
    "day": "Day",
    "gtc": "GTC",
    "fok": "FOK",
    "ioc": "IOC",
}


def _map_order_type(value: str) -> str:
    key = (value or "").lower().replace("-", "_")
    return _ORDER_TYPE_MAP.get(key, value)


def _map_time_in_force(value: str) -> str:
    key = (value or "").lower()
    return _TIF_MAP.get(key, value.upper() if value else "Day")


@lru_cache
def get_snaptrade_service() -> SnapTradeService:
    return SnapTradeService()
