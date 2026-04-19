"""OrderIntentService — the state machine behind every trade.

Every buy/sell request (UI or LLM) becomes an ``order_intents`` row. The
service enforces:

1. **Two-phase execution.** ``preview`` → ``confirm`` → ``submit``. The
   preview returns a SnapTrade ``trade_id`` valid ~5 minutes; we never
   submit without one (except crypto, which has no preview endpoint on
   SnapTrade and submits directly).
2. **Risk checks on confirm.** Per-user caps (``max_order_usd``,
   ``max_daily_usd``), master enable flag, and a disclaimer
   acknowledgement gate are validated before any SnapTrade write.
3. **Market-closed fallback.** Equity intents confirmed while NYSE is
   closed do not error — they transition to ``scheduled_for_market_open``
   with a reminder that fires at the next open. Crypto is unaffected.
4. **Append-only audit log.** Every transition writes a row into
   ``order_intent_events`` so we have immutable provenance if anything
   goes wrong downstream.

The service is fully async. Every Supabase / SnapTrade call is
``asyncio.to_thread``-dispatched so the event loop never blocks.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Iterable

from snaptrade_client import ApiException

from app.backend.core.supabase import (
    get_supabase,
    insert_row,
    select_maybe,
    select_rows,
    update_rows,
)
from app.backend.services.conversations import (
    ConversationService,
    get_conversation_service,
)
from app.backend.services.market_calendar import (
    MarketCalendarService,
    get_market_calendar,
)
from app.backend.services.memory import MemoryService, get_memory_service
from app.backend.services.snaptrade import SnapTradeService, get_snaptrade_service
from app.schemas.conversations import MessageRole
from app.schemas.memory import TradingConfig
from app.schemas.trade import (
    CreateOrderIntent,
    ImpactPreview,
    IntentSource,
    IntentStatus,
    LLMProposedOrder,
    OrderIntent,
)

logger = logging.getLogger(__name__)

_INTENTS_TABLE = "order_intents"
_EVENTS_TABLE = "order_intent_events"
_IMPACT_TTL_SECONDS = 5 * 60  # SnapTrade preview window


class TradingDisabledError(Exception):
    """User has not enabled trading or has not acknowledged the disclaimer."""


class RiskCheckFailed(Exception):
    """A per-user risk cap would be exceeded by this intent."""

    def __init__(self, message: str, *, checks: dict[str, Any]) -> None:
        super().__init__(message)
        self.checks = checks


class IntentNotFound(Exception):
    """Intent doesn't exist or doesn't belong to this user."""


class InvalidStateTransition(Exception):
    """The intent isn't in a state from which the requested action is allowed."""


# ---------------------------------------------------------------------------
# OrderIntentService
# ---------------------------------------------------------------------------


class OrderIntentService:
    def __init__(
        self,
        *,
        snaptrade: SnapTradeService,
        memory: MemoryService,
        conversations: ConversationService,
        calendar: MarketCalendarService,
    ) -> None:
        self._snap = snaptrade
        self._memory = memory
        self._conversations = conversations
        self._calendar = calendar

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    async def create_from_ui(
        self, user_id: str, payload: CreateOrderIntent
    ) -> OrderIntent:
        """Persist a new intent from the UI composer."""
        await self._require_trading_enabled(user_id)
        return await self._insert_intent(
            user_id=user_id,
            source="ui",
            conversation_id=payload.conversation_id,
            asset_class=payload.asset_class,
            account_id=payload.account_id,
            universal_symbol_id=payload.universal_symbol_id,
            symbol=payload.symbol,
            action=payload.action,
            order_type=payload.order_type,
            time_in_force=payload.time_in_force,
            units=payload.units,
            notional_value=payload.notional_value,
            price=payload.price,
            stop=payload.stop,
        )

    async def create_from_llm(
        self,
        user_id: str,
        *,
        conversation_id: str | None,
        proposal: LLMProposedOrder,
    ) -> OrderIntent:
        """Persist a new intent emitted by the LLM via ``propose_order``.

        This never auto-executes. The caller (orchestrator) surfaces the
        intent to the user as an inline confirm card.
        """
        profile = await self._memory.get_profile(user_id)
        trading = _trading_from_profile(profile)
        if not trading.llm_proposals_enabled:
            raise TradingDisabledError(
                "LLM trade proposals are disabled for this user"
            )
        await self._require_trading_enabled(user_id, profile_trading=trading)

        # Resolve account_id: the LLM may have supplied it from the positions
        # block, or may have omitted it entirely. Either way we validate it
        # belongs to this user — if it's missing or invalid we auto-pick.
        account_id = await self._resolve_account_id(
            user_id=user_id,
            proposed_account_id=proposal.account_id,
            symbol=proposal.symbol,
            asset_class=proposal.asset_class,
        )

        # LLM shape is lenient about quantity — default to 1 share market
        # if it omits both so the preview still works. The user can edit
        # before confirm. If the LLM provides BOTH (common when a user says
        # "buy 1 share at $200" — the 200 is a limit price, not a notional),
        # prefer units and drop notional_value: the DB check constraint
        # requires exactly one.
        units = proposal.units
        notional_value = proposal.notional_value
        if units is not None and notional_value is not None:
            logger.info(
                "[trade] LLM provided both units=%s and notional=%s — "
                "keeping units (DB requires exactly one)",
                units,
                notional_value,
            )
            notional_value = None
        if units is None and notional_value is None:
            units = 1.0

        return await self._insert_intent(
            user_id=user_id,
            source="llm",
            conversation_id=conversation_id,
            asset_class=proposal.asset_class,
            account_id=account_id,
            universal_symbol_id=None,
            symbol=proposal.symbol,
            action=proposal.action,
            order_type=proposal.order_type,
            time_in_force=proposal.time_in_force,
            units=units,
            notional_value=notional_value,
            price=proposal.price,
            stop=proposal.stop,
        )

    async def _insert_intent(
        self,
        *,
        user_id: str,
        source: IntentSource,
        conversation_id: str | None,
        asset_class: str,
        account_id: str,
        universal_symbol_id: str | None,
        symbol: str,
        action: str,
        order_type: str,
        time_in_force: str,
        units: float | None,
        notional_value: float | None,
        price: float | None,
        stop: float | None,
    ) -> OrderIntent:
        intent_id = str(uuid.uuid4())
        now = _now_iso()
        row = {
            "id": intent_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "source": source,
            "asset_class": asset_class,
            "account_id": account_id,
            "universal_symbol_id": universal_symbol_id,
            "symbol": symbol.upper(),
            "action": action,
            "order_type": order_type,
            "time_in_force": time_in_force,
            "units": units,
            "notional_value": notional_value,
            "price": price,
            "stop": stop,
            "status": "awaiting_confirmation",
            "created_at": now,
            "updated_at": now,
        }
        inserted = await asyncio.to_thread(insert_row, _INTENTS_TABLE, row)
        intent_row = inserted[0] if inserted else row
        await self._record_event(
            intent_id=intent_id,
            from_status=None,
            to_status="awaiting_confirmation",
            payload={"source": source},
        )
        logger.info(
            "[trade] intent_created id=%s user=%s source=%s symbol=%s %s %s",
            intent_id,
            user_id,
            source,
            symbol.upper(),
            action,
            order_type,
        )
        return _row_to_intent(intent_row)

    # ------------------------------------------------------------------
    # Preview / confirm / cancel
    # ------------------------------------------------------------------

    async def preview(self, user_id: str, intent_id: str) -> OrderIntent:
        """Run SnapTrade's ``get_order_impact`` and store the result.

        Crypto has no impact endpoint on SnapTrade, so we skip the preview
        step and go straight from ``awaiting_confirmation`` to a synthetic
        ``previewed`` state with a best-effort estimated_value. The confirm
        flow handles the two cases uniformly.
        """
        intent = await self._load_intent(user_id, intent_id)
        if intent.status not in ("awaiting_confirmation", "previewed"):
            raise InvalidStateTransition(
                f"cannot preview intent in status={intent.status}"
            )

        if intent.asset_class == "crypto":
            impact = ImpactPreview(
                symbol=intent.symbol,
                action=intent.action,
                order_type=intent.order_type,
                time_in_force=intent.time_in_force,
                units=intent.units,
                price=intent.price,
                estimated_value=_estimate_value(intent.units, intent.price, intent.notional_value),
                warnings=["Crypto orders execute directly — preview is an estimate only."],
            )
            await self._patch_intent(
                intent_id,
                {
                    "status": "previewed",
                    "impact_payload": impact.model_dump(),
                    "impact_expires_at": _iso(_now() + timedelta(seconds=_IMPACT_TTL_SECONDS)),
                    "estimated_value": impact.estimated_value,
                },
            )
            await self._record_event(
                intent_id=intent_id,
                from_status=intent.status,
                to_status="previewed",
                payload={"synthetic": True},
            )
            return await self._load_intent(user_id, intent_id)

        # Equity — SnapTrade's get_order_impact requires universal_symbol_id
        # (a UUID), not a plain ticker. Resolve it via a symbol search if
        # we don't already have it, then persist back so retries are free.
        universal_symbol_id = intent.universal_symbol_id
        if not universal_symbol_id:
            universal_symbol_id = await self._resolve_universal_symbol_id(
                user_id=user_id,
                account_id=intent.account_id,
                symbol=intent.symbol,
            )
            if universal_symbol_id:
                await self._patch_intent(
                    intent_id, {"universal_symbol_id": universal_symbol_id}
                )

        try:
            raw = await asyncio.to_thread(
                self._snap.preview_equity_order,
                user_id,
                account_id=intent.account_id,
                action=intent.action,
                order_type=intent.order_type,
                time_in_force=intent.time_in_force,
                universal_symbol_id=universal_symbol_id,
                symbol=intent.symbol,
                units=intent.units,
                notional_value=intent.notional_value,
                price=intent.price,
                stop=intent.stop,
            )
        except ApiException as e:
            err = _safe_api_error(e)
            await self._patch_intent(
                intent_id, {"status": "failed", "error": err}
            )
            await self._record_event(
                intent_id=intent_id,
                from_status=intent.status,
                to_status="failed",
                payload={"stage": "preview", "error": err},
            )
            raise

        impact = _normalise_impact(raw)
        expires_at = _iso(_now() + timedelta(seconds=_IMPACT_TTL_SECONDS))
        await self._patch_intent(
            intent_id,
            {
                "status": "previewed",
                "snaptrade_trade_id": impact.trade_id,
                "impact_payload": impact.model_dump(),
                "impact_expires_at": expires_at,
                "estimated_value": impact.estimated_value,
            },
        )
        await self._record_event(
            intent_id=intent_id,
            from_status=intent.status,
            to_status="previewed",
            payload={
                "trade_id": impact.trade_id,
                "estimated_value": impact.estimated_value,
            },
        )
        logger.info(
            "[trade] previewed id=%s trade_id=%s est=%s",
            intent_id,
            impact.trade_id,
            impact.estimated_value,
        )
        return await self._load_intent(user_id, intent_id)

    async def confirm(self, user_id: str, intent_id: str) -> OrderIntent:
        """User taps Confirm. Runs risk checks, then either submits
        immediately, schedules for market open (closed equity), or errors.
        """
        intent = await self._load_intent(user_id, intent_id)

        # Allow confirm-from-reminder even if we already transitioned once.
        valid_from = {"previewed", "awaiting_confirmation"}
        if intent.status not in valid_from:
            raise InvalidStateTransition(
                f"cannot confirm intent in status={intent.status}"
            )

        # Re-preview if the impact has expired or never existed.
        if intent.status == "awaiting_confirmation" or _impact_expired(intent):
            intent = await self.preview(user_id, intent_id)

        profile = await self._memory.get_profile(user_id)
        trading = _trading_from_profile(profile)
        await self._run_risk_checks(user_id, intent, trading)

        # Market-closed handling — equity only.
        if intent.asset_class == "equity" and not self._calendar.is_open("equity"):
            next_open = self._calendar.next_open("equity")
            await self._patch_intent(
                intent_id,
                {
                    "status": "scheduled_for_market_open",
                    "scheduled_for": _iso(next_open),
                    "reminder_fires_at": _iso(next_open),
                    "confirmed_at": _now_iso(),
                },
            )
            await self._record_event(
                intent_id=intent_id,
                from_status=intent.status,
                to_status="scheduled_for_market_open",
                payload={"next_open": _iso(next_open)},
            )
            logger.info(
                "[trade] scheduled_for_market_open id=%s next_open=%s",
                intent_id,
                next_open.isoformat(),
            )
            return await self._load_intent(user_id, intent_id)

        # Execute.
        await self._patch_intent(
            intent_id, {"status": "confirmed", "confirmed_at": _now_iso()}
        )
        await self._record_event(
            intent_id=intent_id,
            from_status=intent.status,
            to_status="confirmed",
            payload={},
        )
        return await self._submit(user_id, intent_id)

    async def cancel(self, user_id: str, intent_id: str) -> OrderIntent:
        """Cancel a pending intent (pre-submit) or an open brokerage order."""
        intent = await self._load_intent(user_id, intent_id)

        if intent.status in ("filled", "cancelled", "rejected", "failed"):
            return intent

        if intent.status == "submitted" and intent.snaptrade_order_id:
            try:
                await asyncio.to_thread(
                    self._snap.cancel_equity_order,
                    user_id,
                    account_id=intent.account_id,
                    brokerage_order_id=intent.snaptrade_order_id,
                )
            except ApiException as e:
                err = _safe_api_error(e)
                logger.warning(
                    "[trade] cancel at brokerage failed id=%s err=%s", intent_id, err
                )
                # Fall through — we still mark locally as cancelled and
                # surface the error. The brokerage might reject cancel if
                # the order is already filled.

        await self._patch_intent(
            intent_id,
            {"status": "cancelled", "acknowledged_at": _now_iso()},
        )
        await self._record_event(
            intent_id=intent_id,
            from_status=intent.status,
            to_status="cancelled",
            payload={"by": "user"},
        )
        logger.info("[trade] cancelled id=%s from=%s", intent_id, intent.status)
        return await self._load_intent(user_id, intent_id)

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    async def _submit(self, user_id: str, intent_id: str) -> OrderIntent:
        intent = await self._load_intent(user_id, intent_id)
        try:
            if intent.asset_class == "crypto":
                resp = await asyncio.to_thread(
                    self._snap.place_crypto_order,
                    user_id,
                    account_id=intent.account_id,
                    instrument=intent.symbol,
                    side=intent.action,
                    order_type=intent.order_type,
                    time_in_force=intent.time_in_force,
                    amount=intent.units or intent.notional_value or 0.0,
                    price=intent.price,
                )
            else:
                if not intent.snaptrade_trade_id:
                    raise InvalidStateTransition(
                        "cannot submit equity order without snaptrade_trade_id"
                    )
                resp = await asyncio.to_thread(
                    self._snap.place_equity_order,
                    user_id,
                    trade_id=intent.snaptrade_trade_id,
                    wait_to_confirm=False,
                )
        except ApiException as e:
            err = _safe_api_error(e)
            await self._patch_intent(
                intent_id, {"status": "failed", "error": err}
            )
            await self._record_event(
                intent_id=intent_id,
                from_status=intent.status,
                to_status="failed",
                payload={"stage": "submit", "error": err},
            )
            logger.warning("[trade] submit failed id=%s err=%s", intent_id, err)
            raise

        brokerage_order_id = _extract_order_id(resp)
        await self._patch_intent(
            intent_id,
            {
                "status": "submitted",
                "submitted_at": _now_iso(),
                "snaptrade_order_id": brokerage_order_id,
                "place_payload": _shape_summary(resp),
            },
        )
        await self._record_event(
            intent_id=intent_id,
            from_status="confirmed",
            to_status="submitted",
            payload={"brokerage_order_id": brokerage_order_id},
        )
        logger.info(
            "[trade] submitted id=%s brokerage_order_id=%s",
            intent_id,
            brokerage_order_id,
        )
        return await self._load_intent(user_id, intent_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get(self, user_id: str, intent_id: str) -> OrderIntent:
        return await self._load_intent(user_id, intent_id)

    async def list_for_user(
        self,
        user_id: str,
        *,
        status: str | Iterable[str] | None = None,
        limit: int = 50,
    ) -> list[OrderIntent]:
        rows = await asyncio.to_thread(self._select_rows, user_id, status, limit)
        return [_row_to_intent(r) for r in rows]

    async def reminder_queue(self, user_id: str) -> list[OrderIntent]:
        """Intents that have fired a reminder but haven't been acknowledged.

        Frontend renders these as "Markets just opened — re-confirm?".
        """
        rows = await asyncio.to_thread(_select_reminder_queue, user_id)
        return [_row_to_intent(r) for r in rows]

    async def acknowledge_reminder(self, user_id: str, intent_id: str) -> OrderIntent:
        intent = await self._load_intent(user_id, intent_id)
        await self._patch_intent(intent_id, {"acknowledged_at": _now_iso()})
        await self._record_event(
            intent_id=intent_id,
            from_status=intent.status,
            to_status=intent.status,
            payload={"action": "acknowledged"},
        )
        return await self._load_intent(user_id, intent_id)

    async def status(self, user_id: str) -> dict[str, Any]:
        """Cheap badge poll — no SnapTrade calls."""
        reminder_count, active_count = await asyncio.gather(
            asyncio.to_thread(_count_pending_reminders, user_id),
            asyncio.to_thread(_count_active_intents, user_id),
        )
        profile = await self._memory.get_profile(user_id)
        trading = _trading_from_profile(profile)
        market_open = self._calendar.is_open("equity")
        next_open = self._calendar.next_open("equity") if not market_open else None
        return {
            "pending_reminder_count": reminder_count,
            "active_intent_count": active_count,
            "market_open": market_open,
            "next_market_open": _iso(next_open) if next_open else None,
            "trading_enabled": trading.enabled,
            "disclaimer_acknowledged": trading.disclaimer_acknowledged_at is not None,
        }

    # ------------------------------------------------------------------
    # Scheduler entry point — called from APScheduler every 60 s.
    # ------------------------------------------------------------------

    async def process_due_reminders(self) -> int:
        """Atomically claim due reminders and emit chat notifications.

        Returns the number of intents that flipped from
        ``scheduled_for_market_open`` → ``awaiting_confirmation``.

        The claim uses ``FOR UPDATE SKIP LOCKED`` at the Postgres level so
        multiple concurrent pollers never double-process the same row.
        """
        claimed = await asyncio.to_thread(_claim_due_reminders, 200)
        if not claimed:
            return 0
        logger.info("[trade] reminder_poller claimed=%d", len(claimed))
        for row in claimed:
            intent = _row_to_intent(row)
            await self._record_event(
                intent_id=intent.id,
                from_status="scheduled_for_market_open",
                to_status="awaiting_confirmation",
                payload={"reason": "market_open"},
            )
            await self._emit_reminder_message(intent)
        return len(claimed)

    async def _emit_reminder_message(self, intent: OrderIntent) -> None:
        """Post a ``system`` message into the originating conversation so
        the user sees the reminder inline in chat. Safe if conversation_id
        is None (UI-initiated intents with no thread) — frontend surfaces
        those via ``GET /trade/reminders`` polling instead.
        """
        if not intent.conversation_id:
            return
        try:
            await self._conversations.append_message(
                conversation_id=intent.conversation_id,
                role=MessageRole.system,
                content=(
                    f"Markets just opened — tap to re-confirm your "
                    f"{intent.action} {intent.symbol} order."
                ),
                metadata={
                    "source": "trade_reminder",
                    "intent_id": intent.id,
                    "symbol": intent.symbol,
                    "action": intent.action,
                },
            )
        except Exception:
            logger.exception(
                "[trade] failed to post reminder message for intent=%s",
                intent.id,
            )

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    async def _resolve_account_id(
        self,
        *,
        user_id: str,
        proposed_account_id: str | None,
        symbol: str,
        asset_class: str,
    ) -> str:
        """Return a valid SnapTrade account id for this user.

        Priority:
        1. If the LLM supplied an account_id, verify it actually belongs to
           the user (prevents prompt-injection spoofing a different user's
           account). If valid, use it.
        2. If the LLM omitted it (or supplied an invalid one), pick the first
           account that already holds the ticker (for sells) or the account
           with the highest cash balance (for buys). Fall back to the first
           account in the list if neither heuristic finds a match.

        Raises ``InvalidStateTransition`` if the user has no connected
        accounts at all — there's nothing we can trade on.
        """
        accounts = await asyncio.to_thread(
            self._snap.get_accounts, user_id
        )
        if not accounts:
            raise InvalidStateTransition(
                "No connected brokerage accounts found. "
                "Please link an account via SnapTrade first."
            )

        valid_ids = {a.get("id") for a in accounts if a.get("id")}

        if proposed_account_id and proposed_account_id in valid_ids:
            return proposed_account_id

        if proposed_account_id and proposed_account_id not in valid_ids:
            logger.warning(
                "[trade] LLM proposed unknown account_id=%s for user=%s — "
                "auto-resolving",
                proposed_account_id,
                user_id,
            )

        sym_upper = symbol.upper()
        # Try to find an account that already holds this ticker (ideal for sells
        # and also the most natural account to use for buys of an existing holding).
        try:
            positions = await asyncio.to_thread(self._snap.get_positions, user_id)
            for pos in positions:
                if (
                    pos.get("account_id") in valid_ids
                    and (pos.get("symbol") or "").upper() == sym_upper
                ):
                    return pos["account_id"]
        except Exception:
            logger.warning(
                "[trade] could not load positions for account resolution", exc_info=True
            )

        # Fall back to the account with the largest total balance (most cash to deploy).
        best = max(
            accounts,
            key=lambda a: float(a.get("balance_total") or a.get("balance_cash") or 0),
            default=accounts[0],
        )
        resolved_id = best.get("id", "")
        logger.info(
            "[trade] auto-resolved account_id=%s for user=%s symbol=%s",
            resolved_id,
            user_id,
            sym_upper,
        )
        return resolved_id

    async def _resolve_universal_symbol_id(
        self,
        *,
        user_id: str,
        account_id: str,
        symbol: str,
    ) -> str | None:
        """Look up the SnapTrade universal_symbol_id for a plain ticker.

        SnapTrade's ``get_order_impact`` requires this UUID rather than the
        raw ticker string. We call ``symbol_search_user_account`` scoped to
        the chosen account so we only get symbols that broker supports.

        Returns the first match's id, or ``None`` if the search fails or
        returns no results (caller treats ``None`` as a graceful no-op and
        the SDK will surface the error at submit time).
        """
        try:
            results = await asyncio.to_thread(
                self._snap.search_symbol,
                user_id,
                account_id=account_id,
                query=symbol.upper(),
            )
        except Exception:
            logger.warning(
                "[trade] symbol search failed for %s — proceeding without "
                "universal_symbol_id (SnapTrade may reject the preview)",
                symbol,
                exc_info=True,
            )
            return None

        if not results:
            logger.warning(
                "[trade] no symbol search results for %s in account %s",
                symbol,
                account_id,
            )
            return None

        # Pick the result whose ticker matches exactly; fall back to the
        # first result if there's no exact match (e.g. BRK.B vs BRK/B).
        sym_upper = symbol.upper()
        for result in results:
            # search_symbol returns raw SnapTrade dicts; id is the
            # universal_symbol_id we need.
            result_ticker = (
                result.get("symbol") or result.get("raw_symbol") or ""
            ).upper()
            if result_ticker == sym_upper:
                uid = result.get("id") or result.get("universal_symbol_id")
                if uid:
                    logger.info(
                        "[trade] resolved universal_symbol_id=%s for %s",
                        uid,
                        symbol,
                    )
                    return str(uid)

        # No exact match — take the first result's id.
        first = results[0]
        uid = first.get("id") or first.get("universal_symbol_id")
        if uid:
            logger.info(
                "[trade] resolved universal_symbol_id=%s for %s (first result)",
                uid,
                symbol,
            )
            return str(uid)

        return None

    async def _require_trading_enabled(
        self,
        user_id: str,
        *,
        profile_trading: TradingConfig | None = None,
    ) -> TradingConfig:
        if profile_trading is None:
            profile = await self._memory.get_profile(user_id)
            profile_trading = _trading_from_profile(profile)
        if not profile_trading.enabled:
            raise TradingDisabledError(
                "Trading is not enabled for this user. Toggle it in settings."
            )
        if profile_trading.disclaimer_acknowledged_at is None:
            raise TradingDisabledError(
                "The trading disclaimer must be acknowledged before placing orders."
            )
        return profile_trading

    async def _run_risk_checks(
        self,
        user_id: str,
        intent: OrderIntent,
        trading: TradingConfig,
    ) -> None:
        """Hard-enforce per-user caps. Writes the outcome onto the intent
        so the frontend can display exactly which check blocked a confirm.
        """
        if not trading.enabled:
            raise TradingDisabledError("Trading is disabled")

        estimated = intent.estimated_value
        if estimated is None and intent.price is not None and intent.units is not None:
            estimated = intent.price * intent.units
        if estimated is None and intent.notional_value is not None:
            estimated = intent.notional_value

        checks: dict[str, Any] = {
            "max_order_usd": trading.max_order_usd,
            "max_daily_usd": trading.max_daily_usd,
            "estimated_value": estimated,
        }

        if estimated is not None and estimated > trading.max_order_usd:
            checks["failed"] = "max_order_usd"
            await self._patch_intent(
                intent.id, {"risk_checks": checks}
            )
            raise RiskCheckFailed(
                f"Estimated value ${estimated:,.2f} exceeds per-order cap "
                f"${trading.max_order_usd:,.2f}.",
                checks=checks,
            )

        spent_today = await asyncio.to_thread(_sum_today_submitted_value, user_id)
        projected = spent_today + (estimated or 0.0)
        checks["spent_today"] = spent_today
        checks["projected_daily"] = projected
        if projected > trading.max_daily_usd:
            checks["failed"] = "max_daily_usd"
            await self._patch_intent(intent.id, {"risk_checks": checks})
            raise RiskCheckFailed(
                f"Projected daily volume ${projected:,.2f} exceeds daily cap "
                f"${trading.max_daily_usd:,.2f}.",
                checks=checks,
            )

        checks["passed"] = True
        await self._patch_intent(intent.id, {"risk_checks": checks})

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    async def _load_intent(self, user_id: str, intent_id: str) -> OrderIntent:
        row = await asyncio.to_thread(
            select_maybe,
            _INTENTS_TABLE,
            filters={"id": intent_id, "user_id": user_id},
        )
        if not row:
            raise IntentNotFound(intent_id)
        return _row_to_intent(row)

    async def _patch_intent(
        self, intent_id: str, patch: dict[str, Any]
    ) -> None:
        patch = {**patch, "updated_at": _now_iso()}
        await asyncio.to_thread(
            update_rows,
            _INTENTS_TABLE,
            patch,
            filters={"id": intent_id},
        )

    async def _record_event(
        self,
        *,
        intent_id: str,
        from_status: str | None,
        to_status: str,
        payload: dict[str, Any],
    ) -> None:
        row = {
            "id": str(uuid.uuid4()),
            "intent_id": intent_id,
            "from_status": from_status,
            "to_status": to_status,
            "payload": payload,
            "created_at": _now_iso(),
        }
        try:
            await asyncio.to_thread(insert_row, _EVENTS_TABLE, row)
        except Exception:
            logger.exception(
                "[trade] failed to record event intent=%s %s->%s",
                intent_id,
                from_status,
                to_status,
            )
        # Counter-style log line — cheap for log-based metrics (one per
        # transition). Intentionally contains no user-identifying payload.
        logger.info(
            "[trade] transition intent=%s %s->%s", intent_id, from_status, to_status
        )

    def _select_rows(
        self,
        user_id: str,
        status: str | Iterable[str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        q = (
            get_supabase()
            .table(_INTENTS_TABLE)
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
        )
        if status is not None:
            if isinstance(status, str):
                q = q.eq("status", status)
            else:
                q = q.in_("status", list(status))
        res = q.execute()
        return res.data or []


# ---------------------------------------------------------------------------
# Module-level helpers (run inside to_thread for all blocking DB I/O)
# ---------------------------------------------------------------------------


def _select_reminder_queue(user_id: str) -> list[dict[str, Any]]:
    res = (
        get_supabase()
        .table(_INTENTS_TABLE)
        .select("*")
        .eq("user_id", user_id)
        .not_.is_("reminded_at", "null")
        .is_("acknowledged_at", "null")
        .order("reminded_at", desc=True)
        .limit(50)
        .execute()
    )
    return res.data or []


def _count_pending_reminders(user_id: str) -> int:
    res = (
        get_supabase()
        .table(_INTENTS_TABLE)
        .select("id", count="exact")
        .eq("user_id", user_id)
        .not_.is_("reminded_at", "null")
        .is_("acknowledged_at", "null")
        .execute()
    )
    return res.count if res.count is not None else len(res.data or [])


def _count_active_intents(user_id: str) -> int:
    active = (
        "awaiting_confirmation",
        "previewed",
        "confirmed",
        "scheduled_for_market_open",
        "submitted",
    )
    res = (
        get_supabase()
        .table(_INTENTS_TABLE)
        .select("id", count="exact")
        .eq("user_id", user_id)
        .in_("status", list(active))
        .execute()
    )
    return res.count if res.count is not None else len(res.data or [])


def _sum_today_submitted_value(user_id: str) -> float:
    """Return the sum of today's confirmed + submitted intents' estimated values.

    Used for the ``max_daily_usd`` risk check. Scheduled-for-open intents
    are intentionally excluded — they can't accidentally double-count
    against the cap until the user re-confirms them.
    """
    today_start = datetime.now(tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    rows = (
        get_supabase()
        .table(_INTENTS_TABLE)
        .select("estimated_value")
        .eq("user_id", user_id)
        .in_("status", ["confirmed", "submitted", "filled"])
        .gte("confirmed_at", today_start.isoformat())
        .execute()
        .data
        or []
    )
    total = 0.0
    for row in rows:
        v = row.get("estimated_value")
        if v is not None:
            try:
                total += float(v)
            except (TypeError, ValueError):
                continue
    return total


def _claim_due_reminders(limit: int) -> list[dict[str, Any]]:
    """Atomically flip due reminders from scheduled → awaiting_confirmation.

    Two-step claim using PostgREST:
      1. SELECT candidate ids under the partial index.
      2. UPDATE ... WHERE status='scheduled_for_market_open' AND reminded_at IS NULL
         to move them — PostgreSQL's row-level locking guarantees each row
         can only be flipped by one worker at a time (the UPDATE fails to
         match for any worker that lost the race).

    PostgREST doesn't expose ``FOR UPDATE SKIP LOCKED`` directly, so we rely
    on the ``WHERE reminded_at IS NULL`` predicate + the unique-per-row
    UUID to make the update idempotent: once one worker flips a row, its
    ``reminded_at`` is non-null and any concurrent worker's update matches
    0 rows.
    """
    now_iso = _now_iso()
    sb = get_supabase()
    candidates = (
        sb.table(_INTENTS_TABLE)
        .select("id")
        .eq("status", "scheduled_for_market_open")
        .is_("reminded_at", "null")
        .lte("reminder_fires_at", now_iso)
        .order("reminder_fires_at", desc=False)
        .limit(limit)
        .execute()
        .data
        or []
    )
    if not candidates:
        return []

    ids = [c["id"] for c in candidates if c.get("id")]
    if not ids:
        return []

    # Single UPDATE scoped to the just-selected ids. The idempotency guard
    # is the ``reminded_at IS NULL`` predicate — any worker that arrived
    # second matches zero rows.
    res = (
        sb.table(_INTENTS_TABLE)
        .update(
            {
                "reminded_at": now_iso,
                "status": "awaiting_confirmation",
                "updated_at": now_iso,
            }
        )
        .in_("id", ids)
        .eq("status", "scheduled_for_market_open")
        .is_("reminded_at", "null")
        .execute()
    )
    return res.data or []


# ---------------------------------------------------------------------------
# Row ↔ model mapping
# ---------------------------------------------------------------------------


def _row_to_intent(row: dict[str, Any]) -> OrderIntent:
    impact_data = row.get("impact_payload")
    impact = None
    if impact_data:
        try:
            impact = ImpactPreview.model_validate(impact_data)
        except Exception:
            logger.warning(
                "[trade] failed to parse impact_payload for intent=%s",
                row.get("id"),
            )
            impact = None
    return OrderIntent(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        conversation_id=_opt_str(row.get("conversation_id")),
        source=row["source"],
        asset_class=row["asset_class"],
        account_id=str(row["account_id"]),
        universal_symbol_id=_opt_str(row.get("universal_symbol_id")),
        symbol=str(row["symbol"]),
        action=row["action"],
        order_type=row["order_type"],
        time_in_force=row["time_in_force"],
        units=_opt_float(row.get("units")),
        notional_value=_opt_float(row.get("notional_value")),
        price=_opt_float(row.get("price")),
        stop=_opt_float(row.get("stop")),
        status=row["status"],
        snaptrade_trade_id=_opt_str(row.get("snaptrade_trade_id")),
        snaptrade_order_id=_opt_str(row.get("snaptrade_order_id")),
        impact=impact,
        impact_expires_at=_opt_str(row.get("impact_expires_at")),
        estimated_value=_opt_float(row.get("estimated_value")),
        scheduled_for=_opt_str(row.get("scheduled_for")),
        reminder_fires_at=_opt_str(row.get("reminder_fires_at")),
        reminded_at=_opt_str(row.get("reminded_at")),
        acknowledged_at=_opt_str(row.get("acknowledged_at")),
        confirmed_at=_opt_str(row.get("confirmed_at")),
        submitted_at=_opt_str(row.get("submitted_at")),
        filled_at=_opt_str(row.get("filled_at")),
        error=row.get("error"),
        risk_checks=row.get("risk_checks") or {},
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _trading_from_profile(profile) -> TradingConfig:
    """Extract or synthesise the TradingConfig from whatever shape Supabase
    returned. ``trading`` is a jsonb column added after initial deploys, so
    older profile rows may lack it — default safely to off.
    """
    raw = getattr(profile, "trading", None)
    if isinstance(raw, TradingConfig):
        return raw
    if isinstance(raw, dict):
        try:
            return TradingConfig.model_validate(raw)
        except Exception:
            pass
    return TradingConfig()


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _opt_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _opt_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _impact_expired(intent: OrderIntent) -> bool:
    if not intent.impact_expires_at:
        return True
    try:
        exp = datetime.fromisoformat(intent.impact_expires_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    return exp <= _now()


def _estimate_value(
    units: float | None, price: float | None, notional: float | None
) -> float | None:
    if notional is not None:
        return float(notional)
    if units is not None and price is not None:
        return float(units) * float(price)
    return None


def _normalise_impact(raw: dict[str, Any]) -> ImpactPreview:
    """Extract the fields the UI cares about from SnapTrade's variably-shaped
    impact response. Keeps the whole payload in ``raw`` so nothing is lost.
    """
    trade = raw.get("trade") or {}
    impact_block = raw.get("trade_impact") or raw.get("impact") or {}

    symbol_obj = trade.get("symbol") if isinstance(trade, dict) else None
    symbol = None
    if isinstance(symbol_obj, dict):
        symbol = symbol_obj.get("symbol") or symbol_obj.get("raw_symbol")
    elif isinstance(symbol_obj, str):
        symbol = symbol_obj

    units = trade.get("units") if isinstance(trade, dict) else None
    price = trade.get("price") if isinstance(trade, dict) else None
    estimated = None
    if units is not None and price is not None:
        try:
            estimated = float(units) * float(price)
        except (TypeError, ValueError):
            estimated = None

    commission = impact_block.get("commission") if isinstance(impact_block, dict) else None
    buying_power_effect = impact_block.get("buying_power_effect") or impact_block.get("buying_power") if isinstance(impact_block, dict) else None
    remaining_buying_power = impact_block.get("remaining_buying_power") if isinstance(impact_block, dict) else None

    return ImpactPreview(
        trade_id=(trade.get("id") if isinstance(trade, dict) else None),
        symbol=symbol,
        action=(trade.get("action") if isinstance(trade, dict) else None),
        order_type=(trade.get("order_type") if isinstance(trade, dict) else None),
        time_in_force=(trade.get("time_in_force") if isinstance(trade, dict) else None),
        units=_opt_float(units),
        price=_opt_float(price),
        estimated_value=estimated,
        commission=_opt_float(commission),
        currency=(impact_block.get("currency") if isinstance(impact_block, dict) else None),
        buying_power_effect=_opt_float(buying_power_effect),
        remaining_buying_power=_opt_float(remaining_buying_power),
        expires_at=_iso(_now() + timedelta(seconds=_IMPACT_TTL_SECONDS)),
        warnings=list(impact_block.get("warnings") or []) if isinstance(impact_block, dict) else [],
        raw=raw,
    )


def _extract_order_id(resp: Any) -> str | None:
    """Try a few common SnapTrade response shapes to find the brokerage order id."""
    if not isinstance(resp, dict):
        return None
    for key in ("brokerage_order_id", "id", "order_id"):
        v = resp.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Nested under "trade"
    trade = resp.get("trade") if isinstance(resp, dict) else None
    if isinstance(trade, dict):
        for key in ("brokerage_order_id", "id"):
            v = trade.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _shape_summary(resp: Any) -> dict[str, Any]:
    """Return a redacted shape of the SnapTrade place response for logging.

    Strips any potential secret fields; keeps status and ids only.
    """
    if not isinstance(resp, dict):
        return {"shape": type(resp).__name__}
    keys = {"id", "brokerage_order_id", "order_id", "status", "state", "symbol"}
    out = {k: v for k, v in resp.items() if k in keys}
    trade = resp.get("trade") if isinstance(resp, dict) else None
    if isinstance(trade, dict):
        out["trade"] = {k: v for k, v in trade.items() if k in keys}
    return out


def _safe_api_error(e: ApiException) -> dict[str, Any]:
    """Serialise ``snaptrade_client.ApiException`` without leaking headers."""
    body = getattr(e, "body", None)
    parsed: Any = None
    if isinstance(body, (bytes, bytearray)):
        try:
            parsed = json.loads(body.decode("utf-8", errors="replace"))
        except Exception:
            parsed = body.decode("utf-8", errors="replace")
    elif isinstance(body, str):
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = body
    elif isinstance(body, dict):
        parsed = body
    return {
        "status": getattr(e, "status", None),
        "reason": getattr(e, "reason", None),
        "body": parsed,
    }


# ---------------------------------------------------------------------------
# Singleton wiring
# ---------------------------------------------------------------------------


@lru_cache
def get_order_intent_service() -> OrderIntentService:
    return OrderIntentService(
        snaptrade=get_snaptrade_service(),
        memory=get_memory_service(),
        conversations=get_conversation_service(),
        calendar=get_market_calendar(),
    )


__all__ = [
    "OrderIntentService",
    "get_order_intent_service",
    "TradingDisabledError",
    "RiskCheckFailed",
    "IntentNotFound",
    "InvalidStateTransition",
]
