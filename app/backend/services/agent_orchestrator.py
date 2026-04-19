"""
AgentOrchestrator — single entry point for an incoming user message.

Responsibilities
----------------
1. **Route**: if the message opens with ``/skill``, hand off to
   :class:`SkillSystem`. Otherwise ask :class:`RecommendationAgent` for a
   conversational reply.
2. **Persist**: always append the user message first (so it's durable even if
   the agent layer crashes), then append the assistant reply on success, or a
   graceful fallback message on failure.
3. **Carry metadata**: each turn is tagged with ``{"source": ...}`` so the
   frontend can render skill acknowledgements, recommendation stubs, or plain
   chat replies with distinct UI affordances.

The orchestrator is the *only* place the conversation routes touch higher-
level agents — this keeps the HTTP layer thin and lets us swap agent
implementations without rewriting the routes.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.backend.services.conversations import (
    ConversationService,
    get_conversation_service,
)
from app.backend.services.memory import MemoryService, get_memory_service
from app.backend.services.order_intents import (
    OrderIntentService,
    TradingDisabledError,
    get_order_intent_service,
)
from app.backend.services.recommendations import (
    RecommendationAgent,
    get_recommendation_agent,
)
from app.backend.services.skills import (
    SkillKind,
    SkillResult,
    SkillSystem,
    get_skill_system,
)
from app.schemas.conversations import Message, MessageRole
from app.schemas.trade import LLMProposedOrder

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OrchestratorOutput:
    """What the HTTP layer returns to the client."""

    user_message: Message
    assistant_message: Message


class AgentOrchestrator:
    def __init__(
        self,
        *,
        conversations: ConversationService,
        skills: SkillSystem,
        recommender: RecommendationAgent,
        memory: MemoryService,
        order_intents: OrderIntentService,
    ) -> None:
        self._conversations = conversations
        self._skills = skills
        self._recommender = recommender
        self._memory = memory
        self._order_intents = order_intents

    async def handle_user_message(
        self,
        *,
        user_id: str,
        conversation_id: str,
        content: str,
        user_metadata: dict[str, Any] | None = None,
    ) -> OrchestratorOutput:
        """Persist a user turn, run the right agent, persist the assistant turn."""
        user_message = await self._conversations.append_message(
            conversation_id=conversation_id,
            role=MessageRole.user,
            content=content,
            metadata=user_metadata,
            transcript=(user_metadata or {}).get("transcript") if user_metadata else None,
        )

        try:
            if SkillSystem.detect(content):
                assistant_message = await self._handle_skill(
                    user_id, conversation_id, content
                )
            else:
                assistant_message = await self._handle_chat(
                    user_id, conversation_id, content
                )
        except Exception:
            logger.exception(
                "[orchestrator] agent pipeline failed for conversation=%s",
                conversation_id,
            )
            assistant_message = await self._conversations.append_message(
                conversation_id=conversation_id,
                role=MessageRole.assistant,
                content=(
                    "Sorry — I ran into an error generating a reply. "
                    "Please try again in a moment."
                ),
                metadata={"source": "error"},
            )

        return OrchestratorOutput(
            user_message=user_message,
            assistant_message=assistant_message,
        )

    # --------------------------------------------------------- route: /skill

    async def _handle_skill(
        self, user_id: str, conversation_id: str, content: str
    ) -> Message:
        result: SkillResult = await self._skills.handle(user_id, content)
        metadata: dict[str, Any] = {"source": "skill", "kind": result.kind.value}
        if result.kind == SkillKind.note_appended and result.note is not None:
            metadata["note_id"] = result.note.id
        if result.kind == SkillKind.level_updated and result.profile is not None:
            metadata["experience_level"] = result.profile.experience_level.value
        return await self._conversations.append_message(
            conversation_id=conversation_id,
            role=MessageRole.assistant,
            content=result.reply,
            metadata=metadata,
        )

    # ---------------------------------------------------------- route: chat

    async def _handle_chat(
        self, user_id: str, conversation_id: str, content: str
    ) -> Message:
        # Pull the rolling context *excluding* the turn we just persisted — we
        # pass ``content`` as the final user turn separately, so we want the
        # prior turns here. Redis has the fresh turn on top, so drop that.
        history = await self._conversations.get_context(conversation_id)
        if history and history[-1].get("content") == content:
            history = history[:-1]

        # Allow the LLM to emit a propose_order tool call only when the
        # user has opted in. We gate on both the master ``enabled`` switch
        # and the specific ``llm_proposals_enabled`` toggle so a user can
        # keep trading on for manual orders while disabling LLM suggestions.
        profile = await self._memory.get_profile(user_id)
        trading = getattr(profile, "trading", None)
        trading_enabled = bool(trading and getattr(trading, "enabled", False))
        disclaimer_done = bool(
            trading and getattr(trading, "disclaimer_acknowledged_at", None)
        )
        llm_proposals_on = bool(
            trading and getattr(trading, "llm_proposals_enabled", False)
        )
        # Server-side gate: the LLM is over-eager and will call ``propose_order``
        # for vague questions like "what stocks should I buy in Europe?" We
        # only offer the tool when the message itself reads like a *specific*
        # trade command (direction verb + concrete symbol). Generic market
        # questions fall through to a normal chat reply.
        looks_specific_trade = _looks_like_specific_trade(content)
        allow_tools = (
            trading_enabled
            and disclaimer_done
            and llm_proposals_on
            and looks_specific_trade
        )

        # When trading is turned off or the disclaimer hasn't been acknowledged
        # and the user message looks like a trade request, return a clear
        # actionable message instead of a generic fallback.
        settings_gate_hit = (
            not (trading_enabled and disclaimer_done and llm_proposals_on)
            and looks_specific_trade
        )
        if settings_gate_hit:
            if not trading_enabled or not disclaimer_done:
                gate_msg = (
                    "Trading isn't enabled on your account yet. "
                    "Head to **Settings → Trading** to enable it and acknowledge "
                    "the disclaimer, then ask me again and I'll put together a "
                    "proposed order for your review."
                )
            else:
                # enabled + disclaimer OK, but llm_proposals_on is False
                gate_msg = (
                    "LLM order proposals are currently turned off. "
                    "You can enable them in **Settings → Trading** under "
                    "\"Let Artie suggest orders during chat\", or place the "
                    "order manually from the Orders screen."
                )
            return await self._conversations.append_message(
                conversation_id=conversation_id,
                role=MessageRole.assistant,
                content=gate_msg,
                metadata={"source": "chat", "trade_gated": True},
            )

        outcome = await self._recommender.chat_with_tools(
            user_id=user_id,
            message=content,
            history=history,
            allow_propose_order=allow_tools,
        )
        reply_text = outcome.get("text") or ""
        proposal_payload = outcome.get("proposal")

        if proposal_payload and allow_tools:
            proposal_msg = await self._materialise_proposal(
                user_id=user_id,
                conversation_id=conversation_id,
                proposal_payload=proposal_payload,
                assistant_preface=reply_text,
            )
            if proposal_msg is not None:
                return proposal_msg
            # Fall through to plain chat reply if proposal creation failed.

        if not reply_text:
            reply_text = (
                "I wasn't able to come up with a useful reply for that. "
                "Try asking me about a specific position or news item."
            )
        return await self._conversations.append_message(
            conversation_id=conversation_id,
            role=MessageRole.assistant,
            content=reply_text,
            metadata={"source": "chat"},
        )

    async def _materialise_proposal(
        self,
        *,
        user_id: str,
        conversation_id: str,
        proposal_payload: dict[str, Any],
        assistant_preface: str,
    ) -> Message | None:
        """Convert a ``propose_order`` tool-call payload into a durable
        intent and return an assistant message the UI can render as a
        confirm card. Returns ``None`` if the payload is unusable, so the
        caller falls back to a plain chat reply.
        """
        try:
            proposal = LLMProposedOrder.model_validate(proposal_payload)
        except ValidationError as e:
            logger.warning(
                "[orchestrator] invalid propose_order payload: %s", e.errors()[:3]
            )
            return None

        preview_failed = False
        try:
            intent = await self._order_intents.create_from_llm(
                user_id=user_id,
                conversation_id=conversation_id,
                proposal=proposal,
            )
            # Fire a preview immediately so the confirm card has an
            # estimated value / commission to show without a second round-trip.
            try:
                intent = await self._order_intents.preview(user_id, intent.id)
            except Exception:
                preview_failed = True
                logger.warning(
                    "[orchestrator] preview after LLM proposal failed intent=%s",
                    intent.id,
                    exc_info=True,
                )
                # Cancel the half-baked intent so the UI doesn't see a stale
                # "awaiting_confirmation" row with no impact data — which
                # currently renders as "This order is no longer available".
                try:
                    await self._order_intents.cancel(user_id, intent.id)
                except Exception:
                    logger.warning(
                        "[orchestrator] failed to cancel stale intent=%s",
                        intent.id,
                        exc_info=True,
                    )
        except TradingDisabledError as e:
            return await self._conversations.append_message(
                conversation_id=conversation_id,
                role=MessageRole.assistant,
                content=(
                    "I can't place trades for you until trading is enabled "
                    "and the disclaimer has been acknowledged in settings."
                ),
                metadata={"source": "chat", "error": str(e)},
            )
        except Exception:
            logger.exception(
                "[orchestrator] failed to materialise LLM proposal for user=%s",
                user_id,
            )
            return None

        rationale = (proposal.rationale or assistant_preface or "").strip()

        if preview_failed:
            # We couldn't get a broker preview (e.g. symbol not supported by
            # this account, market closed + broker rejects pre-impact, etc.).
            # Surface a plain chat reply rather than a trade-proposal card so
            # the UI doesn't show "This order is no longer available".
            body = (
                (rationale + "\n\n" if rationale else "")
                + f"I couldn't get a live preview for that {proposal.action.upper()} "
                + f"of {proposal.symbol} from your brokerage right now. "
                + "You can try again in a moment, or place it manually from the "
                + "Orders screen. If the market is closed, I'll still be able "
                + "to schedule it for the next open — just retry once we're back."
            )
            return await self._conversations.append_message(
                conversation_id=conversation_id,
                role=MessageRole.assistant,
                content=body,
                metadata={"source": "chat", "preview_failed": True},
            )

        content_text = (
            (rationale + "\n\n" if rationale else "")
            + f"Proposed {proposal.action} for {proposal.symbol} — tap Confirm to place it."
        )
        return await self._conversations.append_message(
            conversation_id=conversation_id,
            role=MessageRole.assistant,
            content=content_text,
            metadata={
                "source": "trade_proposal",
                "intent_id": intent.id,
                "symbol": intent.symbol,
                "action": intent.action,
                "asset_class": intent.asset_class,
                "impact_expires_at": intent.impact_expires_at,
                "estimated_value": intent.estimated_value,
            },
        )


_TRADE_KEYWORDS = frozenset(
    {
        "buy", "sell", "order", "purchase", "trade", "shares", "share",
        "stock", "crypto", "btc", "eth", "place", "execute", "market order",
        "limit order", "long", "short", "position",
    }
)

# Direction verbs that clearly indicate the user wants to *execute* a trade
# (as opposed to asking a question about trading).
_TRADE_DIRECTION_VERBS = frozenset(
    {"buy", "sell", "purchase", "short", "long", "place", "submit", "execute"}
)

# Words that strongly imply the message is a *question* rather than a command.
# If any of these appear we assume the user wants advice, not an order.
_QUESTION_MARKERS = frozenset(
    {
        "what", "which", "should", "recommend", "suggest", "suggestion",
        "advice", "opinion", "thoughts", "tell me about", "is it", "are there",
        "can you explain", "explain", "why", "how does", "how do",
        "good idea", "worth", "overview", "top ", "best ",
    }
)

# Plain ticker: 1–5 uppercase letters surrounded by word boundaries. We also
# accept a '$' prefix ($AAPL) and crypto pairs like BTC/USD or ETH-USD.
_TICKER_RE = re.compile(r"(?:^|[\s(])\$?([A-Z]{1,5})(?=[\s.,!?)]|$)")
_CRYPTO_PAIR_RE = re.compile(r"\b([A-Z]{2,6})[/\-]([A-Z]{2,6})\b")
# Numeric amount hints: "1 share", "10 units", "$500", "200 dollars".
_AMOUNT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:share|shares|unit|units|lot|lots)\b"
    r"|\$\s*\d+(?:\.\d+)?"
    r"|\b\d+(?:\.\d+)?\s*(?:usd|dollars?|bucks)\b",
    re.IGNORECASE,
)


def _looks_like_trade_request(message: str) -> bool:
    """Permissive heuristic — any trade-adjacent keyword at all.

    Used ONLY to decide whether to show the "enable trading in Settings"
    hint when the feature is off. We deliberately err on the side of
    showing the hint rather than a generic fallback.
    """
    lower = message.lower()
    return any(kw in lower for kw in _TRADE_KEYWORDS)


def _looks_like_specific_trade(message: str) -> bool:
    """Strict heuristic — does this message look like a concrete order?

    We only offer the LLM the ``propose_order`` tool when the answer is
    yes. Requires ALL of:
      * A direction verb (``buy``/``sell``/``purchase``/...).
      * A specific symbol (ticker or crypto pair).
      * Not phrased as a generic question ("what should I buy...",
        "which stocks are good...").

    This keeps general market questions like "what are good stocks to
    buy in Europe" on the plain-chat path instead of firing an order
    proposal the broker will reject.
    """
    lower = message.lower()
    stripped = lower.strip()

    if not any(v in lower for v in _TRADE_DIRECTION_VERBS):
        return False

    has_ticker = bool(
        _TICKER_RE.search(message) or _CRYPTO_PAIR_RE.search(message.upper())
    )
    has_amount = bool(_AMOUNT_RE.search(message))

    if not (has_ticker or has_amount):
        return False

    # Suppress on generic question phrasing unless the user also supplied
    # a concrete amount (e.g. "which is better, 10 NVDA or 20 AMD" is still
    # a question). "Buy 1 share of NVDA" doesn't start with what/which.
    if stripped.startswith(("what", "which", "should", "is ", "are ", "how ")):
        return has_amount and has_ticker
    for marker in _QUESTION_MARKERS:
        if marker in lower and not has_amount:
            return False

    return True


_SINGLETON: AgentOrchestrator | None = None


def get_agent_orchestrator() -> AgentOrchestrator:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = AgentOrchestrator(
            conversations=get_conversation_service(),
            skills=get_skill_system(),
            recommender=get_recommendation_agent(),
            memory=get_memory_service(),
            order_intents=get_order_intent_service(),
        )
    return _SINGLETON


__all__ = [
    "AgentOrchestrator",
    "OrchestratorOutput",
    "get_agent_orchestrator",
]
