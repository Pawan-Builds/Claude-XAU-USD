"""Safe order execution with pending (limit) order lifecycle management.

F-02 fix summary:
- Removed the undefined Mock() reference from the market-fill bookkeeping path.
- Added explicit, idempotent post-fill bookkeeping with retry-on-next-cycle.
- Persisted OrderRequest.mt5_ticket and ExecutedOrder.order_request_id.
- Added deal-history reconciliation for pending fills (ticket mismatch).
- Made ClosedTrade inserts idempotent.
- Restricted close_all_positions to bot-owned magic only.
- Added EntryGate check at the top of execute_signal (defence in depth).

Batch 4 corrections:
- Deleted the duplicated §1d position-sizing block (kept one).
- §1d now sizes against mt5_connector.get_tick() — the canonical quote
  source, matching _process_trade_plans and _prepare_order_request —
  instead of symbol_info.ask/bid. Fails closed if the quote is missing
  or non-positive.
- §1g reads the gate token via getattr(final_gate, "token", None).
  The connector is the enforcement boundary for opening actions (C-03);
  the executor's job is only to pass the token through. This avoids
  turning a missing-token condition into an "uncertain" status that
  blocks the fingerprint duplicate check and needs reconciliation.
- _prepare_order_request calls MT5Connector.pip_size_for() (class static
  method) with an isinstance guard, so the pip size stays a real float
  when the connector instance is patched by tests.

N-H01 additions:
- Candidate-order risk is now reserved atomically before submission
  (RiskManager.try_reserve) and released on any terminal outcome.
- Release paths: _mark_request_status (for terminal statuses),
  _update_order_request (on definitive broker rejection),
  _update_order_request_error (submission_error before broker), and
  _reconcile_orphaned_reservations (crash recovery).
- Reservation stays active during 'pending', 'sent', 'placed',
  'filled_unconfirmed', and 'uncertain' so concurrent submissions
  and crash-restart scenarios cannot both see the same capacity.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

from src.mt5_connector import (
    mt5_connector,
    MT5Connector,
    OrderResult,
    mt5,
)
from src.risk_manager import risk_manager, PositionSizeResult
from src.signal_validator import signal_validator, ValidationResult
from src.trade_manager import trade_manager
from src.config_loader import config
from src.database_manager import (
    db_manager,
    OrderRequest,
    ExecutedOrder,
    OpenPosition,
    ClosedTrade,
    TradingSignal,
    AppSettings,
)
from src.logger import get_logger


logger = get_logger(__name__)


class DuplicateOrderRequest(Exception):
    """Raised when the DB partial-unique index rejects an insert.

    Signals that another active OrderRequest already exists for the
    same (fingerprint, environment). The caller must not submit.
    """


# M-14: every OrderRequest status whose broker execution outcome is
# not yet definitively established. Any row in these states MUST block
# new LIVE openings until reconciliation confirms the outcome. Excludes
# "placed" (a known live pending order is not an unresolved submission)
# and all terminal statuses (rejected / rejected_notfound / cancelled /
# expired / filled / filled_closed).
_UNRESOLVED_STATUSES = (
    "uncertain",
    "submission_error",
    "sent",
    "pending",
    "filled_unconfirmed",
    "unresolved_closed",
)

# Every reconciler (_reconcile_uncertain_and_stuck_requests,
# _reconcile_market_entries, _reconcile_unresolved_closed) applies the
# same 7-day age limit. Once a row ages out of reconciliation scope,
# it must also stop blocking new openings — the operator has had a
# week to intervene.
_RECONCILE_AGE_LIMIT = timedelta(days=7)

# N-H01: statuses whose presence means a risk reservation must remain
# active. Terminal statuses (below) trigger release. "submission_error"
# is deliberately absent from both sets — it is set only by
# _update_order_request_error (pre-submission failure) which releases
# directly rather than routing through _mark_request_status.
_RESERVATION_RELEASABLE_STATUSES = frozenset({
    "filled",
    "filled_closed",
    "unresolved_closed",
    "rejected",
    "rejected_notfound",
    "cancelled",
    "expired",
    "superseded",
})


def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize a datetime to aware UTC; SQLite strips tzinfo on read."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class ExecutionResult:
    success: bool
    signal_id: Optional[int] = None
    order_result: Optional[OrderResult] = None
    position_size: Optional[PositionSizeResult] = None
    validation: Optional[ValidationResult] = None
    error: Optional[str] = None
    pending: bool = False
    pending_order_ticket: Optional[int] = None
    # True when the broker's response could not be definitively
    # interpreted. Callers MUST NOT treat this as a safe retry signal.
    uncertain: bool = False


@dataclass
class CloseAllResult:
    """Result of ``close_all_positions``.

    Iterable like a list for backward compatibility, but carries a
    ``query_ok`` flag so callers can distinguish "no positions to close"
    from "position query failed". H-08 relies on this distinction.
    """

    results: List[OrderResult] = field(default_factory=list)
    query_ok: bool = True

    @property
    def closed(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.success)

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)


@dataclass
class CancelAllResult:
    """Result of ``cancel_all_pending``.

    Iterable like a list for backward compatibility. Carries ``query_ok``
    so the caller can distinguish "nothing to cancel" from "broker query
    failed" (Phase 3D, §6.6). Field semantics mirror ``CloseAllResult``.
    """

    results: List[OrderResult] = field(default_factory=list)
    query_ok: bool = True

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.success)

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)


class OrderExecutor:
    """Places limit/market orders and reconciles their fill state."""

    def __init__(self):
        self.trading_config = config.trading
        self.magic_number = self.trading_config.magic_number
        self.ENVIRONMENT = config.trading.mode.lower()

        # Phase 3D (§6.6): escalation threshold for stuck cancels.
        self._STUCK_CANCEL_THRESHOLD = 3

        # Per-process consecutive failure counter, reset on success.
        # Not persisted — a fresh process gives each ticket a fresh try.
        self._cancel_failures: Dict[int, int] = {}

    # ==================================================================
    # 1. Public entry point
    # ==================================================================

    def execute_signal(
        self,
        signal,
        market_entry: bool = False,
        gate=None,
    ) -> ExecutionResult:
        """Validate, size, submit an order, and bookkeep the result."""

        # --- 1a. Central gate. Must run first. Fail closed. ---
        if gate is None:
            logger.error(
                "execute_signal called without an explicit EntryGate; "
                "refusing to authorize entry."
            )
            return ExecutionResult(
                success=False,
                error="Entry gate not supplied; refusing to submit order.",
            )

        try:
            gate_result = gate.check()
        except Exception as exc:
            logger.exception("Entry gate raised inside execute_signal.")
            return ExecutionResult(
                success=False,
                error=f"Entry gate error: {exc}",
            )

        if not gate_result.ok:
            logger.warning(
                "ORDER BLOCKED BY ENTRY GATE | %s | %s %s",
                gate_result.reason,
                getattr(signal, "symbol", "?"),
                getattr(getattr(signal, "direction", None), "value", "?"),
            )
            return ExecutionResult(
                success=False,
                error=f"Entry gate: {gate_result.reason}",
            )

        # --- 1b. Environment preflight (cheap redundancy) ---
        if not mt5_connector.is_connected():
            return ExecutionResult(
                success=False, error="MT5 is not connected."
            )

        if (
            self.ENVIRONMENT == "live"
            and not config.live_trading_enabled
        ):
            return ExecutionResult(
                success=False,
                error="LIVE trading is disabled (LIVE_TRADING_ENABLED=false).",
            )

        # --- 1c. Signal-quality validation ---
        try:
            validation = signal_validator.validate_signal(
                signal, market_entry=market_entry
            )
        except Exception as exc:
            logger.exception("Signal validation raised.")
            return ExecutionResult(
                success=False,
                error=f"Signal validation error: {exc}",
            )

        if not validation.valid:
            logger.warning("Signal rejected: %s", validation.reason)
            return ExecutionResult(
                success=False,
                validation=validation,
                error=validation.reason,
            )

        # --- 1d. Position sizing (H-02: use expected executable price) ---
        # For market entries the broker fills at the CURRENT ask/bid, not
        # at the planned entry price. Size against the price we actually
        # expect to pay. Never widen risk to compensate for slippage;
        # if the expected price is unavailable, refuse.
        if market_entry:
            # M-12: sizing MUST use a freshness-checked quote, not
            # the raw get_tick(). A stale quote silently widens the
            # real risk far beyond the configured budget.
            tick = mt5_connector.get_fresh_tick()
            if not tick:
                return ExecutionResult(
                    success=False,
                    validation=validation,
                    error=(
                        "Could not retrieve FRESH executable quote "
                        "for size calculation."
                    ),
                )
            try:
                bid = float(tick.get("bid") or 0)
                ask = float(tick.get("ask") or 0)
            except (TypeError, ValueError):
                return ExecutionResult(
                    success=False,
                    validation=validation,
                    error="Broker quote is invalid for size calculation.",
                )
            if bid <= 0 or ask <= 0:
                return ExecutionResult(
                    success=False,
                    validation=validation,
                    error="Broker quote is invalid for size calculation.",
                )
            if signal.direction.value.upper() == "BUY":
                expected_entry = ask
            else:
                expected_entry = bid
        else:
            expected_entry = signal.entry_price

        try:
            position_size = risk_manager.calculate_position_size(
                expected_entry,
                signal.sl_price,
                signal.direction.value,
                signal.risk_percent,
            )
        except Exception as exc:
            logger.exception("Position sizing failed.")
            return ExecutionResult(
                success=False,
                validation=validation,
                error=f"Position sizing error: {exc}",
            )

        if position_size is None or position_size.lot_size <= 0:
            reason = (
                position_size.reason
                if position_size is not None
                else "Risk manager returned no position-size result."
            )
            logger.warning("Position sizing rejected: %s", reason)
            return ExecutionResult(
                success=False,
                position_size=position_size,
                validation=validation,
                error=reason,
            )

        signal.lot_size = position_size.lot_size

        # --- 1e. Fast pre-check (fail-closed) + generate fingerprint ---
        # The partial UNIQUE index on (fingerprint, environment) at
        # INSERT time (see 1g) is the correctness gate. This pre-check
        # is a cheap early exit that ALSO fails closed on DB error, so
        # a database outage produces a clear "duplicate / fail-closed"
        # result rather than a generic signal-persist failure.
        fingerprint = self._generate_fingerprint(signal)
        signal.fingerprint = fingerprint

        if self._is_duplicate(fingerprint):
            logger.warning(
                "Duplicate signal rejected: fingerprint %s "
                "(or DB unavailable; failing closed)", fingerprint,
            )
            return ExecutionResult(
                success=False,
                validation=validation,
                error=(
                    "Duplicate signal already active or recently "
                    "executed."
                ),
            )

        # --- 1e.5 N-H01: atomically reserve candidate risk ---
        # The sizing check in risk_manager.calculate_position_size
        # compares existing + pending + candidate to the cap, but it
        # does not see concurrent in-flight submissions. The atomic
        # reservation below serialises the decision: a second caller
        # starting from the same capacity will see the first caller's
        # reservation and be rejected if the cap would be exceeded.
        try:
            reserve_ok, reserve_reason = risk_manager.try_reserve(
                fingerprint=fingerprint,
                environment=self.ENVIRONMENT,
                risk_amount=float(position_size.risk_amount),
                equity=float(position_size.equity),
            )
        except Exception as exc:
            logger.exception("N-H01: try_reserve raised.")
            reserve_ok, reserve_reason = False, f"Reservation error: {exc}"

        if not reserve_ok:
            logger.warning(
                "N-H01: candidate risk reservation rejected: %s",
                reserve_reason,
            )
            return ExecutionResult(
                success=False,
                position_size=position_size,
                validation=validation,
                error=(
                    f"Risk reservation rejected: {reserve_reason}"
                ),
            )

        # --- 1f. Persist signal ---
        try:
            signal_id = self._save_signal(
                signal, position_size, validation
            )
        except Exception as exc:
            logger.exception("Could not persist signal; order not submitted.")
            self._release_reservation(
                fingerprint, "signal persist failed"
            )
            return ExecutionResult(
                success=False,
                position_size=position_size,
                validation=validation,
                error=f"Could not save signal. Order was not submitted: {exc}",
            )

        # --- 1g. Atomic INSERT for the order request ---
        # The partial UNIQUE index on (fingerprint, environment) WHERE
        # status IN ('placed','pending','sent','filled_unconfirmed',
        # 'uncertain','submission_error') is the single source of
        # truth. Two concurrent processes submitting the same signal
        # will both reach this point, but only one INSERT succeeds.
        try:
            order_request = self._prepare_order_request(
                signal, position_size, market_entry=market_entry
            )
            request_id = self._save_order_request(
                signal_id, order_request
            )
        except DuplicateOrderRequest as dup:
            logger.warning(
                "Duplicate signal blocked by DB: %s", dup
            )
            self._release_reservation(
                fingerprint, "duplicate order request"
            )
            return ExecutionResult(
                success=False,
                signal_id=signal_id,
                position_size=position_size,
                validation=validation,
                error=(
                    "Duplicate signal already active for this "
                    "fingerprint; submission refused."
                ),
            )
        except Exception as exc:
            logger.exception("Could not prepare or persist order request.")
            self._release_reservation(
                fingerprint, "order request persist failed"
            )
            return ExecutionResult(
                success=False,
                signal_id=signal_id,
                position_size=position_size,
                validation=validation,
                error=f"Could not prepare order request: {exc}",
            )

        # --- 1f.5 Central gate re-check immediately before submission ---
        # Defence in depth: pause / kill-switch / shutdown may have flipped
        # between the initial gate check (1a) and now.
        try:
            final_gate = gate.check()
        except Exception as exc:
            logger.exception("Entry gate re-check raised before submission.")
            self._update_order_request_error(
                request_id,
                f"Entry gate re-check error before submission: {exc}",
            )
            return ExecutionResult(
                success=False,
                signal_id=signal_id,
                position_size=position_size,
                validation=validation,
                error=f"Entry gate re-check error: {exc}",
            )

        if not final_gate.ok:
            logger.warning(
                "ORDER BLOCKED BY ENTRY GATE RE-CHECK | %s | %s %s",
                final_gate.reason,
                getattr(signal, "symbol", "?"),
                getattr(getattr(signal, "direction", None), "value", "?"),
            )
            self._mark_request_status(request_id, "rejected")
            return ExecutionResult(
                success=False,
                signal_id=signal_id,
                position_size=position_size,
                validation=validation,
                error=f"Entry gate re-check: {final_gate.reason}",
            )

        # LIVE authorization enforced independently at the submission
        # boundary; never rely solely on the startup check.
        if self.ENVIRONMENT == "live" and not config.live_trading_enabled:
            logger.error(
                "LIVE submission blocked at boundary: "
                "live_trading_enabled is False."
            )
            self._mark_request_status(request_id, "rejected")
            return ExecutionResult(
                success=False,
                signal_id=signal_id,
                position_size=position_size,
                validation=validation,
                error="LIVE submission blocked at boundary.",
            )

        # --- 1f.6 M-12: revalidate risk immediately before submission ---
        # The sizing quote (§1d) and the actual submission price (§1g)
        # are read at different instants. For market entries, an
        # adverse price move widens the SL-distance-from-entry and
        # therefore the real risk above what risk_manager budgeted.
        # Fail closed if the fresh quote has diverged by more than
        # max_slippage_pips. Pending orders fix their entry price at
        # signal time, so this guard applies only to market entries.
        if market_entry:
            adverse_reason = self._revalidate_pre_submission_quote(
                expected_entry, signal.direction.value
            )
            if adverse_reason is not None:
                logger.warning(
                    "ORDER BLOCKED BY PRE-SUBMISSION QUOTE GUARD | %s | "
                    "%s %s",
                    adverse_reason,
                    getattr(signal, "symbol", "?"),
                    getattr(
                        getattr(signal, "direction", None),
                        "value",
                        "?",
                    ),
                )
                self._mark_request_status(request_id, "rejected")
                return ExecutionResult(
                    success=False,
                    signal_id=signal_id,
                    position_size=position_size,
                    validation=validation,
                    error=adverse_reason,
                )

        # --- 1g. Submit ---
        # C-03: pass the gate token through. The connector enforces the
        # token at its submission boundary on opening actions. If the
        # gate supplied no token, we pass None — the connector will fail
        # closed. We do not pre-check here because that would turn a
        # missing-token condition into an "uncertain" status, which blocks
        # the fingerprint check and requires reconciliation.
        gate_token = getattr(final_gate, "token", None)
        try:
            # M-14: pass the current request_id to the connector so
            # its boundary reconciliation check can exclude this
            # request (it is legitimately in 'pending' at this
            # instant). All other unresolved requests still block.
            order_result = mt5_connector.send_order(
                order_request,
                gate_token=gate_token,
                request_id=request_id,
            )
        except Exception as exc:
            # The broker may have accepted the order before the exception.
            # We MUST NOT treat this as a definitive rejection.
            logger.exception("MT5 order submission raised.")
            self._mark_request_uncertain(
                request_id,
                f"Submission exception (outcome unknown): {exc}",
            )
            return ExecutionResult(
                success=False,
                signal_id=signal_id,
                position_size=position_size,
                validation=validation,
                error=f"MT5 submission outcome uncertain: {exc}",
                uncertain=True,
            )

        # --- 1h. Record broker response ---
        #
        # C-02: if the outcome is uncertain we must NOT first call
        # _update_order_request() — that function writes "rejected"
        # when success=False, and the subsequent uncertain-write may
        # fail, leaving a request that looks retry-eligible. Instead,
        # write "uncertain" directly as the one and only state.
        if getattr(order_result, "uncertain", False):
            try:
                self._mark_request_uncertain(
                    request_id,
                    order_result.error_message
                    or "Broker response ambiguous",
                )
            except Exception:
                logger.exception(
                    "CRITICAL: could not persist uncertain state for "
                    "request %s. This fingerprint must be treated as "
                    "permanently blocked until reconciled.",
                    request_id,
                )
                # Fail closed: the caller MUST NOT treat this as
                # retryable. The active-fingerprint row still blocks
                # any subsequent insert attempt.
                return ExecutionResult(
                    success=False,
                    signal_id=signal_id,
                    order_result=order_result,
                    position_size=position_size,
                    validation=validation,
                    error=(
                        "Broker outcome uncertain AND could not persist "
                        "uncertainty. Manual intervention required."
                    ),
                    uncertain=True,
                )
            return ExecutionResult(
                success=False,
                signal_id=signal_id,
                order_result=order_result,
                position_size=position_size,
                validation=validation,
                error=(
                    order_result.error_message
                    or "Broker response ambiguous"
                ),
                uncertain=True,
            )

        # Definitive outcome (success, or a clear non-uncertain reject).
        try:
            self._update_order_request(
                request_id, order_result, market_entry=market_entry
            )
        except Exception:
            logger.exception(
                "Could not update order-request record after MT5 response."
            )

        if not order_result.success:
            error = (
                order_result.error_message
                or order_result.comment
                or "MT5 rejected the order."
            )
            logger.error("Order rejected: %s", error)
            return ExecutionResult(
                success=False,
                signal_id=signal_id,
                order_result=order_result,
                position_size=position_size,
                validation=validation,
                error=error,
            )

        pending_ticket = (
            getattr(order_result, "order_id", None)
            or getattr(order_result, "ticket", None)
        )
        is_pending = not market_entry

        logger.info(
            "%s | ticket=%s | %s %s @ %.5f | vol=%.2f | SL=%.5f | TP=%.5f",
            "PENDING ORDER PLACED" if is_pending else "MARKET ORDER SENT",
            pending_ticket,
            signal.direction.value,
            signal.symbol,
            signal.entry_price,
            position_size.lot_size,
            signal.sl_price,
            signal.tp_price,
        )

        # --- 1i. Market fills: bookkeep immediately; otherwise reconciler ---
        if not is_pending:
            try:
                self._finalise_market_entry(request_id, order_result)
            except Exception:
                logger.exception(
                    "Post-fill bookkeeping for request %s failed; "
                    "reconciler will retry next cycle.",
                    request_id,
                )

        return ExecutionResult(
            success=True,
            signal_id=signal_id,
            order_result=order_result,
            position_size=position_size,
            validation=validation,
            pending=is_pending,
            pending_order_ticket=(
                int(pending_ticket)
                if (is_pending and pending_ticket)
                else None
            ),
        )

    def _generate_fingerprint(self, signal) -> str:
        parts = [
            signal.symbol,
            signal.direction.value,
            signal.strategy.value,
            f"{signal.entry_price:.5f}",
            f"{signal.sl_price:.5f}",
            f"{signal.tp_price:.5f}",
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def _revalidate_pre_submission_quote(
        self,
        expected_entry: float,
        direction: str,
    ) -> Optional[str]:
        """M-12: revalidate the fresh quote immediately before submit.

        Sizing uses one quote; the connector submits with a later one.
        If the price moved adversely (BUY: ask rose; SELL: bid fell)
        by more than `trade_management.max_slippage_pips`, the
        realised loss to the fixed SL exceeds the budgeted risk by an
        unbounded margin. Fail closed in that case.

        Returns None when the submission may proceed, or a
        rejection reason string.
        """
        try:
            expected_entry = float(expected_entry)
        except (TypeError, ValueError):
            return "Expected entry price is invalid for pre-submit check."

        if not (expected_entry > 0):
            return "Expected entry price is invalid for pre-submit check."

        tick = mt5_connector.get_fresh_tick()
        if tick is None:
            return (
                "Fresh quote unavailable immediately before submission."
            )

        try:
            cur_bid = float(tick.get("bid") or 0)
            cur_ask = float(tick.get("ask") or 0)
        except (TypeError, ValueError):
            return "Fresh quote is invalid immediately before submission."

        if cur_bid <= 0 or cur_ask <= 0:
            return "Fresh quote is invalid immediately before submission."

        sym_info = mt5_connector.get_symbol_info()
        pip_size = (
            MT5Connector.pip_size_for(sym_info)
            if sym_info is not None
            else 0.0
        )
        if not isinstance(pip_size, (int, float)) or pip_size <= 0:
            return "Cannot determine pip size for slippage guard."

        try:
            max_slip_pips = float(
                config.trade_management.max_slippage_pips
            )
        except (TypeError, ValueError, AttributeError):
            max_slip_pips = 0.0
        if not (max_slip_pips > 0):
            return "Invalid max_slippage_pips configuration."

        max_adverse = max_slip_pips * pip_size

        d = str(direction).upper()
        if d == "BUY":
            adverse_move = cur_ask - expected_entry
        elif d == "SELL":
            adverse_move = expected_entry - cur_bid
        else:
            return (
                f"Unsupported direction for slippage guard: "
                f"{direction!r}"
            )

        if adverse_move > max_adverse:
            return (
                f"Quote moved adversely by {adverse_move:.5f} "
                f"(> {max_adverse:.5f}) between sizing and "
                f"submission."
            )

        return None

    def _is_duplicate(self, fingerprint: str) -> bool:
        if not fingerprint:
            return False

        try:
            with db_manager.get_session() as session:
                active_request = (
                    session.query(OrderRequest)
                    .filter(
                        OrderRequest.environment == self.ENVIRONMENT,
                        OrderRequest.fingerprint == fingerprint,
                        OrderRequest.status.in_(
                            [
                                "placed",
                                "pending",
                                "sent",
                                "filled_unconfirmed",
                                "uncertain",
                                "submission_error",
                            ]
                        ),
                    )
                    .first()
                )
                if active_request:
                    logger.info(
                        "Duplicate check: active request found for "
                        "fingerprint %s",
                        fingerprint,
                    )
                    return True

                cutoff = datetime.utcnow() - timedelta(hours=24)
                recent_filled = (
                    session.query(OrderRequest)
                    .filter(
                        OrderRequest.environment == self.ENVIRONMENT,
                        OrderRequest.fingerprint == fingerprint,
                        OrderRequest.status.in_(
                            ["filled", "filled_closed", "unresolved_closed"]
                        ),
                        OrderRequest.timestamp >= cutoff,
                    )
                    .first()
                )
                if recent_filled:
                    logger.info(
                        "Duplicate check: recent filled/closed order found "
                        "for fingerprint %s", fingerprint,
                    )
                    return True
        except Exception:
            logger.exception("Duplicate check failed; failing closed.")
            return True

        return False

    # ==================================================================
    # 2. Market-entry finalisation (F-02 core fix)
    # ==================================================================

    def _finalise_market_entry(
        self,
        request_id: int,
        order_result: OrderResult,
    ) -> None:
        """Bookkeep a successful market order idempotently.

        If the position cannot be located yet (rare race), leave the
        request in `filled_unconfirmed` and let the reconciler retry.
        """
        try:
            with db_manager.get_session() as session:
                req = (
                    session.query(OrderRequest)
                    .filter(OrderRequest.id == request_id)
                    .first()
                )
        except Exception:
            logger.exception(
                "Could not load OrderRequest %s during market-fill bookkeeping.",
                request_id,
            )
            return

        if req is None:
            logger.error(
                "OrderRequest %s missing during market-fill bookkeeping.",
                request_id,
            )
            return

        position = self._find_position_for_market_order(order_result)

        if position is None:
            logger.warning(
                "Market order for request %s succeeded but no matching "
                "position was found yet; will retry next cycle.",
                request_id,
            )
            self._mark_request_status(request_id, "filled_unconfirmed")
            return

        # Persist the position ticket on the request for future lookups.
        try:
            with db_manager.get_session() as session:
                row = (
                    session.query(OrderRequest)
                    .filter(OrderRequest.id == request_id)
                    .first()
                )
                if row is not None and row.mt5_ticket is None:
                    row.mt5_ticket = int(position["ticket"])
        except Exception:
            logger.exception(
                "Could not persist mt5_ticket for request %s.", request_id
            )

        self._handle_filled_pending(req, position)
        self._mark_request_status(request_id, "filled")

    def _find_position_for_market_order(
        self, order_result: OrderResult
    ) -> Optional[Dict[str, Any]]:
        order_id = (
            getattr(order_result, "order_id", None)
            or getattr(order_result, "ticket", None)
        )
        if not order_id:
            return None
        return self._find_position_by_order_id(int(order_id))

    def _find_position_by_order_id(
        self, order_id: int
    ) -> Optional[Dict[str, Any]]:
        """Locate a position by the ticket of the order that opened it."""
        positions = mt5_connector.get_positions()
        if positions is None:
            return None

        # Direct match (hedging mode on most retail brokers).
        for pos in positions:
            if int(pos.get("ticket", 0)) == int(order_id):
                return pos

        # Deal-history match (netting mode, or ticket mismatch).
        try:
            deals = mt5_connector.get_history_deals(
                from_date=datetime.now(timezone.utc) - timedelta(days=2),
                to_date=datetime.now(timezone.utc) + timedelta(minutes=1),
            )
        except Exception:
            logger.exception("Deal-history lookup failed.")
            return None

        if deals is None:
            return None

        position_id = None
        for deal in deals:
            if int(deal.get("order") or 0) != int(order_id):
                continue
            pid = deal.get("position_id")
            if pid:
                position_id = int(pid)
                break

        if position_id is None:
            return None

        for pos in positions:
            if int(pos.get("ticket", 0)) == position_id:
                return pos

        return None

    # ==================================================================
    # 3. Pending-order reconciliation
    # ==================================================================

    def has_unresolved_reconciliation(
        self,
        exclude_request_id: Optional[int] = None,
    ) -> bool:
        """True if any prior request has an unresolved execution outcome.

        M-14: expanded from the previous single-status check
        ('uncertain') to cover every status whose broker outcome is
        not yet definitively established:
        uncertain, submission_error, sent, pending,
        filled_unconfirmed, unresolved_closed.

        ``exclude_request_id`` is the request currently being
        submitted. At the moment the LIVE boundary calls this, that
        request is itself at 'pending' — excluding it prevents self-
        block. Prior unresolved requests are still detected.

        Uses the same 7-day age limit as the reconcilers. A row that
        has aged out of reconciliation scope no longer blocks new
        openings (the operator has had a week to intervene, and the
        reconcilers have stopped processing it).

        Fails closed: returns True on DB error so a database outage
        blocks new LIVE openings instead of allowing them through.
        """
        age_limit = datetime.utcnow() - _RECONCILE_AGE_LIMIT
        try:
            with db_manager.get_session() as session:
                query = session.query(OrderRequest).filter(
                    OrderRequest.environment == self.ENVIRONMENT,
                    OrderRequest.status.in_(_UNRESOLVED_STATUSES),
                    OrderRequest.timestamp >= age_limit,
                )
                if exclude_request_id is not None:
                    try:
                        excluded = int(exclude_request_id)
                    except (TypeError, ValueError):
                        excluded = None
                    if excluded is not None:
                        query = query.filter(
                            OrderRequest.id != excluded
                        )
                return query.first() is not None
        except Exception:
            logger.exception(
                "Reconciliation state read failed; failing closed."
            )
            return True

    def reconcile_pending_orders(self) -> None:
        """Align MT5 pending-order state with the database, plus recover
        any market fills whose bookkeeping did not complete."""
        # N-H01: release stale reservations before touching broker state.
        try:
            self._reconcile_orphaned_reservations()
        except Exception:
            logger.exception(
                "N-H01: orphan-reservation reconciliation raised."
            )

        try:
            live_orders = mt5_connector.get_orders()
        except Exception:
            logger.exception("Could not fetch pending orders.")
            live_orders = None

        if live_orders is None:
            logger.error(
                "MT5 pending-order query returned None; "
                "skipping pending reconciliation.",
            )
        else:
            live_tickets = {int(o["ticket"]) for o in live_orders}
            # Phase 3D: clear stuck-cancel markers for tickets no longer
            # present in the broker (filled / expired / cancelled by
            # another path). Prevents the marker from ageing out into a
            # permanent false positive.
            try:
                stuck_map = self._load_stuck_cancels()
                stale = [t for t in stuck_map if t not in live_tickets]
                if stale:
                    for t in stale:
                        stuck_map.pop(t, None)
                    self._save_stuck_cancels(stuck_map)
            except Exception:
                logger.exception(
                    "Could not prune stale stuck-cancel markers."
                )
            pending_requests = self._pending_db_requests()
            if pending_requests is not None:
                for req in pending_requests:
                    ticket = self._request_ticket(req)
                    if ticket is None:
                        continue
                    if ticket in live_tickets:
                        continue
                    self._finalise_pending_request(req, ticket)

            self._cancel_expired_orders(live_orders)
            self._enforce_pending_order_cap(live_orders)

        self._reconcile_market_entries()
        self._reconcile_uncertain_and_stuck_requests()
        self._reconcile_unresolved_closed()

    # ---------- N-H01: orphan-reservation recovery ----------

    def _reconcile_orphaned_reservations(self) -> None:
        """N-H01: release reservations with no holding OrderRequest.

        A reservation becomes an orphan when the process crashes
        between reserving and persisting the OrderRequest row, or
        when the OrderRequest moved to a terminal status via a code
        path that bypassed _mark_request_status / _update_order_request.
        """
        from src.database_manager import RiskReservation as _RR
        holding_statuses = (
            "pending", "sent", "placed",
            "filled_unconfirmed", "uncertain",
        )
        try:
            with db_manager.get_session() as session:
                active = (
                    session.query(_RR)
                    .filter(
                        _RR.environment == self.ENVIRONMENT,
                        _RR.status == "active",
                    )
                    .all()
                )
                if not active:
                    return
                active_fps = [r.fingerprint for r in active]
                holding_rows = (
                    session.query(OrderRequest.fingerprint)
                    .filter(
                        OrderRequest.environment == self.ENVIRONMENT,
                        OrderRequest.fingerprint.in_(active_fps),
                        OrderRequest.status.in_(holding_statuses),
                    )
                    .all()
                )
                holding_set = {
                    row[0] for row in holding_rows if row[0]
                }
        except Exception:
            logger.exception(
                "N-H01: orphan-reservation scan failed."
            )
            return

        for r in active:
            if r.fingerprint not in holding_set:
                try:
                    self._release_reservation(
                        r.fingerprint,
                        "orphan (no holding order request)",
                    )
                except Exception:
                    logger.exception(
                        "N-H01: release of orphan %s raised.",
                        r.fingerprint,
                    )

    # ---------- market-entry recovery ----------

    def _reconcile_market_entries(self) -> None:
        cutoff = datetime.utcnow() - timedelta(days=7)
        try:
            with db_manager.get_session() as session:
                candidates = (
                    session.query(OrderRequest)
                    .filter(
                        OrderRequest.environment == self.ENVIRONMENT,
                        OrderRequest.status.in_(
                            ["filled", "filled_unconfirmed"]
                        ),
                        OrderRequest.timestamp >= cutoff,
                    )
                    .all()
                )
        except Exception:
            logger.exception(
                "Could not query market-entry recovery candidates."
            )
            return

        for req in candidates:
            try:
                if self._is_market_entry_complete(req):
                    if req.status == "filled_unconfirmed":
                        self._mark_request_status(req.id, "filled")
                    continue
                self._retry_finalize_market_entry(req)
            except Exception:
                logger.exception(
                    "Market-entry recovery failed for request %s.", req.id
                )

    def _is_market_entry_complete(self, req: OrderRequest) -> bool:
        if not req.mt5_ticket:
            return False
        executed = db_manager.query_one(
            ExecutedOrder,
            ticket=int(req.mt5_ticket),
            environment=self.ENVIRONMENT,
        )
        if executed is None:
            return False
        open_pos = db_manager.query_one(
            OpenPosition,
            ticket=int(req.mt5_ticket),
            environment=self.ENVIRONMENT,
        )
        return open_pos is not None

    def _retry_finalize_market_entry(self, req: OrderRequest) -> None:
        position = None

        if req.mt5_ticket:
            positions = mt5_connector.get_positions()
            if positions is not None:
                for pos in positions:
                    if int(pos.get("ticket", 0)) == int(req.mt5_ticket):
                        position = pos
                        break

        if position is None:
            response = req.response_data or {}
            order_id = response.get("order_id") or response.get("ticket")
            if order_id:
                position = self._find_position_by_order_id(int(order_id))

        if position is not None:
            try:
                with db_manager.get_session() as session:
                    row = (
                        session.query(OrderRequest)
                        .filter(OrderRequest.id == req.id)
                        .first()
                    )
                    if row is not None and row.mt5_ticket is None:
                        row.mt5_ticket = int(position["ticket"])
            except Exception:
                logger.exception(
                    "Could not persist mt5_ticket for request %s.", req.id
                )

            self._handle_filled_pending(req, position)
            self._mark_request_status(req.id, "filled")
            return

        # NEW: check deal history for a fill+close that never produced a
        # live position we could observe.
        ticket: Optional[int] = None
        if req.mt5_ticket:
            ticket = int(req.mt5_ticket)
        else:
            response = req.response_data or {}
            raw = response.get("order_id") or response.get("ticket")
            if raw:
                try:
                    ticket = int(raw)
                except (TypeError, ValueError):
                    ticket = None

        if ticket is None:
            return

        resolution = self._resolve_via_deal_history(req, ticket)
        if resolution == "filled_closed":
            logger.info(
                "MARKET FILLED+CLOSED | req=%s ticket=%s",
                req.id, ticket,
            )
            self._mark_request_status(req.id, "filled_closed")
        elif resolution == "unresolved_closed":
            logger.warning(
                "MARKET FILLED+CLOSED but ClosedTrade failed | req=%s "
                "ticket=%s", req.id, ticket,
            )
            self._mark_request_status(req.id, "unresolved_closed")
        elif resolution == "live_fill":
            logger.info(
                "MARKET FILLED but position not yet visible | req=%s; "
                "deferring.", req.id,
            )
        # None: no deal found; stay `filled_unconfirmed` and retry.

    # ---------- deal-history fill+close recovery ----------

    def _resolve_via_deal_history(
        self, req: OrderRequest, order_ticket: int,
    ) -> Optional[str]:
        """Classify a request that is neither live nor pending.

        Used for both pending-order finalisation and market-entry
        recovery when the position is not (or no longer) visible in
        get_positions(). Returns:

          * ``"filled_closed"``     — fill+close confirmed, ClosedTrade created
          * ``"unresolved_closed"`` — fill+close confirmed, ClosedTrade failed
          * ``"live_fill"``         — fill confirmed, no close yet (retry)
          * ``None``                — no fill deal found (not executed)
        """
        try:
            deals = mt5_connector.get_history_deals(
                from_date=datetime.now(timezone.utc) - timedelta(days=2),
                to_date=datetime.now(timezone.utc) + timedelta(minutes=1),
            )
        except Exception:
            logger.exception(
                "Deal-history lookup failed for req %s.", req.id,
            )
            return None

        if deals is None:
            logger.warning(
                "Deal history returned None for req %s; deferring.",
                req.id,
            )
            return None

        # Locate the IN deal for this order.
        position_id: Optional[int] = None
        for deal in deals:
            if int(deal.get("order") or 0) != int(order_ticket):
                continue
            entry = str(deal.get("entry", "")).upper()
            if entry != "IN":
                continue
            pid = deal.get("position_id")
            if pid:
                position_id = int(pid)
                break

        if position_id is None:
            return None

        # Is there an OUT deal for the same position?
        has_out = any(
            int(d.get("position_id") or 0) == position_id
            and str(d.get("entry", "")).upper()
                in ("OUT", "OUT_BY", "INOUT")
            for d in deals
        )

        if not has_out:
            return "live_fill"

        # Reconstruct the ClosedTrade from deal history.
        try:
            ok = trade_manager.reconstruct_closed_trade(
                position_id, exit_reason="RECOVERED_FILLED_CLOSED",
            )
        except Exception:
            logger.exception(
                "ClosedTrade reconstruction raised for position %s "
                "(req %s).", position_id, req.id,
            )
            ok = False

        return "filled_closed" if ok else "unresolved_closed"

    # ---------- pending-order reconciliation ----------

    def _pending_db_requests(self) -> Optional[List[OrderRequest]]:
        try:
            with db_manager.get_session() as session:
                return (
                    session.query(OrderRequest)
                    .filter(
                        OrderRequest.environment == self.ENVIRONMENT,
                        OrderRequest.status == "placed",
                    )
                    .all()
                )
        except Exception:
            logger.exception(
                "Could not load pending order requests from DB."
            )
            return None

    @staticmethod
    def _request_ticket(req: OrderRequest) -> Optional[int]:
        """Resolve the broker ticket for a pending request.

        Reads `response_data` first, but falls back to `mt5_ticket` — the
        canonical column populated by `_update_order_request` on success.
        Without the fallback, any request whose `response_data` was not
        persisted (crash between broker response and DB update; JSON
        truncated; legacy row from before `response_data` was written)
        is silently skipped by `_pending_db_requests`, which breaks
        pending-order reconciliation (H-07).
        """
        response = req.response_data or {}
        ticket = (
            response.get("order_id")
            or response.get("ticket")
            or response.get("pending_ticket")
            or getattr(req, "mt5_ticket", None)
        )
        try:
            ticket = int(ticket)
        except (TypeError, ValueError):
            return None
        return ticket if ticket > 0 else None

    def _finalise_pending_request(
        self, req: OrderRequest, ticket: int
    ) -> None:
        try:
            positions = mt5_connector.get_positions()
        except Exception:
            logger.exception(
                "Could not fetch positions for reconciliation."
            )
            positions = None

        if positions is None:
            logger.error(
                "Position query failed while finalising pending order %s; "
                "will retry next cycle.",
                ticket,
            )
            return

        fill_position = None
        for pos in positions:
            if int(pos.get("ticket", 0)) == ticket:
                fill_position = pos
                break

        if fill_position is None:
            fill_position = self._find_position_by_order_id(ticket)

        if fill_position is None:
            fill_position = self._find_unrecorded_position(req, positions)

        if fill_position is not None:
            self._handle_filled_pending(req, fill_position)
            self._mark_request_status(req.id, "filled")
            return

        # NEW: the order may have filled AND closed between cycles —
        # invisible to positions() but visible in deal history.
        resolution = self._resolve_via_deal_history(req, ticket)
        if resolution == "filled_closed":
            logger.info(
                "PENDING FILLED+CLOSED | req=%s ticket=%s",
                req.id, ticket,
            )
            self._mark_request_status(req.id, "filled_closed")
            return
        if resolution == "unresolved_closed":
            logger.warning(
                "PENDING FILLED+CLOSED but ClosedTrade failed | req=%s "
                "ticket=%s", req.id, ticket,
            )
            self._mark_request_status(req.id, "unresolved_closed")
            return
        if resolution == "live_fill":
            # Position should be live; will be located next cycle.
            logger.info(
                "PENDING FILLED but no live position yet | req=%s; "
                "deferring.", req.id,
            )
            return

        # No fill deal → definitively not executed.
        self._handle_cancelled_pending(req)

    def _find_unrecorded_position(
        self, req: OrderRequest, positions: List[Dict]
    ) -> Optional[Dict]:
        candidates: List[Dict] = []
        for pos in positions:
            if int(pos.get("magic", 0)) != int(self.magic_number):
                continue
            if pos.get("symbol") != req.symbol:
                continue
            existing = db_manager.query_one(
                ExecutedOrder,
                ticket=int(pos["ticket"]),
                environment=self.ENVIRONMENT,
            )
            if existing is not None:
                continue
            candidates.append(pos)

        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            logger.error(
                "Ambiguous fill: %d unrecorded positions for req %s; "
                "manual reconciliation required.",
                len(candidates),
                req.id,
            )
        return None

    def _handle_filled_pending(
        self, req: OrderRequest, position: Dict[str, Any]
    ) -> None:
        """Idempotent post-fill bookkeeping. Safe to call repeatedly."""
        ticket = int(position["ticket"])
        logger.info(
            "PENDING/MARKET FILL BOOKKEEPING | req=%s ticket=%s | %s @ %.5f",
            req.id,
            ticket,
            position.get("symbol"),
            float(position.get("price_open", 0.0)),
        )

        self._save_executed_order(position, request_id=req.id)
        self._save_open_position_from_request(position, req)

        if trade_manager.get_position_state(ticket) is None:
            try:
                trade_manager.initialize_position_tracking(
                    ticket,
                    position["type"],
                    float(position["price_open"]),
                    float(position.get("sl") or 0),
                    float(position.get("tp") or 0),
                    float(position["volume"]),
                    (req.comment or "SMC_UNKNOWN").replace("SMC_", ""),
                )
            except Exception:
                logger.exception(
                    "trade_manager.initialize_position_tracking failed for "
                    "ticket %s; reconciler will retry.",
                    ticket,
                )
                return

    def _handle_cancelled_pending(self, req: OrderRequest) -> None:
        logger.info(
            "PENDING ORDER GONE | req=%s | marked cancelled/expired", req.id
        )
        self._mark_request_status(req.id, "cancelled")

    def _release_reservation(
        self, fingerprint: Optional[str], reason: str
    ) -> None:
        """N-H01: release a candidate-risk reservation.

        Best-effort: logs and swallows on failure so a reservation
        release cannot mask the caller's primary result. Stale
        reservations are recovered on the next reconciliation cycle
        by _reconcile_orphaned_reservations.
        """
        if not fingerprint:
            return
        try:
            risk_manager.release_reservation(
                fingerprint, self.ENVIRONMENT, reason,
            )
        except Exception:
            logger.exception(
                "N-H01: release_reservation raised for %s.",
                fingerprint,
            )

    def _mark_request_status(self, request_id: int, status: str) -> None:
        fingerprint: Optional[str] = None
        try:
            with db_manager.get_session() as session:
                request = (
                    session.query(OrderRequest)
                    .filter(OrderRequest.id == request_id)
                    .first()
                )
                if request is not None:
                    request.status = status
                    fingerprint = request.fingerprint
        except Exception:
            logger.exception(
                "Could not update request %s status to %s.",
                request_id,
                status,
            )
            return

        # N-H01: release candidate risk when the underlying order has
        # definitively terminated. Released AFTER the status write
        # commits so a crash between the two leaves the reservation
        # active (fail-closed: blocks new entries rather than
        # undercounting exposure). The next
        # _reconcile_orphaned_reservations cycle will release it.
        if status in _RESERVATION_RELEASABLE_STATUSES:
            self._release_reservation(
                fingerprint, f"status={status}"
            )

    def _cancel_expired_orders(self, live_orders: List[Dict]) -> None:
        if not live_orders:
            return

        max_age = timedelta(hours=config.trading.pending_expiry_hours)
        now = datetime.now(timezone.utc)

        for order in live_orders:
            time_setup = order.get("time_setup")
            if time_setup is None:
                continue
            if time_setup.tzinfo is None:
                time_setup = time_setup.replace(tzinfo=timezone.utc)
            if now - time_setup > max_age:
                ticket = int(order["ticket"])
                logger.info(
                    "Cancelling expired pending order %s", ticket,
                )
                # Phase 3D: route through cancel_pending_order for
                # consistent DB update and failure escalation.
                self.cancel_pending_order(ticket, reason="EXPIRED")

    def _enforce_pending_order_cap(self, live_orders: List[Dict]) -> None:
        cap = int(config.trading.max_pending_orders)
        if cap <= 0 or len(live_orders) <= cap:
            return

        stuck = set(self.get_stuck_cancellations())
        candidates = [
            o for o in live_orders
            if int(o["ticket"]) not in stuck
        ]

        excess = len(live_orders) - cap
        if len(candidates) <= excess:
            # Not enough cancellable candidates — either all excess
            # tickets are stuck, or the cap is being exceeded only by
            # stuck orders. Log and let the natural expiry path handle
            # the newest.
            if stuck:
                logger.warning(
                    "Pending-order cap exceeded (%d > %d) but %d "
                    "stuck order(s) cannot be cancelled. Newest "
                    "orders will expire naturally per "
                    "pending_expiry_hours.",
                    len(live_orders), cap, len(stuck),
                )
            return

        ordered = sorted(
            candidates,
            key=lambda o: (
                o.get("time_setup")
                or datetime.min.replace(tzinfo=timezone.utc)
            ),
        )
        to_cancel = ordered[:excess]

        for order in to_cancel:
            ticket = int(order["ticket"])
            logger.info(
                "Cancelling oldest pending order %s (cap=%s, active=%s)",
                ticket, cap, len(live_orders),
            )
            self.cancel_pending_order(ticket, reason="CAP")

    # ---------- uncertain / stuck-request recovery ----------

    def _reconcile_uncertain_and_stuck_requests(self) -> None:
        """Resolve submissions whose outcome is not definitively known.

        Covers three states that the existing reconciler did not:
          * ``uncertain``         — an exception fired after send_order.
          * ``pending`` / ``sent`` — the process crashed between request
            persistence and response persistence.
        """
        grace = timedelta(minutes=5)
        age_limit = datetime.utcnow() - timedelta(days=7)
        cutoff = datetime.utcnow() - grace

        try:
            with db_manager.get_session() as session:
                candidates = (
                    session.query(OrderRequest)
                    .filter(
                        OrderRequest.environment == self.ENVIRONMENT,
                        OrderRequest.status.in_(
                            ["uncertain", "pending", "sent",
                             "submission_error"]
                        ),
                        OrderRequest.timestamp <= cutoff,
                        OrderRequest.timestamp >= age_limit,
                    )
                    .all()
                )
        except Exception:
            logger.exception(
                "Could not query uncertain/stuck order requests."
            )
            return

        if not candidates:
            return

        try:
            positions = mt5_connector.get_positions()
            orders = mt5_connector.get_orders()
            deals = mt5_connector.get_history_deals(
                from_date=datetime.now(timezone.utc) - timedelta(days=7),
                to_date=datetime.now(timezone.utc) + timedelta(minutes=1),
            )
        except Exception:
            logger.exception(
                "Could not query broker state for uncertain reconciliation."
            )
            return

        if positions is None or orders is None or deals is None:
            logger.warning(
                "Uncertain reconciliation skipped: broker-state query "
                "returned None (positions=%s orders=%s deals=%s).",
                positions is None,
                orders is None,
                deals is None,
            )
            return

        for req in candidates:
            try:
                self._resolve_uncertain_request(
                    req, positions, orders, deals
                )
            except Exception:
                logger.exception(
                    "Could not reconcile uncertain request %s.", req.id
                )

    def _reconcile_unresolved_closed(self) -> None:
        """Retry ClosedTrade reconstruction for requests marked unresolved.

        A request lands in ``unresolved_closed`` when fill+close was
        detected via deal history but the ClosedTrade row could not be
        created (e.g., DB error at that instant). Retried every cycle
        until it succeeds, or until the row ages out of the 7-day window.
        """
        age_limit = datetime.utcnow() - timedelta(days=7)
        try:
            with db_manager.get_session() as session:
                candidates = (
                    session.query(OrderRequest)
                    .filter(
                        OrderRequest.environment == self.ENVIRONMENT,
                        OrderRequest.status == "unresolved_closed",
                        OrderRequest.timestamp >= age_limit,
                    )
                    .all()
                )
        except Exception:
            logger.exception(
                "Could not query unresolved_closed requests."
            )
            return

        for req in candidates:
            try:
                ticket = (
                    int(req.mt5_ticket) if req.mt5_ticket
                    else self._request_ticket(req)
                )
                if not ticket:
                    continue
                resolution = self._resolve_via_deal_history(req, ticket)
                if resolution == "filled_closed":
                    logger.info(
                        "UNRESOLVED_CLOSED resolved | req=%s ticket=%s",
                        req.id, ticket,
                    )
                    self._mark_request_status(req.id, "filled_closed")
                # "unresolved_closed"/"live_fill"/None → leave, retry.
            except Exception:
                logger.exception(
                    "Could not reconcile unresolved_closed req %s.", req.id,
                )

    def _resolve_uncertain_request(
        self,
        req: OrderRequest,
        positions: List[Dict[str, Any]],
        orders: List[Dict[str, Any]],
        deals: List[Dict[str, Any]],
    ) -> None:
        # 1. Matching live pending order → 'placed'
        matching_order = self._find_matching_order(req, orders)
        if matching_order is not None:
            ticket = int(matching_order["ticket"])
            logger.info(
                "UNCERTAIN RESOLVED as pending order | req=%s ticket=%s",
                req.id, ticket,
            )
            try:
                with db_manager.get_session() as session:
                    row = (
                        session.query(OrderRequest)
                        .filter(OrderRequest.id == req.id)
                        .first()
                    )
                    if row is not None:
                        row.mt5_ticket = ticket
                        row.status = "placed"
            except Exception:
                logger.exception(
                    "Could not persist resolution for request %s.", req.id
                )
            return

        # 2. Matching open position → 'filled' with bookkeeping
        matching_position = self._find_matching_position(req, positions)
        if matching_position is not None:
            logger.info(
                "UNCERTAIN RESOLVED as open position | req=%s ticket=%s",
                req.id, int(matching_position["ticket"]),
            )
            self._handle_filled_pending(req, matching_position)
            self._mark_request_status(req.id, "filled")
            return

        # 3. Matching closed deal → 'filled_closed' (or 'unresolved_closed')
        matching_deal = self._find_matching_deal(req, deals)
        if matching_deal is not None:
            position_id = matching_deal.get("position_id")
            reconstructed = False
            if position_id:
                try:
                    reconstructed = trade_manager.reconstruct_closed_trade(
                        int(position_id),
                        exit_reason="RECOVERED_UNCERTAIN_CLOSED",
                    )
                except Exception:
                    logger.exception(
                        "ClosedTrade reconstruction failed for req %s.",
                        req.id,
                    )
                    reconstructed = False
            logger.info(
                "UNCERTAIN RESOLVED as already-closed deal | req=%s | "
                "position_id=%s reconstructed=%s",
                req.id, position_id, reconstructed,
            )
            self._mark_request_status(
                req.id,
                "filled_closed" if reconstructed else "unresolved_closed",
            )
            return

        # 4. Nothing found — only finalise once the request is old enough.
        age = datetime.utcnow() - _to_utc(req.timestamp).replace(tzinfo=None)
        if age < timedelta(minutes=10):
            logger.info(
                "Uncertain request %s not yet resolvable (age=%s); "
                "will retry next cycle.",
                req.id, age,
            )
            return

        logger.warning(
            "UNCERTAIN RESOLVED as not-executed | req=%s age=%s",
            req.id, age,
        )
        self._mark_request_status(req.id, "rejected_notfound")

    # ---------- matching helpers ----------

    @staticmethod
    def _order_type_to_side(value) -> Optional[str]:
        """Normalize an order type (int, string name, or stringified int)
        to ``"BUY"`` or ``"SELL"``. Returns None when unclassifiable.

        Handles three real representations in this codebase:
          * MT5 integer constants  (mt5.ORDER_TYPE_BUY_LIMIT == 2)
          * MT5 string names       (``get_orders()`` -> "BUY_LIMIT")
          * SQLite-coerced strings (``OrderRequest.order_type`` -> "2")
        """
        if value is None:
            return None
        if isinstance(value, str):
            upper = value.strip().upper()
            if upper.startswith("BUY"):
                return "BUY"
            if upper.startswith("SELL"):
                return "SELL"
            try:
                value = int(upper)
            except ValueError:
                return None
        buy_codes = {
            getattr(mt5, "ORDER_TYPE_BUY", None),
            getattr(mt5, "ORDER_TYPE_BUY_LIMIT", None),
            getattr(mt5, "ORDER_TYPE_BUY_STOP", None),
            getattr(mt5, "ORDER_TYPE_BUY_STOP_LIMIT", None),
        }
        sell_codes = {
            getattr(mt5, "ORDER_TYPE_SELL", None),
            getattr(mt5, "ORDER_TYPE_SELL_LIMIT", None),
            getattr(mt5, "ORDER_TYPE_SELL_STOP", None),
            getattr(mt5, "ORDER_TYPE_SELL_STOP_LIMIT", None),
        }
        buy_codes.discard(None)
        sell_codes.discard(None)
        if value in buy_codes:
            return "BUY"
        if value in sell_codes:
            return "SELL"
        return None

    @staticmethod
    def _position_type_to_side(value) -> Optional[str]:
        """Normalize a position type to ``"BUY"`` or ``"SELL"``."""
        if value is None:
            return None
        if isinstance(value, str):
            upper = value.strip().upper()
            if upper == "BUY":
                return "BUY"
            if upper == "SELL":
                return "SELL"
            try:
                value = int(upper)
            except ValueError:
                return None
        if value == getattr(mt5, "POSITION_TYPE_BUY", None):
            return "BUY"
        if value == getattr(mt5, "POSITION_TYPE_SELL", None):
            return "SELL"
        return None

    @staticmethod
    def _order_direction_matches(broker_type, stored_type) -> bool:
        """Both arguments may be ints, MT5 name strings, or stringified ints."""
        b = OrderExecutor._order_type_to_side(broker_type)
        s = OrderExecutor._order_type_to_side(stored_type)
        return b is not None and b == s

    @staticmethod
    def _position_direction_matches(pos_type, stored_order_type) -> bool:
        p = OrderExecutor._position_type_to_side(pos_type)
        s = OrderExecutor._order_type_to_side(stored_order_type)
        return p is not None and p == s

    def _find_matching_order(
        self, req: OrderRequest, orders: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        tol_vol = 0.001
        window = timedelta(minutes=5)
        req_ts = _to_utc(req.timestamp)
        matches: List[Dict[str, Any]] = []

        for order in orders:
            if int(order.get("magic", 0)) != int(req.magic_number):
                continue
            if order.get("symbol") != req.symbol:
                continue
            if abs(
                float(order.get("volume", 0.0)) - float(req.volume)
            ) > tol_vol:
                continue
            if not self._order_direction_matches(
                order.get("type"), req.order_type
            ):
                continue
            time_setup = order.get("time_setup")
            if isinstance(time_setup, datetime):
                delta = abs(
                    (_to_utc(time_setup) - req_ts).total_seconds()
                )
                if delta > window.total_seconds():
                    continue
            matches.append(order)

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.error(
                "Ambiguous uncertain resolution: %d pending orders match "
                "request %s; manual reconciliation required.",
                len(matches), req.id,
            )
        return None

    def _find_matching_position(
        self, req: OrderRequest, positions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        tol_vol = 0.001
        window = timedelta(minutes=10)
        req_ts = _to_utc(req.timestamp)
        matches: List[Dict[str, Any]] = []

        for pos in positions:
            if int(pos.get("magic", 0)) != int(req.magic_number):
                continue
            if pos.get("symbol") != req.symbol:
                continue
            if abs(
                float(pos.get("volume", 0.0)) - float(req.volume)
            ) > tol_vol:
                continue
            if not self._position_direction_matches(
                pos.get("type"), req.order_type
            ):
                continue
            time_open = pos.get("time_open")
            if isinstance(time_open, datetime):
                delta = abs(
                    (_to_utc(time_open) - req_ts).total_seconds()
                )
                if delta > window.total_seconds():
                    continue
            matches.append(pos)

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.error(
                "Ambiguous uncertain resolution: %d positions match "
                "request %s.", len(matches), req.id,
            )
        return None

    def _find_matching_deal(
        self, req: OrderRequest, deals: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Match an IN deal to a request by magic, symbol, volume, side, time.

        get_history_deals returns ``entry`` as a string ("IN"/"OUT"/…), so
        the check is string-based, NOT against mt5.DEAL_ENTRY_IN.
        """
        tol_vol = 0.001
        window = timedelta(minutes=10)
        req_ts = _to_utc(req.timestamp)

        for deal in deals:
            if int(deal.get("magic", 0)) != int(req.magic_number):
                continue
            if deal.get("symbol") != req.symbol:
                continue
            entry = str(deal.get("entry", "")).upper()
            if entry != "IN":
                continue
            if abs(
                float(deal.get("volume", 0.0)) - float(req.volume)
            ) > tol_vol:
                continue
            # Deal type is "BUY"/"SELL" (from _deal_type_to_str); this is
            # the deal direction, which for an IN deal matches the
            # position side.
            if not self._order_direction_matches(
                deal.get("type"), req.order_type
            ):
                continue
            deal_time = deal.get("time")
            if isinstance(deal_time, datetime):
                delta = abs(
                    (_to_utc(deal_time) - req_ts).total_seconds()
                )
                if delta > window.total_seconds():
                    continue
            return deal
        return None

    # ==================================================================
    # 4. Manual cancellation
    # ==================================================================

    def cancel_pending_order(
        self, ticket: int, reason: str = "MANUAL"
    ) -> OrderResult:
        """Cancel a pending order via the broker and reconcile DB state.

        Phase 3D (§6.6) failure classification:
          * ``success``                          → DB row marked ``cancelled``,
                                                    failure counter cleared
          * ``not found``                        → benign terminal state
                                                    (order filled/expired/
                                                    already cancelled);
                                                    reconciler resolves via
                                                    deal history, no retry
          * ``does not belong`` / ``symbol``     → permanent; escalate to
                                                    stuck list immediately
          * everything else                      → retry; escalate after
                                                    ``_STUCK_CANCEL_THRESHOLD``
                                                    consecutive failures
        """
        ticket = int(ticket)
        logger.info(
            "Cancelling pending order %s - reason: %s", ticket, reason,
        )

        try:
            result = mt5_connector.delete_order(ticket)
        except Exception as exc:
            logger.exception("delete_order raised for ticket %s.", ticket)
            result = OrderResult(
                success=False,
                error_message=f"delete_order raised: {exc}",
            )

        if result.success:
            self._clear_cancel_state(ticket)
            self._mark_db_cancelled(ticket)
            return result

        message = (result.error_message or "").lower()

        if "not found" in message:
            # Order is gone (filled, expired, cancelled by broker). No
            # retry needed — the reconciler will resolve via deal history.
            logger.info(
                "Cancel ticket %s: already gone (benign).", ticket,
            )
            self._clear_cancel_state(ticket)
            return result

        if "does not belong" in message or "symbol" in message:
            # Permanent. We cannot cancel this order; there is no retry
            # path. Escalate immediately so the operator sees it.
            logger.critical(
                "PERMANENT cancel failure for ticket %s: %s",
                ticket, result.error_message,
            )
            self._escalate_stuck(ticket, result)
            return result

        # Retryable — increment and escalate after threshold.
        self._record_cancel_failure(ticket, result)
        return result

    def cancel_all_pending(
        self, reason: str = "SHUTDOWN"
    ) -> "CancelAllResult":
        """Cancel every bot-owned pending order.

        Phase 3D (§6.6): returns a ``CancelAllResult`` carrying
        ``query_ok`` so the caller can distinguish "nothing to cancel"
        from "broker query failed". Still iterable as a list for
        backward compatibility.
        """
        try:
            orders = mt5_connector.get_orders()
        except Exception:
            logger.exception(
                "cancel_all_pending: get_orders raised.",
            )
            return CancelAllResult(results=[], query_ok=False)

        if orders is None:
            logger.error(
                "cancel_all_pending: pending-order query failed; "
                "cannot confirm cancellation."
            )
            return CancelAllResult(results=[], query_ok=False)

        results: List[OrderResult] = []
        for order in orders:
            try:
                results.append(
                    self.cancel_pending_order(
                        int(order["ticket"]), reason,
                    )
                )
            except Exception:
                logger.exception(
                    "cancel_pending_order raised for ticket %s.",
                    order.get("ticket"),
                )
                results.append(OrderResult(
                    success=False,
                    error_message="cancel_pending_order raised",
                ))

        return CancelAllResult(results=results, query_ok=True)

    # ---------- Phase 3D: cancel-failure bookkeeping ----------

    def _mark_db_cancelled(self, ticket: int) -> None:
        """Set the DB row for this ticket to ``cancelled``.

        Matches any order-request status that could still be waiting
        on a pending order (not just ``placed``) so a successful cancel
        clears the row regardless of reconciliation lag.
        """
        try:
            with db_manager.get_session() as session:
                candidates = (
                    session.query(OrderRequest)
                    .filter(
                        OrderRequest.environment == self.ENVIRONMENT,
                        OrderRequest.status.in_([
                            "placed", "pending", "sent",
                            "uncertain", "submission_error",
                        ]),
                    )
                    .all()
                )
            for req in candidates:
                if self._request_ticket(req) == int(ticket):
                    self._mark_request_status(req.id, "cancelled")
                    return
        except Exception:
            logger.exception(
                "Could not mark DB cancelled for ticket %s.", ticket,
            )

    def _record_cancel_failure(
        self, ticket: int, result: OrderResult,
    ) -> None:
        ticket = int(ticket)
        count = self._cancel_failures.get(ticket, 0) + 1
        self._cancel_failures[ticket] = count
        logger.warning(
            "Cancel failure %d/%d for ticket %s: %s",
            count, self._STUCK_CANCEL_THRESHOLD,
            ticket, result.error_message,
        )
        if count >= self._STUCK_CANCEL_THRESHOLD:
            self._escalate_stuck(ticket, result)

    def _escalate_stuck(
        self, ticket: int, result: OrderResult,
    ) -> None:
        ticket = int(ticket)
        existing = self._load_stuck_cancels()
        existing[ticket] = {
            "ticket": ticket,
            "failures": self._cancel_failures.get(
                ticket, self._STUCK_CANCEL_THRESHOLD,
            ),
            "last_error": result.error_message or "unknown",
            "last_attempt_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._save_stuck_cancels(existing)
        logger.critical(
            "TICKET %s marked STUCK for cancellation; operator "
            "intervention may be required. Error: %s",
            ticket, result.error_message,
        )

    def _clear_cancel_state(self, ticket: int) -> None:
        ticket = int(ticket)
        self._cancel_failures.pop(ticket, None)
        existing = self._load_stuck_cancels()
        if ticket in existing:
            del existing[ticket]
            self._save_stuck_cancels(existing)

    def _load_stuck_cancels(self) -> Dict[int, Dict[str, Any]]:
        key = f"pending_cancel_stuck_{self.ENVIRONMENT}"
        try:
            setting = db_manager.query_one(AppSettings, key=key)
        except Exception:
            logger.exception("Could not read stuck-cancel marker.")
            return {}
        if setting is None or not setting.value:
            return {}
        try:
            raw = json.loads(setting.value)
            return {int(k): v for k, v in raw.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.exception(
                "Corrupt stuck-cancel marker; ignoring."
            )
            return {}

    def _save_stuck_cancels(
        self, data: Dict[int, Dict[str, Any]],
    ) -> None:
        key = f"pending_cancel_stuck_{self.ENVIRONMENT}"
        payload = json.dumps(
            {str(k): v for k, v in data.items()}
        )
        try:
            with db_manager.get_session() as session:
                setting = (
                    session.query(AppSettings).filter_by(key=key).first()
                )
                if setting is None:
                    session.add(AppSettings(key=key, value=payload))
                else:
                    setting.value = payload
        except Exception:
            logger.exception("Could not persist stuck-cancel marker.")

    def get_stuck_cancellations(self) -> List[int]:
        """Return tickets whose pending-cancel has been escalated.

        Reads from ``AppSettings`` so it reflects state across process
        restarts. Used by the agent health check and dashboard.
        """
        return sorted(self._load_stuck_cancels().keys())

    # ==================================================================
    # 5. Order request preparation
    # ==================================================================

    def _prepare_order_request(
        self,
        signal,
        position_size: PositionSizeResult,
        market_entry: bool = False,
    ) -> Dict[str, Any]:
        symbol = signal.symbol
        direction = signal.direction.value.upper()

        if direction not in ("BUY", "SELL"):
            raise ValueError(f"Unsupported trade direction: {direction}")

        # M-12: unify on the connector's cached symbol spec and
        # freshness-checked quote — the SAME source used for sizing.
        # Mixing raw mt5.symbol_info[_tick]() with connector data can
        # silently disagree (test doubles, cache churn, reconnect).
        symbol_info = mt5_connector.get_symbol_info()
        tick = mt5_connector.get_fresh_tick()

        if symbol_info is None or tick is None:
            raise RuntimeError(
                f"Could not obtain FRESH broker symbol info/quote "
                f"for {symbol}."
            )

        if symbol_info.name != symbol:
            raise RuntimeError(
                f"Broker symbol {symbol_info.name!r} does not match "
                f"request symbol {symbol!r}."
            )

        point = float(symbol_info.point)
        digits = int(symbol_info.digits)
        if point <= 0:
            raise RuntimeError(f"Invalid point size for {symbol}.")

        try:
            tick_bid = float(tick.get("bid") or 0)
            tick_ask = float(tick.get("ask") or 0)
        except (TypeError, ValueError):
            tick_bid = 0.0
            tick_ask = 0.0
        if tick_bid <= 0 or tick_ask <= 0:
            raise RuntimeError(
                f"Fresh broker quote for {symbol} has invalid bid/ask."
            )

        entry = round(float(signal.entry_price), digits)
        sl = round(float(signal.sl_price), digits)
        tp = round(float(signal.tp_price), digits)

        # Call the class static method, not the instance — the instance
        # is often patched in tests and would return a MagicMock.
        pip_size = MT5Connector.pip_size_for(symbol_info)
        if not isinstance(pip_size, (int, float)) or pip_size <= 0:
            raise RuntimeError(f"Broker pip size for {symbol} is invalid.")

        configured_slippage = config.trade_management.max_slippage_pips
        deviation = max(
            1,
            int(round(float(configured_slippage) * pip_size / point)),
        )

        if market_entry:
            order_type = (
                mt5.ORDER_TYPE_BUY
                if direction == "BUY"
                else mt5.ORDER_TYPE_SELL
            )
            return {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "type": order_type,
                "volume": float(position_size.lot_size),
                "sl": sl,
                "tp": tp,
                "deviation": deviation,
                "magic": self.magic_number,
                "comment": f"SMC_{signal.strategy.value}"[:31],
                "fingerprint": signal.fingerprint,
            }

        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY_LIMIT
            if entry >= tick_ask:
                raise RuntimeError(
                    f"BUY_LIMIT entry {entry} is at/above current ask "
                    f"{tick_ask}; setup invalidated before placement."
                )
        else:
            order_type = mt5.ORDER_TYPE_SELL_LIMIT
            if entry <= tick_bid:
                raise RuntimeError(
                    f"SELL_LIMIT entry {entry} is at/below current bid "
                    f"{tick_bid}; setup invalidated before placement."
                )

        return {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "type": order_type,
            "volume": float(position_size.lot_size),
            "price": entry,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": self.magic_number,
            "comment": f"SMC_{signal.strategy.value}"[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "fingerprint": signal.fingerprint,
        }

    # ==================================================================
    # 6. Persistence helpers
    # ==================================================================

    def _save_signal(
        self,
        signal,
        position_size: PositionSizeResult,
        validation: ValidationResult,
    ) -> int:
        db_signal = TradingSignal(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            timeframe=signal.entry_tf,
            direction=signal.direction.value,
            entry_price=signal.entry_price,
            sl_price=signal.sl_price,
            tp_price=signal.tp_price,
            lot_size=position_size.lot_size,
            risk_percent=signal.risk_percent,
            risk_reward=signal.risk_reward,
            strategy=signal.strategy.value,
            confirmations=[
                {
                    "name": c.name,
                    "passed": c.passed,
                    "details": c.details,
                }
                for c in signal.confirmations
            ],
            status="validated",
            environment=self.ENVIRONMENT,
            fingerprint=signal.fingerprint,
        )
        saved_id = db_manager.add(db_signal)
        if not saved_id or int(saved_id) <= 0:
            raise RuntimeError(
                "Database did not return a saved signal ID."
            )
        return int(saved_id)

    def _save_order_request(
        self, signal_id: int, request: Dict[str, Any]
    ) -> int:
        """Atomically INSERT the OrderRequest.

        Raises ``DuplicateOrderRequest`` when the partial UNIQUE index
        ``uq_order_requests_active_fp`` rejects the insert — another
        process has an active request for this fingerprint. Any other
        DB failure propagates so the caller fails closed.
        """
        db_request = OrderRequest(
            signal_id=signal_id,
            magic_number=self.magic_number,
            symbol=request["symbol"],
            order_type=request["type"],
            volume=request["volume"],
            price=request.get("price"),
            sl=request.get("sl"),
            tp=request.get("tp"),
            comment=request.get("comment"),
            request_data=request,
            environment=self.ENVIRONMENT,
            status="pending",
            fingerprint=request.get("fingerprint"),
        )

        ok, row_id = db_manager.insert_order_request_atomic(db_request)
        if not ok:
            raise DuplicateOrderRequest(
                f"Active OrderRequest already exists for fingerprint "
                f"{request.get('fingerprint')} (existing id={row_id})."
            )
        if row_id is None or int(row_id) <= 0:
            raise RuntimeError(
                "Database did not return an order-request ID."
            )
        return int(row_id)

    def _update_order_request(
        self,
        request_id: int,
        result: OrderResult,
        market_entry: bool = False,
    ) -> None:
        rejected_fp: Optional[str] = None
        with db_manager.get_session() as session:
            request = (
                session.query(OrderRequest)
                .filter(OrderRequest.id == request_id)
                .first()
            )
            if request is None:
                logger.error(
                    "Could not find order-request row %s.", request_id
                )
                return

            request.response_data = {
                "success": result.success,
                "ticket": getattr(result, "ticket", None),
                "order_id": getattr(result, "order_id", None),
                "deal_id": getattr(result, "deal_id", None),
                "volume": result.volume,
                "price": result.price,
                "retcode": result.retcode,
                "comment": result.comment,
                "error_code": result.error_code,
                "error_message": result.error_message,
            }

            if result.success:
                ticket = (
                    getattr(result, "order_id", None)
                    or getattr(result, "ticket", None)
                )
                if ticket:
                    request.mt5_ticket = int(ticket)
                request.status = (
                    "filled_unconfirmed" if market_entry else "placed"
                )
                # N-H01: reservation held. It is released later by
                # _mark_request_status when the request transitions
                # to filled/filled_closed/unresolved_closed/cancelled.
            else:
                request.status = "rejected"
                rejected_fp = request.fingerprint

            request.error_code = result.error_code
            request.error_message = result.error_message

        # N-H01: release ONLY after the session commits the rejected
        # status, so a crash between write and release leaves the
        # reservation active (fail-closed).
        if rejected_fp is not None:
            self._release_reservation(rejected_fp, "broker rejection")

    def _update_order_request_error(
        self, request_id: int, message: str
    ) -> None:
        fingerprint: Optional[str] = None
        try:
            with db_manager.get_session() as session:
                request = (
                    session.query(OrderRequest)
                    .filter(OrderRequest.id == request_id)
                    .first()
                )
                if request is not None:
                    request.status = "submission_error"
                    request.error_message = message
                    fingerprint = request.fingerprint
        except Exception:
            logger.exception(
                "Could not record submission exception for %s.", request_id
            )
            return

        # N-H01: this path fires BEFORE send_order (e.g. gate re-check
        # raised). The broker never saw this request; release.
        if fingerprint:
            self._release_reservation(
                fingerprint, "submission_error before broker"
            )

    def _mark_request_uncertain(
        self, request_id: int, message: str
    ) -> None:
        """Atomically transition a request to 'uncertain'.

        Single UPDATE, single transaction. Raises on any failure so
        the caller can fail closed — this is NOT a best-effort write.
        The request must never go through 'rejected' as an
        intermediate state, because 'rejected' is terminal and would
        release the fingerprint.

        Raises:
            RuntimeError if the row is missing or in an unexpected
            state (prevents silent corruption).
            Any SQLAlchemy error propagates.
        """
        with db_manager.get_session() as session:
            request = (
                session.query(OrderRequest)
                .filter(OrderRequest.id == request_id)
                .first()
            )
            if request is None:
                raise RuntimeError(
                    f"Cannot mark uncertain: OrderRequest "
                    f"{request_id} not found."
                )
            if request.status not in ("pending", "sent"):
                raise RuntimeError(
                    f"Refusing uncertain transition for request "
                    f"{request_id} from status {request.status!r}."
                )
            request.status = "uncertain"
            request.error_message = message

    def _save_executed_order(
        self,
        position: Dict[str, Any],
        request_id: Optional[int] = None,
    ) -> None:
        ticket = int(position["ticket"])

        existing = db_manager.query_one(
            ExecutedOrder, ticket=ticket, environment=self.ENVIRONMENT
        )
        if existing is not None:
            if existing.order_request_id is None and request_id is not None:
                try:
                    with db_manager.get_session() as session:
                        row = (
                            session.query(ExecutedOrder)
                            .filter(
                                ExecutedOrder.ticket == ticket,
                                ExecutedOrder.environment == self.ENVIRONMENT,
                            )
                            .first()
                        )
                        if row is not None:
                            row.order_request_id = request_id
                except Exception:
                    logger.exception(
                        "Could not link ExecutedOrder ticket %s to request %s.",
                        ticket,
                        request_id,
                    )
            return

        db_order = ExecutedOrder(
            order_request_id=request_id,
            ticket=ticket,
            magic_number=position.get("magic", self.magic_number),
            symbol=position["symbol"],
            order_type=position["type"],
            volume=position["volume"],
            price_open=position["price_open"],
            price_current=position["price_current"],
            sl=position["sl"],
            tp=position["tp"],
            swap=position["swap"],
            profit=position["profit"],
            commission=position.get("commission", 0.0),
            comment=position.get("comment", ""),
            time_open=position["time_open"],
            environment=self.ENVIRONMENT,
        )
        db_manager.add(db_order)

    def _save_open_position_from_request(
        self, position: Dict[str, Any], request: OrderRequest
    ) -> None:
        ticket = int(position["ticket"])

        existing = db_manager.query_one(
            OpenPosition, ticket=ticket, environment=self.ENVIRONMENT
        )
        if existing is not None:
            return

        entry = float(position["price_open"])
        stop_loss = float(position.get("sl") or 0)
        take_profit = float(position.get("tp") or 0)
        initial_risk = abs(entry - stop_loss)

        db_position = OpenPosition(
            ticket=ticket,
            magic_number=position.get("magic", self.magic_number),
            symbol=position["symbol"],
            direction=position["type"],
            volume=position["volume"],
            price_open=entry,
            price_current=position["price_current"],
            sl=stop_loss,
            tp=take_profit,
            swap=position["swap"],
            profit=position["profit"],
            commission=position.get("commission", 0.0),
            strategy=(request.comment or "SMC_UNKNOWN").replace("SMC_", ""),
            time_open=position["time_open"],
            environment=self.ENVIRONMENT,
            initial_sl=stop_loss,
            initial_tp=take_profit,
            initial_risk_distance=initial_risk,
            remaining_volume=position["volume"],
            highest_r=0.0,
            lowest_r=0.0,
            break_even_activated=False,
            trailing_active=False,
            partial_tp_taken=[],
            management_state={},
        )
        db_manager.add(db_position)

    # ==================================================================
    # 7. Close / modify
    # ==================================================================

    def close_position(
        self, ticket: int, reason: str = "MANUAL"
    ) -> OrderResult:
        logger.info("Closing position %s - reason: %s", ticket, reason)
        result = mt5_connector.close_position(ticket)
        if result.success:
            try:
                self._update_position_on_close(ticket, reason, result)
                trade_manager.remove_position(ticket)
            except Exception:
                logger.exception(
                    "Position %s closed at MT5 but DB cleanup failed.", ticket
                )
        return result

    def close_all_positions(
        self, reason: str = "EMERGENCY"
    ) -> "CloseAllResult":
        """Close bot-owned positions only. Manual positions are untouched.

        H-08: returns a CloseAllResult carrying ``query_ok`` so the
        controller can distinguish "no positions" from "query failed".
        Still iterable as a list for backward compatibility.
        """
        try:
            positions = mt5_connector.get_positions()
        except Exception:
            logger.exception("Cannot close all: position query raised.")
            return CloseAllResult(results=[], query_ok=False)

        if positions is None:
            logger.error("Cannot close all: position query failed.")
            return CloseAllResult(results=[], query_ok=False)

        bot_positions = [
            p for p in positions
            if int(p.get("magic", 0)) == int(self.magic_number)
        ]
        results = [
            self.close_position(int(p["ticket"]), reason)
            for p in bot_positions
        ]
        return CloseAllResult(results=results, query_ok=True)

    def modify_position_sl_tp(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> OrderResult:
        result = mt5_connector.modify_position(ticket, sl, tp)
        if result.success:
            self._update_position_sl_tp(ticket, sl, tp)
        return result

    def _update_position_on_close(
        self, ticket: int, reason: str, close_result: OrderResult
    ) -> None:
        """Delegate F-06 reconstruction to trade_manager.

        If deal history is not yet available, this leaves the OpenPosition
        row in place; trade_manager.reconcile_with_broker() will finalise
        the ClosedTrade on the next health-check cycle.
        """
        success = trade_manager.reconstruct_closed_trade(
            ticket=ticket, exit_reason=reason
        )
        if not success:
            logger.warning(
                "Position %s closed at MT5 but ClosedTrade could not be "
                "reconstructed from deal history yet; reconciler will "
                "retry next cycle.",
                ticket,
            )

    def _update_position_sl_tp(
        self,
        ticket: int,
        sl: Optional[float],
        tp: Optional[float],
    ) -> None:
        with db_manager.get_session() as session:
            order = (
                session.query(ExecutedOrder)
                .filter(
                    ExecutedOrder.ticket == ticket,
                    ExecutedOrder.environment == self.ENVIRONMENT,
                )
                .first()
            )
            if order is not None:
                if sl is not None:
                    order.sl = sl
                if tp is not None:
                    order.tp = tp

            position = (
                session.query(OpenPosition)
                .filter(
                    OpenPosition.ticket == ticket,
                    OpenPosition.environment == self.ENVIRONMENT,
                )
                .first()
            )
            if position is not None:
                if sl is not None:
                    position.sl = sl
                if tp is not None:
                    position.tp = tp


# Global executor instance
order_executor = OrderExecutor()