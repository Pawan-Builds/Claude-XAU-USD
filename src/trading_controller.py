"""Central coordinator for dashboard and out-of-process control.

Every entry action funnels through the trading agent's EntryGate so
that pause, shutdown, account verification, risk, session, news, and
kill-switch checks cannot be bypassed by a dashboard client.

Lifecycle:
- TradingAgent.start()    -> trading_controller.register_agent(...)
- TradingAgent.shutdown() -> trading_controller.unregister_agent()

Before registration, entry actions return 'Agent not registered' and
all state-changing calls are refused. Fail-closed by construction.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Dict, Optional

from src.config_loader import config
from src.mt5_connector import mt5_connector
from src.order_executor import order_executor, ExecutionResult
from src.risk_manager import risk_manager
from src.logger import get_logger

logger = get_logger(__name__)


class ControllerState(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY = "emergency"


@dataclass
class ControllerStatus:
    state: str
    agent_registered: bool
    mt5_connected: bool
    kill_switch: bool
    mode: str
    timestamp: str


class TradingController:
    """Thread-safe facade over the running TradingAgent."""

    def __init__(self):
        self._lock = RLock()
        self._agent = None
        self._gate = None
        self._state = ControllerState.STOPPED

    # ---------------- Lifecycle ----------------

    def register_agent(self, agent, gate) -> None:
        with self._lock:
            self._agent = agent
            self._gate = gate
            self._state = ControllerState.RUNNING
            logger.info("TradingController: agent registered.")

    def unregister_agent(self) -> None:
        with self._lock:
            self._agent = None
            self._gate = None
            # Emergency persists past agent shutdown; operator must clear.
            if self._state != ControllerState.EMERGENCY:
                self._state = ControllerState.STOPPED
            logger.info("TradingController: agent unregistered.")

    # ---------------- Status ----------------

    def status(self) -> ControllerStatus:
        with self._lock:
            try:
                kill = risk_manager.is_kill_switch_active()
            except Exception:
                kill = True  # fail closed in the report
            return ControllerStatus(
                state=self._state.value,
                agent_registered=self._agent is not None,
                mt5_connected=mt5_connector.is_connected(),
                kill_switch=kill,
                mode=str(config.trading.mode).upper(),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    # ---------------- State changes ----------------

    def pause(self) -> Dict[str, Any]:
        with self._lock:
            if self._agent is None:
                return {"ok": False, "reason": "Agent not registered."}
            if self._state == ControllerState.EMERGENCY:
                return {
                    "ok": False,
                    "reason": "Emergency state; clear it first.",
                }
            self._agent._pause_trading()
            self._state = ControllerState.PAUSED
            logger.info("TradingController: paused.")
            return {"ok": True, "state": self._state.value}

    def resume(self) -> Dict[str, Any]:
        with self._lock:
            if self._agent is None:
                return {"ok": False, "reason": "Agent not registered."}
            if self._state == ControllerState.EMERGENCY:
                return {
                    "ok": False,
                    "reason": "Emergency state; clear it first.",
                }
            probe = self._gate.probe()
            if not probe.ok:
                return {"ok": False, "reason": probe.reason}
            self._agent._resume_trading()
            if getattr(self._agent, "_paused", False):
                return {
                    "ok": False,
                    "reason": "Agent refused to resume (state changed).",
                }
            self._state = ControllerState.RUNNING
            logger.info("TradingController: resumed.")
            return {"ok": True, "state": self._state.value}

    def shutdown(self) -> Dict[str, Any]:
        with self._lock:
            if self._agent is None:
                return {"ok": False, "reason": "Agent not registered."}
            self._agent._request_shutdown()
            self._state = ControllerState.STOPPED
            logger.info("TradingController: shutdown requested.")
            return {"ok": True, "state": self._state.value}

    def emergency_stop(self, reason: str = "dashboard") -> Dict[str, Any]:
        """Activate kill switch + pause + cancel pending orders + close
        bot-owned positions, then verify broker state.

        H-08: never reports success unless every sub-operation that
        could be verified was verified. Distinguishes four states:
          * ``emergency``     — procedure engaged and verified
          * ``partial``       — some sub-operations succeeded, some failed
          * ``failed``        — could not activate
          * ``unverifiable``  — close-all's position query failed
        The internal controller state is ALWAYS set to EMERGENCY.

        H-06: bot-owned pending orders are cancelled BEFORE close_all,
        so no pending can fill during the close. A fresh broker query
        after both operations confirms the flattening state via the
        additive ``flattened`` boolean and ``unresolved_exposure`` dict.
        ``state`` continues to mean "procedure engaged and verified";
        ``flattened`` means "broker confirms zero bot-owned exposure".
        """
        with self._lock:
            # 1. Kill switch.
            kill_written = False
            kill_verified = False
            kill_error = None

            try:
                ks_result = risk_manager.set_kill_switch(
                    True, f"Emergency stop: {reason}"
                )

                if ks_result is None:
                    kill_written = True
                    try:
                        kill_verified = bool(
                            risk_manager.is_kill_switch_active()
                        )
                    except Exception:
                        kill_verified = False
                    if not kill_verified:
                        kill_error = (
                            "kill switch returned no result and "
                            "read-back failed"
                        )
                else:
                    ok_attr = getattr(ks_result, "ok", None)
                    active_attr = getattr(ks_result, "active", None)
                    err_attr = getattr(ks_result, "error", None)

                    if ok_attr is None or active_attr is None:
                        kill_written = False
                        kill_verified = False
                        kill_error = (
                            "kill switch returned unexpected type "
                            f"{type(ks_result).__name__}"
                        )
                    else:
                        kill_written = bool(ok_attr)
                        kill_verified = bool(ok_attr and active_attr)
                        if not ok_attr:
                            kill_error = err_attr or (
                                "kill switch write failed."
                            )
                            logger.error(
                                "Emergency stop: kill switch write "
                                "failed: %s", kill_error,
                            )
            except Exception as exc:
                kill_error = f"kill switch write raised: {exc}"
                logger.exception(
                    "Emergency stop: kill switch write raised."
                )

            # 2. Pause agent.
            paused = False
            if self._agent is not None:
                try:
                    self._agent._pause_trading()
                    paused = True
                except Exception:
                    logger.exception(
                        "Emergency stop: could not pause agent."
                    )

            # 3. H-06: cancel bot-owned pending orders FIRST.
            cancel_attempt = self._cancel_bot_pending_orders()

            # 4. Close bot-owned positions.
            close_attempted = False
            query_ok = True
            closed = 0
            failed = 0
            close_error = None

            try:
                close_result = order_executor.close_all_positions(
                    "EMERGENCY"
                )
                close_attempted = True
                closed = close_result.closed
                failed = close_result.failed
                query_ok = close_result.query_ok
            except Exception as exc:
                close_error = f"close all failed: {exc}"
                logger.exception(
                    "Emergency stop: close_all_positions raised."
                )

            # 5. H-06: post-attempt broker verification.
            verification = self._verify_flattened()

            # 6. Classify state (H-08 semantics preserved).
            if not query_ok:
                state = "unverifiable"
                ok = False
            elif kill_verified and close_attempted:
                state = "emergency"
                ok = True
            elif kill_verified or close_attempted:
                state = "partial"
                ok = False
            else:
                state = "failed"
                ok = False

            # 7. H-06: flattened + unresolved exposure.
            # `flattened` is True only when the broker query confirms
            # zero bot-owned positions AND zero bot-owned pending orders
            # AND no unconfirmed cancels. Trust the query, not the
            # per-call success counts.
            flattened = (
                verification["query_ok"]
                and verification["positions_remaining"] == 0
                and verification["pending_remaining"] == 0
                and not cancel_attempt["unconfirmed"]
            )

            unresolved_exposure = {
                "positions_remaining": verification["positions_remaining"],
                "pending_remaining": verification["pending_remaining"],
                "unconfirmed_cancel_tickets": list(
                    cancel_attempt["unconfirmed"]
                ),
            }

            # 8. Always enter EMERGENCY internally so resume is blocked.
            self._state = ControllerState.EMERGENCY

            error = (
                kill_error
                or close_error
                or cancel_attempt["error"]
                or verification["error"]
            )

            logger.critical(
                "TradingController: EMERGENCY STOP | state=%s | "
                "reason=%s | kill_verified=%s | closed=%d failed=%d | "
                "cancelled=%d cancel_failed=%d unconfirmed=%d | "
                "positions_remaining=%s pending_remaining=%s | "
                "flattened=%s",
                state, reason, kill_verified, closed, failed,
                cancel_attempt["cancelled"], cancel_attempt["failed"],
                len(cancel_attempt["unconfirmed"]),
                verification["positions_remaining"],
                verification["pending_remaining"],
                flattened,
            )

            return {
                # Preserved from H-08
                "state": state,
                "ok": ok,
                "kill_switch_written": kill_written,
                "kill_switch_verified": kill_verified,
                "kill_switch": kill_verified,
                "paused": paused,
                "close_attempted": close_attempted,
                "closed": closed,
                "failed": failed,
                "closed_count": closed,
                "failed_count": failed,
                "error": error,
                # H-06 additions
                "cancel_attempted": cancel_attempt["attempted"],
                "cancel_query_ok": cancel_attempt["query_ok"],
                "cancelled": cancel_attempt["cancelled"],
                "cancel_failed": cancel_attempt["failed"],
                "cancel_unconfirmed": list(
                    cancel_attempt["unconfirmed"]
                ),
                "positions_remaining": verification["positions_remaining"],
                "pending_remaining": verification["pending_remaining"],
                "verification_query_ok": verification["query_ok"],
                "flattened": flattened,
                "unresolved_exposure": unresolved_exposure,
            }

    # ---------------- H-06 helpers ----------------

    def _cancel_bot_pending_orders(self) -> Dict[str, Any]:
        """H-06: enumerate, cancel, and verify cancellation of
        bot-owned pending orders.

        Returns a dict with:
            attempted:   bool — the cancel phase ran (enumeration succeeded)
            query_ok:    bool — the unconfirmed list is authoritative
                                (pre-cancel AND post-cancel queries both
                                succeeded)
            cancelled:   int
            failed:      int
            unconfirmed: List[int] — initial tickets still visible on the
                                     broker after the cancel attempt
            error:       Optional[str] — joined diagnostics
        """
        res: Dict[str, Any] = {
            "attempted": False,
            "query_ok": True,
            "cancelled": 0,
            "failed": 0,
            "unconfirmed": [],
            "error": None,
        }
        errors: list = []

        # 1. Enumerate bot-owned pending orders.
        try:
            initial = mt5_connector.get_orders()
        except Exception as exc:
            logger.exception(
                "Emergency stop: pre-cancel order query raised."
            )
            res["query_ok"] = False
            res["error"] = f"pre-cancel order query raised: {exc}"
            return res

        if initial is None:
            logger.error(
                "Emergency stop: pre-cancel order query returned None; "
                "cannot enumerate or verify pending orders."
            )
            res["query_ok"] = False
            res["error"] = "pre-cancel order query returned None"
            return res

        initial_tickets = {int(o["ticket"]) for o in initial}
        res["attempted"] = True

        if not initial_tickets:
            # Verified empty; nothing to cancel, nothing to verify.
            return res

        # 2. Cancel. Cancel's own query_ok is diagnostic; the post-cancel
        # re-query below is authoritative.
        try:
            cancel_res = order_executor.cancel_all_pending("EMERGENCY")
            res["cancelled"] = int(getattr(cancel_res, "succeeded", 0))
            res["failed"] = int(getattr(cancel_res, "failed", 0))
            if not bool(getattr(cancel_res, "query_ok", True)):
                errors.append(
                    "cancel_all_pending reported query_ok=False"
                )
        except Exception as exc:
            logger.exception(
                "Emergency stop: cancel_all_pending raised."
            )
            errors.append(f"cancel_all_pending raised: {exc}")

        # 3. Verify via fresh broker query.
        try:
            remaining = mt5_connector.get_orders()
        except Exception as exc:
            logger.exception(
                "Emergency stop: post-cancel order query raised."
            )
            res["query_ok"] = False
            res["unconfirmed"] = sorted(initial_tickets)
            errors.append(f"post-cancel query raised: {exc}")
            res["error"] = "; ".join(errors) if errors else None
            return res

        if remaining is None:
            res["query_ok"] = False
            res["unconfirmed"] = sorted(initial_tickets)
            errors.append("post-cancel query returned None")
            res["error"] = "; ".join(errors) if errors else None
            return res

        remaining_tickets = {int(o["ticket"]) for o in remaining}
        res["unconfirmed"] = sorted(
            initial_tickets & remaining_tickets
        )

        if errors:
            res["error"] = "; ".join(errors)

        return res

    def _verify_flattened(self) -> Dict[str, Any]:
        """H-06: post-attempt broker verification of bot-owned exposure.

        Fresh queries; not derived from the attempt's own success counts.

        Returns:
            query_ok:            bool — BOTH queries succeeded
            positions_remaining: int  — -1 on query failure
            pending_remaining:   int  — -1 on query failure
            error:               Optional[str]
        """
        res: Dict[str, Any] = {
            "query_ok": True,
            "positions_remaining": 0,
            "pending_remaining": 0,
            "error": None,
        }
        errors: list = []

        try:
            positions = mt5_connector.get_positions()
            if positions is None:
                res["query_ok"] = False
                res["positions_remaining"] = -1
                errors.append("position query returned None")
            else:
                res["positions_remaining"] = len(positions)
        except Exception as exc:
            logger.exception(
                "Emergency stop: post-close position query raised."
            )
            res["query_ok"] = False
            res["positions_remaining"] = -1
            errors.append(f"position query raised: {exc}")

        try:
            orders = mt5_connector.get_orders()
            if orders is None:
                res["query_ok"] = False
                res["pending_remaining"] = -1
                errors.append("order query returned None")
            else:
                res["pending_remaining"] = len(orders)
        except Exception as exc:
            logger.exception(
                "Emergency stop: post-close order query raised."
            )
            res["query_ok"] = False
            res["pending_remaining"] = -1
            errors.append(f"order query raised: {exc}")

        if errors:
            res["error"] = "; ".join(errors)

        return res

    def clear_emergency(self) -> Dict[str, Any]:
        with self._lock:
            if self._state != ControllerState.EMERGENCY:
                return {"ok": False, "reason": "Not in emergency state."}
            self._state = (
                ControllerState.PAUSED
                if self._agent is not None
                else ControllerState.STOPPED
            )
            logger.warning(
                "TradingController: emergency cleared. "
                "Kill switch still requires explicit clear."
            )
            return {
                "ok": True,
                "state": self._state.value,
                "kill_switch_still_active": risk_manager.is_kill_switch_active(),
            }

    def set_kill_switch(
        self, active: bool, reason: str = "operator"
    ) -> Dict[str, Any]:
        """Manual kill-switch control.

        Never reports ok=True unless RiskManager's read-back
        confirmed the state. On any failure returns ok=False with
        the confirmed/previous state and an error message.
        """
        result = risk_manager.set_kill_switch(active, reason)
        if not result.ok:
            logger.error(
                "Kill switch change FAILED | requested=%s | active=%s "
                "| error=%s",
                active, result.active, result.error,
            )
            return {
                "ok": False,
                "active": result.active,
                "error": result.error or "Kill switch write failed.",
            }
        return {"ok": True, "active": result.active}

    # ---------------- Position operations ----------------

    def close_position(self, ticket: int) -> Dict[str, Any]:
        result = order_executor.close_position(
            int(ticket), "MANUAL_DASHBOARD"
        )
        return {
            "success": bool(result.success),
            "error": result.error_message,
        }

    def close_all(self) -> Dict[str, Any]:
        try:
            result = order_executor.close_all_positions(
                "MANUAL_DASHBOARD_ALL"
            )
        except Exception as exc:
            logger.exception("close_all raised.")
            return {
                "ok": False,
                "closed": 0,
                "failed": 0,
                "query_ok": False,
                "error": f"close_all raised: {exc}",
            }

        return {
            "ok": result.query_ok and result.failed == 0,
            "closed": result.closed,
            "failed": result.failed,
            "query_ok": result.query_ok,
            "error": (
                None if result.query_ok
                else "Position query failed; cannot confirm all closed."
            ),
        }

    def close_all_for_weekend(self) -> Dict[str, Any]:
        """Close bot-owned positions ahead of the weekend.

        Distinct from ``close_all`` in its reason, so the ClosedTrade
        rows can be distinguished from manual closes. Idempotency is
        enforced by the caller (main.py) via a persisted weekend marker;
        this method itself will close whatever is currently open.
        """
        try:
            result = order_executor.close_all_positions("WEEKEND")
        except Exception as exc:
            logger.exception("close_all_for_weekend raised.")
            return {
                "ok": False,
                "closed": 0,
                "failed": 0,
                "query_ok": False,
                "error": f"close_all_for_weekend raised: {exc}",
            }

        return {
            "ok": result.query_ok and result.failed == 0,
            "closed": result.closed,
            "failed": result.failed,
            "query_ok": result.query_ok,
            "error": (
                None if result.query_ok
                else "Position query failed; cannot confirm all closed."
            ),
        }

    # ---------------- Entry ----------------

    def submit_manual_trade(self, signal) -> ExecutionResult:
        with self._lock:
            if self._state == ControllerState.EMERGENCY:
                return ExecutionResult(
                    success=False,
                    error="Emergency stop is active. Clear it before trading.",
                )
            if self._gate is None:
                return ExecutionResult(
                    success=False,
                    error="Agent not registered; manual trades refused.",
                )
            gate = self._gate
        return order_executor.execute_signal(signal, gate=gate)


trading_controller = TradingController()