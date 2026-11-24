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
        """Activate kill switch + pause + close bot-owned positions.

        H-08: never reports success unless every sub-operation that
        could be verified was verified. Distinguishes four states:
          * ``emergency``     — fully activated and verified
          * ``partial``       — some sub-operations succeeded, some failed
          * ``failed``        — could not activate; new entries may still
                                be blocked by internal state, but the
                                report is honest
          * ``unverifiable``  — position query failed; cannot confirm
                                whether positions remain open
        The internal controller state is ALWAYS set to EMERGENCY so
        resume is blocked regardless of the reported state.
        """
        with self._lock:
            # 1. Kill switch: write + read-back verification.
            kill_written = False
            kill_verified = False
            kill_error = None

            try:
                risk_manager.set_kill_switch(
                    True, f"Emergency stop: {reason}"
                )
                kill_written = True
            except Exception as exc:
                kill_error = f"kill switch write failed: {exc}"
                logger.exception(
                    "Emergency stop: could not set kill switch."
                )

            try:
                kill_verified = bool(risk_manager.is_kill_switch_active())
            except Exception as exc:
                kill_verified = False
                if kill_error is None:
                    kill_error = f"kill switch read failed: {exc}"

            # 2. Pause agent (may be None).
            paused = False
            if self._agent is not None:
                try:
                    self._agent._pause_trading()
                    paused = True
                except Exception:
                    logger.exception(
                        "Emergency stop: could not pause agent."
                    )

            # 3. Close bot-owned positions.
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

            # 4. Classify state honestly.
            #
            # `state="emergency"` means the procedure engaged and was
            # verified (kill switch written AND read back True; close-all
            # attempted; position query OK). Per-close `failed` outcomes
            # are reported in the payload for the operator to triage but
            # do NOT downgrade the state — a broker rejecting one close
            # is not the same failure as the kill switch failing to write.
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

            # 5. Always enter EMERGENCY internally so resume is blocked.
            self._state = ControllerState.EMERGENCY

            error = kill_error or close_error
            logger.critical(
                "TradingController: EMERGENCY STOP | state=%s | "
                "reason=%s | kill_verified=%s | closed=%d failed=%d "
                "query_ok=%s",
                state, reason, kill_verified, closed, failed, query_ok,
            )

            return {
                "state": state,
                "ok": ok,
                "kill_switch_written": kill_written,
                "kill_switch_verified": kill_verified,
                "kill_switch": kill_verified,  # back-compat
                "paused": paused,
                "close_attempted": close_attempted,
                "closed": closed,
                "failed": failed,
                "closed_count": closed,
                "failed_count": failed,
                "error": error,
            }

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
        """Manual kill switch control. Deactivation requires explicit flag."""
        try:
            risk_manager.set_kill_switch(active, reason)
        except Exception:
            logger.exception("Could not set kill switch.")
            return {"ok": False, "reason": "Kill switch write failed."}
        return {"ok": True, "active": active}

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