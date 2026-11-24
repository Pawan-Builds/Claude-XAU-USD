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
        with self._lock:
            # 1. Durable kill switch.
            try:
                risk_manager.set_kill_switch(
                    True, f"Emergency stop: {reason}"
                )
            except Exception:
                logger.exception(
                    "Emergency stop: could not set kill switch."
                )
            # 2. Pause the agent (optional; may be None).
            if self._agent is not None:
                try:
                    self._agent._pause_trading()
                except Exception:
                    logger.exception(
                        "Emergency stop: could not pause agent."
                    )
            # 3. Close bot-owned positions regardless.
            closed = failed = 0
            try:
                results = order_executor.close_all_positions("EMERGENCY")
                closed = sum(1 for r in results if r.success)
                failed = sum(1 for r in results if not r.success)
            except Exception:
                logger.exception(
                    "Emergency stop: close_all_positions failed."
                )
            self._state = ControllerState.EMERGENCY
            logger.critical(
                "TradingController: EMERGENCY STOP | reason=%s | "
                "closed=%d failed=%d",
                reason, closed, failed,
            )
            return {
                "ok": True,
                "state": self._state.value,
                "closed": closed,
                "failed": failed,
                "kill_switch": True,
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
        results = order_executor.close_all_positions(
            "MANUAL_DASHBOARD_ALL"
        )
        return {
            "closed": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
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