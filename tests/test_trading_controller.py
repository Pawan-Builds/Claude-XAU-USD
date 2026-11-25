"""Tests for the central TradingController (F-03)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.conftest import StubGate


@pytest.fixture
def ctrl_harness(monkeypatch):
    from src import trading_controller as tc_module
    from src.order_executor import CloseAllResult

    # H-08: kill switch mock must reflect its own write state so that
    # read-back verification returns True after a successful write and
    # False after a failed one.
    kill_state = {"active": False}

    rm = MagicMock(name="risk_manager")
    rm.is_kill_switch_active.side_effect = lambda: kill_state["active"]

    def _set_kill(active, reason=""):
        kill_state["active"] = bool(active)

    rm.set_kill_switch.side_effect = _set_kill

    from src.order_executor import CancelAllResult

    oe = MagicMock(name="order_executor")
    oe.close_all_positions.return_value = CloseAllResult(
        results=[], query_ok=True,
    )
    oe.cancel_all_pending.return_value = CancelAllResult(
        results=[], query_ok=True,
    )
    oe.close_position.return_value = SimpleNamespace(
        success=True, error_message=None
    )
    oe.execute_signal.return_value = SimpleNamespace(
        success=True, error=None
    )

    mt5 = MagicMock(name="mt5_connector")
    mt5.is_connected.return_value = True
    # H-06: emergency_stop queries both before and after cancel/close.
    # Default to empty so tests that don't exercise H-06 behaviour are
    # unaffected; individual tests override via .return_value or
    # .side_effect.
    mt5.get_positions.return_value = []
    mt5.get_orders.return_value = []

    fake_config = SimpleNamespace(
        trading=SimpleNamespace(mode="DEMO"),
    )

    monkeypatch.setattr(tc_module, "risk_manager", rm)
    monkeypatch.setattr(tc_module, "order_executor", oe)
    monkeypatch.setattr(tc_module, "mt5_connector", mt5)
    monkeypatch.setattr(tc_module, "config", fake_config)

    from src.trading_controller import TradingController
    controller = TradingController()

    return SimpleNamespace(
        controller=controller, rm=rm, oe=oe, mt5=mt5,
        kill_state=kill_state,
    )


def _mock_agent():
    agent = MagicMock(name="agent")
    agent._paused = True
    agent._pause_trading.side_effect = lambda: setattr(
        agent, "_paused", True
    )
    agent._resume_trading.side_effect = lambda: setattr(
        agent, "_paused", False
    )
    agent._request_shutdown.side_effect = lambda: None
    return agent


# ---------------- Lifecycle ----------------


def test_initial_state_is_stopped(ctrl_harness):
    status = ctrl_harness.controller.status()
    assert status.state == "stopped"
    assert status.agent_registered is False


def test_register_and_unregister(ctrl_harness):
    agent = _mock_agent()
    gate = StubGate(ok=True)

    ctrl_harness.controller.register_agent(agent, gate)
    status = ctrl_harness.controller.status()
    assert status.state == "running"
    assert status.agent_registered is True

    ctrl_harness.controller.unregister_agent()
    status = ctrl_harness.controller.status()
    assert status.state == "stopped"
    assert status.agent_registered is False


def test_unregister_does_not_clear_emergency(ctrl_harness):
    agent = _mock_agent()
    gate = StubGate(ok=True)
    ctrl_harness.controller.register_agent(agent, gate)
    ctrl_harness.controller.emergency_stop(reason="test")

    ctrl_harness.controller.unregister_agent()
    assert ctrl_harness.controller.status().state == "emergency"


# ---------------- Pause / resume ----------------


def test_pause_calls_agent_and_sets_state(ctrl_harness):
    agent = _mock_agent()
    agent._paused = False
    ctrl_harness.controller.register_agent(agent, StubGate(ok=True))

    result = ctrl_harness.controller.pause()

    assert result["ok"] is True
    assert result["state"] == "paused"
    agent._pause_trading.assert_called_once()


def test_pause_refused_without_agent(ctrl_harness):
    result = ctrl_harness.controller.pause()
    assert result["ok"] is False
    assert "not registered" in result["reason"].lower()


def test_pause_refused_in_emergency(ctrl_harness):
    agent = _mock_agent()
    ctrl_harness.controller.register_agent(agent, StubGate(ok=True))
    ctrl_harness.controller.emergency_stop(reason="test")

    result = ctrl_harness.controller.pause()
    assert result["ok"] is False
    assert "emergency" in result["reason"].lower()


def test_resume_blocked_when_gate_fails(ctrl_harness):
    agent = _mock_agent()
    ctrl_harness.controller.register_agent(
        agent, StubGate(ok=False, reason="session closed")
    )

    result = ctrl_harness.controller.resume()
    assert result["ok"] is False
    assert "session" in result["reason"].lower()
    agent._resume_trading.assert_not_called()


def test_resume_succeeds_when_gate_passes(ctrl_harness):
    agent = _mock_agent()
    ctrl_harness.controller.register_agent(agent, StubGate(ok=True))

    result = ctrl_harness.controller.resume()

    assert result["ok"] is True
    assert result["state"] == "running"
    agent._resume_trading.assert_called_once()


# ---------------- Shutdown ----------------


def test_shutdown_calls_agent(ctrl_harness):
    agent = _mock_agent()
    ctrl_harness.controller.register_agent(agent, StubGate(ok=True))

    result = ctrl_harness.controller.shutdown()

    assert result["ok"] is True
    assert result["state"] == "stopped"
    agent._request_shutdown.assert_called_once()


# ---------------- Emergency ----------------


def test_emergency_stop_sets_kill_switch_and_closes_all(ctrl_harness):
    from src.order_executor import CloseAllResult
    agent = _mock_agent()
    ctrl_harness.controller.register_agent(agent, StubGate(ok=True))
    ctrl_harness.oe.close_all_positions.return_value = CloseAllResult(
        results=[
            SimpleNamespace(success=True, error_message=None),
            SimpleNamespace(success=False, error_message="rejected"),
        ],
        query_ok=True,
    )

    result = ctrl_harness.controller.emergency_stop(reason="test")

    assert result["ok"] is True
    assert result["state"] == "emergency"
    assert result["closed"] == 1
    assert result["failed"] == 1
    assert result["kill_switch_verified"] is True
    ctrl_harness.rm.set_kill_switch.assert_called_once()
    agent._pause_trading.assert_called_once()


def test_emergency_stop_works_without_agent(ctrl_harness):
    # No agent registered. Kill switch + close must still execute.
    result = ctrl_harness.controller.emergency_stop(reason="test")
    assert result["ok"] is True
    ctrl_harness.rm.set_kill_switch.assert_called_once()
    ctrl_harness.oe.close_all_positions.assert_called_once()


def test_emergency_stop_does_not_raise_if_kill_switch_write_fails(
    ctrl_harness,
):
    ctrl_harness.rm.set_kill_switch.side_effect = RuntimeError("db down")
    result = ctrl_harness.controller.emergency_stop(reason="test")
    # H-08: kill-switch write failed → must NOT report success.
    assert result["ok"] is False
    assert result["state"] == "partial"
    assert result["kill_switch_verified"] is False
    assert result["close_attempted"] is True
    ctrl_harness.oe.close_all_positions.assert_called_once()


def test_clear_emergency_resets_state(ctrl_harness):
    agent = _mock_agent()
    ctrl_harness.controller.register_agent(agent, StubGate(ok=True))
    ctrl_harness.controller.emergency_stop(reason="test")

    result = ctrl_harness.controller.clear_emergency()

    assert result["ok"] is True
    assert result["state"] == "paused"


def test_clear_emergency_noop_when_not_in_emergency(ctrl_harness):
    result = ctrl_harness.controller.clear_emergency()
    assert result["ok"] is False


# ---------------- Manual trades ----------------


def test_manual_trade_refused_without_agent(ctrl_harness):
    result = ctrl_harness.controller.submit_manual_trade(
        SimpleNamespace()
    )
    assert result.success is False
    assert "not registered" in result.error.lower()
    ctrl_harness.oe.execute_signal.assert_not_called()


def test_manual_trade_refused_in_emergency(ctrl_harness):
    agent = _mock_agent()
    ctrl_harness.controller.register_agent(agent, StubGate(ok=True))
    ctrl_harness.controller.emergency_stop(reason="test")

    result = ctrl_harness.controller.submit_manual_trade(
        SimpleNamespace()
    )
    assert result.success is False
    assert "emergency" in result.error.lower()
    ctrl_harness.oe.execute_signal.assert_not_called()


def test_manual_trade_uses_agent_gate(ctrl_harness):
    agent = _mock_agent()
    gate = StubGate(ok=True)
    ctrl_harness.controller.register_agent(agent, gate)

    ctrl_harness.controller.submit_manual_trade(SimpleNamespace())

    ctrl_harness.oe.execute_signal.assert_called_once()
    kwargs = ctrl_harness.oe.execute_signal.call_args.kwargs
    assert kwargs["gate"] is gate

# ---------------- H-06: pending-order cancellation + flattening ----------------


def test_emergency_stop_cancels_bot_pending_orders(ctrl_harness):
    agent = _mock_agent()
    ctrl_harness.controller.register_agent(agent, StubGate(ok=True))

    from src.order_executor import CancelAllResult

    ctrl_harness.mt5.get_orders.side_effect = [
        [{"ticket": 100}, {"ticket": 101}],   # pre-cancel enumerate
        [],                                    # post-cancel verify
        [],                                    # _verify_flattened
    ]
    ctrl_harness.oe.cancel_all_pending.return_value = CancelAllResult(
        results=[
            SimpleNamespace(success=True, error_message=None),
            SimpleNamespace(success=True, error_message=None),
        ],
        query_ok=True,
    )

    result = ctrl_harness.controller.emergency_stop(reason="test")

    ctrl_harness.oe.cancel_all_pending.assert_called_once_with("EMERGENCY")
    assert result["cancel_attempted"] is True
    assert result["cancelled"] == 2
    assert result["cancel_failed"] == 0
    assert result["cancel_unconfirmed"] == []
    assert result["cancel_query_ok"] is True


def test_emergency_stop_verifies_via_fresh_query(ctrl_harness):
    """Post-cancel verification must re-query the broker, not trust
    cancel_all_pending's own report."""
    agent = _mock_agent()
    ctrl_harness.controller.register_agent(agent, StubGate(ok=True))

    from src.order_executor import CancelAllResult

    # Two initial orders, but one is STILL present after cancel.
    # cancel_all_pending claims both succeeded — the reconciler catches
    # the divergence.
    ctrl_harness.mt5.get_orders.side_effect = [
        [{"ticket": 100}, {"ticket": 101}],   # pre-cancel
        [{"ticket": 101}],                     # post-cancel: 101 survived
        [{"ticket": 101}],                     # _verify_flattened
    ]
    ctrl_harness.oe.cancel_all_pending.return_value = CancelAllResult(
        results=[
            SimpleNamespace(success=True, error_message=None),
            SimpleNamespace(success=True, error_message=None),
        ],
        query_ok=True,
    )

    result = ctrl_harness.controller.emergency_stop(reason="test")

    assert result["cancel_unconfirmed"] == [101]
    assert result["flattened"] is False
    assert result["pending_remaining"] == 1


def test_emergency_stop_flattened_true_when_all_verified(ctrl_harness):
    agent = _mock_agent()
    ctrl_harness.controller.register_agent(agent, StubGate(ok=True))

    # Clean state: nothing pending, nothing open.
    result = ctrl_harness.controller.emergency_stop(reason="test")

    assert result["flattened"] is True
    assert result["positions_remaining"] == 0
    assert result["pending_remaining"] == 0
    assert result["unresolved_exposure"] == {
        "positions_remaining": 0,
        "pending_remaining": 0,
        "unconfirmed_cancel_tickets": [],
    }


def test_emergency_stop_flattened_false_when_position_remains(ctrl_harness):
    agent = _mock_agent()
    ctrl_harness.controller.register_agent(agent, StubGate(ok=True))

    # Position still open after close_all attempt.
    ctrl_harness.mt5.get_positions.side_effect = [
        [{"ticket": 200}],   # _verify_flattened
    ]

    result = ctrl_harness.controller.emergency_stop(reason="test")

    assert result["flattened"] is False
    assert result["positions_remaining"] == 1


def test_emergency_stop_flattened_false_when_pending_remains(ctrl_harness):
    agent = _mock_agent()
    ctrl_harness.controller.register_agent(agent, StubGate(ok=True))

    # Pending order that cancel could not clear.
    ctrl_harness.mt5.get_orders.side_effect = [
        [{"ticket": 300}],   # pre-cancel
        [{"ticket": 300}],   # post-cancel: cancel failed
        [{"ticket": 300}],   # _verify_flattened
    ]

    from src.order_executor import CancelAllResult
    ctrl_harness.oe.cancel_all_pending.return_value = CancelAllResult(
        results=[SimpleNamespace(success=False, error_message="rejected")],
        query_ok=True,
    )

    result = ctrl_harness.controller.emergency_stop(reason="test")

    assert result["flattened"] is False
    assert result["pending_remaining"] == 1
    assert 300 in result["cancel_unconfirmed"]


def test_emergency_stop_handles_pre_cancel_query_failure(ctrl_harness):
    agent = _mock_agent()
    ctrl_harness.controller.register_agent(agent, StubGate(ok=True))

    ctrl_harness.mt5.get_orders.side_effect = RuntimeError("broker down")

    result = ctrl_harness.controller.emergency_stop(reason="test")

    assert result["cancel_attempted"] is False
    assert result["cancel_query_ok"] is False
    assert result["flattened"] is False
    # Close still ran despite the cancel failure.
    ctrl_harness.oe.close_all_positions.assert_called_once()


def test_emergency_stop_handles_post_cancel_query_failure(ctrl_harness):
    agent = _mock_agent()
    ctrl_harness.controller.register_agent(agent, StubGate(ok=True))

    from src.order_executor import CancelAllResult

    call_seq = iter([
        [{"ticket": 400}],   # pre-cancel OK
        RuntimeError("timeout"),   # post-cancel query fails
    ])

    def _get_orders():
        v = next(call_seq)
        if isinstance(v, Exception):
            raise v
        return v

    ctrl_harness.mt5.get_orders.side_effect = _get_orders
    ctrl_harness.oe.cancel_all_pending.return_value = CancelAllResult(
        results=[SimpleNamespace(success=True, error_message=None)],
        query_ok=True,
    )

    result = ctrl_harness.controller.emergency_stop(reason="test")

    # Unknown post-cancel state → treat all initial tickets as
    # unconfirmed; flattened must be False.
    assert result["cancel_query_ok"] is False
    assert result["cancel_unconfirmed"] == [400]
    assert result["flattened"] is False


def test_emergency_stop_preserves_existing_response_fields(ctrl_harness):
    """H-08 back-compat: all fields that existed before H-06 must
    still be present."""
    agent = _mock_agent()
    ctrl_harness.controller.register_agent(agent, StubGate(ok=True))

    result = ctrl_harness.controller.emergency_stop(reason="test")

    for key in (
        "state", "ok", "kill_switch_written", "kill_switch_verified",
        "kill_switch", "paused", "close_attempted", "closed", "failed",
        "closed_count", "failed_count", "error",
    ):
        assert key in result, f"missing back-compat field: {key}"