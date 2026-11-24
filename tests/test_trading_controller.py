"""Tests for the central TradingController (F-03)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.conftest import StubGate


@pytest.fixture
def ctrl_harness(monkeypatch):
    from src import trading_controller as tc_module

    rm = MagicMock(name="risk_manager")
    rm.is_kill_switch_active.return_value = False
    rm.set_kill_switch.return_value = None

    oe = MagicMock(name="order_executor")
    oe.close_all_positions.return_value = []
    oe.close_position.return_value = SimpleNamespace(
        success=True, error_message=None
    )
    oe.execute_signal.return_value = SimpleNamespace(
        success=True, error=None
    )

    mt5 = MagicMock(name="mt5_connector")
    mt5.is_connected.return_value = True

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
        controller=controller,
        rm=rm,
        oe=oe,
        mt5=mt5,
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
    agent = _mock_agent()
    ctrl_harness.controller.register_agent(agent, StubGate(ok=True))
    ctrl_harness.oe.close_all_positions.return_value = [
        SimpleNamespace(success=True, error_message=None),
        SimpleNamespace(success=False, error_message="rejected"),
    ]

    result = ctrl_harness.controller.emergency_stop(reason="test")

    assert result["ok"] is True
    assert result["state"] == "emergency"
    assert result["closed"] == 1
    assert result["failed"] == 1
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
    # Emergency must still execute close and set state.
    assert result["ok"] is True
    assert result["state"] == "emergency"
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