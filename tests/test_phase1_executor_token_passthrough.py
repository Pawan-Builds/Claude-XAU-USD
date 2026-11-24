"""Phase 1: verify execute_signal threads the final gate token to
send_order."""

from types import SimpleNamespace

from src.order_executor import order_executor
from src.signal_validator import ValidationResult
from src import gate_token


def _signal(direction="BUY"):
    class _D:
        value = direction
    class _S:
        value = "BOS_OB"
    return SimpleNamespace(
        symbol="GOLD", direction=_D(), strategy=_S(),
        entry_price=2000.0, sl_price=1990.0, tp_price=2020.0,
        risk_percent=1.0, risk_reward=2.0, lot_size=0.0,
        fingerprint="", confirmations=[], timestamp=None,
        entry_tf="m5",
    )


class _ConfiguredGate:
    def __init__(self, token):
        self._token = token
        self._calls = 0

    def check(self):
        self._calls += 1
        return SimpleNamespace(ok=True, reason="", token=self._token)


def test_executor_passes_final_gate_token_to_send_order(monkeypatch):
    gate_token.reset_for_tests()
    tok = gate_token.issue()

    captured = {}

    def _send(_req, gate_token=None):
        captured["token"] = gate_token
        return SimpleNamespace(success=True, order_id=1)

    monkeypatch.setattr(
        "src.order_executor.mt5_connector.is_connected", lambda: True,
    )
    # §1d reads get_tick(), not get_symbol_info().
    monkeypatch.setattr(
        "src.order_executor.mt5_connector.get_tick",
        lambda: {"bid": 2004.5, "ask": 2005.0, "spread": 0.5},
    )
    monkeypatch.setattr(
        "src.order_executor.signal_validator.validate_signal",
        lambda *_, **__: ValidationResult(valid=True, reason="", checks={}),
    )
    monkeypatch.setattr(
        "src.order_executor.risk_manager.calculate_position_size",
        lambda *_, **__: SimpleNamespace(lot_size=0.1, reason=""),
    )
    monkeypatch.setattr(order_executor, "_save_signal", lambda *a, **kw: 1)
    monkeypatch.setattr(
        order_executor, "_save_order_request", lambda *a, **kw: 1,
    )
    monkeypatch.setattr(
        order_executor, "_update_order_request", lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        order_executor, "_mark_request_uncertain", lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        order_executor, "_finalise_market_entry", lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        order_executor, "_is_duplicate", lambda *a, **kw: False,
    )
    monkeypatch.setattr(
        order_executor, "_prepare_order_request",
        lambda *a, **kw: {
            "symbol": "GOLD", "type": 1, "volume": 0.1,
            "sl": 1990.0, "tp": 2020.0, "comment": "SMC_TEST",
            "fingerprint": "x",
        },
    )
    monkeypatch.setattr(
        "src.order_executor.mt5_connector.send_order", _send,
    )

    order_executor.execute_signal(
        _signal("BUY"), market_entry=True, gate=_ConfiguredGate(tok),
    )

    assert captured["token"] == tok


def test_executor_passes_none_when_gate_has_no_token(monkeypatch):
    """Old gate stubs (no .token attr) must still reach the broker —
    the connector enforces the token at its boundary, not the executor."""
    captured = {"called": False, "token": "sentinel"}

    def _send(_req, gate_token=None):
        captured["called"] = True
        captured["token"] = gate_token
        return SimpleNamespace(success=True, order_id=1)

    monkeypatch.setattr(
        "src.order_executor.mt5_connector.is_connected", lambda: True,
    )
    monkeypatch.setattr(
        "src.order_executor.mt5_connector.get_tick",
        lambda: {"bid": 2004.5, "ask": 2005.0, "spread": 0.5},
    )
    monkeypatch.setattr(
        "src.order_executor.signal_validator.validate_signal",
        lambda *_, **__: ValidationResult(valid=True, reason="", checks={}),
    )
    monkeypatch.setattr(
        "src.order_executor.risk_manager.calculate_position_size",
        lambda *_, **__: SimpleNamespace(lot_size=0.1, reason=""),
    )
    monkeypatch.setattr(order_executor, "_save_signal", lambda *a, **kw: 1)
    monkeypatch.setattr(
        order_executor, "_save_order_request", lambda *a, **kw: 1,
    )
    monkeypatch.setattr(
        order_executor, "_update_order_request", lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        order_executor, "_mark_request_uncertain", lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        order_executor, "_finalise_market_entry", lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        order_executor, "_is_duplicate", lambda *a, **kw: False,
    )
    monkeypatch.setattr(
        order_executor, "_prepare_order_request",
        lambda *a, **kw: {
            "symbol": "GOLD", "type": 1, "volume": 0.1,
            "sl": 1990.0, "tp": 2020.0, "comment": "SMC_TEST",
            "fingerprint": "x",
        },
    )
    monkeypatch.setattr(
        "src.order_executor.mt5_connector.send_order", _send,
    )

    class _TokenlessGate:
        def check(self):
            return SimpleNamespace(ok=True, reason="")

    order_executor.execute_signal(
        _signal("BUY"), market_entry=True, gate=_TokenlessGate(),
    )

    assert captured["called"] is True
    assert captured["token"] is None