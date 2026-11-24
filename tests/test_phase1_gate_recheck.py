"""Phase 1 / C-03 regression: gate and LIVE must be re-checked
immediately before broker submission."""

from types import SimpleNamespace

import pytest

from src.order_executor import order_executor
from src.signal_validator import ValidationResult


class _Gate:
    def __init__(self, results):
        # results: list of SimpleNamespace(ok, reason) returned in order
        self._results = list(results)
        self.calls = 0

    def check(self):
        self.calls += 1
        return self._results.pop(0) if self._results else SimpleNamespace(
            ok=True, reason="", token="stub-token"
        )


def _signal():
    class _Dir:
        value = "BUY"
    class _Strat:
        value = "BOS_OB"
    return SimpleNamespace(
        symbol="GOLD", direction=_Dir(), strategy=_Strat(),
        entry_price=2000.0, sl_price=1990.0, tp_price=2020.0,
        risk_percent=1.0, risk_reward=2.0, lot_size=0.0,
        fingerprint="", confirmations=[], timestamp=None,
        entry_tf="m5",
    )


def _stub_pipeline(monkeypatch, sent):
    monkeypatch.setattr(
        "src.order_executor.mt5_connector.is_connected", lambda: True,
    )
    # §1d now reads mt5_connector.get_tick(), not get_symbol_info().
    monkeypatch.setattr(
        "src.order_executor.mt5_connector.get_tick",
        lambda: {"bid": 2004.5, "ask": 2005.0, "spread": 0.5},
    )
    monkeypatch.setattr(
        "src.order_executor.signal_validator.validate_signal",
        lambda *_, **__: ValidationResult(
            valid=True, reason="", checks={}
        ),
    )
    monkeypatch.setattr(
        "src.order_executor.risk_manager.calculate_position_size",
        lambda *_, **__: SimpleNamespace(lot_size=0.1, reason=""),
    )

    def _send(_req, gate_token=None):
        sent["called"] = True
        sent["token"] = gate_token
        return SimpleNamespace(success=True, order_id=1)

    monkeypatch.setattr(
        "src.order_executor.mt5_connector.send_order", _send,
    )
    monkeypatch.setattr(order_executor, "_save_signal", lambda *a, **kw: 1)
    monkeypatch.setattr(
        order_executor, "_save_order_request", lambda *a, **kw: 1,
    )
    monkeypatch.setattr(
        order_executor, "_mark_request_status", lambda *a, **kw: None,
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


def test_gate_recheck_blocks_submission(monkeypatch):
    gate = _Gate([
        SimpleNamespace(ok=True, reason="", token="stub-token"),
        SimpleNamespace(ok=False, reason="kill-switch activated", token=""),
    ])
    sent = {"called": False, "token": None}
    _stub_pipeline(monkeypatch, sent)

    result = order_executor.execute_signal(
        _signal(), market_entry=True, gate=gate,
    )

    assert result.success is False
    assert "Entry gate re-check" in (result.error or "")
    assert sent["called"] is False
    assert gate.calls == 2


def test_gate_recheck_passes_token_when_ok(monkeypatch):
    gate = _Gate([
        SimpleNamespace(ok=True, reason="", token="stub-token"),
        SimpleNamespace(ok=True, reason="", token="final-token"),
    ])
    sent = {"called": False, "token": None}
    _stub_pipeline(monkeypatch, sent)

    result = order_executor.execute_signal(
        _signal(), market_entry=True, gate=gate,
    )

    assert result.success is True
    assert sent["called"] is True
    assert sent["token"] == "final-token"
    assert gate.calls == 2