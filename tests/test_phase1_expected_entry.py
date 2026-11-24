"""Phase 1 / H-02 regression: market-entry sizing uses the expected
executable quote from mt5_connector.get_tick() — ask for BUY, bid for
SELL. Pending entries size against the plan entry."""

from types import SimpleNamespace

import pytest

from src.order_executor import order_executor
from src.signal_validator import ValidationResult


def _signal(direction="BUY"):
    class _Dir:
        value = direction
    class _Strat:
        value = "BOS_OB"
    return SimpleNamespace(
        symbol="GOLD", direction=_Dir(), strategy=_Strat(),
        entry_price=2000.0, sl_price=1990.0, tp_price=2020.0,
        risk_percent=1.0, risk_reward=2.0, lot_size=0.0,
        fingerprint="", confirmations=[], timestamp=None,
        entry_tf="m5",
    )


class _Gate:
    def check(self):
        return SimpleNamespace(ok=True, reason="", token="stub-token")


def _patch_send_pipeline(monkeypatch, captured):
    def _size(entry, sl, direction, risk):
        captured["entry"] = entry
        return SimpleNamespace(lot_size=0.1, reason="")

    monkeypatch.setattr(
        "src.order_executor.mt5_connector.is_connected", lambda: True,
    )
    monkeypatch.setattr(
        "src.order_executor.signal_validator.validate_signal",
        lambda *_, **__: ValidationResult(valid=True, reason="", checks={}),
    )
    monkeypatch.setattr(
        "src.order_executor.risk_manager.calculate_position_size", _size,
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
        "src.order_executor.mt5_connector.send_order",
        lambda *_, **kw: SimpleNamespace(success=True, order_id=1),
    )


def test_market_buy_sizes_against_ask(monkeypatch):
    captured = {}
    _patch_send_pipeline(monkeypatch, captured)
    # §1d reads get_tick() now — ask=2005.0 is the expected fill for a
    # market BUY.
    monkeypatch.setattr(
        "src.order_executor.mt5_connector.get_tick",
        lambda: {"bid": 2004.5, "ask": 2005.0, "spread": 0.5},
    )
    order_executor.execute_signal(
        _signal("BUY"), market_entry=True, gate=_Gate(),
    )
    assert captured["entry"] == 2005.0


def test_market_sell_sizes_against_bid(monkeypatch):
    captured = {}
    _patch_send_pipeline(monkeypatch, captured)
    monkeypatch.setattr(
        "src.order_executor.mt5_connector.get_tick",
        lambda: {"bid": 2004.5, "ask": 2005.0, "spread": 0.5},
    )
    order_executor.execute_signal(
        _signal("SELL"), market_entry=True, gate=_Gate(),
    )
    assert captured["entry"] == 2004.5


def test_pending_sizes_against_plan_entry(monkeypatch):
    captured = {}
    _patch_send_pipeline(monkeypatch, captured)
    # Market price drift; pending sizing must ignore get_tick().
    monkeypatch.setattr(
        "src.order_executor.mt5_connector.get_tick",
        lambda: {"bid": 2009.5, "ask": 2010.0, "spread": 0.5},
    )
    order_executor.execute_signal(
        _signal("BUY"), market_entry=False, gate=_Gate(),
    )
    assert captured["entry"] == 2000.0


def test_missing_gate_blocks_entry(monkeypatch):
    monkeypatch.setattr(
        "src.order_executor.mt5_connector.is_connected", lambda: True,
    )
    sent = {"called": False}

    def _send(*_, **__):
        sent["called"] = True
        return SimpleNamespace(success=True, order_id=1)

    monkeypatch.setattr(
        "src.order_executor.mt5_connector.send_order", _send,
    )

    result = order_executor.execute_signal(
        _signal("BUY"), market_entry=True, gate=None,
    )
    assert result.success is False
    assert "Entry gate not supplied" in (result.error or "")
    assert sent["called"] is False


def test_missing_quote_blocks_market_entry(monkeypatch):
    """§1d fails closed when get_tick() returns None."""
    captured = {}
    _patch_send_pipeline(monkeypatch, captured)
    monkeypatch.setattr(
        "src.order_executor.mt5_connector.get_tick", lambda: None,
    )
    result = order_executor.execute_signal(
        _signal("BUY"), market_entry=True, gate=_Gate(),
    )
    assert result.success is False
    assert "quote" in (result.error or "").lower()
    assert "entry" not in captured