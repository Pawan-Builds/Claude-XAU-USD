"""Phase 1 / C-03 regression: LIVE boundary + HMAC gate token.

SAFETY: no test in this module may ever reach mt5.order_send.
"""

from types import SimpleNamespace

import MetaTrader5 as mt5
import pytest

from src.mt5_connector import mt5_connector
from src import gate_token


def _fake_config(mode="LIVE", live_enabled=False, demo_only=False):
    trading = SimpleNamespace(
        mode=mode, symbol="GOLD", magic_number=123456789,
        risk_percent=1.0, min_risk_reward=2.0, max_positions=3,
        max_daily_trades=20, max_daily_loss_percent=5.0,
        max_drawdown_percent=10.0, max_consecutive_losses=5,
        max_holding_hours=48, entry_order_type="limit",
        pending_expiry_hours=4, max_pending_orders=5,
        pending_orders_on_shutdown="keep",
    )
    safety = SimpleNamespace(
        demo_only=demo_only, verify_demo_on_startup=True,
        kill_switch_enabled=True, max_errors_before_pause=10,
        health_check_interval=60, trading_mode=mode,
    )
    return SimpleNamespace(
        trading=trading, safety=safety,
        live_trading_enabled=live_enabled,
        mt5=SimpleNamespace(
            login=12345, server="XM.LIVE", password="",
            path="", timeout=60000,
            retry_attempts=3, retry_delay=5,
        ),
    )


def _fake_account():
    return SimpleNamespace(
        login=12345, server="XM.LIVE", is_demo=False,
        trade_mode="REAL", trade_allowed=True, trade_expert=True,
        balance=10000, equity=10000, margin=0, free_margin=10000,
        margin_level=0, leverage=100, currency="USD",
        name="test", company="XM", margin_mode="RETAIL_HEDGING",
    )


@pytest.fixture(autouse=True)
def _reset_tokens():
    gate_token.reset_for_tests()
    yield
    gate_token.reset_for_tests()


def _prep_connector(monkeypatch, cfg, account):
    monkeypatch.setattr("src.mt5_connector.config", cfg)
    monkeypatch.setattr(mt5_connector, "mt5_config", cfg.mt5)
    monkeypatch.setattr(mt5_connector, "trading_config", cfg.trading)
    monkeypatch.setattr(mt5_connector, "_initialized", True)
    monkeypatch.setattr(mt5_connector, "_symbol_name", "GOLD")
    monkeypatch.setattr(mt5_connector, "update_account_info", lambda: True)
    monkeypatch.setattr(mt5_connector, "_account_info", account)
    monkeypatch.setattr(mt5_connector, "_trading_precheck", lambda: None)

    def _no_send(*_a, **_kw):
        raise AssertionError("mt5.order_send must not be reached in tests")
    monkeypatch.setattr("src.mt5_connector.mt5.order_send", _no_send)
    monkeypatch.setattr(
        "src.mt5_connector.mt5.symbol_info_tick",
        lambda *_a, **_kw: None,
    )


# ---- LIVE boundary ----

def test_live_market_open_blocked_when_disabled(monkeypatch):
    _prep_connector(monkeypatch, _fake_config(), _fake_account())
    # Even with a valid token, the LIVE boundary must block first.
    tok = gate_token.issue()
    result = mt5_connector._send_order_request(
        {
            "action": mt5.TRADE_ACTION_DEAL,
            "type": mt5.ORDER_TYPE_BUY,
            "volume": 0.1, "symbol": "GOLD",
        },
        gate_token=tok,
    )
    assert result.success is False
    assert "LIVE_TRADING_ENABLED" in (result.error_message or "")


def test_live_pending_open_blocked_when_disabled(monkeypatch):
    _prep_connector(monkeypatch, _fake_config(), _fake_account())
    tok = gate_token.issue()
    result = mt5_connector._send_order_request(
        {
            "action": mt5.TRADE_ACTION_PENDING,
            "type": mt5.ORDER_TYPE_BUY_LIMIT,
            "price": 2000.0, "volume": 0.1, "symbol": "GOLD",
        },
        gate_token=tok,
    )
    assert result.success is False
    assert "LIVE_TRADING_ENABLED" in (result.error_message or "")


def test_live_close_not_blocked_when_disabled(monkeypatch):
    _prep_connector(monkeypatch, _fake_config(), _fake_account())
    result = mt5_connector._send_order_request({
        "action": mt5.TRADE_ACTION_DEAL,
        "position": 999,
        "type": mt5.ORDER_TYPE_SELL,
        "volume": 0.1, "symbol": "GOLD",
    })
    assert "LIVE_TRADING_ENABLED" not in (result.error_message or "")
    assert "gate token" not in (result.error_message or "").lower()


# ---- Gate token ----

def test_demo_open_without_token_blocked_by_token_check(monkeypatch):
    acct = _fake_account()
    acct.is_demo = True
    acct.trade_mode = "DEMO"
    cfg = _fake_config(mode="DEMO", live_enabled=False, demo_only=True)
    _prep_connector(monkeypatch, cfg, acct)
    result = mt5_connector._send_order_request({
        "action": mt5.TRADE_ACTION_DEAL,
        "type": mt5.ORDER_TYPE_BUY,
        "volume": 0.1, "symbol": "GOLD",
    })
    assert "LIVE_TRADING_ENABLED" not in (result.error_message or "")
    assert "gate token" in (result.error_message or "").lower()


def test_demo_open_with_valid_token_reaches_broker_layer(monkeypatch):
    acct = _fake_account()
    acct.is_demo = True
    acct.trade_mode = "DEMO"
    cfg = _fake_config(mode="DEMO", live_enabled=False, demo_only=True)
    _prep_connector(monkeypatch, cfg, acct)
    tok = gate_token.issue()
    result = mt5_connector._send_order_request(
        {
            "action": mt5.TRADE_ACTION_DEAL,
            "type": mt5.ORDER_TYPE_BUY,
            "volume": 0.1, "symbol": "GOLD",
        },
        gate_token=tok,
    )
    # Quote is None → blocked with a quote error, NOT a token error
    assert "LIVE_TRADING_ENABLED" not in (result.error_message or "")
    assert "gate token" not in (result.error_message or "").lower()


def test_token_is_single_use(monkeypatch):
    acct = _fake_account()
    acct.is_demo = True
    acct.trade_mode = "DEMO"
    cfg = _fake_config(mode="DEMO", live_enabled=False, demo_only=True)
    _prep_connector(monkeypatch, cfg, acct)
    tok = gate_token.issue()
    mt5_connector._send_order_request(
        {
            "action": mt5.TRADE_ACTION_DEAL,
            "type": mt5.ORDER_TYPE_BUY,
            "volume": 0.1, "symbol": "GOLD",
        },
        gate_token=tok,
    )
    # Second use of the same token must be rejected as consumed
    result = mt5_connector._send_order_request(
        {
            "action": mt5.TRADE_ACTION_DEAL,
            "type": mt5.ORDER_TYPE_BUY,
            "volume": 0.1, "symbol": "GOLD",
        },
        gate_token=tok,
    )
    assert "gate token" in (result.error_message or "").lower()