"""C-03 regression tests: full LIVE authorization at the submission
boundary.

Verifies every required condition independently at the boundary that
actually calls ``mt5.order_send``:
  * live_trading_enabled
  * live_trading_confirmation
  * live_allowed_account membership
  * server match
  * demo_only bypass prevention
  * kill-switch readable & inactive
  * no unresolved reconciliation
  * single-use token (missing / consumed)
  * closing actions bypass opening authorization

All MT5 calls are mocked. No terminal, no account, no orders.
"""

from types import SimpleNamespace

import pytest

from src.mt5_connector import (
    MT5Connector,
    AccountInfo,
    OrderResult,
    mt5,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def live_config(monkeypatch):
    """LIVE mode fully authorized in configuration.

    `config.trading`, `config.mt5`, `config.safety` are rebuilt as
    plain SimpleNamespace objects at the module level of every module
    that reads them (mt5_connector, risk_manager, order_executor).

    Why not patch `config.trading.mode` directly? Because
    `config.trading` returns a fresh sub-config on each access in this
    codebase. Patching `mode` on one instance does nothing for the
    next access, so `_assert_live_opening_authorized` would still see
    the original (non-LIVE) mode.

    Replacing the module-level `config` binding with a stable
    SimpleNamespace guarantees that every `config.<sub>` read returns
    the same fake across the whole call path.
    """
    from src.config_loader import config as real_config
    from src import mt5_connector as mc
    from src import risk_manager as rm
    from src import order_executor as oe
    from types import SimpleNamespace

    real_trading = real_config.trading
    real_mt5 = real_config.mt5

    fake = SimpleNamespace(
        trading=SimpleNamespace(
            mode="LIVE",
            magic_number=getattr(real_trading, "magic_number", 0),
        ),
        mt5=SimpleNamespace(
            server="XM-REAL",
            path=getattr(real_mt5, "path", None),
            login=getattr(real_mt5, "login", None),
            password=getattr(real_mt5, "password", None),
            timeout=getattr(real_mt5, "timeout", 0),
            retry_attempts=getattr(real_mt5, "retry_attempts", 0),
            retry_delay=getattr(real_mt5, "retry_delay", 0),
        ),
        safety=SimpleNamespace(demo_only=False),
        live_trading_enabled=True,
        live_trading_confirmation="I-CONFIRM",
        live_allowed_account="12345",
    )

    monkeypatch.setattr(mc, "config", fake)
    monkeypatch.setattr(rm, "config", fake)
    monkeypatch.setattr(oe, "config", fake)

    yield fake


@pytest.fixture
def safe_externals(monkeypatch):
    """Patch the two external safety getters to 'all clear'."""
    from src.risk_manager import risk_manager
    from src.order_executor import order_executor

    monkeypatch.setattr(
        risk_manager, "get_kill_switch_state_strict", lambda: False,
    )
    monkeypatch.setattr(
        order_executor, "has_unresolved_reconciliation", lambda: False,
    )
    yield risk_manager, order_executor


def _make_connector(login=12345, server="XM-REAL", is_demo=False):
    c = MT5Connector()
    c._account_info = AccountInfo(
        login=login,
        server=server,
        name="test",
        company="XM",
        currency="USD",
        balance=10000.0,
        equity=10000.0,
        margin=0.0,
        free_margin=10000.0,
        margin_level=0.0,
        leverage=100,
        trade_mode="DEMO" if is_demo else "REAL",
        trade_allowed=True,
        trade_expert=True,
        margin_mode="HEDGING",
        is_demo=is_demo,
    )
    return c


def _opening_request():
    return {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": "XAUUSD",
        "type": mt5.ORDER_TYPE_BUY,
        "volume": 0.1,
    }


# ---------------------------------------------------------------------------
# 1. LIVE_TRADING_ENABLED must be True
# ---------------------------------------------------------------------------

def test_live_open_blocked_when_trading_disabled(
    monkeypatch, live_config, safe_externals
):
    monkeypatch.setattr(live_config, "live_trading_enabled", False,
                        raising=False)
    c = _make_connector()
    reason = c._assert_live_opening_authorized()
    assert reason is not None
    assert "LIVE_TRADING_ENABLED" in reason


# ---------------------------------------------------------------------------
# 2. LIVE_TRADING_CONFIRMATION must be present
# ---------------------------------------------------------------------------

def test_live_open_blocked_when_confirmation_missing(
    monkeypatch, live_config, safe_externals
):
    monkeypatch.setattr(live_config, "live_trading_confirmation", "",
                        raising=False)
    c = _make_connector()
    reason = c._assert_live_opening_authorized()
    assert reason is not None
    assert "LIVE_TRADING_CONFIRMATION" in reason


# ---------------------------------------------------------------------------
# 3. Account must be in LIVE_ALLOWED_ACCOUNT
# ---------------------------------------------------------------------------

def test_live_open_blocked_when_account_not_allowlisted(
    monkeypatch, live_config, safe_externals
):
    monkeypatch.setattr(live_config, "live_allowed_account", "99999",
                        raising=False)
    c = _make_connector(login=12345)
    reason = c._assert_live_opening_authorized()
    assert reason is not None
    assert "LIVE_ALLOWED_ACCOUNT" in reason


# ---------------------------------------------------------------------------
# 4. Connected server must match configured server
# ---------------------------------------------------------------------------

def test_live_open_blocked_when_server_mismatch(
    monkeypatch, live_config, safe_externals
):
    c = _make_connector(server="XM-DIFFERENT")
    reason = c._assert_live_opening_authorized()
    assert reason is not None
    assert "server" in reason.lower()


# ---------------------------------------------------------------------------
# 5. DEMO-only must not be bypassed
# ---------------------------------------------------------------------------

def test_live_open_blocked_when_demo_only(
    monkeypatch, live_config, safe_externals
):
    monkeypatch.setattr(live_config.safety, "demo_only", True,
                        raising=False)
    c = _make_connector()
    reason = c._assert_live_opening_authorized()
    assert reason is not None
    assert "DEMO-only" in reason or "demo_only" in reason.lower()


# ---------------------------------------------------------------------------
# 6. Kill switch active → block
# ---------------------------------------------------------------------------

def test_live_open_blocked_when_kill_switch_active(
    monkeypatch, live_config, safe_externals
):
    rm, _ = safe_externals
    monkeypatch.setattr(
        rm, "get_kill_switch_state_strict", lambda: True,
    )
    c = _make_connector()
    reason = c._assert_live_opening_authorized()
    assert reason is not None
    assert "Kill switch" in reason


# ---------------------------------------------------------------------------
# 7. Kill switch state unreadable → block (fail closed)
# ---------------------------------------------------------------------------

def test_live_open_blocked_when_kill_switch_unreadable(
    monkeypatch, live_config, safe_externals
):
    rm, _ = safe_externals
    monkeypatch.setattr(
        rm, "get_kill_switch_state_strict", lambda: None,
    )
    c = _make_connector()
    reason = c._assert_live_opening_authorized()
    assert reason is not None
    assert "unreadable" in reason.lower()


# ---------------------------------------------------------------------------
# 8. Unresolved reconciliation → block
# ---------------------------------------------------------------------------

def test_live_open_blocked_when_unresolved_reconciliation(
    monkeypatch, live_config, safe_externals
):
    _, oe = safe_externals
    monkeypatch.setattr(
        oe, "has_unresolved_reconciliation", lambda: True,
    )
    c = _make_connector()
    reason = c._assert_live_opening_authorized()
    assert reason is not None
    assert "reconciliation" in reason.lower()


# ---------------------------------------------------------------------------
# 9. Missing token blocks the opening submission at the boundary
# ---------------------------------------------------------------------------

def test_live_open_blocked_when_no_token(
    monkeypatch, live_config, safe_externals
):
    c = _make_connector()
    monkeypatch.setattr(c, "_trading_precheck", lambda: None)

    def no_send(*args, **kwargs):
        raise AssertionError("order_send must not be reached")

    monkeypatch.setattr(mt5, "order_send", no_send)

    result = c._send_order_request(_opening_request(), gate_token=None)
    assert result.success is False
    assert "token" in (result.error_message or "").lower()


# ---------------------------------------------------------------------------
# 10. Already-consumed token blocks the opening submission
# ---------------------------------------------------------------------------

def test_live_open_blocked_when_token_already_consumed(
    monkeypatch, live_config, safe_externals
):
    from src import gate_token as gt

    token = gt.issue()
    assert gt.consume(token) is True  # pre-consume

    c = _make_connector()
    monkeypatch.setattr(c, "_trading_precheck", lambda: None)

    def no_send(*args, **kwargs):
        raise AssertionError("order_send must not be reached")

    monkeypatch.setattr(mt5, "order_send", no_send)

    result = c._send_order_request(_opening_request(), gate_token=token)
    assert result.success is False
    assert "token" in (result.error_message or "").lower()


# ---------------------------------------------------------------------------
# 11. Closing actions bypass opening authorization entirely
# ---------------------------------------------------------------------------

def test_close_bypasses_opening_authorization(
    monkeypatch, live_config, safe_externals
):
    c = _make_connector()
    monkeypatch.setattr(c, "_trading_precheck", lambda: None)
    monkeypatch.setattr(
        c, "_get_filling_mode",
        lambda symbol: mt5.ORDER_FILLING_IOC,
    )

    calls = []

    def spy():
        calls.append(True)
        return "SHOULD_NOT_BE_CALLED"

    monkeypatch.setattr(c, "_assert_live_opening_authorized", spy)

    tick = SimpleNamespace(bid=2000.0, ask=2000.5)
    monkeypatch.setattr(mt5, "symbol_info_tick", lambda symbol: tick)

    def fake_send(payload):
        return SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE,
            comment="closed",
            order=999,
            deal=888,
            volume=0.1,
            price=2000.0,
        )

    monkeypatch.setattr(mt5, "order_send", fake_send)

    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": "XAUUSD",
        "type": mt5.ORDER_TYPE_SELL,
        "volume": 0.1,
        "position": 111,   # closing
    }
    result = c._send_order_request(close_request, gate_token=None)
    assert calls == [], (
        "Closing action must not invoke opening authorization"
    )
    assert result.success is True