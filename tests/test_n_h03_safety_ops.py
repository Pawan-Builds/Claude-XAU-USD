"""Focused N-H03: connector precheck must not block safety operations."""
from unittest.mock import MagicMock, patch

import pytest

from src.mt5_connector import MT5Connector, mt5


@pytest.fixture
def terminal_up():
    """Patch mt5.terminal_info to indicate a running terminal."""
    with patch.object(mt5, "terminal_info", return_value=MagicMock()):
        yield


def _make_connector(
    *,
    initialized=True,
    account_login=12345,
    account_server="TEST",
    account_is_demo=True,
    account_trade_allowed=True,
    account_trade_expert=True,
    update_ok=True,
):
    c = MT5Connector.__new__(MT5Connector)
    c._initialized = initialized
    c.mt5_config = MagicMock(login=12345, server="TEST")
    c.trading_config = MagicMock(magic_number=1)
    c._symbol_name = "GOLD"
    c.state = None
    account = MagicMock(
        login=account_login,
        server=account_server,
        is_demo=account_is_demo,
        trade_allowed=account_trade_allowed,
        trade_expert=account_trade_expert,
        trade_mode="DEMO" if account_is_demo else "REAL",
    )
    c._account_info = account
    c.update_account_info = MagicMock(return_value=update_ok)
    return c


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_classify_market_open_is_increase():
    c = _make_connector()
    req = {"action": mt5.TRADE_ACTION_DEAL, "type": mt5.ORDER_TYPE_BUY}
    assert c._classify_operation(req) == "increase"


def test_classify_pending_is_increase():
    c = _make_connector()
    req = {
        "action": mt5.TRADE_ACTION_PENDING,
        "type": mt5.ORDER_TYPE_BUY_LIMIT,
    }
    assert c._classify_operation(req) == "increase"


def test_classify_close_is_reduce():
    c = _make_connector()
    with patch.object(
        c,
        "_get_live_position_info",
        return_value={
            "ticket": 999, "type": "BUY", "volume": 0.1,
            "magic": 1, "symbol": "GOLD",
        },
    ):
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "type": mt5.ORDER_TYPE_SELL,
            "position": 999,
            "volume": 0.1,
        }
        assert c._classify_operation(req) == "reduce"


def test_classify_same_side_with_position_is_unknown():
    c = _make_connector()
    with patch.object(
        c,
        "_get_live_position_info",
        return_value={
            "ticket": 999, "type": "BUY", "volume": 0.1,
            "magic": 1, "symbol": "GOLD",
        },
    ):
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "type": mt5.ORDER_TYPE_BUY,
            "position": 999,
            "volume": 0.1,
        }
        assert c._classify_operation(req) == "unknown"


def test_classify_oversized_close_is_unknown():
    c = _make_connector()
    with patch.object(
        c,
        "_get_live_position_info",
        return_value={
            "ticket": 999, "type": "BUY", "volume": 0.1,
            "magic": 1, "symbol": "GOLD",
        },
    ):
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "type": mt5.ORDER_TYPE_SELL,
            "position": 999,
            "volume": 0.5,   # > position volume
        }
        assert c._classify_operation(req) == "unknown"


def test_classify_sltp_is_protect():
    c = _make_connector()
    req = {"action": mt5.TRADE_ACTION_SLTP, "position": 999}
    assert c._classify_operation(req) == "protect"


def test_classify_remove_is_reduce():
    c = _make_connector()
    req = {"action": mt5.TRADE_ACTION_REMOVE, "order": 999}
    assert c._classify_operation(req) == "reduce"


def test_classify_pending_with_position_is_unknown():
    c = _make_connector()
    req = {
        "action": mt5.TRADE_ACTION_PENDING,
        "type": mt5.ORDER_TYPE_BUY_LIMIT,
        "position": 999,
    }
    assert c._classify_operation(req) == "unknown"


def test_classify_unknown_action_is_unknown():
    c = _make_connector()
    assert c._classify_operation({"action": 99999}) == "unknown"


# ---------------------------------------------------------------------------
# _safety_precheck must NOT block on entry-only restrictions
# ---------------------------------------------------------------------------

def test_demo_config_real_account_does_not_block_safety_op(terminal_up):
    c = _make_connector(account_is_demo=False)
    assert c._safety_precheck() is None


def test_demo_only_flag_does_not_block_safety_op(terminal_up):
    c = _make_connector(account_is_demo=False)
    with patch("src.mt5_connector.config") as mock_config:
        mock_config.safety.demo_only = True
        mock_config.trading.mode = "DEMO"
        assert c._safety_precheck() is None


def test_trade_allowed_false_does_not_block_safety_op(terminal_up):
    c = _make_connector(account_trade_allowed=False)
    assert c._safety_precheck() is None


def test_trade_expert_false_does_not_block_safety_op(terminal_up):
    c = _make_connector(account_trade_expert=False)
    assert c._safety_precheck() is None


# ---------------------------------------------------------------------------
# _safety_precheck must still enforce identity / connectivity
# ---------------------------------------------------------------------------

def test_login_mismatch_blocks(terminal_up):
    c = _make_connector(account_login=99999)
    reason = c._safety_precheck()
    assert reason is not None
    assert "login" in reason.lower() or "mismatch" in reason.lower()


def test_server_mismatch_blocks(terminal_up):
    c = _make_connector(account_server="OTHER")
    reason = c._safety_precheck()
    assert reason is not None
    assert "server" in reason.lower() or "mismatch" in reason.lower()


def test_update_failure_blocks(terminal_up):
    c = _make_connector(update_ok=False)
    reason = c._safety_precheck()
    assert reason is not None
    assert "refresh" in reason.lower() or "account" in reason.lower()


def test_uninitialized_blocks(terminal_up):
    c = _make_connector(initialized=False)
    reason = c._safety_precheck()
    assert reason is not None
    assert "initialized" in reason.lower()


def test_terminal_missing_blocks():
    c = _make_connector()
    with patch.object(mt5, "terminal_info", return_value=None):
        reason = c._safety_precheck()
        assert reason is not None
        assert "terminal" in reason.lower()