"""Focused M-12 test: quote consistency between sizing and submission."""
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.order_executor import OrderExecutor
from src.mt5_connector import MT5Connector


def _fresh_tick(bid, ask, age_s=1):
    return {
        "bid": bid,
        "ask": ask,
        "time": datetime.now(timezone.utc) - timedelta(seconds=age_s),
    }


def _symbol_info(point=0.01, digits=2, name="GOLD"):
    info = MagicMock()
    info.point = point
    info.digits = digits
    info.name = name
    info.volume_min = 0.01
    info.volume_max = 100.0
    info.volume_step = 0.01
    info.tick_size = 0.01
    info.tick_value = 1.0
    info.contract_size = 100.0
    return info


def _patch_connector(tick_seq, sym_info):
    patcher = patch("src.order_executor.mt5_connector")
    mock_conn = patcher.start()
    mock_conn.get_fresh_tick.side_effect = tick_seq
    mock_conn.get_symbol_info.return_value = sym_info
    return patcher, mock_conn


# --- test 1: fresh sizing tick required -------------------------------------

def test_sizing_rejects_stale_quote():
    executor = OrderExecutor()
    signal = MagicMock()
    signal.symbol = "GOLD"
    signal.direction.value = "BUY"
    signal.entry_price = 2000.0
    signal.sl_price = 1990.0
    signal.tp_price = 2020.0
    signal.risk_percent = 1.0
    signal.strategy.value = "s"
    signal.confirmations = []
    signal.timestamp = datetime.now(timezone.utc)
    signal.entry_tf = "m5"
    signal.risk_reward = 2.0
    signal.fingerprint = "fp"

    gate = MagicMock()
    gate.check.return_value = MagicMock(ok=True, token="tok")

    # Validation runs BEFORE sizing; stub it so the sizing path is reached.
    fake_validation = MagicMock(valid=True, reason="")

    with patch("src.order_executor.mt5_connector") as mock_conn, \
         patch("src.order_executor.signal_validator") as mock_sv:
        mock_conn.is_connected.return_value = True
        mock_conn.get_fresh_tick.return_value = None   # stale!
        mock_sv.validate_signal.return_value = fake_validation

        result = executor.execute_signal(
            signal, market_entry=True, gate=gate
        )

    assert not result.success
    assert "FRESH" in (result.error or "")


# --- test 2: guard helper rejects adverse movement --------------------------

@pytest.mark.parametrize(
    "direction,expected_entry,bid,ask,expect_reject",
    [
        # BUY: ask moved from 2000.0 to 2005.0 = 50 pips on gold (pip=0.10)
        ("BUY", 2000.0, 2004.9, 2005.0, True),
        # BUY: ask moved 1 pip adverse (0.10) -> within 3-pip tolerance
        ("BUY", 2000.0, 1999.9, 2000.1, False),
        # SELL: bid dropped from 2000.0 to 1994.9 -> 51 pips adverse
        ("SELL", 2000.0, 1994.9, 1995.0, True),
        # SELL: bid dropped 1 pip -> within tolerance
        ("SELL", 2000.0, 1999.9, 2000.0, False),
    ],
)
def test_guard_rejects_adverse_move(direction, expected_entry, bid, ask, expect_reject):
    executor = OrderExecutor()
    sym_info = _symbol_info(point=0.01, digits=2, name="GOLD")

    with patch("src.order_executor.mt5_connector") as mock_conn:
        mock_conn.get_fresh_tick.return_value = _fresh_tick(bid, ask)
        mock_conn.get_symbol_info.return_value = sym_info

        reason = executor._revalidate_pre_submission_quote(
            expected_entry, direction
        )

    if expect_reject:
        assert reason is not None, "expected rejection, got pass"
        assert "adversely" in reason.lower()
    else:
        assert reason is None, f"unexpected rejection: {reason}"


# --- test 3: guard fails closed on missing/stale quote ----------------------

@pytest.mark.parametrize("bad_quote", [None])
def test_guard_fails_closed_on_missing_tick(bad_quote):
    executor = OrderExecutor()
    with patch("src.order_executor.mt5_connector") as mock_conn:
        mock_conn.get_fresh_tick.return_value = bad_quote
        mock_conn.get_symbol_info.return_value = _symbol_info()

        reason = executor._revalidate_pre_submission_quote(2000.0, "BUY")

    assert reason is not None
    assert "Fresh quote" in reason


def test_guard_fails_closed_on_invalid_pip():
    executor = OrderExecutor()
    with patch("src.order_executor.mt5_connector") as mock_conn:
        mock_conn.get_fresh_tick.return_value = _fresh_tick(1999.9, 2000.0)
        mock_conn.get_symbol_info.return_value = None   # pip unavailable

        reason = executor._revalidate_pre_submission_quote(2000.0, "BUY")

    assert reason is not None
    assert "pip size" in reason.lower()


# --- test 4: _prepare_order_request uses fresh tick; fails if None ---------

def test_prepare_order_request_fails_without_fresh_quote():
    executor = OrderExecutor()
    signal = MagicMock()
    signal.symbol = "GOLD"
    signal.direction.value = "BUY"
    signal.entry_price = 2000.0
    signal.sl_price = 1990.0
    signal.tp_price = 2020.0
    signal.strategy.value = "s"
    signal.fingerprint = "fp"

    size = MagicMock(lot_size=0.1)

    with patch("src.order_executor.mt5_connector") as mock_conn:
        mock_conn.get_symbol_info.return_value = _symbol_info()
        mock_conn.get_fresh_tick.return_value = None

        with pytest.raises(RuntimeError) as exc:
            executor._prepare_order_request(signal, size, market_entry=True)

    assert "FRESH" in str(exc.value)


# --- test 5: connector-level submission boundary fails on stale tick -------

def test_connector_send_order_fails_on_stale_deal_quote():
    # Test the connector boundary directly.
    from src import mt5_connector as mod
    connector = mod.MT5Connector.__new__(mod.MT5Connector)
    connector._symbol_name = "GOLD"
    connector._initialized = True
    connector._account_info = None
    connector.mt5_config = MagicMock()
    connector.trading_config = MagicMock(magic_number=1)
    connector.state = mod.ConnectionState.AUTHENTICATED

    # Force _trading_precheck and live/recon guards to pass by stub.
    connector._trading_precheck = lambda: None
    connector._assert_reconciliation_clear = lambda: None
    connector._assert_live_opening_authorized = lambda: None

    request = {
        "action": mod.mt5.TRADE_ACTION_DEAL,
        "symbol": "GOLD",
        "type": mod.mt5.ORDER_TYPE_BUY,
        "volume": 0.1,
        "sl": 1990.0,
        "tp": 2020.0,
    }

    with patch.object(connector, "get_fresh_tick", return_value=None):
        # Short-circuit the gate-token gate so we isolate the M-12 check.
        with patch("src.gate_token.consume", return_value=True):
            result = connector._send_order_request(
                request, gate_token="tok"
            )

    assert not result.success
    assert "FRESH" in (result.error_message or "")