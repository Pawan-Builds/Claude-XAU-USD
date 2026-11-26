"""Focused M-14 test: unresolved order statuses block LIVE openings.

LIVE is never enabled here — all tests are unit-level or mocked.
"""
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------
# constant coverage
# ---------------------------------------------------------------------

def test_unresolved_statuses_cover_all_required():
    from src.order_executor import _UNRESOLVED_STATUSES

    expected = {
        "uncertain",
        "submission_error",
        "sent",
        "pending",
        "filled_unconfirmed",
        "unresolved_closed",
    }
    assert set(_UNRESOLVED_STATUSES) == expected


def test_unresolved_statuses_exclude_placed_and_terminal():
    from src.order_executor import _UNRESOLVED_STATUSES

    # 'placed' is a known live pending order, not an unresolved
    # submission — must not block new opens.
    assert "placed" not in _UNRESOLVED_STATUSES
    for terminal in (
        "rejected", "rejected_notfound",
        "cancelled", "expired",
        "filled", "filled_closed",
    ):
        assert terminal not in _UNRESOLVED_STATUSES


# ---------------------------------------------------------------------
# has_unresolved_reconciliation: filter shape
# ---------------------------------------------------------------------

def _make_session_mock(rows):
    session = MagicMock()
    query = MagicMock()
    session.query.return_value = query
    query.filter.return_value = query
    query.first.return_value = rows[0] if rows else None
    ctx = MagicMock()
    ctx.return_value.__enter__.return_value = session
    return ctx, query


def test_has_unresolved_returns_true_when_row_present():
    from src.order_executor import OrderExecutor

    executor = OrderExecutor()
    ctx, _ = _make_session_mock([MagicMock(id=1, status="uncertain")])

    with patch(
        "src.order_executor.db_manager.get_session", ctx
    ):
        assert executor.has_unresolved_reconciliation() is True


def test_has_unresolved_returns_false_when_no_row():
    from src.order_executor import OrderExecutor

    executor = OrderExecutor()
    ctx, _ = _make_session_mock([])

    with patch(
        "src.order_executor.db_manager.get_session", ctx
    ):
        assert executor.has_unresolved_reconciliation() is False


def test_has_unresolved_adds_second_filter_for_exclusion():
    from src.order_executor import OrderExecutor

    executor = OrderExecutor()
    ctx, query = _make_session_mock([])

    with patch(
        "src.order_executor.db_manager.get_session", ctx
    ):
        # Without exclusion: exactly one filter call.
        executor.has_unresolved_reconciliation()
        assert query.filter.call_count == 1

        # With exclusion: a second filter call is chained.
        query.filter.reset_mock()
        executor.has_unresolved_reconciliation(exclude_request_id=42)
        assert query.filter.call_count == 2


def test_has_unresolved_fails_closed_on_db_error():
    from src.order_executor import OrderExecutor

    executor = OrderExecutor()

    with patch(
        "src.order_executor.db_manager.get_session",
        side_effect=RuntimeError("db down"),
    ):
        # Any DB exception → fail closed (True).
        assert executor.has_unresolved_reconciliation() is True


# ---------------------------------------------------------------------
# connector boundary forwards request_id
# ---------------------------------------------------------------------

def _live_connector_stub():
    from src.mt5_connector import MT5Connector

    connector = MT5Connector.__new__(MT5Connector)
    connector.mt5_config = MagicMock(server="S")
    connector._account_info = MagicMock(login=123, server="S")
    return connector


def _live_config_stub():
    cfg = MagicMock()
    cfg.trading.mode = "LIVE"
    cfg.live_trading_enabled = True
    cfg.live_trading_confirmation = "I_ACCEPT_LIVE_TRADING_RISK"
    cfg.live_allowed_account = 123
    cfg.safety.demo_only = False
    return cfg


def test_live_boundary_passes_request_id_to_unresolved_check():
    connector = _live_connector_stub()
    cfg = _live_config_stub()

    with patch("src.mt5_connector.config", cfg), \
         patch(
             "src.risk_manager.risk_manager.get_kill_switch_state_strict",
             return_value=False,
         ), \
         patch(
             "src.order_executor.order_executor.has_unresolved_reconciliation",
             return_value=False,
         ) as mock_unresolved:
        result = connector._assert_live_opening_authorized(
            request_id=99
        )

    assert result is None  # authorized
    mock_unresolved.assert_called_once_with(exclude_request_id=99)


def test_live_boundary_blocks_when_unresolved_present():
    connector = _live_connector_stub()
    cfg = _live_config_stub()

    with patch("src.mt5_connector.config", cfg), \
         patch(
             "src.risk_manager.risk_manager.get_kill_switch_state_strict",
             return_value=False,
         ), \
         patch(
             "src.order_executor.order_executor.has_unresolved_reconciliation",
             return_value=True,
         ):
        result = connector._assert_live_opening_authorized(
            request_id=99
        )

    assert result is not None
    assert "reconciliation" in result.lower()


# ---------------------------------------------------------------------
# execute_signal forwards request_id to send_order
# ---------------------------------------------------------------------

def test_execute_signal_forwards_request_id_to_send_order():
    from src.order_executor import OrderExecutor

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

    fake_validation = MagicMock(valid=True, reason="")
    fake_size = MagicMock(lot_size=0.1, reason="OK")
    fake_order_result = MagicMock(
        success=True,
        uncertain=False,
        order_id=1234,
        ticket=1234,
        volume=0.1,
        price=2000.0,
        retcode=0,
        comment="",
        error_code=None,
        error_message=None,
    )

    with patch("src.order_executor.mt5_connector") as mock_conn, \
         patch("src.order_executor.signal_validator") as mock_sv, \
         patch("src.order_executor.risk_manager") as mock_rm, \
         patch.object(executor, "_save_signal", return_value=1), \
         patch.object(executor, "_save_order_request", return_value=42), \
         patch.object(executor, "_prepare_order_request", return_value={}), \
         patch.object(executor, "_is_duplicate", return_value=False), \
         patch.object(executor, "_update_order_request"), \
         patch.object(executor, "_finalise_market_entry"):
        mock_conn.is_connected.return_value = True
        mock_conn.get_fresh_tick.return_value = {
            "bid": 1999.5, "ask": 2000.0,
        }
        mock_conn.get_symbol_info.return_value = MagicMock(
            point=0.01, digits=2, name="GOLD",
        )
        mock_conn.send_order.return_value = fake_order_result

        mock_sv.validate_signal.return_value = fake_validation
        mock_rm.calculate_position_size.return_value = fake_size

        executor.execute_signal(
            signal, market_entry=True, gate=gate
        )

        assert mock_conn.send_order.called
        kwargs = mock_conn.send_order.call_args.kwargs
        assert kwargs.get("request_id") == 42, (
            f"execute_signal must forward request_id to send_order; "
            f"got kwargs={kwargs!r}"
        )