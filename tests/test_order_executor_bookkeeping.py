"""Regression tests for the F-02 bookkeeping fixes.

Uses a real in-memory SQLite database so we test the actual persistence
logic, not a choreography of mocks. Only MT5 and trade_manager are mocked.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.conftest import StubGate


@pytest.fixture
def harness(monkeypatch, sample_signal, sample_order_result, sample_position):
    from src import order_executor as oe_module
    from src.database_manager import DatabaseManager

    # ----------------------------------------------------------------
    # In-memory SQLite. Each test gets a fresh schema.
    # ----------------------------------------------------------------
    db = DatabaseManager("sqlite:///:memory:")
    db.create_tables()

    # ----------------------------------------------------------------
    # Fake MetaTrader5 package, used directly by _prepare_order_request.
    # ----------------------------------------------------------------
    fake_mt5 = SimpleNamespace(
        symbol_info=lambda sym: SimpleNamespace(point=0.01, digits=2),
        symbol_info_tick=lambda sym: SimpleNamespace(bid=2004.90, ask=2005.00),
        ORDER_TYPE_BUY=0,
        ORDER_TYPE_SELL=1,
        ORDER_TYPE_BUY_LIMIT=2,
        ORDER_TYPE_SELL_LIMIT=3,
        TRADE_ACTION_DEAL=4,
        TRADE_ACTION_PENDING=5,
        ORDER_TIME_GTC=0,
    )

    # ----------------------------------------------------------------
    # mt5_connector: mock. Its `send_order` returns the sample success.
    # ----------------------------------------------------------------
    mt5_connector = MagicMock(name="mt5_connector")
    mt5_connector.is_connected.return_value = True
    mt5_connector.send_order.return_value = sample_order_result
    mt5_connector.get_positions.return_value = [sample_position]
    mt5_connector.get_history_deals.return_value = []
    mt5_connector.get_orders.return_value = []
    mt5_connector.get_symbol_info.return_value = SimpleNamespace(
        name="GOLD", point=0.01, digits=2, tick_size=0.01, tick_value=1.0,
        volume_min=0.01, volume_max=100.0, volume_step=0.01,
        contract_size=100.0, margin_initial=0.0,
    )
    mt5_connector.pip_size_for.return_value = 0.1

    # ----------------------------------------------------------------
    # trade_manager: mock.
    # ----------------------------------------------------------------
    tm = MagicMock(name="trade_manager")
    tm.get_position_state.return_value = None
    tm.initialize_position_tracking.return_value = True
    tm.remove_position.return_value = True

    # ----------------------------------------------------------------
    # risk_manager and signal_validator: mock.
    # ----------------------------------------------------------------
    rm = MagicMock(name="risk_manager")
    rm.calculate_position_size.return_value = SimpleNamespace(
        lot_size=0.10, reason="OK"
    )
    sv = MagicMock(name="signal_validator")
    sv.validate_signal.return_value = SimpleNamespace(valid=True, reason="OK")

    # ----------------------------------------------------------------
    # config: replace the singleton with a namespace, not properties.
    # ----------------------------------------------------------------
    fake_config = SimpleNamespace(
        trading=SimpleNamespace(
            magic_number=123456789,
            mode="DEMO",
            pending_expiry_hours=4,
            max_pending_orders=5,
        ),
        trade_management=SimpleNamespace(max_slippage_pips=3.0),
        live_trading_enabled=False,
    )

    # ----------------------------------------------------------------
    # Apply patches on the order_executor module namespace.
    # ----------------------------------------------------------------
    monkeypatch.setattr(oe_module, "db_manager", db)
    monkeypatch.setattr(oe_module, "mt5", fake_mt5)
    monkeypatch.setattr(oe_module, "mt5_connector", mt5_connector)
    monkeypatch.setattr(oe_module, "trade_manager", tm)
    monkeypatch.setattr(oe_module, "risk_manager", rm)
    monkeypatch.setattr(oe_module, "signal_validator", sv)
    monkeypatch.setattr(oe_module, "config", fake_config)

    from src.order_executor import OrderExecutor

    executor = OrderExecutor()

    return SimpleNamespace(
        executor=executor,
        mt5_connector=mt5_connector,
        db=db,
        tm=tm,
        rm=rm,
        sv=sv,
        config=fake_config,
        signal=sample_signal,
        result=sample_order_result,
        position=sample_position,
    )


# ----------------------------------------------------------------- market entry


def test_market_entry_happy_path_bookkeeps_immediately(harness):
    gate = StubGate(ok=True)

    result = harness.executor.execute_signal(
        harness.signal, market_entry=True, gate=gate
    )

    assert result.success is True, result.error
    assert harness.mt5_connector.send_order.call_count == 1
    assert harness.tm.initialize_position_tracking.call_count == 1
    assert gate.calls["check"] == 1

    # Verify persistence actually happened.
    from src.database_manager import OrderRequest, ExecutedOrder, OpenPosition
    reqs = harness.db.query(OrderRequest)
    assert len(reqs) == 1
    assert reqs[0].status == "filled"
    assert reqs[0].mt5_ticket == 9001

    orders = harness.db.query(ExecutedOrder)
    assert len(orders) == 1
    assert orders[0].ticket == 9001
    assert orders[0].order_request_id == reqs[0].id

    positions = harness.db.query(OpenPosition)
    assert len(positions) == 1
    assert positions[0].ticket == 9001


def test_market_entry_blocked_by_gate(harness):
    gate = StubGate(ok=False, reason="paused")

    result = harness.executor.execute_signal(
        harness.signal, market_entry=True, gate=gate
    )

    assert result.success is False
    assert "gate" in (result.error or "").lower()
    harness.mt5_connector.send_order.assert_not_called()


def test_market_entry_when_position_not_found_marks_unconfirmed(harness):
    harness.mt5_connector.get_positions.return_value = []
    harness.mt5_connector.get_history_deals.return_value = []
    gate = StubGate(ok=True)

    result = harness.executor.execute_signal(
        harness.signal, market_entry=True, gate=gate
    )

    assert result.success is True, result.error
    harness.mt5_connector.send_order.assert_called_once()
    harness.tm.initialize_position_tracking.assert_not_called()

    from src.database_manager import OrderRequest
    reqs = harness.db.query(OrderRequest)
    assert len(reqs) == 1
    assert reqs[0].status == "filled_unconfirmed"


# ----------------------------------------------------------------- pending order


def test_pending_order_placed_without_market_bookkeeping(harness):
    gate = StubGate(ok=True)

    result = harness.executor.execute_signal(
        harness.signal, market_entry=False, gate=gate
    )

    assert result.success is True, result.error
    assert result.pending is True
    harness.tm.initialize_position_tracking.assert_not_called()

    from src.database_manager import OrderRequest
    reqs = harness.db.query(OrderRequest)
    assert len(reqs) == 1
    assert reqs[0].status == "placed"


# ----------------------------------------------------------------- failure modes


def test_duplicate_check_fails_closed_when_db_raises(harness, monkeypatch):
    from src import order_executor as oe_module

    class BrokenDB:
        def get_session(self):
            raise RuntimeError("DB down")

        def add(self, obj):
            raise RuntimeError("DB down")

        def query_one(self, *args, **kwargs):
            raise RuntimeError("DB down")

        def query(self, *args, **kwargs):
            raise RuntimeError("DB down")

    monkeypatch.setattr(oe_module, "db_manager", BrokenDB())

    gate = StubGate(ok=True)
    result = harness.executor.execute_signal(
        harness.signal, market_entry=True, gate=gate
    )

    assert result.success is False
    assert "duplicate" in (result.error or "").lower()


def test_reconcile_market_entries_is_noop_when_nothing_incomplete(harness):
    harness.executor.reconcile_pending_orders()
    harness.tm.initialize_position_tracking.assert_not_called()


# ----------------------------------------------------------------- close paths


def test_close_all_positions_skips_manual_positions(harness):
    harness.mt5_connector.get_positions.return_value = [
        {**harness.position, "ticket": 1, "magic": 123456789},  # bot
        {**harness.position, "ticket": 2, "magic": 999999},     # manual
    ]
    harness.mt5_connector.close_position.return_value = SimpleNamespace(
        success=True, price=2000.0
    )

    closed = harness.executor.close_all_positions(reason="EMERGENCY")

    ticket_calls = [
        call.args[0]
        for call in harness.mt5_connector.close_position.call_args_list
    ]
    assert ticket_calls == [1]
    assert len(closed) == 1