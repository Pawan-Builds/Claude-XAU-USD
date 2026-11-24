"""Tests for F-05: broker <-> DB position reconciliation."""

from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def harness(monkeypatch):
    from src import trade_manager as tm_module
    from src.database_manager import DatabaseManager

    db = DatabaseManager("sqlite:///:memory:")
    db.create_tables()

    mt5 = MagicMock(name="mt5_connector")
    mt5.get_symbol_info.return_value = SimpleNamespace(
        name="GOLD", point=0.01, digits=2, tick_size=0.01, tick_value=1.0,
        volume_min=0.01, volume_max=100.0, volume_step=0.01,
        contract_size=100.0,
    )
    mt5.pip_size_for.return_value = 0.1

    fake_config = SimpleNamespace(
        trading=SimpleNamespace(
            magic_number=123456789, mode="DEMO",
            max_holding_hours=48, pending_expiry_hours=4,
            max_pending_orders=5, pending_orders_on_shutdown="keep",
        ),
        trade_management=SimpleNamespace(
            break_even_enabled=True, break_even_trigger_r=1.0,
            break_even_buffer_pips=5.0, trailing_stop_enabled=True,
            trailing_stop_start_r=1.5, trailing_stop_step_pips=10.0,
            partial_tp_enabled=True, partial_tp_levels=[],
            max_slippage_pips=3.0, max_spread_pips=5.0,
        ),
        risk=SimpleNamespace(
            default_risk_percent=1.0, max_risk_percent=1.0,
            max_total_open_risk_percent=4.0,
        ),
    )

    monkeypatch.setattr(tm_module, "db_manager", db)
    monkeypatch.setattr(tm_module, "mt5_connector", mt5)
    monkeypatch.setattr(tm_module, "config", fake_config)

    from src.trade_manager import TradeManager
    manager = TradeManager()

    return SimpleNamespace(
        manager=manager, db=db, mt5=mt5, config=fake_config,
    )


def _entry_deal(ticket, price=2000.0, volume=0.1, when=None):
    return {
        "ticket": 8001,
        "order": 9001,
        "position_id": ticket,
        "magic": 123456789,
        "symbol": "GOLD",
        "type": "BUY",
        "entry": "IN",
        "volume": volume,
        "price": price,
        "commission": -0.70,
        "swap": 0.0,
        "profit": 0.0,
        "comment": "SMC_liquidity_sweep_fvg",
        "time": when or datetime.now(timezone.utc) - timedelta(hours=2),
    }


def _exit_deal(ticket, price=2010.0, volume=0.1, when=None):
    return {
        "ticket": 8002,
        "order": 9002,
        "position_id": ticket,
        "magic": 123456789,
        "symbol": "GOLD",
        "type": "BUY",
        "entry": "OUT",
        "volume": volume,
        "price": price,
        "commission": -0.70,
        "swap": 0.0,
        "profit": 100.0,
        "comment": "SMC_BOT_CLOSE",
        "time": when or datetime.now(timezone.utc) - timedelta(minutes=5),
    }


def _live_position(ticket, sl=1995.0, tp=2010.0, volume=0.1):
    return {
        "ticket": ticket,
        "magic": 123456789,
        "symbol": "GOLD",
        "type": "BUY",
        "volume": volume,
        "price_open": 2000.0,
        "price_current": 2005.0,
        "sl": sl,
        "tp": tp,
        "swap": 0.0,
        "profit": 50.0,
        "commission": 0.0,
        "comment": "SMC_liquidity_sweep_fvg",
        "time_open": datetime.now(timezone.utc) - timedelta(hours=2),
        "time_update": datetime.now(timezone.utc),
    }


def test_reconcile_returns_false_when_mt5_query_fails(harness):
    harness.mt5.get_positions.return_value = None
    assert harness.manager.reconcile_with_broker() is False


def test_reconcile_noop_when_no_positions(harness):
    harness.mt5.get_positions.return_value = []
    assert harness.manager.reconcile_with_broker() is True


def test_reconcile_recovers_orphan_fill(harness):
    """MT5 has a position but DB has no ExecutedOrder/OpenPosition."""
    harness.mt5.get_positions.return_value = [_live_position(ticket=5001)]
    harness.mt5.get_history_deals.return_value = [_entry_deal(5001)]

    assert harness.manager.reconcile_with_broker() is True

    from src.database_manager import ExecutedOrder, OpenPosition
    executed = harness.db.query(ExecutedOrder)
    assert len(executed) == 1
    assert executed[0].ticket == 5001
    assert executed[0].price_open == 2000.0
    assert executed[0].environment == "demo"

    open_rows = harness.db.query(OpenPosition)
    assert len(open_rows) == 1
    assert open_rows[0].ticket == 5001
    assert open_rows[0].strategy == "RECOVERED"

    state = harness.manager.get_position_state(5001)
    assert state is not None
    assert state["direction"] == "BUY"


def test_reconcile_reconstructs_missing_closure(harness):
    """MT5 no longer has the position; DB still has OpenPosition."""
    from src.database_manager import (
        ExecutedOrder, OpenPosition, ClosedTrade,
    )

    harness.db.add(ExecutedOrder(
        order_request_id=None, ticket=5002, magic_number=123456789,
        symbol="GOLD", order_type="BUY", volume=0.1,
        price_open=2000.0, price_current=2000.0,
        sl=1995.0, tp=2010.0, swap=0.0, profit=0.0, commission=0.0,
        comment="SMC_liquidity_sweep_fvg",
        time_open=datetime.now(timezone.utc) - timedelta(hours=2),
        environment="demo",
    ))
    harness.db.add(OpenPosition(
        ticket=5002, magic_number=123456789, symbol="GOLD",
        direction="BUY", volume=0.1, price_open=2000.0,
        price_current=2000.0, sl=1995.0, tp=2010.0, swap=0.0,
        profit=0.0, commission=0.0, strategy="liquidity_sweep_fvg",
        time_open=datetime.now(timezone.utc) - timedelta(hours=2),
        environment="demo", initial_sl=1995.0, initial_tp=2010.0,
        initial_risk_distance=5.0, remaining_volume=0.1,
        highest_r=0.0, lowest_r=0.0,
    ))

    harness.mt5.get_positions.return_value = []
    harness.mt5.get_history_deals.return_value = [
        _entry_deal(5002),
        _exit_deal(5002),
    ]

    assert harness.manager.reconcile_with_broker() is True

    closed = harness.db.query(ClosedTrade)
    assert len(closed) == 1
    ct = closed[0]
    assert ct.ticket == 5002
    assert ct.price_close == 2010.0
    # net_profit = exit.profit + all commissions + all swaps
    #            = 100.0 + (-0.70 - 0.70) + 0.0
    #            = 98.60
    assert abs(ct.net_profit - 98.6) < 0.01
    assert abs(ct.realized_r - 2.0) < 0.01  # (2010-2000)/5

    open_rows = harness.db.query(OpenPosition)
    assert len(open_rows) == 0


def test_reconcile_idempotent(harness):
    """Running reconcile twice must not duplicate ClosedTrade."""
    from src.database_manager import ClosedTrade

    harness.mt5.get_positions.return_value = []
    harness.mt5.get_history_deals.return_value = [
        _entry_deal(5003),
        _exit_deal(5003),
    ]
    # No OpenPosition means step 2 doesn't try to reconstruct.
    # Directly exercise the reconstruction API.
    harness.manager.reconstruct_closed_trade(5003, exit_reason="SL")
    harness.manager.reconstruct_closed_trade(5003, exit_reason="SL")

    assert len(harness.db.query(ClosedTrade)) == 1


def test_reconcile_leaves_orphan_untracked_when_no_deals(harness):
    """MT5 has a position but history query returns nothing."""
    harness.mt5.get_positions.return_value = [_live_position(ticket=5004)]
    harness.mt5.get_history_deals.return_value = []

    assert harness.manager.reconcile_with_broker() is True

    from src.database_manager import ExecutedOrder, OpenPosition
    assert len(harness.db.query(ExecutedOrder)) == 0
    assert len(harness.db.query(OpenPosition)) == 0
    assert harness.manager.get_position_state(5004) is None