"""Phase 3B: pending orders and market orders that fill AND close
between reconciliation cycles must produce a ClosedTrade, never be
silently marked `cancelled`/`filled_unconfirmed` forever.
"""

from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _entry_deal(ticket, order_ticket, price=2000.0, volume=0.1):
    return {
        "ticket": 8001, "order": order_ticket, "position_id": ticket,
        "magic": 123456789, "symbol": "GOLD", "type": "BUY",
        "entry": "IN", "volume": volume, "price": price,
        "commission": -0.70, "swap": 0.0, "profit": 0.0,
        "comment": "SMC_liquidity_sweep_fvg",
        "time": datetime.now(timezone.utc) - timedelta(hours=2),
    }


def _exit_deal(ticket, price=2010.0, volume=0.1):
    return {
        "ticket": 8002, "order": 9002, "position_id": ticket,
        "magic": 123456789, "symbol": "GOLD", "type": "BUY",
        "entry": "OUT", "volume": volume, "price": price,
        "commission": -0.70, "swap": 0.0, "profit": 100.0,
        "comment": "SMC_BOT_CLOSE",
        "time": datetime.now(timezone.utc) - timedelta(minutes=5),
    }


@pytest.fixture
def harness(monkeypatch, sample_signal, sample_order_result):
    from src import order_executor as oe_module
    from src.database_manager import DatabaseManager

    db = DatabaseManager("sqlite:///:memory:")
    db.create_tables()

    fake_mt5 = SimpleNamespace(
        symbol_info=lambda sym: SimpleNamespace(point=0.01, digits=2),
        symbol_info_tick=lambda sym: SimpleNamespace(bid=2004.9, ask=2005.0),
        ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1,
        ORDER_TYPE_BUY_LIMIT=2, ORDER_TYPE_SELL_LIMIT=3,
        ORDER_TYPE_BUY_STOP=4, ORDER_TYPE_SELL_STOP=5,
        ORDER_TYPE_BUY_STOP_LIMIT=6, ORDER_TYPE_SELL_STOP_LIMIT=7,
        TRADE_ACTION_DEAL=1, TRADE_ACTION_PENDING=5,
        ORDER_TIME_GTC=0,
    )

    mt5c = MagicMock(name="mt5_connector")
    mt5c.is_connected.return_value = True
    mt5c.send_order.return_value = sample_order_result
    mt5c.get_positions.return_value = []
    mt5c.get_orders.return_value = []
    mt5c.get_history_deals.return_value = []
    mt5c.get_symbol_info.return_value = SimpleNamespace(
        name="GOLD", point=0.01, digits=2, tick_size=0.01, tick_value=1.0,
        volume_min=0.01, volume_max=100.0, volume_step=0.01,
        contract_size=100.0, margin_initial=0.0,
    )

    from src.trade_manager import TradeManager
    tm = TradeManager()
    monkeypatch.setattr(
        "src.trade_manager.mt5_connector", mt5c,
    )
    # CRITICAL: TradeManager keeps a reference to src.trade_manager.db_manager.
    # Redirect it to the in-memory DB. Without this line, every
    # ClosedTrade insert from reconstruct_closed_trade() lands in
    # database/trading.db (production), not in the test DB.
    monkeypatch.setattr("src.trade_manager.db_manager", db)
    tm_reconstruct = MagicMock(wraps=tm.reconstruct_closed_trade,
                               return_value=True)

    rm = MagicMock(name="risk_manager")
    rm.calculate_position_size.return_value = SimpleNamespace(
        lot_size=0.10, reason="OK",
    )
    sv = MagicMock(name="signal_validator")
    sv.validate_signal.return_value = SimpleNamespace(valid=True, reason="OK")

    fake_config = SimpleNamespace(
        trading=SimpleNamespace(
            magic_number=123456789, mode="DEMO",
            pending_expiry_hours=4, max_pending_orders=5,
        ),
        trade_management=SimpleNamespace(max_slippage_pips=3.0),
        live_trading_enabled=False,
    )

    monkeypatch.setattr(oe_module, "db_manager", db)
    monkeypatch.setattr(oe_module, "mt5", fake_mt5)
    monkeypatch.setattr(oe_module, "mt5_connector", mt5c)
    monkeypatch.setattr(oe_module, "trade_manager", tm)
    monkeypatch.setattr(oe_module, "risk_manager", rm)
    monkeypatch.setattr(oe_module, "signal_validator", sv)
    monkeypatch.setattr(oe_module, "config", fake_config)

    from src.order_executor import OrderExecutor
    executor = OrderExecutor()

    return SimpleNamespace(
        executor=executor, db=db, mt5=mt5c, tm=tm,
        config=fake_config, tm_reconstruct=tm_reconstruct,
    )


def _insert_request(db, *, status, order_type=0, ticket=None,
                    timestamp=None):
    """Insert an OrderRequest with the minimum fields the reconciler reads."""
    from src.database_manager import OrderRequest

    row = OrderRequest(
        timestamp=timestamp or datetime.utcnow(),
        signal_id=None, magic_number=123456789, symbol="GOLD",
        order_type=order_type, volume=0.1,
        price=2000.0, sl=1990.0, tp=2020.0, comment="SMC_TEST",
        request_data={}, response_data={}, status=status,
        environment="demo", fingerprint="fp",
        mt5_ticket=ticket,
    )
    return db.add(row)


# ---------- Bug 1: direction matching now works ----------

def test_order_type_to_side_handles_int_string_and_stringified_int():
    from src.order_executor import OrderExecutor

    assert OrderExecutor._order_type_to_side(0) == "BUY"          # int
    assert OrderExecutor._order_type_to_side("BUY_LIMIT") == "BUY"
    assert OrderExecutor._order_type_to_side("BUY") == "BUY"
    assert OrderExecutor._order_type_to_side("0") == "BUY"        # SQLite
    assert OrderExecutor._order_type_to_side("1") == "SELL"
    assert OrderExecutor._order_type_to_side(None) is None
    assert OrderExecutor._order_type_to_side("garbage") is None


def test_order_direction_matches_mixed_representations():
    from src.order_executor import OrderExecutor

    assert OrderExecutor._order_direction_matches(0, "BUY_LIMIT")
    assert OrderExecutor._order_direction_matches("BUY_LIMIT", 0)
    assert OrderExecutor._order_direction_matches("0", "BUY_LIMIT")
    assert not OrderExecutor._order_direction_matches(0, "SELL_LIMIT")


def test_position_direction_matches_mixed_representations():
    from src.order_executor import OrderExecutor

    assert OrderExecutor._position_direction_matches("BUY", 0)
    assert OrderExecutor._position_direction_matches(0, "BUY_LIMIT")
    assert not OrderExecutor._position_direction_matches("BUY", "SELL_LIMIT")


def test_find_matching_deal_now_uses_string_entry(harness):
    """Regression: mt5 DEAL_ENTRY_IN is an int; deals carry 'IN'.

    Aligns deal/request timestamps inside the 10-minute matching
    window so the test exercises the entry-string comparison, not the
    window filter.
    """
    from src.database_manager import OrderRequest

    deal_time = datetime.now(timezone.utc) - timedelta(minutes=2)
    deal = {
        "ticket": 8001, "order": 9001, "position_id": 5001,
        "magic": 123456789, "symbol": "GOLD", "type": "BUY",
        "entry": "IN", "volume": 0.1, "price": 2000.0,
        "commission": -0.70, "swap": 0.0, "profit": 0.0,
        "comment": "SMC_TEST", "time": deal_time,
    }
    req = OrderRequest(
        # 30 seconds earlier → well inside the 10-minute window.
        timestamp=deal_time.replace(tzinfo=None) - timedelta(seconds=30),
        magic_number=123456789, symbol="GOLD",
        order_type=0, volume=0.1, environment="demo",
    )

    assert harness.executor._find_matching_deal(req, [deal]) is deal

# ---------- Bug 2: pending fill+close is recovered ----------

def test_pending_filled_and_closed_creates_closed_trade(harness):
    """A pending order that filled and closed → filled_closed + ClosedTrade."""
    from src.database_manager import OrderRequest, ClosedTrade
    from src.order_executor import OrderExecutor

    # Arrange: pending order was placed (ticket 9001), then filled (position 5001)
    # then closed. Broker shows no live order, no live position.
    req_id = _insert_request(
        harness.db, status="placed", order_type=2, ticket=9001,
        timestamp=datetime.utcnow() - timedelta(hours=1),
    )
    harness.mt5.get_orders.return_value = []
    harness.mt5.get_positions.return_value = []
    harness.mt5.get_history_deals.return_value = [
        _entry_deal(ticket=5001, order_ticket=9001),
        _exit_deal(ticket=5001),
    ]

    # Act
    harness.executor.reconcile_pending_orders()

    # Assert: request status is terminal, ClosedTrade exists.
    with harness.db.get_session() as s:
        req = s.query(OrderRequest).filter(OrderRequest.id == req_id).first()
        assert req.status == "filled_closed"

    closed = harness.db.query(ClosedTrade)
    assert len(closed) == 1
    assert closed[0].ticket == 5001


def test_pending_never_filled_still_marked_cancelled(harness):
    """A pending order that never filled is still cancelled (no regression)."""
    from src.database_manager import OrderRequest, ClosedTrade

    req_id = _insert_request(
        harness.db, status="placed", order_type=2, ticket=9001,
        timestamp=datetime.utcnow() - timedelta(hours=1),
    )
    harness.mt5.get_orders.return_value = []
    harness.mt5.get_positions.return_value = []
    harness.mt5.get_history_deals.return_value = []  # no fill, no close

    harness.executor.reconcile_pending_orders()

    with harness.db.get_session() as s:
        req = s.query(OrderRequest).filter(OrderRequest.id == req_id).first()
        assert req.status == "cancelled"

    assert len(harness.db.query(ClosedTrade)) == 0


def test_pending_fill_only_defers(harness):
    """Fill confirmed but no OUT deal → do NOT mark cancelled; retry."""
    from src.database_manager import OrderRequest

    req_id = _insert_request(
        harness.db, status="placed", order_type=2, ticket=9001,
        timestamp=datetime.utcnow() - timedelta(hours=1),
    )
    harness.mt5.get_orders.return_value = []
    harness.mt5.get_positions.return_value = []
    harness.mt5.get_history_deals.return_value = [
        _entry_deal(ticket=5001, order_ticket=9001),
    ]

    harness.executor.reconcile_pending_orders()

    with harness.db.get_session() as s:
        req = s.query(OrderRequest).filter(OrderRequest.id == req_id).first()
        assert req.status == "placed"  # unchanged → retry next cycle


# ---------- Bug 2: market fill+close is recovered ----------

def test_market_filled_and_closed_creates_closed_trade(harness):
    """A market order that filled and closed → filled_closed + ClosedTrade."""
    from src.database_manager import OrderRequest, ClosedTrade

    req_id = _insert_request(
        harness.db, status="filled_unconfirmed", order_type=0, ticket=9002,
        timestamp=datetime.utcnow() - timedelta(minutes=10),
    )
    harness.mt5.get_orders.return_value = []
    harness.mt5.get_positions.return_value = []
    harness.mt5.get_history_deals.return_value = [
        _entry_deal(ticket=5002, order_ticket=9002),
        _exit_deal(ticket=5002),
    ]

    harness.executor.reconcile_pending_orders()

    with harness.db.get_session() as s:
        req = s.query(OrderRequest).filter(OrderRequest.id == req_id).first()
        assert req.status == "filled_closed"

    closed = harness.db.query(ClosedTrade)
    assert len(closed) == 1
    assert closed[0].ticket == 5002


def test_market_still_live_found_not_regressed(harness):
    """Market order whose position is still visible → status 'filled'."""
    from src.database_manager import OrderRequest

    req_id = _insert_request(
        harness.db, status="filled_unconfirmed", order_type=0, ticket=9003,
        timestamp=datetime.utcnow() - timedelta(minutes=10),
    )
    harness.mt5.get_positions.return_value = [{
        "ticket": 9003, "magic": 123456789, "symbol": "GOLD",
        "type": "BUY", "volume": 0.1, "price_open": 2000.0,
        "price_current": 2005.0, "sl": 1990.0, "tp": 2020.0,
        "swap": 0.0, "profit": 50.0, "commission": 0.0,
        "comment": "SMC_TEST",
        "time_open": datetime.now(timezone.utc) - timedelta(minutes=10),
        "time_update": datetime.now(timezone.utc),
    }]

    harness.executor.reconcile_pending_orders()

    with harness.db.get_session() as s:
        req = s.query(OrderRequest).filter(OrderRequest.id == req_id).first()
        assert req.status == "filled"


# ---------- unresolved_closed retries ----------

def test_unresolved_closed_retried_and_promoted(harness, monkeypatch):
    """A row in unresolved_closed is retried and promoted to filled_closed."""
    from src.database_manager import OrderRequest

    req_id = _insert_request(
        harness.db, status="unresolved_closed", order_type=0, ticket=9004,
        timestamp=datetime.utcnow() - timedelta(minutes=30),
    )
    harness.mt5.get_history_deals.return_value = [
        _entry_deal(ticket=5004, order_ticket=9004),
        _exit_deal(ticket=5004),
    ]

    harness.executor.reconcile_pending_orders()

    with harness.db.get_session() as s:
        req = s.query(OrderRequest).filter(OrderRequest.id == req_id).first()
        assert req.status == "filled_closed"


def test_unresolved_closed_kept_when_reconstruction_fails(
    harness, monkeypatch,
):
    """If ClosedTrade fails again, the row stays in unresolved_closed."""
    from src.database_manager import OrderRequest

    req_id = _insert_request(
        harness.db, status="unresolved_closed", order_type=0, ticket=9005,
        timestamp=datetime.utcnow() - timedelta(minutes=30),
    )
    harness.mt5.get_history_deals.return_value = [
        _entry_deal(ticket=5005, order_ticket=9005),
        _exit_deal(ticket=5005),
    ]

    # Force reconstruction to fail on this pass.
    broken = MagicMock(return_value=False)
    monkeypatch.setattr(
        "src.order_executor.trade_manager.reconstruct_closed_trade", broken,
    )

    harness.executor.reconcile_pending_orders()

    with harness.db.get_session() as s:
        req = s.query(OrderRequest).filter(OrderRequest.id == req_id).first()
        assert req.status == "unresolved_closed"
    broken.assert_called_once()