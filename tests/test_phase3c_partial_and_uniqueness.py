"""Phase 3C: partial-exit volume refresh + ClosedTrade uniqueness.

Covers §6.7 (do not silently zero remaining_volume on a failed broker
query) and §6.8 (a ticket may produce at most one ClosedTrade row per
environment).
"""

from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ===================================================================
# §6.7 — remaining_volume refresh
# ===================================================================


@pytest.fixture
def tm_harness(monkeypatch):
    from src import trade_manager as tm_module
    from src.database_manager import DatabaseManager

    db = DatabaseManager("sqlite:///:memory:")
    db.create_tables()

    mt5 = MagicMock(name="mt5_connector")
    mt5.get_symbol_info.return_value = SimpleNamespace(
        name="GOLD", point=0.01, digits=2,
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

    # Seed a tracked position directly (bypass initialize_position_tracking
    # so we don't touch the DB for these unit-level checks).
    manager._managed_positions[7001] = {
        "direction": "BUY",
        "entry_price": 2000.0,
        "initial_sl": 1995.0,
        "current_sl": 1995.0,
        "initial_tp": 2010.0,
        "current_tp": 2010.0,
        "initial_risk_distance": 5.0,
        "volume": 0.10,
        "remaining_volume": 0.10,
        "strategy": "test",
        "break_even_activated": False,
        "trailing_active": False,
        "partial_tps_taken": [],
        "highest_r": 0.0,
        "lowest_r": 0.0,
    }

    return SimpleNamespace(manager=manager, mt5=mt5)


def test_refresh_leaves_volume_when_query_returns_none(tm_harness):
    """Query failure must NOT zero remaining_volume."""
    tm_harness.mt5.get_positions.return_value = None

    ok = tm_harness.manager._refresh_remaining_volume(7001)

    assert ok is False
    state = tm_harness.manager.get_position_state(7001)
    assert state["remaining_volume"] == 0.10  # unchanged


def test_refresh_leaves_volume_when_query_raises(tm_harness):
    """Exception during query must NOT zero remaining_volume."""
    tm_harness.mt5.get_positions.side_effect = RuntimeError("broker down")

    ok = tm_harness.manager._refresh_remaining_volume(7001)

    assert ok is False
    state = tm_harness.manager.get_position_state(7001)
    assert state["remaining_volume"] == 0.10


def test_refresh_zeros_when_position_genuinely_gone(tm_harness):
    """Empty result from a *successful* query means position closed."""
    tm_harness.mt5.get_positions.return_value = []

    ok = tm_harness.manager._refresh_remaining_volume(7001)

    assert ok is True
    state = tm_harness.manager.get_position_state(7001)
    assert state["remaining_volume"] == 0.0


def test_refresh_updates_when_position_present(tm_harness):
    """Successful query with the position live reflects broker volume."""
    tm_harness.mt5.get_positions.return_value = [{
        "ticket": 7001, "magic": 123456789, "symbol": "GOLD",
        "type": "BUY", "volume": 0.04,
        "price_open": 2000.0, "price_current": 2005.0,
        "sl": 1995.0, "tp": 2010.0, "swap": 0.0, "profit": 0.0,
        "commission": 0.0, "comment": "SMC_TEST",
        "time_open": datetime.now(timezone.utc) - timedelta(hours=1),
        "time_update": datetime.now(timezone.utc),
    }]

    ok = tm_harness.manager._refresh_remaining_volume(7001)

    assert ok is True
    state = tm_harness.manager.get_position_state(7001)
    assert state["remaining_volume"] == 0.04


def test_refresh_returns_false_for_untracked_ticket(tm_harness):
    tm_harness.mt5.get_positions.return_value = []
    assert tm_harness.manager._refresh_remaining_volume(9999) is False


# ===================================================================
# §6.8 — ClosedTrade composite uniqueness
# ===================================================================


@pytest.fixture
def db_harness(tmp_path):
    from src.database_manager import DatabaseManager

    db_url = f"sqlite:///{tmp_path / 'phase3c.db'}"
    db = DatabaseManager(database_url=db_url)
    db.create_tables()
    yield db
    db.close()


def _closed_trade(ticket: int, environment: str = "demo", price_close=2010.0):
    from src.database_manager import ClosedTrade

    return ClosedTrade(
        ticket=ticket,
        magic_number=123456789,
        symbol="GOLD",
        direction="BUY",
        volume=0.10,
        price_open=2000.0,
        price_close=price_close,
        sl=1995.0,
        tp=2010.0,
        profit=100.0,
        swap=0.0,
        commission=-1.40,
        net_profit=98.6,
        risk_percent=1.0,
        realized_r=2.0,
        strategy="test",
        entry_reason=None,
        exit_reason="TP",
        time_open=datetime.now(timezone.utc) - timedelta(hours=1),
        time_close=datetime.now(timezone.utc),
        duration_hours=1.0,
        environment=environment,
    )


def test_unique_index_exists_after_create_tables(db_harness):
    """Migration must have created the composite unique index."""
    from sqlalchemy import text

    with db_harness.get_session() as session:
        rows = session.execute(text(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'index' AND name = 'ix_closed_trades_ticket_env'"
        )).fetchall()
    assert len(rows) == 1


def test_duplicate_ticket_env_rejected_by_index(db_harness):
    """Inserting two ClosedTrades with same (ticket, environment) must
    raise IntegrityError, not silently succeed."""
    from sqlalchemy.exc import IntegrityError

    db_harness.add(_closed_trade(ticket=6001, environment="demo"))

    with pytest.raises(IntegrityError):
        db_harness.add(_closed_trade(ticket=6001, environment="demo"))


def test_same_ticket_across_environments_allowed(db_harness):
    """A ticket can exist once in DEMO and once in LIVE — separately."""
    db_harness.add(_closed_trade(ticket=6002, environment="demo"))
    db_harness.add(_closed_trade(ticket=6002, environment="live"))

    with db_harness.get_session() as session:
        from src.database_manager import ClosedTrade
        count = session.query(ClosedTrade).filter(
            ClosedTrade.ticket == 6002,
        ).count()
    assert count == 2


def test_migration_skips_index_when_duplicates_exist(tmp_path):
    """If the DB already contains duplicates, migration must NOT fail;
    it must log a warning and skip index creation."""
    from sqlalchemy import text
    from src.database_manager import DatabaseManager

    db_url = f"sqlite:///{tmp_path / 'prepolluted.db'}"

    # 1. Create schema WITHOUT the unique index by using a plain engine.
    from src.database_manager import Base
    from sqlalchemy import create_engine
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    # 2. Insert two duplicate (ticket, environment) rows directly.
    with engine.begin() as conn:
        for _ in range(2):
            conn.execute(text(
                "INSERT INTO closed_trades "
                "(ticket, magic_number, symbol, direction, volume, "
                " price_open, price_close, sl, tp, profit, swap, "
                " commission, net_profit, risk_percent, realized_r, "
                " strategy, time_open, time_close, duration_hours, "
                " environment) "
                "VALUES (7001, 123456789, 'GOLD', 'BUY', 0.1, "
                "        2000.0, 2010.0, 1995.0, 2010.0, 100.0, 0.0, "
                "        -1.4, 98.6, 1.0, 2.0, 'test', "
                "        '2024-01-01 00:00:00', '2024-01-01 01:00:00', "
                "        1.0, 'demo')"
            ))
    engine.dispose()

    # 3. Now open via DatabaseManager and run the migration.
    db = DatabaseManager(database_url=db_url)
    db.create_tables()   # Must NOT raise on duplicates.

    # 4. Index must NOT exist (skipped due to duplicates).
    with db.get_session() as session:
        rows = session.execute(text(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'index' AND name = 'ix_closed_trades_ticket_env'"
        )).fetchall()
    assert len(rows) == 0
    db.close()


def test_migration_is_idempotent(db_harness):
    """Running create_tables three times must not raise."""
    db_harness.create_tables()
    db_harness.create_tables()
    db_harness.create_tables()
    # Implicit assertion: no exception raised.