"""Phase 3D (§6.6): pending-order cancellation failures must be
classified, escalated, and reported honestly — never silently retried
forever or reported as success when the broker query failed."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _order_result(success, message=None):
    return SimpleNamespace(
        success=success, error_message=message, retcode=None,
    )


@pytest.fixture
def harness(monkeypatch):
    from src import order_executor as oe_module
    from src.database_manager import DatabaseManager

    db = DatabaseManager("sqlite:///:memory:")
    db.create_tables()
    monkeypatch.setattr(oe_module, "db_manager", db)

    mt5c = MagicMock(name="mt5_connector")
    mt5c.is_connected.return_value = True
    mt5c.get_orders.return_value = []
    mt5c.delete_order.return_value = _order_result(True)

    fake_config = SimpleNamespace(
        trading=SimpleNamespace(
            magic_number=123456789, mode="DEMO",
            pending_expiry_hours=4, max_pending_orders=5,
        ),
        trade_management=SimpleNamespace(max_slippage_pips=3.0),
        live_trading_enabled=False,
    )

    monkeypatch.setattr(oe_module, "mt5_connector", mt5c)
    monkeypatch.setattr(oe_module, "config", fake_config)

    from src.order_executor import OrderExecutor
    return SimpleNamespace(
        executor=OrderExecutor(), db=db, mt5=mt5c,
    )


def _insert_placed_request(db, ticket, status="placed"):
    from src.database_manager import OrderRequest
    return db.add(OrderRequest(
        timestamp=datetime.utcnow(),
        magic_number=123456789, symbol="GOLD",
        order_type="BUY_LIMIT", volume=0.1, price=2000.0,
        sl=1990.0, tp=2020.0, comment="SMC_TEST",
        environment="demo", fingerprint="fp",
        mt5_ticket=ticket, status=status,
    ))


# ---------- cancel_all_pending: honest query reporting ----------


def test_cancel_all_pending_reports_query_failure(harness):
    """If get_orders() returns None, query_ok must be False."""
    harness.mt5.get_orders.return_value = None

    result = harness.executor.cancel_all_pending(reason="TEST")

    assert result.query_ok is False
    assert len(result) == 0


def test_cancel_all_pending_iterable_and_countable(harness):
    """Backward-compat: len() and iteration still work."""
    harness.mt5.get_orders.return_value = [
        {"ticket": 100},
        {"ticket": 101},
    ]

    result = harness.executor.cancel_all_pending(reason="TEST")

    assert result.query_ok is True
    assert len(result) == 2
    assert result.succeeded == 2
    assert [r.success for r in result] == [True, True]


# ---------- cancel_pending_order: classification ----------


def test_cancel_pending_order_marks_db_on_success(harness):
    from src.database_manager import OrderRequest
    req_id = _insert_placed_request(harness.db, ticket=5001)
    harness.mt5.delete_order.return_value = _order_result(True)

    result = harness.executor.cancel_pending_order(5001, reason="TEST")

    assert result.success is True
    with harness.db.get_session() as s:
        req = s.query(OrderRequest).filter(OrderRequest.id == req_id).first()
        assert req.status == "cancelled"


def test_cancel_pending_order_benign_on_not_found(harness):
    """'not found' does not increment the failure counter."""
    harness.mt5.delete_order.return_value = _order_result(
        False, "Pending order not found",
    )

    result = harness.executor.cancel_pending_order(5002, reason="TEST")

    assert result.success is False
    assert 5002 not in harness.executor._cancel_failures
    assert harness.executor.get_stuck_cancellations() == []


def test_cancel_pending_order_records_retryable_failure(harness):
    harness.mt5.delete_order.return_value = _order_result(
        False, "broker rejected",
    )

    harness.executor.cancel_pending_order(5003, reason="TEST")

    assert harness.executor._cancel_failures[5003] == 1
    assert harness.executor.get_stuck_cancellations() == []


def test_cancel_pending_order_escalates_after_threshold(harness):
    harness.mt5.delete_order.return_value = _order_result(
        False, "broker rejected",
    )

    for _ in range(3):
        harness.executor.cancel_pending_order(5004, reason="TEST")

    stuck = harness.executor.get_stuck_cancellations()
    assert 5004 in stuck


def test_cancel_pending_order_escalates_immediately_on_permanent(harness):
    """'does not belong to this bot' is permanent — no 3-strike wait."""
    harness.mt5.delete_order.return_value = _order_result(
        False, "Pending order does not belong to this bot",
    )

    harness.executor.cancel_pending_order(5005, reason="TEST")

    stuck = harness.executor.get_stuck_cancellations()
    assert 5005 in stuck


def test_stuck_ticket_cleared_on_success(harness):
    """After escalation, a later successful cancel clears the marker."""
    # Escalate first.
    harness.mt5.delete_order.return_value = _order_result(
        False, "broker rejected",
    )
    for _ in range(3):
        harness.executor.cancel_pending_order(5006, reason="TEST")
    assert 5006 in harness.executor.get_stuck_cancellations()

    # Now the broker accepts.
    harness.mt5.delete_order.return_value = _order_result(True)
    harness.executor.cancel_pending_order(5006, reason="TEST")

    assert 5006 not in harness.executor.get_stuck_cancellations()
    assert 5006 not in harness.executor._cancel_failures


# ---------- expiry / cap use cancel_pending_order ----------


def test_expired_orders_use_cancel_pending_order(harness, monkeypatch):
    """_cancel_expired_orders must route through cancel_pending_order
    (so DB updates and failure escalation both fire)."""
    calls = []
    original = harness.executor.cancel_pending_order

    def spy(ticket, reason=""):
        calls.append((ticket, reason))
        return original(ticket, reason)

    monkeypatch.setattr(harness.executor, "cancel_pending_order", spy)

    old = datetime.now(timezone.utc) - timedelta(hours=10)
    harness.executor._cancel_expired_orders([
        {"ticket": 7001, "time_setup": old},
    ])

    assert calls == [(7001, "EXPIRED")]


def test_cap_uses_cancel_pending_order(harness, monkeypatch):
    calls = []
    original = harness.executor.cancel_pending_order

    def spy(ticket, reason=""):
        calls.append((ticket, reason))
        return original(ticket, reason)

    monkeypatch.setattr(harness.executor, "cancel_pending_order", spy)

    now = datetime.now(timezone.utc)
    # cap = 5. 7 live orders → cancel 2 oldest.
    live = [
        {"ticket": 100 + i, "time_setup": now - timedelta(hours=10 - i)}
        for i in range(7)
    ]
    harness.executor._enforce_pending_order_cap(live)

    assert [c[0] for c in calls] == [100, 101]
    assert all(c[1] == "CAP" for c in calls)


def test_cap_skips_stuck_orders(harness, monkeypatch):
    """Stuck tickets are excluded from cap enforcement; newest survive."""
    # Mark 100 (oldest) as stuck.
    harness.mt5.delete_order.return_value = _order_result(
        False, "broker rejected",
    )
    for _ in range(3):
        harness.executor.cancel_pending_order(100, reason="PRESEED")
    assert 100 in harness.executor.get_stuck_cancellations()

    calls = []
    original = harness.executor.cancel_pending_order

    def spy(ticket, reason=""):
        calls.append(ticket)
        return original(ticket, reason)

    monkeypatch.setattr(harness.executor, "cancel_pending_order", spy)

    now = datetime.now(timezone.utc)
    live = [
        {"ticket": 100 + i, "time_setup": now - timedelta(hours=10 - i)}
        for i in range(7)
    ]
    harness.executor._enforce_pending_order_cap(live)

    # 7 > 5 → need to trim 2; 100 is stuck so we trim 101 and 102.
    assert 100 not in calls
    assert calls == [101, 102]


# ---------- stuck-state pruning ----------


def test_stuck_marker_pruned_when_order_gone(harness):
    """If broker no longer reports the stuck ticket, marker is cleared."""
    harness.mt5.delete_order.return_value = _order_result(
        False, "broker rejected",
    )
    for _ in range(3):
        harness.executor.cancel_pending_order(6001, reason="PRESEED")
    assert 6001 in harness.executor.get_stuck_cancellations()

    # Broker no longer reports ticket 6001 → reconcile should prune.
    harness.mt5.get_orders.return_value = []
    harness.executor.reconcile_pending_orders()

    assert 6001 not in harness.executor.get_stuck_cancellations()