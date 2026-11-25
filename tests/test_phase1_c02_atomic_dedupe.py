"""C-02 regression tests: atomic dedupe + uncertain state persistence.

Verifies:
  * uncertain transition is a SINGLE write, never rejected-then-uncertain
  * uncertain transition raises on DB failure (caller can fail closed)
  * uncertain transition refuses from an unexpected terminal status
  * atomic insert blocks an existing pending row
  * atomic insert blocks an existing uncertain row
  * atomic insert ALLOWS after a rejected row (terminal)
  * atomic insert ALLOWS after a filled row (terminal)
  * the partial UNIQUE index exists with the exact WHERE clause

All tests use an isolated temp SQLite DB; no broker, no MT5.
"""

import pytest

from src.database_manager import (
    DatabaseManager,
    OrderRequest,
)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Isolated DatabaseManager with the full schema, bound to both the
    database_manager module AND the order_executor module."""
    db_path = tmp_path / "c02_test.db"
    mgr = DatabaseManager(database_url=f"sqlite:///{db_path}")
    mgr.create_tables()

    import src.database_manager as dm_mod
    import src.order_executor as oe_mod

    monkeypatch.setattr(dm_mod, "db_manager", mgr)
    monkeypatch.setattr(oe_mod, "db_manager", mgr)

    yield mgr
    mgr.close()


def _make_request(fingerprint, status="pending", **overrides):
    kwargs = dict(
        signal_id=None,
        magic_number=12345,
        symbol="XAUUSD",
        order_type="BUY",
        volume=0.1,
        environment="demo",
        fingerprint=fingerprint,
        status=status,
    )
    kwargs.update(overrides)
    return OrderRequest(**kwargs)


# ---------------------------------------------------------------------------
# 1. Uncertain transition is atomic (no intermediate rejected)
# ---------------------------------------------------------------------------

def test_uncertain_transition_persists_uncertain_not_rejected(
    temp_db, monkeypatch
):
    from src.order_executor import OrderExecutor

    executor = OrderExecutor()

    status_calls = []
    orig = executor._mark_request_status

    def spy(request_id, status):
        status_calls.append((request_id, status))
        return orig(request_id, status)

    monkeypatch.setattr(executor, "_mark_request_status", spy)

    req = _make_request("fp-atomic", status="pending")
    req_id = temp_db.add(req)

    executor._mark_request_uncertain(req_id, "broker timeout")

    row = temp_db.query_one(OrderRequest, id=req_id)
    assert row.status == "uncertain"
    assert row.error_message == "broker timeout"

    rejected_calls = [
        (rid, s) for rid, s in status_calls if s == "rejected"
    ]
    assert rejected_calls == [], (
        f"Uncertain transition must not pass through 'rejected'. "
        f"Unexpected calls: {rejected_calls}"
    )


# ---------------------------------------------------------------------------
# 2. Uncertain transition raises on DB failure (fail closed)
# ---------------------------------------------------------------------------

def test_uncertain_transition_raises_on_db_failure(
    temp_db, monkeypatch
):
    from src.order_executor import OrderExecutor

    executor = OrderExecutor()

    class BrokenDB:
        def get_session(self):
            raise RuntimeError("DB down")

    import src.order_executor as oe_mod
    monkeypatch.setattr(oe_mod, "db_manager", BrokenDB())

    # Must raise, not swallow.
    with pytest.raises(Exception):
        executor._mark_request_uncertain(1, "broker timeout")


# ---------------------------------------------------------------------------
# 3. Uncertain transition refuses from a terminal status
# ---------------------------------------------------------------------------

def test_uncertain_transition_raises_from_terminal_status(temp_db):
    from src.order_executor import OrderExecutor

    executor = OrderExecutor()
    req = _make_request("fp-terminal", status="rejected")
    req_id = temp_db.add(req)

    with pytest.raises(RuntimeError, match="Refusing uncertain transition"):
        executor._mark_request_uncertain(req_id, "late arrival")

    row = temp_db.query_one(OrderRequest, id=req_id)
    assert row.status == "rejected", "Status must remain terminal"


# ---------------------------------------------------------------------------
# 4. Atomic insert blocks a pending active duplicate
# ---------------------------------------------------------------------------

def test_atomic_insert_duplicate_pending_returns_false(temp_db):
    first = _make_request("fp-dup-pending", status="pending")
    ok, row_id = temp_db.insert_order_request_atomic(first)
    assert ok is True
    assert row_id is not None

    second = _make_request("fp-dup-pending", status="pending")
    ok2, existing = temp_db.insert_order_request_atomic(second)
    assert ok2 is False
    assert existing == row_id, (
        f"Expected existing id {row_id}, got {existing}"
    )


# ---------------------------------------------------------------------------
# 5. Atomic insert blocks an uncertain active duplicate
# ---------------------------------------------------------------------------

def test_atomic_insert_duplicate_uncertain_returns_false(temp_db):
    first = _make_request("fp-dup-uncertain", status="uncertain")
    ok, row_id = temp_db.insert_order_request_atomic(first)
    assert ok is True

    second = _make_request("fp-dup-uncertain", status="pending")
    ok2, existing = temp_db.insert_order_request_atomic(second)
    assert ok2 is False
    assert existing == row_id


# ---------------------------------------------------------------------------
# 6. Atomic insert allows a new attempt after a terminal 'rejected'
# ---------------------------------------------------------------------------

def test_atomic_insert_allows_after_rejected(temp_db):
    first = _make_request("fp-reuse-rejected", status="rejected")
    ok, _ = temp_db.insert_order_request_atomic(first)
    assert ok is True

    second = _make_request("fp-reuse-rejected", status="pending")
    ok2, row_id2 = temp_db.insert_order_request_atomic(second)
    assert ok2 is True
    assert row_id2 is not None


# ---------------------------------------------------------------------------
# 7. Atomic insert allows a new attempt after a terminal 'filled'
# ---------------------------------------------------------------------------

def test_atomic_insert_allows_after_filled(temp_db):
    first = _make_request("fp-reuse-filled", status="filled")
    ok, _ = temp_db.insert_order_request_atomic(first)
    assert ok is True

    second = _make_request("fp-reuse-filled", status="pending")
    ok2, _ = temp_db.insert_order_request_atomic(second)
    assert ok2 is True


# ---------------------------------------------------------------------------
# 8. Partial unique index exists with the expected WHERE filter
# ---------------------------------------------------------------------------

def test_partial_unique_index_exists_with_active_status_filter(temp_db):
    import sqlite3

    db_path = temp_db.database_url.replace("sqlite:///", "", 1)
    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE name = ?",
            ("uq_order_requests_active_fp",),
        ).fetchone()
    finally:
        con.close()

    assert row is not None, "Partial unique index not found"
    sql = row[0]
    for required in (
        "UNIQUE",
        "fingerprint",
        "environment",
        "'placed'",
        "'pending'",
        "'sent'",
        "'filled_unconfirmed'",
        "'uncertain'",
        "'submission_error'",
    ):
        assert required in sql, (
            f"Index SQL missing required token: {required!r}\n"
            f"Full SQL: {sql}"
        )