"""Kill-switch persistence regression tests.

Verifies:
  * set_kill_switch returns ok=True only when read-back agrees
  * write failure → ok=False and error surfaced (never swallowed)
  * read-back unavailable → ok=False
  * read-back mismatch → ok=False with the observed state
  * failed deactivation leaves the persisted state active
  * TradingController.set_kill_switch surfaces failures to callers
  * TradingController.set_kill_switch reports success only when verified
  * emergency_stop does NOT report ok=True when the kill-switch write failed
"""

from types import SimpleNamespace

import pytest

from src.database_manager import (
    DatabaseManager,
    AppSettings,
)
from src.risk_manager import RiskManager, KillSwitchResult


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "ks_test.db"
    mgr = DatabaseManager(database_url=f"sqlite:///{db_path}")
    mgr.create_tables()

    import src.risk_manager as rm_mod
    monkeypatch.setattr(rm_mod, "db_manager", mgr)

    yield mgr
    mgr.close()


# ---------------------------------------------------------------------------
# 1. Success path — write + read-back agreement
# ---------------------------------------------------------------------------

def test_set_kill_switch_success_verified(temp_db):
    rm = RiskManager()
    assert rm.get_kill_switch_state_strict() is False

    result = rm.set_kill_switch(True, "test")
    assert result.ok is True
    assert result.active is True
    assert rm.get_kill_switch_state_strict() is True

    result = rm.set_kill_switch(False, "test")
    assert result.ok is True
    assert result.active is False
    assert rm.get_kill_switch_state_strict() is False


# ---------------------------------------------------------------------------
# 2. Write failure → ok=False, error surfaced
# ---------------------------------------------------------------------------

def test_set_kill_switch_write_failure_returns_ok_false(
    temp_db, monkeypatch
):
    rm = RiskManager()

    class FailOnWrite:
        def get_session(self):
            raise RuntimeError("disk full")

        def query_one(self, *args, **kwargs):
            return temp_db.query_one(*args, **kwargs)

    import src.risk_manager as rm_mod
    monkeypatch.setattr(rm_mod, "db_manager", FailOnWrite())

    result = rm.set_kill_switch(True, "test")
    assert result.ok is False
    assert "disk full" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# 3. Read-back unavailable → ok=False
# ---------------------------------------------------------------------------

def test_set_kill_switch_readback_unavailable_returns_ok_false(
    temp_db, monkeypatch
):
    rm = RiskManager()
    monkeypatch.setattr(
        rm, "get_kill_switch_state_strict", lambda: None,
    )
    result = rm.set_kill_switch(True, "test")
    assert result.ok is False
    assert "read-back" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# 4. Read-back mismatch → ok=False and observed state reported
# ---------------------------------------------------------------------------

def test_set_kill_switch_readback_mismatch_returns_ok_false(
    temp_db, monkeypatch
):
    rm = RiskManager()
    monkeypatch.setattr(
        rm, "get_kill_switch_state_strict", lambda: False,
    )
    result = rm.set_kill_switch(True, "test")
    assert result.ok is False
    assert result.active is False
    assert "mismatch" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# 5. Failed deactivation leaves persisted state active
# ---------------------------------------------------------------------------

def test_failed_deactivation_leaves_state_active(
    temp_db, monkeypatch
):
    rm = RiskManager()
    rm.set_kill_switch(True, "setup")
    assert rm.get_kill_switch_state_strict() is True

    real_db = temp_db

    class FailOnWrite:
        def get_session(self):
            raise RuntimeError("write refused")

        def query_one(self, *args, **kwargs):
            return real_db.query_one(*args, **kwargs)

    import src.risk_manager as rm_mod
    monkeypatch.setattr(rm_mod, "db_manager", FailOnWrite())

    result = rm.set_kill_switch(False, "test")
    assert result.ok is False

    # Persisted state is untouched (still active).
    row = real_db.query_one(AppSettings, key="kill_switch")
    assert row is not None
    assert row.value == "true"


# ---------------------------------------------------------------------------
# 6. TradingController surfaces failure (never reports success on failure)
# ---------------------------------------------------------------------------

def test_controller_set_kill_switch_returns_failure_on_persist_failure(
    monkeypatch,
):
    from src.trading_controller import TradingController
    import src.trading_controller as tc_mod

    ctrl = TradingController()

    fake_rm = SimpleNamespace(
        set_kill_switch=lambda active, reason="": KillSwitchResult(
            ok=False, active=False, error="db down",
        ),
    )
    monkeypatch.setattr(tc_mod, "risk_manager", fake_rm)

    result = ctrl.set_kill_switch(True)
    assert result["ok"] is False
    assert result["error"] == "db down"
    assert result["active"] is False


# ---------------------------------------------------------------------------
# 7. Controller reports success only when RiskManager verified
# ---------------------------------------------------------------------------

def test_controller_set_kill_switch_returns_success_only_when_verified(
    monkeypatch,
):
    from src.trading_controller import TradingController
    import src.trading_controller as tc_mod

    ctrl = TradingController()

    fake_rm = SimpleNamespace(
        set_kill_switch=lambda active, reason="": KillSwitchResult(
            ok=True, active=True,
        ),
    )
    monkeypatch.setattr(tc_mod, "risk_manager", fake_rm)

    result = ctrl.set_kill_switch(True)
    assert result["ok"] is True
    assert result["active"] is True
    assert "error" not in result


# ---------------------------------------------------------------------------
# 8. emergency_stop does not report ok=True when kill-switch write failed
# ---------------------------------------------------------------------------

def test_emergency_stop_reports_not_ok_on_ks_failure(monkeypatch):
    from src.trading_controller import TradingController
    from src.order_executor import CloseAllResult
    import src.trading_controller as tc_mod

    ctrl = TradingController()

    fake_rm = SimpleNamespace(
        set_kill_switch=lambda active, reason="": KillSwitchResult(
            ok=False, active=False, error="db down",
        ),
        is_kill_switch_active=lambda: False,
    )
    fake_oe = SimpleNamespace(
        close_all_positions=lambda reason: CloseAllResult(
            results=[], query_ok=True,
        ),
    )
    monkeypatch.setattr(tc_mod, "risk_manager", fake_rm)
    monkeypatch.setattr(tc_mod, "order_executor", fake_oe)

    result = ctrl.emergency_stop(reason="test")
    assert result["ok"] is False
    assert result["kill_switch_verified"] is False
    assert result["kill_switch_written"] is False