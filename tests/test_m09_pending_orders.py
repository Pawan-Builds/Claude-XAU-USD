"""M-09 regression tests: shutdown/weekend pending-order handling.

Asserts:
  * the configured default is "cancel" (settings.yaml + code fallback);
  * close_all_for_weekend cancels pending orders BEFORE closing positions;
  * close_all_for_weekend verifies post-close pending & position counts;
  * close_all_for_weekend returns ok=False when any pending or position
    remains, when cancel failed, or when any query failed;
  * _handle_weekend_close does NOT persist the idempotency marker when
    pending orders remain;
  * shutdown verifies post-cancel state and logs CRITICAL when pending
    orders survive.

Tests use MagicMock and TradingAgent.__new__ to avoid constructing the
full agent (which would load real config and touch MT5).
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_close_result(closed=0, failed=0, query_ok=True):
    return SimpleNamespace(closed=closed, failed=failed, query_ok=query_ok)


def _make_cancel_result(succeeded=0, failed=0, query_ok=True):
    return SimpleNamespace(
        succeeded=succeeded, failed=failed, query_ok=query_ok,
        results=[None] * (succeeded + failed),
    )


# ---------------------------------------------------------------------
# Settings default
# ---------------------------------------------------------------------

def test_settings_default_pending_policy_is_cancel():
    settings = yaml.safe_load(
        (PROJECT_ROOT / "config" / "settings.yaml").read_text(
            encoding="utf-8"
        )
    )
    policy = settings.get("trading", {}).get(
        "pending_orders_on_shutdown"
    )
    assert policy == "cancel", (
        "settings.yaml must default pending_orders_on_shutdown to "
        "'cancel' (M-09). Found: {!r}".format(policy)
    )


# ---------------------------------------------------------------------
# close_all_for_weekend behaviour
# ---------------------------------------------------------------------

def _patch_controller(monkeypatch):
    """Return a controller whose collaborators are controlled mocks."""
    from src import trading_controller as tmod

    calls = []

    fake_exec = MagicMock()
    fake_exec.cancel_all_pending.side_effect = lambda reason: (
        calls.append(("cancel", reason))
        or _make_cancel_result(succeeded=1)
    )
    fake_exec.close_all_positions.side_effect = lambda reason: (
        calls.append(("close", reason))
        or _make_close_result(closed=1)
    )
    monkeypatch.setattr(tmod, "order_executor", fake_exec)

    fake_conn = MagicMock()
    fake_conn.get_orders.return_value = []
    fake_conn.get_positions.return_value = []
    monkeypatch.setattr(tmod, "mt5_connector", fake_conn)

    return tmod.trading_controller, calls


def test_weekend_close_cancels_pending_before_closing(monkeypatch):
    ctrl, calls = _patch_controller(monkeypatch)
    result = ctrl.close_all_for_weekend()
    assert result["ok"] is True
    assert [c[0] for c in calls] == ["cancel", "close"], (
        "close_all_for_weekend must cancel pending orders BEFORE "
        f"closing positions. Calls: {calls}"
    )
    assert result["cancelled"] == 1
    assert result["closed"] == 1


def test_weekend_close_verifies_post_state(monkeypatch):
    ctrl, _ = _patch_controller(monkeypatch)
    # Override: post-close query returns one pending order.
    ctrl_conn = None
    for k, v in monkeypatch._setitem:
        pass  # placeholder, we re-patch below
    from src import trading_controller as tmod
    tmod.mt5_connector.get_orders.return_value = [{"ticket": 42}]
    tmod.mt5_connector.get_positions.return_value = []

    result = ctrl.close_all_for_weekend()
    assert result["pending_remaining"] == 1
    assert result["ok"] is False, (
        "close_all_for_weekend must report ok=False when a pending "
        "order survives the cancel attempt."
    )


def test_weekend_close_fails_on_cancel_failure(monkeypatch):
    from src import trading_controller as tmod
    fake_exec = MagicMock()
    fake_exec.cancel_all_pending.return_value = _make_cancel_result(
        succeeded=0, failed=1,
    )
    fake_exec.close_all_positions.return_value = _make_close_result(
        closed=1,
    )
    monkeypatch.setattr(tmod, "order_executor", fake_exec)
    fake_conn = MagicMock()
    fake_conn.get_orders.return_value = [{"ticket": 42}]
    fake_conn.get_positions.return_value = []
    monkeypatch.setattr(tmod, "mt5_connector", fake_conn)

    result = tmod.trading_controller.close_all_for_weekend()
    assert result["ok"] is False
    assert result["cancel_failed"] == 1
    assert result["pending_remaining"] == 1


def test_weekend_close_fails_when_query_returns_none(monkeypatch):
    from src import trading_controller as tmod
    fake_exec = MagicMock()
    fake_exec.cancel_all_pending.return_value = _make_cancel_result()
    fake_exec.close_all_positions.return_value = _make_close_result()
    monkeypatch.setattr(tmod, "order_executor", fake_exec)
    fake_conn = MagicMock()
    fake_conn.get_orders.return_value = None
    fake_conn.get_positions.return_value = []
    monkeypatch.setattr(tmod, "mt5_connector", fake_conn)

    result = tmod.trading_controller.close_all_for_weekend()
    assert result["ok"] is False
    assert result["query_ok"] is False
    assert result["pending_remaining"] == -1


# ---------------------------------------------------------------------
# _handle_weekend_close does not persist the marker on failure
# ---------------------------------------------------------------------

def test_handle_weekend_close_does_not_mark_when_pending_remain(
    monkeypatch,
):
    from src.main import TradingAgent
    from src import main as mmod
    from datetime import datetime as real_datetime, timezone

    agent = TradingAgent.__new__(TradingAgent)
    agent._WEEKEND_MARKER_KEY = "last_weekend_close_date"
    agent._telegram_ready = False

    # 1. Fake config with a fixed weekday-close window.
    fake_config = SimpleNamespace(
        sessions=SimpleNamespace(
            weekend_protection=True,
            close_before_weekend_hours=4,
        ),
    )
    monkeypatch.setattr(mmod, "config", fake_config, raising=False)

    # 2. Force weekend-protection to be active.
    monkeypatch.setattr(
        mmod.market_data,
        "is_weekend_protection_active",
        lambda: True,
    )

    # 3. Controller returns a result with a surviving pending order.
    fake_controller = MagicMock()
    fake_controller.close_all_for_weekend.return_value = {
        "ok": False,
        "closed": 0,
        "failed": 0,
        "cancelled": 0,
        "cancel_failed": 0,
        "pending_remaining": 1,
        "positions_remaining": 0,
        "query_ok": True,
        "error": None,
    }
    monkeypatch.setattr(mmod, "trading_controller", fake_controller)

    # 4. Fake DB manager: record any get_session() attempt, which is
    #    the code path used to persist the weekend marker.
    session_attempts: list = []

    class _FakeSession:
        def __enter__(self):
            session_attempts.append("entered")
            return self
        def __exit__(self, *args):
            return False
        def query(self, *args, **kwargs):
            return self
        def filter_by(self, **kw):
            return self
        def first(self):
            return None
        def add(self, obj):
            session_attempts.append(("add", type(obj).__name__))

    class _FakeDbManager:
        def query_one(self, *a, **kw):
            # No existing marker: proceed past the idempotency guard.
            return None
        def get_session(self):
            return _FakeSession()

    monkeypatch.setattr(
        mmod, "db_manager", _FakeDbManager(), raising=False
    )

    # 5. Pin "now" to a Friday so week_key is computed and the
    #    Monday-through-Thursday early return is not triggered.
    fixed_now = real_datetime(2025, 1, 3, 20, 0, tzinfo=timezone.utc)

    class _FakeDatetime:
        @staticmethod
        def now(tz=None):
            return fixed_now if tz is None else fixed_now.astimezone(tz)

    monkeypatch.setattr(mmod, "datetime", _FakeDatetime)

    agent._handle_weekend_close()

    # Assertions:
    fake_controller.close_all_for_weekend.assert_called_once()
    assert session_attempts == [], (
        "weekend marker must NOT be persisted when pending orders "
        "remain (pending_remaining=1); _handle_weekend_close must "
        "return before writing the marker so the close retries next "
        f"cycle. Observed db writes: {session_attempts}"
    )


# ---------------------------------------------------------------------
# Shutdown policy default and verification
# ---------------------------------------------------------------------

def test_shutdown_default_policy_is_cancel():
    """Sanity: code fallback must also be 'cancel', not 'keep'."""
    src = (PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
    assert 'getattr(config.trading, "pending_orders_on_shutdown", "cancel")' in src, (
        "shutdown() must use 'cancel' as the fail-safe default policy."
    )
    assert 'pending_orders_on_shutdown", "keep"' not in src, (
        "'keep' must not remain as a hardcoded fallback default."
    )


def test_shutdown_verifies_pending_state_after_cancel():
    src = (PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
    assert "SHUTDOWN PENDING VERIFICATION FAILED" in src, (
        "shutdown must log CRITICAL when pending orders survive the "
        "cancel attempt (M-09 verification gate)."
    )
    assert "remaining = mt5_connector.get_orders()" in src, (
        "shutdown must re-query get_orders() to verify cancellation."
    )