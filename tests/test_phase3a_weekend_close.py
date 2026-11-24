"""Phase 3A: bot-owned positions must be closed once per weekend when
config.sessions.close_before_weekend_hours > 0."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Wire a fresh in-memory DB and mock the agent's collaborators."""
    from src import main as main_mod
    from src.database_manager import DatabaseManager

    db = DatabaseManager("sqlite:///:memory:")
    db.create_tables()
    monkeypatch.setattr(main_mod, "db_manager", db)

    # Config: replace the module-level `config` object in main_mod with
    # a namespace. `config.sessions` is a @property on the real Config
    # class (no setter), so patching the property on the instance is not
    # possible. Replacing `main_mod.config` is how every other test in
    # this suite mocks configuration.
    cfg_sessions = SimpleNamespace(
        weekend_protection=True,
        close_before_weekend_hours=4,
        timezone="UTC",
    )
    fake_config = SimpleNamespace(
        sessions=cfg_sessions,
        trading=SimpleNamespace(mode="DEMO", magic_number=123456789),
    )
    monkeypatch.setattr(main_mod, "config", fake_config)

    # Weekend state — the knob that gates the close call.
    md = MagicMock(name="market_data")
    md.is_weekend_protection_active.return_value = False
    monkeypatch.setattr(main_mod, "market_data", md)

    # The close call itself.
    tc = MagicMock(name="trading_controller")
    tc.close_all_for_weekend.return_value = {
        "ok": True, "closed": 2, "failed": 0,
        "query_ok": True, "error": None,
    }
    monkeypatch.setattr(main_mod, "trading_controller", tc)

    # Avoid spinning up the real Telegram client.
    agent = main_mod.TradingAgent.__new__(main_mod.TradingAgent)
    agent._telegram_ready = False
    agent._WEEKEND_MARKER_KEY = "last_weekend_close_date"

    return SimpleNamespace(
        agent=agent,
        cfg_sessions=cfg_sessions,
        md=md,
        tc=tc,
        db=db,
        main_mod=main_mod,
    )


def _set_weekend_true(harness):
    harness.md.is_weekend_protection_active.return_value = True


def _set_weekend_false(harness):
    harness.md.is_weekend_protection_active.return_value = False


def _force_now(monkeypatch, when):
    """Patch datetime.now inside src.main to return `when`."""
    import src.main as main_mod

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return when

    monkeypatch.setattr(main_mod, "datetime", _Frozen, raising=False)


# ---------- negative cases ----------

def test_noop_when_weekend_protection_disabled(harness):
    harness.cfg_sessions.weekend_protection = False
    _set_weekend_true(harness)
    harness.agent._handle_weekend_close()
    harness.tc.close_all_for_weekend.assert_not_called()


def test_noop_when_close_hours_is_zero(harness):
    harness.cfg_sessions.close_before_weekend_hours = 0
    _set_weekend_true(harness)
    harness.agent._handle_weekend_close()
    harness.tc.close_all_for_weekend.assert_not_called()


def test_noop_when_not_yet_weekend(harness):
    _set_weekend_false(harness)
    harness.agent._handle_weekend_close()
    harness.tc.close_all_for_weekend.assert_not_called()


# ---------- behaviour on a weekend ----------

def test_closes_when_weekend_active_and_not_yet_closed(
    harness, monkeypatch,
):
    _force_now(
        monkeypatch,
        datetime(2025, 1, 4, 21, 0, tzinfo=timezone.utc),  # Saturday
    )
    _set_weekend_true(harness)

    harness.agent._handle_weekend_close()

    harness.tc.close_all_for_weekend.assert_called_once()


def test_marker_persists_and_prevents_repeat(harness, monkeypatch):
    _force_now(
        monkeypatch,
        datetime(2025, 1, 4, 21, 0, tzinfo=timezone.utc),
    )
    _set_weekend_true(harness)

    harness.agent._handle_weekend_close()
    assert harness.tc.close_all_for_weekend.call_count == 1

    # Second call on the same weekend: marker prevents double-fire.
    harness.agent._handle_weekend_close()
    assert harness.tc.close_all_for_weekend.call_count == 1

    # Marker was written.
    from src.database_manager import AppSettings
    marker = harness.db.query_one(
        AppSettings, key="last_weekend_close_date",
    )
    assert marker is not None
    assert marker.value == "2025-01-03"  # Friday of that week


def test_marker_is_reusable_next_weekend(harness, monkeypatch):
    _force_now(
        monkeypatch,
        datetime(2025, 1, 4, 21, 0, tzinfo=timezone.utc),  # Sat 2025-01-04
    )
    _set_weekend_true(harness)
    harness.agent._handle_weekend_close()
    assert harness.tc.close_all_for_weekend.call_count == 1

    # Next weekend (Sat 2025-01-11): different marker → fires again.
    _force_now(
        monkeypatch,
        datetime(2025, 1, 11, 21, 0, tzinfo=timezone.utc),
    )
    harness.agent._handle_weekend_close()
    assert harness.tc.close_all_for_weekend.call_count == 2


# ---------- failure handling ----------

def test_does_not_mark_done_when_query_failed(harness, monkeypatch):
    _force_now(
        monkeypatch,
        datetime(2025, 1, 4, 21, 0, tzinfo=timezone.utc),
    )
    _set_weekend_true(harness)
    harness.tc.close_all_for_weekend.return_value = {
        "ok": False, "closed": 0, "failed": 0,
        "query_ok": False, "error": "position query failed",
    }

    harness.agent._handle_weekend_close()
    assert harness.tc.close_all_for_weekend.call_count == 1

    # Second call: still weekend, marker was NOT written → retries.
    harness.agent._handle_weekend_close()
    assert harness.tc.close_all_for_weekend.call_count == 2


def test_does_not_mark_done_when_some_closes_failed(harness, monkeypatch):
    _force_now(
        monkeypatch,
        datetime(2025, 1, 4, 21, 0, tzinfo=timezone.utc),
    )
    _set_weekend_true(harness)
    harness.tc.close_all_for_weekend.return_value = {
        "ok": False, "closed": 1, "failed": 1,
        "query_ok": True, "error": "one close rejected",
    }

    harness.agent._handle_weekend_close()
    assert harness.tc.close_all_for_weekend.call_count == 1

    harness.agent._handle_weekend_close()
    assert harness.tc.close_all_for_weekend.call_count == 2