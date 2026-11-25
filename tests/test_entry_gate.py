"""Tests for the central fail-closed entry gate.

Config is a singleton with read-only properties, so we replace the
module-level reference with a mutable SimpleNamespace fake.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _make_fake_config(mode="DEMO", max_daily_trades=20):
    return SimpleNamespace(
        trading=SimpleNamespace(
            mode=mode,
            max_daily_trades=max_daily_trades,
        ),
        safety=SimpleNamespace(demo_only=True),
    )


@pytest.fixture
def gate_harness(monkeypatch):
    """Build an EntryGate whose dependencies are all controllable."""
    from src import entry_gate as eg_module

    mt5 = MagicMock(name="mt5_connector")
    risk = MagicMock(name="risk_manager")
    sess = MagicMock(name="session_filter")
    news = MagicMock(name="news_filter")
    mkt = MagicMock(name="market_data")
    dbm = MagicMock(name="db_manager")

    # Default happy path.
    mt5.is_connected.return_value = True
    mt5.update_account_info.return_value = True
    mt5.get_account_info.return_value = SimpleNamespace(
        trade_mode="DEMO",
        is_demo=True,
        trade_allowed=True,
        trade_expert=True,
    )
    risk.is_kill_switch_active.return_value = False
    risk.check_risk_limits.return_value = {"ok": True, "checks": {}}
    sess.is_trading_allowed.return_value = True
    news.should_block_trading.return_value = {
        "block": False,
        "reason": "no events",
        "events": [],
        "data_available": True,
    }
    mkt.is_weekend_protection_active.return_value = False
    mkt.is_high_spread.return_value = False

    # DB session: no trades today.
    fake_session = MagicMock(name="db_session")
    fake_session.query.return_value.filter.return_value.count.return_value = 0
    fake_session.query.return_value.filter.return_value.first.return_value = None
    dbm.get_session.return_value.__enter__.return_value = fake_session
    dbm.get_session.return_value.__exit__.return_value = False

    # Replace module-level singletons with controllable stubs.
    monkeypatch.setattr(eg_module, "mt5_connector", mt5)
    monkeypatch.setattr(eg_module, "risk_manager", risk)
    monkeypatch.setattr(eg_module, "session_filter", sess)
    monkeypatch.setattr(eg_module, "news_filter", news)
    monkeypatch.setattr(eg_module, "market_data", mkt)
    monkeypatch.setattr(eg_module, "db_manager", dbm)

    # Replace the Config singleton itself, not its properties.
    fake_config = _make_fake_config()
    monkeypatch.setattr(eg_module, "config", fake_config)
    # H-05: mock trade_manager so the new reconciliation check passes
    # by default. Tests that want to exercise reconciliation failure
    # override tm.get_unresolved_tickets_strict.return_value = None or
    # [123].
    tm = MagicMock(name="trade_manager")
    tm.get_unresolved_tickets_strict.return_value = []
    tm.has_unresolved_reconciliation.return_value = False
    monkeypatch.setattr(eg_module, "trade_manager", tm)
    from src.entry_gate import EntryGate

    paused = {"v": False}
    shutdown = {"v": False}
    gate = EntryGate(
        is_paused_getter=lambda: paused["v"],
        is_shutdown_getter=lambda: shutdown["v"],
    )

    return SimpleNamespace(
        gate=gate,
        mt5=mt5,
        risk=risk,
        sess=sess,
        news=news,
        mkt=mkt,
        dbm=dbm,
        config=fake_config,
        session=fake_session,
        paused=paused,
        shutdown=shutdown,
    )


# ---------------------------------------------------------------- happy path


def test_gate_passes_when_everything_is_healthy(gate_harness):
    result = gate_harness.gate.check()
    assert result.ok, result.reason
    assert result.checks.get("account_verified") is True


# ---------------------------------------------------------------- pause / shutdown


def test_check_blocks_when_paused(gate_harness):
    gate_harness.paused["v"] = True
    result = gate_harness.gate.check()
    assert result.ok is False
    assert "paused" in result.reason.lower()


def test_probe_ignores_pause(gate_harness):
    gate_harness.paused["v"] = True
    result = gate_harness.gate.probe()
    assert result.ok is True


def test_gate_blocks_on_shutdown(gate_harness):
    gate_harness.shutdown["v"] = True
    result = gate_harness.gate.check()
    assert result.ok is False
    assert "shutdown" in result.reason.lower()


# ---------------------------------------------------------------- MT5 / account


def test_gate_blocks_when_mt5_disconnected(gate_harness):
    gate_harness.mt5.is_connected.return_value = False
    result = gate_harness.gate.check()
    assert result.ok is False
    assert "not connected" in result.reason.lower()


def test_gate_blocks_on_demo_real_mismatch(gate_harness):
    gate_harness.mt5.get_account_info.return_value = SimpleNamespace(
        trade_mode="REAL",
        is_demo=False,
        trade_allowed=True,
        trade_expert=True,
    )
    result = gate_harness.gate.check()
    assert result.ok is False
    assert "demo" in result.reason.lower()


def test_gate_blocks_on_live_demo_mismatch(gate_harness):
    gate_harness.config.trading.mode = "LIVE"
    gate_harness.mt5.get_account_info.return_value = SimpleNamespace(
        trade_mode="DEMO",
        is_demo=True,
        trade_allowed=True,
        trade_expert=True,
    )
    result = gate_harness.gate.check()
    assert result.ok is False
    # "Mode LIVE but connected account is DEMO"
    assert "live" in result.reason.lower() or "demo" in result.reason.lower()


def test_gate_blocks_when_trade_expert_disabled(gate_harness):
    gate_harness.mt5.get_account_info.return_value = SimpleNamespace(
        trade_mode="DEMO",
        is_demo=True,
        trade_allowed=True,
        trade_expert=False,
    )
    result = gate_harness.gate.check()
    assert result.ok is False
    assert "expert" in result.reason.lower()


# ---------------------------------------------------------------- kill switch


def test_gate_blocks_when_kill_switch_active(gate_harness):
    gate_harness.risk.is_kill_switch_active.return_value = True
    result = gate_harness.gate.check()
    assert result.ok is False
    assert "kill" in result.reason.lower()


def test_gate_blocks_when_kill_switch_query_raises(gate_harness):
    gate_harness.risk.is_kill_switch_active.side_effect = RuntimeError("DB down")
    result = gate_harness.gate.check()
    assert result.ok is False


# ---------------------------------------------------------------- session / weekend / spread / news


def test_gate_blocks_outside_session(gate_harness):
    gate_harness.sess.is_trading_allowed.return_value = False
    result = gate_harness.gate.check()
    assert result.ok is False
    assert "session" in result.reason.lower()


def test_gate_blocks_on_weekend(gate_harness):
    gate_harness.mkt.is_weekend_protection_active.return_value = True
    result = gate_harness.gate.check()
    assert result.ok is False
    assert "weekend" in result.reason.lower()


def test_gate_blocks_on_high_spread(gate_harness):
    gate_harness.mkt.is_high_spread.return_value = True
    result = gate_harness.gate.check()
    assert result.ok is False
    assert "spread" in result.reason.lower()


def test_gate_blocks_on_news(gate_harness):
    gate_harness.news.should_block_trading.return_value = {
        "block": True,
        "reason": "NFP window",
        "events": [{"event": "NFP"}],
        "data_available": True,
    }
    result = gate_harness.gate.check()
    assert result.ok is False
    assert "nfp" in result.reason.lower() or "news" in result.reason.lower()


def test_gate_blocks_when_news_filter_raises(gate_harness):
    gate_harness.news.should_block_trading.side_effect = RuntimeError("boom")
    result = gate_harness.gate.check()
    assert result.ok is False


# ---------------------------------------------------------------- risk


def test_gate_blocks_on_risk_failure(gate_harness):
    gate_harness.risk.check_risk_limits.return_value = {
        "ok": False,
        "checks": {"daily_loss": False},
    }
    result = gate_harness.gate.check()
    assert result.ok is False
    assert "risk" in result.reason.lower()


def test_gate_blocks_on_risk_exception(gate_harness):
    gate_harness.risk.check_risk_limits.side_effect = RuntimeError("boom")
    result = gate_harness.gate.check()
    assert result.ok is False


# ---------------------------------------------------------------- daily trade count


def test_gate_blocks_on_daily_trade_limit(gate_harness):
    gate_harness.session.query.return_value.filter.return_value.count.return_value = 999
    result = gate_harness.gate.check()
    assert result.ok is False
    assert "daily trade" in result.reason.lower()