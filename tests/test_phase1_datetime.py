"""Phase 1 / C-01 regression: TradePlan expiry comparison must not raise
TypeError when mixing aware UTC "now" with SQLite-rounded naive datetimes."""

from datetime import datetime, timezone, timedelta

import pytest

from src.database_manager import db_manager, TradePlan, Base
from src.main import TradingAgent
from src.config_loader import config


@pytest.fixture()
def sqlite_memory(monkeypatch, tmp_path):
    # Point the singleton at a throwaway file DB; never touch the real one.
    db_url = f"sqlite:///{tmp_path / 'phase1.db'}"
    monkeypatch.setattr(config.database, "url", db_url, raising=False)
    db_manager.database_url = db_url
    db_manager._init_engine()
    Base.metadata.create_all(bind=db_manager.engine)
    yield db_manager
    db_manager.close()


def _persist_plan(expiry: datetime, status: str = "WAITING") -> int:
    plan = TradePlan(
        timestamp=datetime.now(timezone.utc),
        symbol="GOLD",
        direction="BUY",
        strategy="BOS_OB",
        entry_price=2000.0,
        sl_price=1990.0,
        tp_price=2020.0,
        risk_reward=2.0,
        status=status,
        expiry_time=expiry,
        environment="demo",
    )
    return db_manager.add(plan)


def _reload(plan_id: int) -> TradePlan:
    return db_manager.get(TradePlan, plan_id)


def test_persisted_plan_is_naive_after_read(sqlite_memory):
    plan_id = _persist_plan(datetime.now(timezone.utc) + timedelta(hours=1))
    reloaded = _reload(plan_id)
    assert reloaded is not None
    # SQLAlchemy SQLite returns naive; this is what previously blew up.
    assert reloaded.expiry_time.tzinfo is None


def test_as_utc_restores_utc_tzinfo():
    naive = datetime(2026, 11, 24, 12, 0, 0)
    assert TradingAgent._as_utc(naive).tzinfo is timezone.utc


def test_as_utc_preserves_aware():
    aware = datetime(2026, 11, 24, 12, 0, 0, tzinfo=timezone.utc)
    assert TradingAgent._as_utc(aware) == aware


def test_expired_plan_compares_without_raising(sqlite_memory):
    plan_id = _persist_plan(datetime.now(timezone.utc) - timedelta(minutes=5))
    plan = _reload(plan_id)
    now = datetime.now(timezone.utc)
    # Direct comparison would raise TypeError; the guard must not.
    assert now >= TradingAgent._as_utc(plan.expiry_time)


def test_non_expired_plan_compares_without_raising(sqlite_memory):
    plan_id = _persist_plan(datetime.now(timezone.utc) + timedelta(hours=1))
    plan = _reload(plan_id)
    now = datetime.now(timezone.utc)
    assert not (now >= TradingAgent._as_utc(plan.expiry_time))