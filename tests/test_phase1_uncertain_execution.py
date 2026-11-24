"""Phase 1 / C-02 regression: uncertain submission lifecycle."""

from datetime import datetime, timedelta, timezone

import pytest

from src.database_manager import (
    db_manager, Base, OrderRequest, TradingSignal,
)
from src.order_executor import ExecutionResult, order_executor, _to_utc
from src.config_loader import config


@pytest.fixture()
def sqlite_memory(monkeypatch, tmp_path):
    db_url = f"sqlite:///{tmp_path / 'phase1_c02.db'}"
    monkeypatch.setattr(config.database, "url", db_url, raising=False)
    db_manager.database_url = db_url
    db_manager._init_engine()
    Base.metadata.create_all(bind=db_manager.engine)
    yield db_manager
    db_manager.close()


def _insert_request(status="pending", age_seconds=600):
    signal = TradingSignal(
        timestamp=datetime.utcnow(),
        symbol="GOLD",
        timeframe="m5",
        direction="BUY",
        entry_price=2000.0,
        sl_price=1990.0,
        tp_price=2020.0,
        lot_size=0.1,
        risk_percent=1.0,
        risk_reward=2.0,
        strategy="BOS_OB",
        environment="demo",
        fingerprint="deadbeef",
    )
    sig_id = db_manager.add(signal)
    req = OrderRequest(
        timestamp=datetime.utcnow() - timedelta(seconds=age_seconds),
        signal_id=sig_id,
        magic_number=int(config.trading.magic_number),
        symbol="GOLD",
        order_type=1,  # TRADE_ACTION_DEAL style sentinel; not asserted here
        volume=0.1,
        price=2000.0,
        sl=1990.0,
        tp=2020.0,
        comment="SMC_TEST",
        status=status,
        environment="demo",
        fingerprint="deadbeef",
    )
    return db_manager.add(req)


def test_to_utc_is_idempotent():
    naive = datetime(2026, 11, 24, 12, 0, 0)
    aware = _to_utc(naive)
    assert aware.tzinfo is timezone.utc
    assert _to_utc(aware) == aware


def test_uncertain_status_blocks_duplicates(sqlite_memory):
    _insert_request(status="uncertain", age_seconds=60)
    assert order_executor._is_duplicate("deadbeef") is True


def test_submission_error_status_blocks_duplicates(sqlite_memory):
    _insert_request(status="submission_error", age_seconds=60)
    assert order_executor._is_duplicate("deadbeef") is True


def test_rejected_notfound_does_not_block_duplicates(sqlite_memory):
    _insert_request(status="rejected_notfound", age_seconds=60)
    assert order_executor._is_duplicate("deadbeef") is False


def test_uncertain_request_resolved_to_rejected_notfound(
    sqlite_memory, monkeypatch
):
    _insert_request(status="uncertain", age_seconds=3600)

    # Empty broker state; grace exceeded → must finalise as not-executed.
    monkeypatch.setattr(
        "src.mt5_connector.mt5_connector.get_positions",
        lambda: [],
    )
    monkeypatch.setattr(
        "src.mt5_connector.mt5_connector.get_orders",
        lambda: [],
    )
    monkeypatch.setattr(
        "src.mt5_connector.mt5_connector.get_history_deals",
        lambda **_: [],
    )

    order_executor._reconcile_uncertain_and_stuck_requests()

    with db_manager.get_session() as s:
        req = s.query(OrderRequest).first()
        assert req.status == "rejected_notfound"


def test_uncertain_request_younger_than_grace_is_preserved(
    sqlite_memory, monkeypatch
):
    # age_seconds=60 is < 5-minute grace cutoff; must NOT be finalised.
    _insert_request(status="uncertain", age_seconds=60)

    monkeypatch.setattr(
        "src.mt5_connector.mt5_connector.get_positions", lambda: [],
    )
    monkeypatch.setattr(
        "src.mt5_connector.mt5_connector.get_orders", lambda: [],
    )
    monkeypatch.setattr(
        "src.mt5_connector.mt5_connector.get_history_deals", lambda **_: [],
    )

    order_executor._reconcile_uncertain_and_stuck_requests()

    with db_manager.get_session() as s:
        req = s.query(OrderRequest).first()
        assert req.status == "uncertain"