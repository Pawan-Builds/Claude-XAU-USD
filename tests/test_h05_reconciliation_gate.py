"""H-05 regression tests: fail-closed unresolved-reconciliation state."""

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.database_manager import DatabaseManager, AppSettings


# ---------------------------------------------------------------------------
# Shared DB fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tm_env(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    db = DatabaseManager(database_url=f"sqlite:///{tmp.name}")
    db.create_tables()

    with patch("src.trade_manager.db_manager", db):
        from src.trade_manager import TradeManager
        # `environment` is a read-only property derived from
        # config.trading.mode. Patch it on the class so tests get a
        # deterministic "demo" value and are independent of ambient
        # config. monkeypatch restores it at teardown.
        monkeypatch.setattr(
            TradeManager, "environment",
            property(lambda self: "demo"),
        )
        tm = TradeManager()
        yield SimpleNamespace(tm=tm, db=db)

    try:
        db.close()
    except Exception:
        pass
    try:
        os.unlink(tmp.name)
    except OSError:
        pass


def _write_marker(db, env, payload_str):
    with db.get_session() as session:
        row = session.query(AppSettings).filter_by(
            key=f"unresolved_recon_{env}"
        ).first()
        if row is None:
            session.add(AppSettings(
                key=f"unresolved_recon_{env}", value=payload_str,
            ))
        else:
            row.value = payload_str


# ---------------------------------------------------------------------------
# get_unresolved_tickets_strict
# ---------------------------------------------------------------------------

class TestStrictMarkerRead:

    def test_missing_marker_returns_none(self, tm_env):
        assert tm_env.tm.get_unresolved_tickets_strict() is None

    def test_empty_value_returns_none(self, tm_env):
        _write_marker(tm_env.db, "demo", "")
        assert tm_env.tm.get_unresolved_tickets_strict() is None

    def test_corrupt_json_returns_none(self, tm_env):
        _write_marker(tm_env.db, "demo", "not-json")
        assert tm_env.tm.get_unresolved_tickets_strict() is None

    def test_missing_tickets_key_returns_none(self, tm_env):
        _write_marker(tm_env.db, "demo", json.dumps({"errors": "oops"}))
        assert tm_env.tm.get_unresolved_tickets_strict() is None

    def test_non_int_tickets_returns_none(self, tm_env):
        _write_marker(tm_env.db, "demo", json.dumps(
            {"tickets": ["abc"], "errors": ""}
        ))
        assert tm_env.tm.get_unresolved_tickets_strict() is None

    def test_valid_empty_returns_empty_list(self, tm_env):
        _write_marker(tm_env.db, "demo", json.dumps(
            {"tickets": [], "errors": ""}
        ))
        assert tm_env.tm.get_unresolved_tickets_strict() == []

    def test_valid_tickets_returns_list(self, tm_env):
        _write_marker(tm_env.db, "demo", json.dumps(
            {"tickets": [123, 456], "errors": ""}
        ))
        assert tm_env.tm.get_unresolved_tickets_strict() == [123, 456]

    def test_db_query_error_returns_none(self, tm_env, monkeypatch):
        class BrokenDB:
            def query_one(self, *a, **kw):
                raise RuntimeError("DB down")
        monkeypatch.setattr("src.trade_manager.db_manager", BrokenDB())
        assert tm_env.tm.get_unresolved_tickets_strict() is None


# ---------------------------------------------------------------------------
# has_unresolved_reconciliation fails closed
# ---------------------------------------------------------------------------

class TestHasUnresolvedFailsClosed:

    def test_marker_missing_returns_true(self, tm_env):
        assert tm_env.tm.has_unresolved_reconciliation() is True

    def test_marker_corrupt_returns_true(self, tm_env):
        _write_marker(tm_env.db, "demo", "garbage")
        assert tm_env.tm.has_unresolved_reconciliation() is True

    def test_marker_empty_list_returns_false(self, tm_env):
        _write_marker(tm_env.db, "demo", json.dumps(
            {"tickets": [], "errors": ""}
        ))
        assert tm_env.tm.has_unresolved_reconciliation() is False

    def test_marker_with_tickets_returns_true(self, tm_env):
        _write_marker(tm_env.db, "demo", json.dumps(
            {"tickets": [123], "errors": ""}
        ))
        assert tm_env.tm.has_unresolved_reconciliation() is True


# ---------------------------------------------------------------------------
# EntryGate reconciliation check
# ---------------------------------------------------------------------------

class TestEntryGateReconciliation:

    @pytest.fixture
    def gate_harness(self, monkeypatch):
        from src import entry_gate as eg_module

        mt5 = MagicMock()
        mt5.is_connected.return_value = True
        mt5.update_account_info.return_value = True
        mt5.get_account_info.return_value = SimpleNamespace(
            trade_mode="DEMO", is_demo=True,
            trade_allowed=True, trade_expert=True,
        )
        risk = MagicMock()
        risk.is_kill_switch_active.return_value = False
        risk.check_risk_limits.return_value = {"ok": True, "checks": {}}
        sess = MagicMock()
        sess.is_trading_allowed.return_value = True
        news = MagicMock()
        news.should_block_trading.return_value = {
            "block": False, "reason": "ok", "events": [],
            "data_available": True,
        }
        mkt = MagicMock()
        mkt.is_weekend_protection_active.return_value = False
        mkt.is_high_spread.return_value = False
        dbm = MagicMock()
        fake_session = MagicMock()
        fake_session.query.return_value.filter.return_value.count.return_value = 0
        dbm.get_session.return_value.__enter__.return_value = fake_session
        dbm.get_session.return_value.__exit__.return_value = False

        tm = MagicMock()
        tm.get_unresolved_tickets_strict.return_value = []
        tm.has_unresolved_reconciliation.return_value = False

        monkeypatch.setattr(eg_module, "mt5_connector", mt5)
        monkeypatch.setattr(eg_module, "risk_manager", risk)
        monkeypatch.setattr(eg_module, "session_filter", sess)
        monkeypatch.setattr(eg_module, "news_filter", news)
        monkeypatch.setattr(eg_module, "market_data", mkt)
        monkeypatch.setattr(eg_module, "db_manager", dbm)
        monkeypatch.setattr(eg_module, "trade_manager", tm)
        monkeypatch.setattr(eg_module, "config", SimpleNamespace(
            trading=SimpleNamespace(
                mode="DEMO", max_daily_trades=20,
            ),
            safety=SimpleNamespace(demo_only=True),
        ))

        from src.entry_gate import EntryGate
        paused = {"v": False}
        shutdown = {"v": False}
        gate = EntryGate(
            is_paused_getter=lambda: paused["v"],
            is_shutdown_getter=lambda: shutdown["v"],
        )
        return SimpleNamespace(gate=gate, tm=tm, paused=paused,
                                shutdown=shutdown)

    def test_gate_passes_when_marker_empty(self, gate_harness):
        gate_harness.tm.get_unresolved_tickets_strict.return_value = []
        result = gate_harness.gate.check()
        assert result.ok, result.reason
        assert result.checks.get("reconciliation") is True

    def test_gate_blocks_when_marker_unresolved(self, gate_harness):
        gate_harness.tm.get_unresolved_tickets_strict.return_value = [123]
        result = gate_harness.gate.check()
        assert result.ok is False
        assert "reconciliation" in result.reason.lower()

    def test_gate_blocks_when_marker_unknown(self, gate_harness):
        gate_harness.tm.get_unresolved_tickets_strict.return_value = None
        result = gate_harness.gate.check()
        assert result.ok is False
        assert "unknown" in result.reason.lower()

    def test_probe_also_blocks_on_unknown(self, gate_harness):
        """H-05: probe() must enforce the same reconciliation check
        as check(), otherwise _health_check will un-pause incorrectly."""
        gate_harness.tm.get_unresolved_tickets_strict.return_value = None
        result = gate_harness.gate.probe()
        assert result.ok is False
        assert "reconciliation" in result.reason.lower() or \
            "unknown" in result.reason.lower()

    def test_probe_passes_when_marker_empty(self, gate_harness):
        gate_harness.tm.get_unresolved_tickets_strict.return_value = []
        result = gate_harness.gate.probe()
        assert result.ok, result.reason


# ---------------------------------------------------------------------------
# MT5Connector._assert_reconciliation_clear and the send_order boundary
# ---------------------------------------------------------------------------

class TestConnectorBoundary:

    def _connector_ready(self):
        """Build a connected MT5Connector ready for _send_order_request."""
        from src.mt5_connector import MT5Connector, AccountInfo
        from src.config_loader import config

        conn = MT5Connector()
        conn._initialized = True
        conn._symbol_name = "XAUUSD"
        conn._account_info = AccountInfo(
            login=int(config.mt5.login) if config.mt5.login else 12345,
            server=config.mt5.server or "XMGlobal-MT5 5",
            name="T", company="XM", currency="USD",
            balance=10000.0, equity=10000.0, margin=0.0,
            free_margin=10000.0, margin_level=0.0, leverage=100,
            trade_mode="DEMO", trade_allowed=True, trade_expert=True,
            margin_mode="RETAIL_NETTING", is_demo=True,
        )
        return conn

    def test_assert_clear_returns_none_on_empty(self, tm_env):
        _write_marker(tm_env.db, "demo", json.dumps(
            {"tickets": [], "errors": ""}
        ))
        conn = self._connector_ready()
        with patch("src.trade_manager.trade_manager", tm_env.tm), \
             patch("src.trade_manager.db_manager", tm_env.db):
            assert conn._assert_reconciliation_clear() is None

    def test_assert_clear_returns_reason_on_tickets(self, tm_env):
        _write_marker(tm_env.db, "demo", json.dumps(
            {"tickets": [42], "errors": ""}
        ))
        conn = self._connector_ready()
        with patch("src.trade_manager.trade_manager", tm_env.tm), \
             patch("src.trade_manager.db_manager", tm_env.db):
            reason = conn._assert_reconciliation_clear()
            assert reason is not None
            assert "42" in reason

    def test_assert_clear_returns_reason_on_unknown(self, tm_env):
        conn = self._connector_ready()
        # Marker missing → strict returns None
        with patch("src.trade_manager.trade_manager", tm_env.tm), \
             patch("src.trade_manager.db_manager", tm_env.db):
            reason = conn._assert_reconciliation_clear()
            assert reason is not None
            assert "unknown" in reason.lower()

    def test_send_order_blocks_opening_with_unresolved_marker(
        self, tm_env
    ):
        """The opening boundary must refuse to submit even if the entry
        gate was bypassed.

        `_trading_precheck` is patched to None so the test isolates
        the reconciliation gate; other safety steps are exercised in
        their own tests.
        """
        _write_marker(tm_env.db, "demo", json.dumps(
            {"tickets": [42], "errors": ""}
        ))
        conn = self._connector_ready()

        opening_request = {
            "action": 1,   # TRADE_ACTION_DEAL sentinel
            "symbol": "XAUUSD",
            "type": 0,     # BUY sentinel
            "volume": 0.1,
        }

        with patch("src.trade_manager.trade_manager", tm_env.tm), \
             patch("src.trade_manager.db_manager", tm_env.db), \
             patch.object(conn, "_trading_precheck", return_value=None), \
             patch("src.mt5_connector.mt5") as mock_mt5:
            mock_mt5.TRADE_ACTION_DEAL = 1
            mock_mt5.TRADE_ACTION_PENDING = 5
            result = conn._send_order_request(opening_request)

        assert result.success is False
        assert "reconciliation" in (result.error_message or "").lower()

    def test_send_order_skips_reconciliation_check_for_closing(
        self, tm_env
    ):
        """H-05: closing actions bypass the reconciliation check, so
        emergency management still works during recovery.

        Verified by spying on `_assert_reconciliation_clear`: it must
        NOT be called when the payload carries a `position` key.
        """
        _write_marker(tm_env.db, "demo", json.dumps(
            {"tickets": [42], "errors": ""}
        ))
        conn = self._connector_ready()

        closing_request = {
            "action": 1,       # TRADE_ACTION_DEAL
            "position": 999,   # ← closing action
            "symbol": "XAUUSD",
            "type": 1,         # SELL (closes BUY)
            "volume": 0.1,
        }

        with patch("src.trade_manager.trade_manager", tm_env.tm), \
             patch("src.trade_manager.db_manager", tm_env.db), \
             patch.object(conn, "_trading_precheck", return_value=None), \
             patch.object(
                 conn, "_assert_reconciliation_clear"
             ) as mock_recon, \
             patch("src.mt5_connector.mt5") as mock_mt5:
            mock_mt5.TRADE_ACTION_DEAL = 1
            mock_mt5.TRADE_ACTION_PENDING = 5
            mock_mt5.TRADE_ACTION_SLTP = 6
            mock_mt5.TRADE_ACTION_REMOVE = 8
            mock_mt5.ORDER_TYPE_BUY = 0
            mock_mt5.ORDER_TYPE_SELL = 1
            mock_mt5.ORDER_FILLING_IOC = 1
            mock_mt5.symbol_info_tick.return_value = SimpleNamespace(
                bid=2000.0, ask=2000.5,
            )
            mock_mt5.symbol_info.return_value = SimpleNamespace(
                filling_mode=0, point=0.01, digits=2,
            )
            mock_mt5.order_send.return_value = None
            mock_mt5.last_error.return_value = (10004, "no conn")
            conn._send_order_request(closing_request)

        # The reconciliation check must NOT run for a closing action.
        mock_recon.assert_not_called()