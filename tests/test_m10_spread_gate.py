"""M-10 regression tests: market-data reliability and spread gate.

Agreed scope:
  A. market_data.is_market_data_reliable() integrated into EntryGate.
  B. Missing, zero, unavailable, or stale spread fails closed for
     new entries.
  Safety boundary: entry checks must not block position closes,
     partial closes, SL/TP adjustments, or emergency-stop actions.

Focused coverage per finding:
  1. Unreliable market data blocks a new entry.
  2. Missing/unavailable spread blocks a new entry.
  3. Zero spread blocks a new entry.
  4. Stale spread blocks a new entry.
  5. Valid fresh data and spread pass the relevant checks.
  6. Position-management and emergency actions unaffected.
"""

from types import SimpleNamespace

import src.entry_gate as gate_mod
from src.entry_gate import EntryGate


# ---------------------------------------------------------------------
# Harness: gate whose every unrelated check passes
# ---------------------------------------------------------------------

def _pass_through_fixtures(
    monkeypatch,
    *,
    reliability=(True, "ok"),
    spread=(1.5, "ok"),
):
    """Set up EntryGate with all unrelated checks passing.

    Only reliability and spread are varied per test:
      reliability: (ok, reason) tuple, or an Exception to raise.
      spread:      (pips, status) tuple, or an Exception to raise.
    """
    fake_account = SimpleNamespace(
        trade_mode="DEMO",
        is_demo=True,
        trade_allowed=True,
        trade_expert=True,
    )

    class FakeConn:
        def is_connected(self): return True
        def update_account_info(self): return True
        def get_account_info(self): return fake_account

    class FakeRisk:
        def is_kill_switch_active(self): return False
        def check_risk_limits(self): return {"ok": True, "checks": {}}

    class FakeTradeManager:
        def get_unresolved_tickets_strict(self): return []

    class FakeSessionFilter:
        def is_trading_allowed(self): return True

    class FakeMarketData:
        def is_weekend_protection_active(self): return False
        def is_market_data_reliable(self):
            if isinstance(reliability, Exception):
                raise reliability
            return reliability
        def get_spread_status(self):
            if isinstance(spread, Exception):
                raise spread
            return spread

    class FakeNews:
        def should_block_trading(self): return {"block": False}

    class FakeConfig:
        class trading:
            mode = "DEMO"
            max_daily_trades = 20
        class trade_management:
            max_spread_pips = 5.0

    class FakeSession:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def query(self, *a, **kw): return self
        def filter(self, *a, **kw): return self
        def count(self): return 0

    class FakeDb:
        def get_session(self): return FakeSession()

    monkeypatch.setattr(gate_mod, "mt5_connector", FakeConn())
    monkeypatch.setattr(gate_mod, "risk_manager", FakeRisk())
    monkeypatch.setattr(gate_mod, "trade_manager", FakeTradeManager())
    monkeypatch.setattr(gate_mod, "session_filter", FakeSessionFilter())
    monkeypatch.setattr(gate_mod, "market_data", FakeMarketData())
    monkeypatch.setattr(gate_mod, "news_filter", FakeNews())
    monkeypatch.setattr(gate_mod, "config", FakeConfig())
    monkeypatch.setattr(gate_mod, "db_manager", FakeDb())
    monkeypatch.setattr(
        gate_mod, "_issue_gate_token", lambda: "test-token"
    )

    gate = EntryGate(
        is_paused_getter=lambda: False,
        is_shutdown_getter=lambda: False,
    )
    return gate


# ---------------------------------------------------------------------
# 1. Unreliable market data blocks a new entry
# ---------------------------------------------------------------------

def test_unreliable_market_data_blocks_entry(monkeypatch):
    gate = _pass_through_fixtures(
        monkeypatch,
        reliability=(False, "h1 stale: last closed candle ended 500min ago"),
    )
    result = gate.check()
    assert result.ok is False
    assert result.token == ""
    assert "market data" in result.reason.lower()
    assert result.checks.get("market_data") is False


def test_reliability_lookup_exception_fails_closed(monkeypatch):
    gate = _pass_through_fixtures(
        monkeypatch,
        reliability=RuntimeError("is_market_data_reliable blew up"),
    )
    result = gate.check()
    assert result.ok is False
    assert result.token == ""
    assert "reliability" in result.reason.lower()


# ---------------------------------------------------------------------
# 2. Missing or unavailable spread blocks a new entry
# ---------------------------------------------------------------------

def test_unavailable_spread_blocks_entry(monkeypatch):
    gate = _pass_through_fixtures(
        monkeypatch, spread=(None, "unavailable"),
    )
    result = gate.check()
    assert result.ok is False
    assert result.token == ""
    assert "unavailable" in result.reason.lower()


def test_missing_spread_value_blocks_entry(monkeypatch):
    # Defensive: status says "ok" but the value is None.
    gate = _pass_through_fixtures(
        monkeypatch, spread=(None, "ok"),
    )
    result = gate.check()
    assert result.ok is False
    assert "missing" in result.reason.lower()


def test_spread_lookup_exception_fails_closed(monkeypatch):
    gate = _pass_through_fixtures(
        monkeypatch,
        spread=RuntimeError("get_spread_status blew up"),
    )
    result = gate.check()
    assert result.ok is False
    assert "spread" in result.reason.lower()


# ---------------------------------------------------------------------
# 3. Zero spread blocks a new entry
# ---------------------------------------------------------------------

def test_zero_spread_blocks_entry(monkeypatch):
    gate = _pass_through_fixtures(
        monkeypatch, spread=(0.0, "ok"),
    )
    result = gate.check()
    assert result.ok is False
    assert result.token == ""
    assert "zero" in result.reason.lower()


def test_negative_spread_blocks_entry(monkeypatch):
    gate = _pass_through_fixtures(
        monkeypatch, spread=(-0.5, "ok"),
    )
    result = gate.check()
    assert result.ok is False
    reason = result.reason.lower()
    assert "zero" in reason or "negative" in reason


# ---------------------------------------------------------------------
# 4. Stale spread blocks a new entry
# ---------------------------------------------------------------------

def test_stale_spread_blocks_entry(monkeypatch):
    gate = _pass_through_fixtures(
        monkeypatch, spread=(None, "stale"),
    )
    result = gate.check()
    assert result.ok is False
    assert result.token == ""
    assert "stale" in result.reason.lower()


# ---------------------------------------------------------------------
# 5. Valid fresh data and spread pass the relevant checks
# ---------------------------------------------------------------------

def test_valid_data_and_spread_pass(monkeypatch):
    gate = _pass_through_fixtures(
        monkeypatch,
        reliability=(True, "ok"),
        spread=(1.5, "ok"),
    )
    result = gate.check()
    assert result.ok is True, (
        f"gate blocked unexpectedly: {result.reason}"
    )
    assert result.token == "test-token"
    assert result.checks.get("market_data") is True
    assert result.checks.get("spread") is True


def test_max_spread_boundary_passes(monkeypatch):
    # Exactly at max should pass (strict > comparison).
    gate = _pass_through_fixtures(
        monkeypatch,
        reliability=(True, "ok"),
        spread=(5.0, "ok"),
    )
    result = gate.check()
    assert result.ok is True, (
        f"gate blocked at exact max spread: {result.reason}"
    )


def test_above_max_spread_blocks(monkeypatch):
    gate = _pass_through_fixtures(
        monkeypatch,
        reliability=(True, "ok"),
        spread=(5.01, "ok"),
    )
    result = gate.check()
    assert result.ok is False
    assert "exceeds" in result.reason.lower()


# ---------------------------------------------------------------------
# 6. Safety boundary: position management / emergency unaffected
# ---------------------------------------------------------------------

def test_safety_boundary_management_methods_skip_gate():
    """Entry-gate failures must not block close/emergency paths.

    Static check: the relevant methods must not reference the entry
    gate, so a failing gate cannot short-circuit them.
    """
    import inspect
    from src.main import TradingAgent
    from src.order_executor import OrderExecutor
    from src.trading_controller import TradingController

    gate_refs = (
        "entry_gate.check",
        "_entry_gate.check",
        "_entry_gate.probe",
        "gate.check",
    )

    for name in ("close_all_positions", "close_position"):
        method = getattr(OrderExecutor, name, None)
        assert method is not None, f"OrderExecutor.{name} missing"
        src = inspect.getsource(method)
        for ref in gate_refs:
            assert ref not in src, (
                f"OrderExecutor.{name} references {ref!r}; "
                "position closes must not depend on the entry gate."
            )

    for name in (
        "close_all",
        "close_all_for_weekend",
        "close_position",
        "emergency_stop",
        "clear_emergency",
    ):
        method = getattr(TradingController, name, None)
        assert method is not None, f"TradingController.{name} missing"
        src = inspect.getsource(method)
        for ref in gate_refs:
            assert ref not in src, (
                f"TradingController.{name} references {ref!r}; "
                "close/emergency actions must not depend on the "
                "entry gate."
            )

    agent_src = inspect.getsource(TradingAgent._manage_positions)
    for ref in gate_refs:
        assert ref not in agent_src, (
            f"TradingAgent._manage_positions references {ref!r}; "
            "position management must not depend on the entry gate."
        )