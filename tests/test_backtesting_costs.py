"""M-04 tests: broker-aware execution cost model.

Covers spread, slippage, tick-value P&L, volume sizing, commission,
swap, fail-closed on missing broker spec.

Run with: pytest tests/test_backtesting_costs.py -v

Tests instantiate BacktestEngine and set _broker_spec / _cost_config
directly. No MT5 connection required.
"""

from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import pytest

from src.backtesting.engine import BacktestEngine


def _engine_with_spec(
    *,
    tick_size=0.01, tick_value=1.0, contract_size=100.0,
    volume_min=0.01, volume_max=50.0, volume_step=0.01,
    point=0.01, pip_size=0.1,
    swap_long=-5.0, swap_short=2.0,
    spread_pips=2.0, slippage_pips=1.0,
    commission_per_lot=7.0, commission_per_side=True,
    swap_enabled=True, swap_unit="usd_per_lot_per_day",
):
    engine = BacktestEngine()
    engine._broker_spec = {
        "tick_size": tick_size, "tick_value": tick_value,
        "contract_size": contract_size,
        "volume_min": volume_min, "volume_max": volume_max,
        "volume_step": volume_step,
        "point": point, "pip_size": pip_size,
        "swap_long": swap_long, "swap_short": swap_short,
    }
    engine._cost_config = {
        "spread_pips": spread_pips, "slippage_pips": slippage_pips,
        "commission_per_lot": commission_per_lot,
        "commission_per_side": commission_per_side,
        "swap_enabled": swap_enabled, "swap_unit": swap_unit,
    }
    engine.pip_value = pip_size
    engine.point_value = point
    return engine


# ---------------------------------------------------------------------------
# _calculate_gross_pl: broker-aware formula
# ---------------------------------------------------------------------------

class TestGrossPL:

    def test_buy_profit_uses_tick_economics(self):
        engine = _engine_with_spec()
        # BUY entry 100.00, exit 102.00, volume 0.1.
        # delta = 2.00, tick_size=0.01, tick_value=1.0 -> 200 ticks
        # -> 200 * 1.0 * 0.1 = $20.
        pl = engine._calculate_gross_pl("BUY", 100.00, 102.00, 0.1)
        assert pl == pytest.approx(20.0)

    def test_sell_profit_uses_tick_economics(self):
        engine = _engine_with_spec()
        # SELL entry 102.00, exit 100.00, volume 0.1.
        pl = engine._calculate_gross_pl("SELL", 102.00, 100.00, 0.1)
        assert pl == pytest.approx(20.0)

    def test_buy_loss(self):
        engine = _engine_with_spec()
        pl = engine._calculate_gross_pl("BUY", 100.00, 99.00, 0.1)
        assert pl == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# _calculate_volume: risk budget, floor rounding, min/max
# ---------------------------------------------------------------------------

class TestVolumeCalculation:

    def test_exact_fit_is_returned(self):
        engine = _engine_with_spec()
        # Risk $20, sl_distance=2.00, pl_per_lot=100 per price.
        # Raw volume = 20 / (2.00 * 100) = 0.10. Exact step match.
        vol = engine._calculate_volume(20.0, 2.00)
        assert vol == pytest.approx(0.10)

    def test_rounds_down_not_nearest(self):
        engine = _engine_with_spec()
        # Risk $21, sl_distance=2.00 -> raw = 0.105.
        # Nearest-0.01 would give 0.11 (risk $22). Floor -> 0.10.
        vol = engine._calculate_volume(21.0, 2.00)
        assert vol == pytest.approx(0.10)
        # Confirm risk at 0.10 is within budget, at 0.11 is not.
        assert 0.10 * 2.00 * 100.0 <= 21.0
        assert 0.11 * 2.00 * 100.0 > 21.0

    def test_below_min_returns_none(self):
        engine = _engine_with_spec()
        # Risk $0.50 -> raw = 0.0025. Below volume_min=0.01.
        vol = engine._calculate_volume(0.50, 2.00)
        assert vol is None

    def test_zero_risk_returns_none(self):
        engine = _engine_with_spec()
        assert engine._calculate_volume(0.0, 2.00) is None

    def test_zero_sl_returns_none(self):
        engine = _engine_with_spec()
        assert engine._calculate_volume(20.0, 0.0) is None

    def test_caps_at_volume_max(self):
        engine = _engine_with_spec(volume_max=5.0)
        # Risk $10000, sl_distance=2.00 -> raw = 50.0. Capped at 5.0.
        vol = engine._calculate_volume(10000.0, 2.00)
        assert vol == pytest.approx(5.0)

    def test_respects_custom_volume_step(self):
        engine = _engine_with_spec(volume_step=0.1, volume_min=0.1)
        # Risk $50 -> raw = 0.25. Floor to 0.1-step -> 0.2.
        vol = engine._calculate_volume(50.0, 2.00)
        assert vol == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# _apply_entry_costs / _apply_exit_costs
# ---------------------------------------------------------------------------

class TestEntryExitCosts:

    def test_buy_entry_charged_half_spread_and_slippage(self):
        engine = _engine_with_spec()  # 2 pips spread, 1 pip slip, pip 0.1
        # half_spread = 2 * 0.1 / 2 = 0.10; slippage = 1 * 0.1 = 0.10.
        fill = engine._apply_entry_costs("BUY", 100.00)
        assert fill == pytest.approx(100.20)

    def test_sell_entry_charged_half_spread_and_slippage(self):
        engine = _engine_with_spec()
        fill = engine._apply_entry_costs("SELL", 100.00)
        assert fill == pytest.approx(99.80)

    def test_buy_sl_exit_has_slippage(self):
        engine = _engine_with_spec()
        # is_limit=False -> slippage applied -> 99.00 - 0.10 = 98.90
        fill = engine._apply_exit_costs("BUY", 99.00, is_limit=False)
        assert fill == pytest.approx(98.90)

    def test_buy_tp_exit_has_no_slippage(self):
        engine = _engine_with_spec()
        fill = engine._apply_exit_costs("BUY", 102.00, is_limit=True)
        assert fill == pytest.approx(102.00)

    def test_sell_sl_exit_has_slippage(self):
        engine = _engine_with_spec()
        fill = engine._apply_exit_costs("SELL", 101.00, is_limit=False)
        assert fill == pytest.approx(101.10)

    def test_sell_tp_exit_has_no_slippage(self):
        engine = _engine_with_spec()
        fill = engine._apply_exit_costs("SELL", 98.00, is_limit=True)
        assert fill == pytest.approx(98.00)


# ---------------------------------------------------------------------------
# _apply_swap
# ---------------------------------------------------------------------------

class TestSwap:

    def test_buy_swap_applied_daily(self):
        engine = _engine_with_spec(swap_long=-5.0)
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t1 = t0 + timedelta(days=3)
        swap = engine._apply_swap("BUY", 0.10, t0, t1)
        # -5.0 * 0.10 * 3 = -1.50
        assert swap == pytest.approx(-1.50)

    def test_sell_swap_applied_daily(self):
        engine = _engine_with_spec(swap_short=2.0)
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t1 = t0 + timedelta(days=2)
        swap = engine._apply_swap("SELL", 0.10, t0, t1)
        # 2.0 * 0.10 * 2 = 0.40
        assert swap == pytest.approx(0.40)

    def test_less_than_one_day_no_swap(self):
        engine = _engine_with_spec()
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t1 = t0 + timedelta(hours=6)
        assert engine._apply_swap("BUY", 0.10, t0, t1) == 0.0

    def test_swap_disabled_returns_zero(self):
        engine = _engine_with_spec(swap_enabled=False)
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t1 = t0 + timedelta(days=5)
        assert engine._apply_swap("BUY", 0.10, t0, t1) == 0.0

    def test_swap_unit_none_returns_zero(self):
        engine = _engine_with_spec(swap_unit="none")
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t1 = t0 + timedelta(days=5)
        assert engine._apply_swap("BUY", 0.10, t0, t1) == 0.0


# ---------------------------------------------------------------------------
# _load_broker_spec: fail closed on missing/invalid broker info
# ---------------------------------------------------------------------------

class TestBrokerSpecLoading:

    def _stub(self, **fields):
        return SimpleNamespace(
            tick_size=fields.get("tick_size", 0.01),
            tick_value=fields.get("tick_value", 1.0),
            contract_size=fields.get("contract_size", 100.0),
            volume_min=fields.get("volume_min", 0.01),
            volume_max=fields.get("volume_max", 50.0),
            volume_step=fields.get("volume_step", 0.01),
            point=fields.get("point", 0.01),
            digits=fields.get("digits", 2),
            name=fields.get("name", "GOLD"),
            swap_long=fields.get("swap_long", -5.0),
            swap_short=fields.get("swap_short", 2.0),
        )

    def test_none_returns_none(self, monkeypatch):
        from src.backtesting import engine as eng_module
        monkeypatch.setattr(
            eng_module.mt5_connector, "get_symbol_info", lambda: None,
            raising=False,
        )
        engine = BacktestEngine()
        assert engine._load_broker_spec() is None

    def test_exception_returns_none(self, monkeypatch):
        from src.backtesting import engine as eng_module
        def _raise():
            raise RuntimeError("boom")
        monkeypatch.setattr(
            eng_module.mt5_connector, "get_symbol_info", _raise,
            raising=False,
        )
        engine = BacktestEngine()
        assert engine._load_broker_spec() is None

    def test_invalid_tick_size_returns_none(self, monkeypatch):
        from src.backtesting import engine as eng_module
        stub = self._stub(tick_size=0.0)
        monkeypatch.setattr(
            eng_module.mt5_connector, "get_symbol_info", lambda: stub,
            raising=False,
        )
        engine = BacktestEngine()
        assert engine._load_broker_spec() is None

    def test_valid_spec_is_returned(self, monkeypatch):
        from src.backtesting import engine as eng_module
        stub = self._stub()
        monkeypatch.setattr(
            eng_module.mt5_connector, "get_symbol_info", lambda: stub,
            raising=False,
        )
        engine = BacktestEngine()
        spec = engine._load_broker_spec()
        assert spec is not None
        assert spec["tick_size"] == 0.01
        assert spec["pip_size"] > 0
        assert spec["volume_min"] == 0.01


# ---------------------------------------------------------------------------
# _coerce_non_negative
# ---------------------------------------------------------------------------

class TestCoerce:

    def test_valid_float(self):
        assert BacktestEngine._coerce_non_negative(2.5, 1.0) == 2.5

    def test_negative_falls_back(self):
        assert BacktestEngine._coerce_non_negative(-1.0, 0.5) == 0.5

    def test_nan_falls_back(self):
        assert (
            BacktestEngine._coerce_non_negative(float("nan"), 0.5) == 0.5
        )

    def test_string_falls_back(self):
        assert (
            BacktestEngine._coerce_non_negative("bad", 0.5) == 0.5
        )


# ---------------------------------------------------------------------------
# Fail-closed in run_backtest when broker spec missing
# ---------------------------------------------------------------------------

class TestFailClosed:

    def test_run_backtest_fails_when_broker_spec_missing(
        self, monkeypatch,
    ):
        from src.backtesting import engine as eng_module

        monkeypatch.setattr(
            eng_module.mt5_connector, "get_symbol_info", lambda: None,
            raising=False,
        )

        config_stub = SimpleNamespace(
            id=1, name="test",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            initial_balance=10000.0,
            risk_percent=2.0,
            timeframes=["m5"],
            commission_per_lot=7.0,
        )

        engine = BacktestEngine()
        report = engine.run_backtest(config_stub)

        assert engine.get_status().value == "failed"
        assert report.total_trades == 0
        assert engine.get_assumptions_applied() == {}


# ---------------------------------------------------------------------------
# assumptions_applied transparency
# ---------------------------------------------------------------------------

class TestAssumptions:

    def test_preserves_legacy_compat_fields(self):
        engine = _engine_with_spec()
        # The old public attributes still exist and are readable.
        assert hasattr(engine, "pip_value")
        assert hasattr(engine, "point_value")
        assert hasattr(engine, "assumptions_applied")

    def test_get_assumptions_applied_returns_copy(self):
        engine = _engine_with_spec()
        engine.assumptions_applied = {"spread_model": "half_per_side"}
        copy = engine.get_assumptions_applied()
        copy["spread_model"] = "MUTATED"
        assert engine.assumptions_applied["spread_model"] == "half_per_side"