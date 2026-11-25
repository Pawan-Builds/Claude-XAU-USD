"""Phase 4A: isolated backtest engine bug regressions."""

from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock
import asyncio

import pandas as pd
import pytest


@pytest.fixture
def engine_harness(monkeypatch):
    """Build a BacktestEngine with a deterministic initial state."""
    from src.backtesting import engine as engine_module

    # Patch global config pieces the engine reads.
    fake_config = SimpleNamespace(
        trading=SimpleNamespace(
            symbol="GOLD", magic_number=123456789, mode="DEMO",
            max_positions=2,
        ),
        risk=SimpleNamespace(default_risk_percent=1.0, max_risk_percent=1.0),
        backtesting=SimpleNamespace(
            commission_per_lot=7.0, spread_pips=2.0, slippage_pips=1.0,
            swap_enabled=True, default_balance=10000.0,
        ),
        smc=SimpleNamespace(strategies=["liquidity_sweep_fvg"]),
    )
    monkeypatch.setattr(engine_module, "config", fake_config)

    from src.backtesting.engine import BacktestEngine, BacktestState
    engine = BacktestEngine()
    engine.current_state = BacktestState(
        current_index=0,
        current_time=datetime.now(timezone.utc),
        equity=10000.0,
        balance=10000.0,
        open_positions=[],
        closed_trades=[],
        daily_pl=0.0,
        max_equity=10000.0,
        max_drawdown=0.0,
        max_drawdown_pct=0.0,
    )
    return SimpleNamespace(engine=engine, engine_module=engine_module)


def _signal(direction="BUY", entry=2000.0, sl=1990.0, tp=2020.0):
    class _Dir:
        def __init__(self, v): self.value = v
    class _Strat:
        def __init__(self, v): self.value = v
    return SimpleNamespace(
        direction=_Dir(direction),
        strategy=_Strat("liquidity_sweep_fvg"),
        entry_price=entry, sl_price=sl, tp_price=tp,
        risk_percent=1.0, risk_reward=2.0, lot_size=0.0,
        fingerprint="", confirmations=[], timestamp=None,
    )


def _candle(close=2000.0):
    return pd.Series({
        "open": close, "high": close + 5, "low": close - 5,
        "close": close, "tick_volume": 100,
    })


def _position(direction="BUY", entry=2000.0, volume=0.1):
    from src.backtesting.engine import BacktestPosition
    return BacktestPosition(
        ticket=1, direction=direction, entry_price=entry,
        sl=entry - 10 if direction == "BUY" else entry + 10,
        tp=entry + 20 if direction == "BUY" else entry - 20,
        volume=volume, entry_time=datetime.now(timezone.utc),
        entry_index=0, strategy="test", risk_percent=1.0,
    )


# ---------- 4-01: no duplicate signal evaluation ----------


def test_check_signals_calls_execute_once(engine_harness, monkeypatch):
    """Removing the duplicate body → analyze runs once, signals execute once."""
    engine = engine_harness.engine
    calls = {"execute": 0, "analyze": 0}

    def fake_execute(sig, candle, t, cfg):
        calls["execute"] += 1

    monkeypatch.setattr(engine, "_execute_signal", fake_execute)

    fake_smc = MagicMock()
    def fake_analyze():
        calls["analyze"] += 1
        return [_signal()]  # exactly one signal
    fake_smc.analyze_all_strategies = fake_analyze

    # Minimal df to feed the mock: one bar; entry price (2000) is inside.
    df = pd.DataFrame(
        {"open": [2000.0], "high": [2005.0], "low": [1995.0],
         "close": [2000.0], "tick_volume": [100]},
        index=[pd.Timestamp(datetime.now(timezone.utc))],
    )
    engine.candles["m5"] = df
    engine.candles["m15"] = df
    engine.candles["h1"] = df
    engine.candles["h4"] = df

    fake_cfg = SimpleNamespace(commission_per_lot=7.0)

    engine._check_signals(df, 0, fake_smc, fake_cfg)

    assert calls["analyze"] == 1
    assert calls["execute"] == 1


# ---------- 4-02: market_data restored on exception ----------


def test_check_signals_restores_market_data_on_exception(
    engine_harness, monkeypatch,
):
    from src.market_data import market_data

    engine = engine_harness.engine
    orig_get_candles = market_data.get_candles
    orig_get_closed = market_data.get_closed_candles
    orig_latest = market_data.get_latest_candle
    orig_atr = market_data.calculate_atr

    fake_smc = MagicMock()
    fake_smc.analyze_all_strategies.side_effect = RuntimeError("boom")

    df = pd.DataFrame(
        {"open": [2000.0], "high": [2005.0], "low": [1995.0],
         "close": [2000.0], "tick_volume": [100]},
        index=[pd.Timestamp(datetime.now(timezone.utc))],
    )
    engine.candles["m5"] = df
    engine.candles["m15"] = df
    engine.candles["h1"] = df
    engine.candles["h4"] = df

    with pytest.raises(RuntimeError):
        engine._check_signals(df, 0, fake_smc, SimpleNamespace())

    # Phase 4A: originals restored even when the analysis raises.
    assert market_data.get_candles is orig_get_candles
    assert market_data.get_closed_candles is orig_get_closed
    assert market_data.get_latest_candle is orig_latest
    assert market_data.calculate_atr is orig_atr


# ---------- 4-03: commission from passed-in config ----------


def test_check_exits_uses_config_commission(engine_harness):
    """Per-backtest commission is used, not global config."""
    engine = engine_harness.engine
    engine.current_state.open_positions = [_position("BUY", 2000.0)]
    # Current candle's low hits the SL (1990).
    candle = pd.Series({
        "open": 2000.0, "high": 2001.0, "low": 1989.0,
        "close": 1995.0, "tick_volume": 100,
    })

    fake_cfg = SimpleNamespace(commission_per_lot=99.0)

    engine._check_exits(
        candle, datetime.now(timezone.utc), fake_cfg,
    )

    # Trade should be closed at SL.
    assert engine.current_state.open_positions == []
    # The exact commission cost shows up as a reduction in net_profit.
    # With SL, gross is negative; commission makes it more negative.
    # Verify trade.net_profit is roughly:
    #   gross - spread_cost - commission = (-100) - 1 - 9.9 = -110.9
    # If it used the global (7.0), commission would be 0.7, net ≈ -101.7.
    # The 99 in config_per_lot must dominate the difference.
    closed_trade = engine.current_state.closed_trades[0]
    assert closed_trade.net_profit < -100.0


def test_close_all_positions_uses_config_commission(engine_harness):
    engine = engine_harness.engine
    engine.current_state.open_positions = [_position("BUY", 2000.0)]
    candle = pd.Series({
        "open": 2000.0, "high": 2005.0, "low": 1995.0,
        "close": 2000.0, "tick_volume": 100,
    })

    fake_cfg = SimpleNamespace(commission_per_lot=99.0)
    engine._close_all_positions(
        candle, datetime.now(timezone.utc), fake_cfg,
    )

    closed_trade = engine.current_state.closed_trades[0]
    # With 99 per lot, net_profit should be much lower than with 7.
    assert closed_trade.net_profit < -10.0


# ---------- 4-04: SELL marked at ask ----------


def test_update_equity_marks_sell_at_ask(engine_harness):
    engine = engine_harness.engine
    engine.current_state.balance = 10000.0
    engine.current_state.open_positions = [_position("SELL", 2000.0)]

    candle = pd.Series({
        "open": 2000.0, "high": 2005.0, "low": 1995.0,
        "close": 2000.0, "tick_volume": 100,
    })

    engine._update_equity(candle)

    # spread = 0.20, ask = 2000.10, pip = 0.1
    # SELL PL in pips = (2000 - 2000.10)/0.1 = -1.0 pip
    # tick_value = 10.0, volume = 0.1 → PL = -1.0 * 10 * 0.1 = -1.0
    # Equity = 10000 - 1.0 = 9999.0
    assert engine.current_state.equity < 10000.0
    assert abs(engine.current_state.equity - 9999.0) < 0.01


# ---------- 4-05: max_positions enforced at fill ----------


def test_execute_signal_rejects_when_max_reached(engine_harness, monkeypatch):
    engine = engine_harness.engine
    # Pre-fill up to the cap.
    engine.current_state.open_positions = [
        _position("BUY", 2000.0), _position("BUY", 2000.0),
    ]  # max_positions = 2

    candle = pd.Series({
        "open": 2000.0, "high": 2005.0, "low": 1995.0,
        "close": 2000.0, "tick_volume": 100,
    })
    fake_cfg = SimpleNamespace(risk_percent=1.0)

    engine._execute_signal(
        _signal(), candle, datetime.now(timezone.utc), fake_cfg,
    )

    # No new position should have been added.
    assert len(engine.current_state.open_positions) == 2


# ---------- 4-06 / 4-07: dashboard int + async handling ----------


def test_run_backtest_async_is_a_coroutine():
    """run_backtest_async must be a coroutine function (async def)."""
    from src.backtesting.engine import run_backtest_async
    assert asyncio.iscoroutinefunction(run_backtest_async)


def test_run_backtest_async_fetches_config_by_id(monkeypatch):
    """Passing an integer config_id must fetch the config, not receive it."""
    from src.backtesting import engine as engine_module

    seen = {}

    class FakeEngine:
        def __init__(self, config_id=None):
            seen["config_id"] = config_id

        def load_data(self, symbol, timeframes, start, end):
            seen["load_data_args"] = (symbol, timeframes)
            return True

        def run_backtest(self, config):
            seen["run_backtest_config"] = config
            return SimpleNamespace(id=42, net_profit=0.0)

    fake_bt_config = SimpleNamespace(
        id=7, timeframes=["m5"], start_date=1, end_date=2,
    )

    fake_db = MagicMock()
    fake_db.get.return_value = fake_bt_config

    monkeypatch.setattr(engine_module, "BacktestEngine", FakeEngine)
    monkeypatch.setattr(engine_module, "db_manager", fake_db)

    report = asyncio.run(
        engine_module.run_backtest_async(config_id=7)
    )

    assert report.id == 42
    assert seen["config_id"] == 7
    assert seen["run_backtest_config"] is fake_bt_config


def test_run_backtest_async_propagates_exceptions(monkeypatch):
    """An exception in the worker must propagate to the caller."""
    from src.backtesting import engine as engine_module

    class FailingEngine:
        def __init__(self, config_id=None):
            pass

        def load_data(self, *args, **kwargs):
            return True

        def run_backtest(self, config):
            raise RuntimeError("engine blew up")

    fake_db = MagicMock()
    fake_db.get.return_value = SimpleNamespace(
        id=7, timeframes=["m5"], start_date=1, end_date=2,
    )

    monkeypatch.setattr(engine_module, "BacktestEngine", FailingEngine)
    monkeypatch.setattr(engine_module, "db_manager", fake_db)

    with pytest.raises(RuntimeError, match="engine blew up"):
        asyncio.run(engine_module.run_backtest_async(config_id=7))


def test_run_backtest_async_raises_for_missing_config(monkeypatch):
    from src.backtesting import engine as engine_module

    fake_db = MagicMock()
    fake_db.get.return_value = None
    monkeypatch.setattr(engine_module, "db_manager", fake_db)

    with pytest.raises(ValueError, match="not found"):
        asyncio.run(engine_module.run_backtest_async(config_id=999))