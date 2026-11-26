
"""Focused N-H04 tests: backtesting must not mutate live-trading state.

These tests do not connect to MT5 or place broker orders.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src import market_data as md_module
from src.market_data import market_data
from src.smc_context import (
    AnalyzerProxy,
    set_smc_analyzer_context,
    reset_smc_analyzer_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_candles(start, count, base_price=2000.0):
    """Create deterministic M5 candles."""
    index = pd.date_range(
        start=start,
        periods=count,
        freq="5min",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "open": [base_price + i for i in range(count)],
            "high": [base_price + i + 2 for i in range(count)],
            "low": [base_price + i - 2 for i in range(count)],
            "close": [base_price + i + 1 for i in range(count)],
            "tick_volume": [100] * count,
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# 1. Live market_data methods must never be monkeypatched
# ---------------------------------------------------------------------------

def test_market_data_methods_not_patched():
    """The backtest provider must not replace shared singleton methods."""
    from src.backtesting.engine import BacktestDataProvider

    original_get_candles = market_data.get_candles
    original_get_closed = market_data.get_closed_candles
    original_atr = market_data.calculate_atr

    provider = BacktestDataProvider({})

    token = md_module.set_backtest_data_provider(provider)

    try:
        assert market_data.get_candles == original_get_candles
        assert market_data.get_closed_candles == original_get_closed
        assert market_data.calculate_atr == original_atr
    finally:
        md_module.reset_backtest_data_provider(token)


# ---------------------------------------------------------------------------
# 2. Backtest data providers must be isolated between threads
# ---------------------------------------------------------------------------

def test_provider_isolated_per_thread():
    from src.backtesting.engine import BacktestDataProvider

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    provider_a = BacktestDataProvider(
        {"m5": make_candles(start, 5, 2000.0)}
    )
    provider_b = BacktestDataProvider(
        {"m5": make_candles(start, 5, 3000.0)}
    )

    def read_provider(provider):
        token = md_module.set_backtest_data_provider(provider)

        try:
            candles = market_data.get_candles("m5", 5)
            return [c.open for c in candles]
        finally:
            md_module.reset_backtest_data_provider(token)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(read_provider, provider_a)
        future_b = executor.submit(read_provider, provider_b)

        result_a = future_a.result()
        result_b = future_b.result()

    assert result_a
    assert result_b

    assert result_a[0] == pytest.approx(2000.0)
    assert result_b[0] == pytest.approx(3000.0)

    assert result_a != result_b


# ---------------------------------------------------------------------------
# 3. Simulated clocks must not leak between providers
# ---------------------------------------------------------------------------

def test_provider_clock_is_independent():
    from src.backtesting.engine import BacktestDataProvider

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    candles = make_candles(start, 10)

    provider_a = BacktestDataProvider({"m5": candles})
    provider_b = BacktestDataProvider({"m5": candles})

    provider_a.set_current_time(candles.index[3])
    provider_b.set_current_time(candles.index[7])

    assert provider_a._current_time == candles.index[3]
    assert provider_b._current_time == candles.index[7]

    assert len(provider_a.get_candles("m5", 100)) == 4
    assert len(provider_b.get_candles("m5", 100)) == 8


# ---------------------------------------------------------------------------
# 4. Closed-candle reads must exclude the current forming candle
# ---------------------------------------------------------------------------

def test_provider_excludes_forming_candle():
    from src.backtesting.engine import BacktestDataProvider

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = make_candles(start, 5)

    provider = BacktestDataProvider({"m5": candles})
    provider.set_current_time(candles.index[3])

    closed = provider.get_closed_candles("m5", 100)

    assert len(closed) == 3
    assert closed[-1].timestamp == candles.index[2]

    latest = provider.get_latest_closed_candle("m5")

    assert latest is not None
    assert latest.timestamp == candles.index[2]


# ---------------------------------------------------------------------------
# 5. Analyzer contexts must be isolated between threads
# ---------------------------------------------------------------------------

def test_analyzer_context_isolated_per_thread():
    proxy = AnalyzerProxy("market_structure")

    analyzer_a = MagicMock(name="analyzer_a")
    analyzer_b = MagicMock(name="analyzer_b")

    def read_analyzer(analyzer):
        token = set_smc_analyzer_context(
            {"market_structure": analyzer}
        )

        try:
            return proxy
        finally:
            reset_smc_analyzer_context(token)

    def read_attribute(analyzer):
        token = set_smc_analyzer_context(
            {"market_structure": analyzer}
        )

        try:
            return proxy.some_attribute
        finally:
            reset_smc_analyzer_context(token)

    analyzer_a.some_attribute = "A"
    analyzer_b.some_attribute = "B"

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(read_attribute, analyzer_a)
        future_b = executor.submit(read_attribute, analyzer_b)

        result_a = future_a.result()
        result_b = future_b.result()

    assert result_a == "A"
    assert result_b == "B"


# ---------------------------------------------------------------------------
# 6. Analyzer context must be restored after exceptions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 6. Analyzer context must be restored after exceptions
# ---------------------------------------------------------------------------

def test_analyzer_context_reset_after_exception():
    proxy = AnalyzerProxy("market_structure")

    analyzer = MagicMock()
    analyzer.some_attribute = "backtest"

    token = set_smc_analyzer_context(
        {"market_structure": analyzer}
    )

    try:
        assert proxy.some_attribute == "backtest"

        raise RuntimeError("simulated backtest failure")

    except RuntimeError:
        pass

    finally:
        reset_smc_analyzer_context(token)

    # After reset, the proxy resolves to the default analyzer.
    # The default analyzer should not have the mock-only attribute.
    with pytest.raises(AttributeError):
        _ = proxy.some_attribute

# ---------------------------------------------------------------------------
# 7. Backtest signal analysis must run exactly once per iteration
# ---------------------------------------------------------------------------

def test_check_signals_runs_analysis_once():
    from src.backtesting.engine import (
        BacktestEngine,
        BacktestDataProvider,
    )

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = make_candles(start, 5)

    engine = BacktestEngine()
    engine._backtest_provider = BacktestDataProvider({"m5": df})

    smc_engine = MagicMock()
    smc_engine.analyze_all_strategies.return_value = []

    engine._execute_signal = MagicMock()

    engine._check_signals(
        df_m5=df,
        current_index=2,
        smc_engine=smc_engine,
        config=MagicMock(),
    )

    assert smc_engine.analyze_all_strategies.call_count == 1
    engine._execute_signal.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Missing isolated provider must fail closed
# ---------------------------------------------------------------------------

def test_check_signals_without_provider_does_not_analyze():
    from src.backtesting.engine import BacktestEngine

    engine = BacktestEngine()
    engine._backtest_provider = None

    smc_engine = MagicMock()

    df = make_candles(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        5,
    )

    engine._check_signals(
        df_m5=df,
        current_index=2,
        smc_engine=smc_engine,
        config=MagicMock(),
    )

    smc_engine.analyze_all_strategies.assert_not_called()


# ---------------------------------------------------------------------------
# 9. Analysis exceptions must not leave a changed provider clock context
# ---------------------------------------------------------------------------

def test_check_signals_handles_analysis_exception():
    from src.backtesting.engine import (
        BacktestEngine,
        BacktestDataProvider,
    )

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = make_candles(start, 5)

    engine = BacktestEngine()
    provider = BacktestDataProvider({"m5": df})
    engine._backtest_provider = provider

    smc_engine = MagicMock()
    smc_engine.analyze_all_strategies.side_effect = RuntimeError(
        "simulated analysis error"
    )

    engine._check_signals(
        df_m5=df,
        current_index=2,
        smc_engine=smc_engine,
        config=MagicMock(),
    )

    assert provider._current_time == df.index[2]
    assert smc_engine.analyze_all_strategies.call_count == 1


# ---------------------------------------------------------------------------
# 10. Nested analyzer contexts must restore the outer context
# ---------------------------------------------------------------------------

def test_nested_analyzer_context_restores_outer_context():
    proxy = AnalyzerProxy("market_structure")

    outer_analyzer = MagicMock()
    outer_analyzer.some_attribute = "outer"

    inner_analyzer = MagicMock()
    inner_analyzer.some_attribute = "inner"

    outer_token = set_smc_analyzer_context(
        {"market_structure": outer_analyzer}
    )

    try:
        assert proxy.some_attribute == "outer"

        inner_token = set_smc_analyzer_context(
            {"market_structure": inner_analyzer}
        )

        try:
            assert proxy.some_attribute == "inner"
        finally:
            reset_smc_analyzer_context(inner_token)

        assert proxy.some_attribute == "outer"

    finally:
        reset_smc_analyzer_context(outer_token)
