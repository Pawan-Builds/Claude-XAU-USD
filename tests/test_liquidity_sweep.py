"""Tests for M-01: liquidity sweep detection with strict pierce + reclaim.

Coverage:
- Exact touch (no strict pierce) is not consumed.
- Strict pierce with reclaim is classified as a genuine "sweep".
- Strict pierce without reclaim is classified as a "breakout".
- Close exactly at the level is a breakout (strict reclaim required).
- Pierce buffer and reclaim buffer are honored.
- A breakout does NOT appear in recent_sweeps / was_*_swept.
- Config values are propagated into LiquidityAnalyzer, with safe
  defaults for missing / invalid / out-of-range settings.
- Input validation rejects mismatched arrays and negative buffers.

Run with: pytest tests/test_liquidity_sweep.py -v
"""

from types import SimpleNamespace

import numpy as np
import pytest

from src.indicators import (
    LiquidityLevel,
    check_liquidity_sweep,
)
from src.liquidity import LiquidityAnalyzer


def _buy_level(price: float) -> LiquidityLevel:
    return LiquidityLevel(
        price=price,
        type="buy_side",
        strength=2,
        indices=[0, 1],
    )


def _sell_level(price: float) -> LiquidityLevel:
    return LiquidityLevel(
        price=price,
        type="sell_side",
        strength=2,
        indices=[0, 1],
    )


# ---------------------------------------------------------------------------
# Buy-side classification
# ---------------------------------------------------------------------------

class TestBuySideClassification:

    def test_exact_touch_is_not_consumed(self):
        levels = [_buy_level(100.0)]
        highs = np.array([99.0, 100.0])  # exact touch, no strict pierce
        lows = np.array([98.0, 99.0])
        closes = np.array([98.5, 99.5])

        check_liquidity_sweep(levels, [], highs, lows, closes, 1)

        assert not levels[0].swept
        assert levels[0].sweep_kind is None
        assert levels[0].sweep_index is None

    def test_wick_below_level_is_not_consumed(self):
        levels = [_buy_level(100.0)]
        highs = np.array([99.0, 99.5])
        lows = np.array([98.0, 98.5])
        closes = np.array([98.5, 99.0])

        check_liquidity_sweep(levels, [], highs, lows, closes, 1)

        assert not levels[0].swept

    def test_strict_pierce_with_reclaim_is_sweep(self):
        levels = [_buy_level(100.0)]
        highs = np.array([99.0, 101.0])  # pierces above 100
        lows = np.array([98.0, 99.0])
        closes = np.array([98.5, 99.5])  # closes back below 100

        check_liquidity_sweep(levels, [], highs, lows, closes, 1)

        assert levels[0].swept
        assert levels[0].sweep_kind == "sweep"
        assert levels[0].sweep_index == 1

    def test_strict_pierce_without_reclaim_is_breakout(self):
        levels = [_buy_level(100.0)]
        highs = np.array([99.0, 101.0])
        lows = np.array([98.0, 99.0])
        closes = np.array([98.5, 100.5])  # closes above 100

        check_liquidity_sweep(levels, [], highs, lows, closes, 1)

        assert levels[0].swept
        assert levels[0].sweep_kind == "breakout"
        assert levels[0].sweep_index == 1

    def test_close_exactly_at_level_is_breakout(self):
        # Strict reclaim requires close < level.price.
        levels = [_buy_level(100.0)]
        highs = np.array([99.0, 101.0])
        lows = np.array([98.0, 99.0])
        closes = np.array([98.5, 100.0])

        check_liquidity_sweep(levels, [], highs, lows, closes, 1)

        assert levels[0].swept
        assert levels[0].sweep_kind == "breakout"

    def test_pierce_buffer_suppresses_shallow_pierce(self):
        levels = [_buy_level(100.0)]
        highs = np.array([99.0, 100.5])  # 0.5 pierce
        lows = np.array([98.0, 99.0])
        closes = np.array([98.5, 99.5])

        check_liquidity_sweep(
            levels, [], highs, lows, closes, 1,
            pierce_buffer=1.0,
        )

        assert not levels[0].swept

    def test_pierce_buffer_allows_sufficient_pierce(self):
        levels = [_buy_level(100.0)]
        highs = np.array([99.0, 101.5])  # 1.5 pierce
        lows = np.array([98.0, 99.0])
        closes = np.array([98.5, 99.5])

        check_liquidity_sweep(
            levels, [], highs, lows, closes, 1,
            pierce_buffer=1.0,
        )

        assert levels[0].swept
        assert levels[0].sweep_kind == "sweep"

    def test_reclaim_buffer_requires_deeper_close_back(self):
        levels = [_buy_level(100.0)]
        highs = np.array([99.0, 101.0])
        lows = np.array([98.0, 99.0])
        closes = np.array([98.5, 99.5])  # 0.5 below level

        check_liquidity_sweep(
            levels, [], highs, lows, closes, 1,
            reclaim_buffer=1.0,  # require 1.0 below level
        )

        assert levels[0].swept
        assert levels[0].sweep_kind == "breakout"  # not enough reclaim


# ---------------------------------------------------------------------------
# Sell-side classification
# ---------------------------------------------------------------------------

class TestSellSideClassification:

    def test_exact_touch_is_not_consumed(self):
        levels = [_sell_level(100.0)]
        highs = np.array([101.0, 101.0])
        lows = np.array([99.5, 100.0])  # exact touch
        closes = np.array([100.5, 100.5])

        check_liquidity_sweep([], levels, highs, lows, closes, 1)

        assert not levels[0].swept

    def test_strict_pierce_with_reclaim_is_sweep(self):
        levels = [_sell_level(100.0)]
        highs = np.array([101.0, 101.0])
        lows = np.array([99.5, 99.0])  # pierces below 100
        closes = np.array([100.5, 100.5])  # closes back above 100

        check_liquidity_sweep([], levels, highs, lows, closes, 1)

        assert levels[0].swept
        assert levels[0].sweep_kind == "sweep"
        assert levels[0].sweep_index == 1

    def test_strict_pierce_without_reclaim_is_breakout(self):
        levels = [_sell_level(100.0)]
        highs = np.array([101.0, 101.0])
        lows = np.array([99.5, 99.0])
        closes = np.array([100.5, 99.5])  # closes below 100

        check_liquidity_sweep([], levels, highs, lows, closes, 1)

        assert levels[0].swept
        assert levels[0].sweep_kind == "breakout"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_mismatched_lengths_raise(self):
        levels = [_buy_level(100.0)]
        highs = np.array([99.0, 101.0])
        lows = np.array([98.0])
        closes = np.array([98.5, 99.5])

        with pytest.raises(ValueError, match="equal lengths"):
            check_liquidity_sweep(levels, [], highs, lows, closes, 1)

    def test_negative_pierce_buffer_raises(self):
        levels = [_buy_level(100.0)]
        highs = np.array([99.0, 101.0])
        lows = np.array([98.0, 99.0])
        closes = np.array([98.5, 99.5])

        with pytest.raises(ValueError, match="pierce_buffer"):
            check_liquidity_sweep(
                levels, [], highs, lows, closes, 1,
                pierce_buffer=-0.1,
            )

    def test_negative_reclaim_buffer_raises(self):
        levels = [_buy_level(100.0)]
        highs = np.array([99.0, 101.0])
        lows = np.array([98.0, 99.0])
        closes = np.array([98.5, 99.5])

        with pytest.raises(ValueError, match="reclaim_buffer"):
            check_liquidity_sweep(
                levels, [], highs, lows, closes, 1,
                reclaim_buffer=-0.1,
            )


# ---------------------------------------------------------------------------
# Breakout must not feed recent_sweeps / was_*_swept
# ---------------------------------------------------------------------------

class TestRecentSweepsCollection:

    def _candle(self):
        return SimpleNamespace(timestamp="2024-01-01T00:00:00")

    def test_breakout_not_in_recent_sweeps(self):
        analyzer = LiquidityAnalyzer()
        level = LiquidityLevel(
            price=100.0,
            type="buy_side",
            strength=2,
            indices=[0, 1],
            swept=True,
            sweep_index=5,
            sweep_kind="breakout",
        )

        sweeps = analyzer._collect_recent_sweeps(
            [level], [], self._candle(), 5
        )

        assert sweeps == []

    def test_genuine_sweep_in_recent_sweeps(self):
        analyzer = LiquidityAnalyzer()
        level = LiquidityLevel(
            price=100.0,
            type="buy_side",
            strength=2,
            indices=[0, 1],
            swept=True,
            sweep_index=5,
            sweep_kind="sweep",
        )

        sweeps = analyzer._collect_recent_sweeps(
            [level], [], self._candle(), 5
        )

        assert len(sweeps) == 1
        assert sweeps[0]["type"] == "buy_side_sweep"
        assert sweeps[0]["price"] == 100.0

    def test_older_sweep_not_in_recent_sweeps(self):
        analyzer = LiquidityAnalyzer()
        level = LiquidityLevel(
            price=100.0,
            type="buy_side",
            strength=2,
            indices=[0, 1],
            swept=True,
            sweep_index=3,  # older than current
            sweep_kind="sweep",
        )

        sweeps = analyzer._collect_recent_sweeps(
            [level], [], self._candle(), 5
        )

        assert sweeps == []


# ---------------------------------------------------------------------------
# Configuration propagation
# ---------------------------------------------------------------------------

class TestConfigPropagation:

    def test_reads_buffers_from_config(self, monkeypatch):
        from src import liquidity as liquidity_module

        class FakeLiquiditySection:
            sweep_pierce_buffer_pips = 2.5
            sweep_reclaim_buffer_pips = 1.5

        class FakeConfig:
            liquidity = FakeLiquiditySection()

        monkeypatch.setattr(
            liquidity_module, "config", FakeConfig(), raising=False
        )

        analyzer = LiquidityAnalyzer()
        pierce, reclaim = analyzer._get_sweep_buffers_pips()

        assert pierce == 2.5
        assert reclaim == 1.5

    def test_defaults_when_section_missing(self, monkeypatch):
        from src import liquidity as liquidity_module

        class FakeConfig:
            pass  # no liquidity attribute

        monkeypatch.setattr(
            liquidity_module, "config", FakeConfig(), raising=False
        )

        analyzer = LiquidityAnalyzer()
        pierce, reclaim = analyzer._get_sweep_buffers_pips()

        assert pierce == 0.0
        assert reclaim == 0.0

    def test_invalid_values_fall_back_to_zero(self, monkeypatch):
        from src import liquidity as liquidity_module

        class FakeLiquiditySection:
            sweep_pierce_buffer_pips = "not-a-number"
            sweep_reclaim_buffer_pips = -5.0  # negative is invalid

        class FakeConfig:
            liquidity = FakeLiquiditySection()

        monkeypatch.setattr(
            liquidity_module, "config", FakeConfig(), raising=False
        )

        analyzer = LiquidityAnalyzer()
        pierce, reclaim = analyzer._get_sweep_buffers_pips()

        assert pierce == 0.0
        assert reclaim == 0.0

    def test_non_finite_values_fall_back_to_zero(self, monkeypatch):
        from src import liquidity as liquidity_module

        class FakeLiquiditySection:
            sweep_pierce_buffer_pips = float("nan")
            sweep_reclaim_buffer_pips = float("inf")

        class FakeConfig:
            liquidity = FakeLiquiditySection()

        monkeypatch.setattr(
            liquidity_module, "config", FakeConfig(), raising=False
        )

        analyzer = LiquidityAnalyzer()
        pierce, reclaim = analyzer._get_sweep_buffers_pips()

        assert pierce == 0.0
        assert reclaim == 0.0