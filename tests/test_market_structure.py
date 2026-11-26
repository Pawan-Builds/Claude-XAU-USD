"""Tests for M-03: mutually-exclusive BOS/CHoCH current state,
configurable event lookback, and preservation of close-based
structural-break confirmation.

Run with: pytest tests/test_market_structure.py -v

The base series (see `_build_bullish_base`) is 45 candles with a
confirmed bullish structure:

  swing high at  5  (102.0)
  swing low  at 11  ( 98.0)
  swing high at 17  (104.0)   <- higher high
  swing low  at 23  ( 99.0)   <- higher low

=> trend = "bullish"
=> latest_swing_high = (17, 104.0)
=> latest_swing_low  = (23,  99.0)
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from src.indicators import analyze_market_structure, MarketStructure
from src.market_structure import (
    MarketStructureAnalyzer,
    StructureState,
)


# ---------------------------------------------------------------------------
# Series builder
# ---------------------------------------------------------------------------

def _build_bullish_base(n: int = 45):
    """Standard bullish series with the four swing points documented
    in the module docstring."""
    base = 100.0
    o = np.full(n, base, dtype=float)
    h = np.full(n, base + 0.5, dtype=float)
    l = np.full(n, base - 0.5, dtype=float)
    c = np.full(n, base, dtype=float)

    h[5] = 102.0
    l[11] = 98.0
    h[17] = 104.0
    l[23] = 99.0

    return o, h, l, c


def _call(o, h, l, c, event_lookback: int = 10):
    return analyze_market_structure(
        h, l, c,
        lookback=50,
        swing_lookback=5,
        event_lookback=event_lookback,
    )


# ---------------------------------------------------------------------------
# Baseline sanity: the synthetic series produces the documented swings
# ---------------------------------------------------------------------------

class TestBaselineSeries:

    def test_trend_is_bullish(self):
        o, h, l, c = _build_bullish_base()
        result = _call(o, h, l, c)
        assert result.trend == "bullish"

    def test_no_events_in_baseline(self):
        o, h, l, c = _build_bullish_base()
        result = _call(o, h, l, c)
        assert result.bos_detected is False
        assert result.choch_detected is False
        assert result.last_bos_index is None
        assert result.last_choch_index is None


# ---------------------------------------------------------------------------
# M-03 (i): mutually-exclusive current state
# ---------------------------------------------------------------------------

class TestMutualExclusivity:

    def _choch_more_recent_series(self):
        """BOS at 35, CHoCH at 40 -> CHoCH wins."""
        o, h, l, c = _build_bullish_base()

        # Flatten highs from 30-40 to prevent a new swing high from
        # forming on the BOS candle.
        h[30:41] = 107.0
        c[35] = 106.0            # BOS (close > latest_swing_high 104)

        # Two equal lows around the CHoCH candle prevent a new swing
        # low from forming at 40 (fractal requires a *unique* min).
        l[38] = 98.5
        l[40] = 98.5
        c[40] = 98.8             # CHoCH (close < latest_swing_low 99)

        return o, h, l, c

    def _bos_more_recent_series(self):
        """CHoCH at 35, BOS at 40 -> BOS wins."""
        o, h, l, c = _build_bullish_base()

        h[30:41] = 107.0
        l[33] = 98.5
        l[35] = 98.5
        c[35] = 98.8             # CHoCH (close < latest_swing_low 99)

        c[40] = 106.0            # BOS (close > latest_swing_high 104)

        return o, h, l, c

    def test_choch_more_recent_wins(self):
        o, h, l, c = self._choch_more_recent_series()
        result = _call(o, h, l, c)

        assert result.trend == "bullish"
        assert result.choch_detected is True
        assert result.last_choch_index == 40
        assert result.last_choch_direction == "bearish"

        # BOS is no longer the current state, but its historical
        # index and direction remain available.
        assert result.bos_detected is False
        assert result.last_bos_index == 35
        assert result.last_bos_direction == "bullish"

    def test_bos_more_recent_wins(self):
        o, h, l, c = self._bos_more_recent_series()
        result = _call(o, h, l, c)

        assert result.trend == "bullish"
        assert result.bos_detected is True
        assert result.last_bos_index == 40
        assert result.last_bos_direction == "bullish"

        assert result.choch_detected is False
        assert result.last_choch_index == 35
        assert result.last_choch_direction == "bearish"

    def test_is_helpers_reflect_current_state_after_choch(self):
        """Verify is_bos_* / is_choch_* on the analyzer cache."""
        o, h, l, c = self._choch_more_recent_series()
        result = _call(o, h, l, c)

        analyzer = MarketStructureAnalyzer()
        analyzer.structure_cache["h1"] = StructureState(
            timeframe="h1",
            timestamp=datetime.now(timezone.utc),
            structure=result,
            swing_highs=[],
            swing_lows=[],
            bias="bullish",
            key_levels={},
        )

        assert analyzer.is_bos_bullish("h1") is False
        assert analyzer.is_choch_bearish("h1") is True
        assert analyzer.is_choch_bullish("h1") is False
        assert analyzer.is_bos_bearish("h1") is False


# ---------------------------------------------------------------------------
# M-03 (iii): configurable event lookback
# ---------------------------------------------------------------------------

class TestEventLookback:

    def _series(self):
        o, h, l, c = _build_bullish_base()
        h[30:41] = 107.0
        c[35] = 106.0            # BOS at 35
        l[38] = 98.5
        l[40] = 98.5
        c[40] = 98.8             # CHoCH at 40
        return o, h, l, c

    def test_default_lookback_catches_both(self):
        o, h, l, c = self._series()
        result = _call(o, h, l, c, event_lookback=10)

        # Both events detected -> CHoCH wins (more recent).
        assert result.last_bos_index == 35
        assert result.last_choch_index == 40
        assert result.choch_detected is True
        assert result.bos_detected is False

    def test_tight_lookback_excludes_older_event(self):
        o, h, l, c = self._series()
        result = _call(o, h, l, c, event_lookback=5)

        # check_start = 45 - 5 = 40 -> only candle 40 evaluated.
        assert result.last_bos_index is None
        assert result.last_bos_direction is None
        assert result.bos_detected is False

        assert result.last_choch_index == 40
        assert result.choch_detected is True

    def test_wide_lookback_includes_older_event(self):
        o, h, l, c = self._series()
        result = _call(o, h, l, c, event_lookback=20)

        # check_start = 45 - 20 = 25 -> both still visible.
        assert result.last_bos_index == 35
        assert result.last_choch_index == 40

    def test_invalid_lookback_raises(self):
        o, h, l, c = _build_bullish_base()
        with pytest.raises(ValueError, match="event_lookback"):
            _call(o, h, l, c, event_lookback=0)


# ---------------------------------------------------------------------------
# Close-based structural-break confirmation is preserved
# (wick-only breach is not a BOS / CHoCH)
# ---------------------------------------------------------------------------

class TestWickOnlyBreach:

    def test_wick_above_swing_high_is_not_bos(self):
        """Candle 35 pierces 104 on its high (wick) but closes at 100
        — well below the swing high. That must NOT fire BOS."""
        o, h, l, c = _build_bullish_base()
        # Two equal highs prevent a new swing high from forming at 35
        # (fractal requires a *unique* max).
        h[33] = 105.0
        h[35] = 105.0
        c[35] = 100.0            # close < 104 -> not a BOS

        result = _call(o, h, l, c)

        assert result.bos_detected is False
        assert result.last_bos_index is None
        assert result.choch_detected is False
        assert result.last_choch_index is None

    def test_close_exactly_at_swing_high_is_not_bos(self):
        """Strict comparison: close == swing_high.price is not a break."""
        o, h, l, c = _build_bullish_base()
        h[30:41] = 107.0
        c[35] = 104.0            # exactly at latest_swing_high.price

        result = _call(o, h, l, c)

        assert result.bos_detected is False
        assert result.last_bos_index is None

    def test_close_exactly_at_swing_low_is_not_choch(self):
        """Strict comparison: close == swing_low.price is not a break."""
        o, h, l, c = _build_bullish_base()
        h[30:41] = 107.0
        l[38] = 99.0             # match swing_low price to avoid a
        l[40] = 99.0             # new swing low from forming
        c[40] = 99.0             # exactly at latest_swing_low.price

        result = _call(o, h, l, c)

        assert result.choch_detected is False
        assert result.last_choch_index is None


# ---------------------------------------------------------------------------
# Bearish branch: symmetric behavior
# ---------------------------------------------------------------------------

class TestBearishSymmetric:

    def _build_bearish_base(self, n: int = 45):
        base = 100.0
        o = np.full(n, base, dtype=float)
        h = np.full(n, base + 0.5, dtype=float)
        l = np.full(n, base - 0.5, dtype=float)
        c = np.full(n, base, dtype=float)

        # Lower lows and lower highs -> bearish trend.
        l[5] = 98.0              # swing low
        h[11] = 102.0            # swing high
        l[17] = 97.0             # swing low (lower low)
        h[23] = 101.0            # swing high (lower high)

        return o, h, l, c

    def test_bearish_bos_and_choch_mutual_exclusive(self):
        o, h, l, c = self._build_bearish_base()
        # Flatten lows to prevent a new swing low on the BOS candle.
        l[30:41] = 93.0

        # BOS at 35 (bearish): close below latest_swing_low 97
        c[35] = 95.0
        # CHoCH at 40 (bullish): close above latest_swing_high 101
        h[38] = 102.5
        h[40] = 102.5
        c[40] = 101.5

        result = _call(o, h, l, c)

        assert result.trend == "bearish"
        # CHoCH is more recent (40 > 35) -> current state is CHoCH.
        assert result.choch_detected is True
        assert result.last_choch_index == 40
        assert result.last_choch_direction == "bullish"
        assert result.bos_detected is False
        assert result.last_bos_index == 35
        assert result.last_bos_direction == "bearish"


# ---------------------------------------------------------------------------
# Config propagation into MarketStructureAnalyzer
# ---------------------------------------------------------------------------

class TestConfigPropagation:

    def test_reads_value_from_config(self, monkeypatch):
        from src import market_structure as ms_module

        class FakeSMC:
            structure_event_lookback = 7

        class FakeConfig:
            smc = FakeSMC()

        monkeypatch.setattr(
            ms_module, "config", FakeConfig(), raising=False
        )

        analyzer = ms_module.MarketStructureAnalyzer()
        assert analyzer.structure_event_lookback == 7

    def test_missing_key_defaults_to_10(self, monkeypatch):
        from src import market_structure as ms_module

        class FakeSMC:
            pass  # no structure_event_lookback attribute

        class FakeConfig:
            smc = FakeSMC()

        monkeypatch.setattr(
            ms_module, "config", FakeConfig(), raising=False
        )

        analyzer = ms_module.MarketStructureAnalyzer()
        assert analyzer.structure_event_lookback == 10

    def test_invalid_value_falls_back_to_10(self, monkeypatch):
        from src import market_structure as ms_module

        for bad in (-1, 0, "not-a-number", True, 3.7):
            class FakeSMC:
                structure_event_lookback = bad

            class FakeConfig:
                smc = FakeSMC()

            monkeypatch.setattr(
                ms_module, "config", FakeConfig(), raising=False
            )

            analyzer = ms_module.MarketStructureAnalyzer()
            # 3.7 is coerced to int(3.7)=3 by _read_event_lookback?
            # No: isinstance(3.7, bool) is False, int(3.7)=3, 3 >= 1
            # -> 3. So 3.7 is technically coerced to 3, which is a
            # valid positive int. Adjust the expectation accordingly.
            if bad == 3.7:
                assert analyzer.structure_event_lookback == 3
            else:
                assert analyzer.structure_event_lookback == 10