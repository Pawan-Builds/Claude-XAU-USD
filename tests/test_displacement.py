"""Tests for M-02: ATR-relative displacement detection.

Run with: pytest tests/test_displacement.py -v

All test prices are in "absolute" units. Point/pip conversions are
supplied explicitly so tests do not depend on a live broker.
"""

import numpy as np
import pytest

from src.indicators import detect_displacement, _compute_atr_series


def _build_candles(bar_high_low_range: float, bar_body: float,
                   n: int = 40, close_at_end: float = 100.0):
    """Synthetic OHLC where every candle has the given range and body.

    Bodies alternate bullish/bearish and are symmetric around a fixed
    midpoint, so ATR is stable and small deterministic shifts are
    easy to introduce.
    """
    mid = 100.0
    o = np.empty(n, dtype=float)
    h = np.empty(n, dtype=float)
    l = np.empty(n, dtype=float)
    c = np.empty(n, dtype=float)
    for i in range(n):
        if i % 2 == 0:
            o[i] = mid - bar_body / 2.0
            c[i] = mid + bar_body / 2.0
        else:
            o[i] = mid + bar_body / 2.0
            c[i] = mid - bar_body / 2.0
        h[i] = mid + bar_high_low_range / 2.0
        l[i] = mid - bar_high_low_range / 2.0
    return o, h, l, c


def _inject_spike(o, h, l, c, index: int, body: float, range_: float):
    """Overwrite one candle with a directional spike of given body/range."""
    mid = 100.0
    o[index] = mid - body / 2.0
    c[index] = mid + body / 2.0
    h[index] = mid + range_ / 2.0
    l[index] = mid - range_ / 2.0
    return o, h, l, c


# ---------------------------------------------------------------------------
# ATR computation
# ---------------------------------------------------------------------------

class TestATRSeries:

    def test_insufficient_data_returns_nan(self):
        o, h, l, c = _build_candles(1.0, 0.1, n=10)
        atr = _compute_atr_series(h, l, c, period=14)
        # n=10 < period+1=15 => all NaN
        assert np.all(np.isnan(atr))

    def test_stable_range_yields_expected_atr(self):
        o, h, l, c = _build_candles(1.0, 0.1, n=40)
        atr = _compute_atr_series(h, l, c, period=14)
        # After the period warm-up, ATR should be ~= range = 1.0
        valid = atr[~np.isnan(atr)]
        assert len(valid) > 0
        assert np.allclose(valid, 1.0, atol=1e-9)

    def test_atr_at_i_does_not_include_candle_i(self):
        # Compare ATR at index 39 in two datasets that only differ at
        # index 39. ATR[39] must be identical because it uses [25..38].
        o, h, l, c = _build_candles(1.0, 0.1, n=40)
        atr_a = _compute_atr_series(h, l, c, period=14)

        o2, h2, l2, c2 = _build_candles(1.0, 0.1, n=40)
        h2[39] = 200.0
        l2[39] = 0.0
        atr_b = _compute_atr_series(h2, l2, c2, period=14)

        assert np.isclose(atr_a[39], atr_b[39], equal_nan=True)


# ---------------------------------------------------------------------------
# Displacement detection
# ---------------------------------------------------------------------------

class TestDisplacementDetection:

    def test_insufficient_data_returns_empty(self):
        o, h, l, c = _build_candles(1.0, 0.1, n=10)
        out = detect_displacement(
            o, h, l, c, lookback=10, atr_period=14,
            atr_multiplier=1.5, min_body_pips=0, point_value=0.01,
        )
        assert out == []

    def test_ordinary_candle_not_flagged(self):
        o, h, l, c = _build_candles(1.0, 0.1, n=40)
        out = detect_displacement(
            o, h, l, c, lookback=10, atr_period=14,
            atr_multiplier=1.5, min_body_pips=0, point_value=0.01,
        )
        assert out == []

    def test_clear_displacement_flagged(self):
        o, h, l, c = _build_candles(1.0, 0.1, n=40)
        _inject_spike(o, h, l, c, index=39, body=2.0, range_=2.5)
        out = detect_displacement(
            o, h, l, c, lookback=10, atr_period=14,
            atr_multiplier=1.5, min_body_pips=0, point_value=0.01,
        )
        assert len(out) == 1
        assert out[0]["index"] == 39
        assert out[0]["direction"] == "bullish"
        assert out[0]["atr_ratio"] >= 1.5

    def test_high_volatility_candle_not_flagged(self):
        """The central M-02 fix: an ordinary candle during a high-
        volatility burst must NOT count as displacement.
        ATR = 5.0, body = 2.0, multiplier = 1.5 -> 2.0 < 7.5 -> no.
        """
        o, h, l, c = _build_candles(5.0, 2.0, n=40)
        out = detect_displacement(
            o, h, l, c, lookback=10, atr_period=14,
            atr_multiplier=1.5, min_body_pips=0, point_value=0.01,
        )
        assert out == []

    def test_min_body_floor_filters_tiny_bodies(self):
        """Tiny ATR, tiny body — ATR ratio passes but absolute floor
        rejects. point_value=0.01, min_body_pips=10 => 0.10 floor.
        """
        o, h, l, c = _build_candles(0.05, 0.01, n=40)
        _inject_spike(o, h, l, c, index=39, body=0.08, range_=0.09)
        # Without floor: body 0.08 > 1.5*ATR(~0.05)=0.075 -> would flag
        out_no_floor = detect_displacement(
            o, h, l, c, lookback=10, atr_period=14,
            atr_multiplier=1.5, min_body_pips=0, point_value=0.01,
        )
        assert len(out_no_floor) == 1

        # With floor: min_body_price = 10*0.01 = 0.10 > 0.08 -> no flag
        out_with_floor = detect_displacement(
            o, h, l, c, lookback=10, atr_period=14,
            atr_multiplier=1.5, min_body_pips=10, point_value=0.01,
        )
        assert out_with_floor == []

    def test_bearish_direction_correct(self):
        o, h, l, c = _build_candles(1.0, 0.1, n=40)
        # Overwrite index 39 with a strong bearish body.
        mid = 100.0
        o[39] = mid + 1.0
        c[39] = mid - 1.0
        h[39] = mid + 1.2
        l[39] = mid - 1.2
        out = detect_displacement(
            o, h, l, c, lookback=10, atr_period=14,
            atr_multiplier=1.5, min_body_pips=0, point_value=0.01,
        )
        assert len(out) == 1
        assert out[0]["direction"] == "bearish"

    def test_output_includes_legacy_keys(self):
        o, h, l, c = _build_candles(1.0, 0.1, n=40)
        _inject_spike(o, h, l, c, index=39, body=2.0, range_=2.5)
        out = detect_displacement(
            o, h, l, c, lookback=10, atr_period=14,
            atr_multiplier=1.5, min_body_pips=0, point_value=0.01,
        )
        assert "ratio" in out[0]
        assert "avg_body" in out[0]
        assert "atr_ratio" in out[0]
        assert "atr" in out[0]
        assert "body_size" in out[0]

    def test_no_mutation_of_inputs(self):
        o, h, l, c = _build_candles(1.0, 0.1, n=40)
        o0, h0, l0, c0 = o.copy(), h.copy(), l.copy(), c.copy()
        detect_displacement(
            o, h, l, c, lookback=10, atr_period=14,
            atr_multiplier=1.5, min_body_pips=0, point_value=0.01,
        )
        assert np.array_equal(o, o0)
        assert np.array_equal(h, h0)
        assert np.array_equal(l, l0)
        assert np.array_equal(c, c0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_mismatched_lengths_raise(self):
        o, h, l, c = _build_candles(1.0, 0.1, n=20)
        with pytest.raises(ValueError, match="equal lengths"):
            detect_displacement(o, h, l, c[:-1])

    def test_negative_multiplier_raises(self):
        o, h, l, c = _build_candles(1.0, 0.1, n=20)
        with pytest.raises(ValueError, match="atr_multiplier"):
            detect_displacement(
                o, h, l, c, atr_multiplier=-0.1,
            )

    def test_zero_atr_period_raises(self):
        o, h, l, c = _build_candles(1.0, 0.1, n=20)
        with pytest.raises(ValueError, match="atr_period"):
            detect_displacement(o, h, l, c, atr_period=0)

    def test_negative_min_body_raises(self):
        o, h, l, c = _build_candles(1.0, 0.1, n=20)
        with pytest.raises(ValueError, match="min_body_pips"):
            detect_displacement(o, h, l, c, min_body_pips=-1.0)

    def test_non_positive_point_value_raises(self):
        o, h, l, c = _build_candles(1.0, 0.1, n=20)
        with pytest.raises(ValueError, match="point_value"):
            detect_displacement(o, h, l, c, point_value=0.0)


# ---------------------------------------------------------------------------
# No lookahead: altering candles after index i must not change result at i
# ---------------------------------------------------------------------------

class TestNoLookahead:

    def test_altered_future_does_not_change_past_detection(self):
        base_o, base_h, base_l, base_c = _build_candles(1.0, 0.1, n=40)
        _inject_spike(base_o, base_h, base_l, base_c,
                      index=39, body=2.0, range_=2.5)

        out_a = detect_displacement(
            base_o, base_h, base_l, base_c,
            lookback=10, atr_period=14,
            atr_multiplier=1.5, min_body_pips=0, point_value=0.01,
        )
        assert len(out_a) == 1
        idx = out_a[0]["index"]

        # Introduce a massive spike AFTER idx in a fresh copy.
        mod_o, mod_h, mod_l, mod_c = (
            base_o.copy(), base_h.copy(),
            base_l.copy(), base_c.copy(),
        )
        if idx + 1 < 40:
            _inject_spike(mod_o, mod_h, mod_l, mod_c,
                          index=idx + 1, body=100.0, range_=120.0)

        out_b = detect_displacement(
            mod_o, mod_h, mod_l, mod_c,
            lookback=10, atr_period=14,
            atr_multiplier=1.5, min_body_pips=0, point_value=0.01,
        )
        # Displacement at idx must be unchanged by future candles.
        a = next((d for d in out_a if d["index"] == idx), None)
        b = next((d for d in out_b if d["index"] == idx), None)
        assert a is not None
        assert b is not None
        assert a["atr"] == b["atr"]
        assert a["body_size"] == b["body_size"]