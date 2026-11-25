"""Technical indicators for SMC analysis.

Conventions used throughout this module:
- Buy-side liquidity (BSL) is above equal highs / swing highs.
- Sell-side liquidity (SSL) is below equal lows / swing lows.
- A bullish FVG is fully mitigated when price reaches its bottom.
- A bearish FVG is fully mitigated when price reaches its top.
- A bullish order block is fully mitigated when price reaches its bottom.
- A bearish order block is fully mitigated when price reaches its top.

Swing points are only confirmed when the required candles to their right
are available. The caller is responsible for supplying completed candles
when confirmed-candle analysis is required.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Shared input validation
# ---------------------------------------------------------------------------

def _as_float_array(values: Any, name: str) -> np.ndarray:
    """Convert input to a one-dimensional float array and validate it."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")

    return array


def _validate_ohlc(
    opens: Any,
    highs: Any,
    lows: Any,
    closes: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert and validate OHLC arrays."""
    open_array = _as_float_array(opens, "opens")
    high_array = _as_float_array(highs, "highs")
    low_array = _as_float_array(lows, "lows")
    close_array = _as_float_array(closes, "closes")

    lengths = {
        len(open_array),
        len(high_array),
        len(low_array),
        len(close_array),
    }

    if len(lengths) != 1:
        raise ValueError("OHLC arrays must all have the same length")

    if np.any(high_array < low_array):
        raise ValueError("A candle has a high below its low")

    if np.any(high_array < np.maximum(open_array, close_array)):
        raise ValueError("A candle high is below its open or close")

    if np.any(low_array > np.minimum(open_array, close_array)):
        raise ValueError("A candle low is above its open or close")

    return open_array, high_array, low_array, close_array


def _validate_lookback(lookback: int, name: str = "lookback") -> int:
    """Validate a positive integer lookback."""
    if isinstance(lookback, bool) or not isinstance(lookback, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")

    if lookback < 1:
        raise ValueError(f"{name} must be at least 1")

    return int(lookback)


def _validate_index(current_index: int, length: int) -> int:
    """Validate a candle index."""
    if isinstance(current_index, bool) or not isinstance(
        current_index, (int, np.integer)
    ):
        raise ValueError("current_index must be an integer")

    current_index = int(current_index)

    if current_index < 0 or current_index >= length:
        raise IndexError(
            f"current_index {current_index} is outside array length {length}"
        )

    return current_index


# ---------------------------------------------------------------------------
# Swing points
# ---------------------------------------------------------------------------

@dataclass
class SwingPoint:
    index: int
    price: float
    type: str  # "high" or "low"
    timestamp: Any = None
    strength: int = 1


def find_swing_highs(
    highs: np.ndarray,
    lookback: int = 5
) -> List[SwingPoint]:
    """Find confirmed swing highs using a symmetrical fractal pattern."""
    high_array = _as_float_array(highs, "highs")
    lookback = _validate_lookback(lookback)

    swings: List[SwingPoint] = []
    n = len(high_array)

    if n < (2 * lookback + 1):
        return swings

    for i in range(lookback, n - lookback):
        window = high_array[i - lookback:i + lookback + 1]
        center = high_array[i]

        if center == np.max(window) and np.count_nonzero(window == center) == 1:
            swings.append(
                SwingPoint(index=i, price=float(center), type="high")
            )

    return swings


def find_swing_lows(
    lows: np.ndarray,
    lookback: int = 5
) -> List[SwingPoint]:
    """Find confirmed swing lows using a symmetrical fractal pattern."""
    low_array = _as_float_array(lows, "lows")
    lookback = _validate_lookback(lookback)

    swings: List[SwingPoint] = []
    n = len(low_array)

    if n < (2 * lookback + 1):
        return swings

    for i in range(lookback, n - lookback):
        window = low_array[i - lookback:i + lookback + 1]
        center = low_array[i]

        if center == np.min(window) and np.count_nonzero(window == center) == 1:
            swings.append(
                SwingPoint(index=i, price=float(center), type="low")
            )

    return swings


def find_swing_points(
    highs: np.ndarray,
    lows: np.ndarray,
    lookback: int = 5
) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """Find both swing highs and swing lows."""
    return (
        find_swing_highs(highs, lookback),
        find_swing_lows(lows, lookback),
    )


# ---------------------------------------------------------------------------
# Market structure: BOS and CHoCH
# ---------------------------------------------------------------------------

@dataclass
class MarketStructure:
    trend: str  # "bullish", "bearish", or "ranging"
    bos_detected: bool
    choch_detected: bool
    last_bos_index: Optional[int]
    last_choch_index: Optional[int]
    last_bos_direction: Optional[str]
    last_choch_direction: Optional[str]
    swing_highs: List[SwingPoint]
    swing_lows: List[SwingPoint]
    structure_levels: Dict[str, float]


def analyze_market_structure(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    lookback: int = 50,
    swing_lookback: int = 5
) -> MarketStructure:
    """Analyze confirmed swing structure and recent BOS/CHoCH events."""
    high_array = _as_float_array(highs, "highs")
    low_array = _as_float_array(lows, "lows")
    close_array = _as_float_array(closes, "closes")

    if not (len(high_array) == len(low_array) == len(close_array)):
        raise ValueError("highs, lows, and closes must have equal lengths")

    lookback = _validate_lookback(lookback)
    swing_lookback = _validate_lookback(swing_lookback)

    if len(close_array) == 0:
        raise ValueError("Market structure cannot be calculated from empty data")

    if np.any(high_array < low_array):
        raise ValueError("A candle has a high below its low")

    all_highs, all_lows = find_swing_points(
        high_array, low_array, swing_lookback
    )

    start_index = max(0, len(close_array) - lookback)

    recent_highs = [s for s in all_highs if s.index >= start_index]
    recent_lows = [s for s in all_lows if s.index >= start_index]

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return MarketStructure(
            trend="ranging",
            bos_detected=False,
            choch_detected=False,
            last_bos_index=None,
            last_choch_index=None,
            last_bos_direction=None,
            last_choch_direction=None,
            swing_highs=recent_highs,
            swing_lows=recent_lows,
            structure_levels={},
        )

    # Trend = direction of the LAST TWO swing pivots on each side.
    # This is more robust than counting older pivots because the most
    # recent two are the ones the market is currently respecting.
    def _last_direction(swings: List[SwingPoint]) -> int:
        if len(swings) < 2:
            return 0
        diff = swings[-1].price - swings[-2].price
        if diff > 0:
            return 1
        if diff < 0:
            return -1
        return 0

    high_dir = _last_direction(recent_highs)
    low_dir = _last_direction(recent_lows)

    if high_dir > 0 and low_dir > 0:
        trend = "bullish"
    elif high_dir < 0 and low_dir < 0:
        trend = "bearish"
    elif high_dir == 0 and low_dir == 0:
        trend = "ranging"
    else:
        combined = high_dir + low_dir
        if combined > 0:
            trend = "bullish"
        elif combined < 0:
            trend = "bearish"
        else:
            trend = "ranging"

    bos_detected = False
    choch_detected = False
    last_bos_index: Optional[int] = None
    last_choch_index: Optional[int] = None
    last_bos_direction: Optional[str] = None
    last_choch_direction: Optional[str] = None

    latest_swing_high = recent_highs[-1]
    latest_swing_low = recent_lows[-1]

    check_start = max(0, len(close_array) - 10)

    if trend == "bullish":
        for i in range(check_start, len(close_array)):
            if i <= latest_swing_high.index:
                continue

            if close_array[i] > latest_swing_high.price:
                bos_detected = True
                last_bos_index = i
                last_bos_direction = "bullish"

            elif close_array[i] < latest_swing_low.price:
                choch_detected = True
                last_choch_index = i
                last_choch_direction = "bearish"

    elif trend == "bearish":
        for i in range(check_start, len(close_array)):
            if i <= latest_swing_low.index:
                continue

            if close_array[i] < latest_swing_low.price:
                bos_detected = True
                last_bos_index = i
                last_bos_direction = "bearish"

            elif close_array[i] > latest_swing_high.price:
                choch_detected = True
                last_choch_index = i
                last_choch_direction = "bullish"

    structure_levels: Dict[str, float] = {}

    if recent_highs:
        structure_levels["recent_high"] = float(recent_highs[-1].price)
        if len(recent_highs) >= 2:
            structure_levels["previous_high"] = float(recent_highs[-2].price)

    if recent_lows:
        structure_levels["recent_low"] = float(recent_lows[-1].price)
        if len(recent_lows) >= 2:
            structure_levels["previous_low"] = float(recent_lows[-2].price)

    return MarketStructure(
        trend=trend,
        bos_detected=bos_detected,
        choch_detected=choch_detected,
        last_bos_index=last_bos_index,
        last_choch_index=last_choch_index,
        last_bos_direction=last_bos_direction,
        last_choch_direction=last_choch_direction,
        swing_highs=recent_highs,
        swing_lows=recent_lows,
        structure_levels=structure_levels,
    )


# ---------------------------------------------------------------------------
# Fair Value Gaps
# ---------------------------------------------------------------------------

@dataclass
class FVG:
    index: int
    type: str  # "bullish" or "bearish"
    top: float
    bottom: float
    timestamp: Any = None
    mitigated: bool = False
    mitigation_index: Optional[int] = None


def detect_fvgs(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    min_size_pips: float = 5.0,
    point_value: float = 0.01
) -> List[FVG]:
    """Detect three-candle Fair Value Gaps."""
    open_array, high_array, low_array, close_array = _validate_ohlc(
        opens, highs, lows, closes
    )

    if not np.isfinite(min_size_pips) or min_size_pips < 0:
        raise ValueError("min_size_pips must be a non-negative number")

    if not np.isfinite(point_value) or point_value <= 0:
        raise ValueError("point_value must be greater than zero")

    min_size = float(min_size_pips) * float(point_value)
    fvgs: List[FVG] = []

    for i in range(2, len(open_array)):
        bullish_bottom = high_array[i - 2]
        bullish_top = low_array[i]

        if bullish_top > bullish_bottom:
            gap_size = bullish_top - bullish_bottom
            if gap_size >= min_size:
                fvgs.append(
                    FVG(
                        index=i,
                        type="bullish",
                        top=float(bullish_top),
                        bottom=float(bullish_bottom),
                    )
                )

        bearish_bottom = high_array[i]
        bearish_top = low_array[i - 2]

        if bearish_top > bearish_bottom:
            gap_size = bearish_top - bearish_bottom
            if gap_size >= min_size:
                fvgs.append(
                    FVG(
                        index=i,
                        type="bearish",
                        top=float(bearish_top),
                        bottom=float(bearish_bottom),
                    )
                )

    return fvgs


def check_fvg_mitigation(
    fvgs: List[FVG],
    highs: np.ndarray,
    lows: np.ndarray,
    current_index: int
) -> List[FVG]:
    """Update FVGs when a later candle completely fills the gap."""
    high_array = _as_float_array(highs, "highs")
    low_array = _as_float_array(lows, "lows")

    if len(high_array) != len(low_array):
        raise ValueError("highs and lows must have equal lengths")

    if len(high_array) == 0:
        return fvgs

    current_index = _validate_index(current_index, len(high_array))

    for fvg in fvgs:
        if fvg.mitigated:
            continue
        if fvg.type not in ("bullish", "bearish"):
            continue
        if fvg.bottom >= fvg.top:
            continue

        start = max(0, fvg.index + 1)
        if start > current_index:
            continue

        for i in range(start, current_index + 1):
            if fvg.type == "bullish" and low_array[i] <= fvg.bottom:
                fvg.mitigated = True
                fvg.mitigation_index = i
                break
            if fvg.type == "bearish" and high_array[i] >= fvg.top:
                fvg.mitigated = True
                fvg.mitigation_index = i
                break

    return fvgs


# ---------------------------------------------------------------------------
# Order Blocks
# ---------------------------------------------------------------------------

@dataclass
class OrderBlock:
    index: int
    type: str  # "bullish" or "bearish"
    top: float
    bottom: float
    timestamp: Any = None
    strength: float = 1.0
    tested: bool = False
    mitigation_index: Optional[int] = None


def detect_order_blocks(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    lookback: int = 20,
    displacement_threshold: float = 1.5
) -> List[OrderBlock]:
    """Detect candidate order blocks before strong directional moves."""

    open_array, high_array, low_array, close_array = _validate_ohlc(
        opens, highs, lows, closes
    )

    lookback = _validate_lookback(lookback)

    if not np.isfinite(displacement_threshold) or displacement_threshold <= 0:
        raise ValueError("displacement_threshold must be greater than zero")

    obs: List[OrderBlock] = []
    n = len(open_array)

    if lookback < 1:
        raise ValueError("lookback must be at least 1")

    start_index = lookback

    for i in range(start_index, n - 1):
        baseline_start = max(0, i - lookback)
        previous_bodies = np.abs(
            close_array[baseline_start:i]
            - open_array[baseline_start:i]
        )
        if len(previous_bodies) == 0:
            continue

        avg_size = float(np.mean(previous_bodies))
        if avg_size <= 0:
            continue

        future_end = min(i + 5, n)
        future_closes = close_array[i + 1:future_end]

        if len(future_closes) == 0:
            continue

        if close_array[i] < open_array[i]:
            upward_displacement = float(
                np.max(future_closes) - open_array[i]
            )
            strength = upward_displacement / avg_size
            if strength >= displacement_threshold:
                obs.append(
                    OrderBlock(
                        index=i,
                        type="bullish",
                        top=float(high_array[i]),
                        bottom=float(low_array[i]),
                        strength=float(strength),
                    )
                )

        elif close_array[i] > open_array[i]:
            downward_displacement = float(
                open_array[i] - np.min(future_closes)
            )
            strength = downward_displacement / avg_size
            if strength >= displacement_threshold:
                obs.append(
                    OrderBlock(
                        index=i,
                        type="bearish",
                        top=float(high_array[i]),
                        bottom=float(low_array[i]),
                        strength=float(strength),
                    )
                )

    return obs


def check_ob_mitigation(
    obs: List[OrderBlock],
    highs: np.ndarray,
    lows: np.ndarray,
    current_index: int
) -> List[OrderBlock]:
    """Mark order blocks fully mitigated by a later candle."""
    high_array = _as_float_array(highs, "highs")
    low_array = _as_float_array(lows, "lows")

    if len(high_array) != len(low_array):
        raise ValueError("highs and lows must have equal lengths")

    if len(high_array) == 0:
        return obs

    current_index = _validate_index(current_index, len(high_array))

    for ob in obs:
        if ob.tested:
            continue
        if ob.type not in ("bullish", "bearish"):
            continue
        if ob.bottom >= ob.top:
            continue

        start = max(0, ob.index + 1)
        if start > current_index:
            continue

        for i in range(start, current_index + 1):
            if ob.type == "bullish" and low_array[i] <= ob.bottom:
                ob.tested = True
                ob.mitigation_index = i
                break
            if ob.type == "bearish" and high_array[i] >= ob.top:
                ob.tested = True
                ob.mitigation_index = i
                break

    return obs


# ---------------------------------------------------------------------------
# Liquidity: equal highs and equal lows
# ---------------------------------------------------------------------------

@dataclass
class LiquidityLevel:
    price: float
    type: str  # "buy_side" or "sell_side"
    strength: int
    indices: List[int]
    timestamp: Any = None
    swept: bool = False
    sweep_index: Optional[int] = None
    # M-01: classification of the event that consumed this level.
    #   "sweep"    = wick pierced the level and the same closed candle
    #                closed back on the origin side (genuine liquidity
    #                sweep; eligible to trigger strategies).
    #   "breakout" = wick pierced the level and the close stayed
    #                beyond it (structural breakout; consumed, i.e.
    #                excluded from active_levels, but MUST NOT be
    #                reported as a sweep).
    #   None       = level is still active.
    #
    # Added as the last field with a default so existing keyword-arg
    # constructions in detect_liquidity_levels() are unchanged.
    sweep_kind: Optional[str] = None


def _cluster_price_levels(
    values: np.ndarray,
    start_index: int,
    tolerance: float
) -> List[Tuple[float, List[int]]]:
    """Cluster nearby prices and return average price plus candle indices."""
    indexed_values = [
        (float(values[i]), i) for i in range(start_index, len(values))
    ]
    indexed_values.sort(key=lambda item: item[0])

    clusters: List[List[Tuple[float, int]]] = []

    for price, index in indexed_values:
        if not clusters:
            clusters.append([(price, index)])
            continue

        current_cluster = clusters[-1]
        cluster_mean = float(np.mean([item[0] for item in current_cluster]))

        if abs(price - cluster_mean) <= tolerance:
            current_cluster.append((price, index))
        else:
            clusters.append([(price, index)])

    result: List[Tuple[float, List[int]]] = []

    for cluster in clusters:
        average_price = float(np.mean([item[0] for item in cluster]))
        indices = sorted(item[1] for item in cluster)
        result.append((average_price, indices))

    return result


def detect_liquidity_levels(
    highs: np.ndarray,
    lows: np.ndarray,
    lookback: int = 100,
    tolerance_pips: float = 2.0,
    point_value: float = 0.01
) -> Tuple[List[LiquidityLevel], List[LiquidityLevel]]:
    """Detect buy-side and sell-side liquidity from equal highs/lows."""

    high_array = _as_float_array(highs, "highs")
    low_array = _as_float_array(lows, "lows")

    if len(high_array) != len(low_array):
        raise ValueError("highs and lows must have equal lengths")

    lookback = _validate_lookback(lookback)

    if not np.isfinite(tolerance_pips) or tolerance_pips < 0:
        raise ValueError("tolerance_pips must be non-negative")

    if not np.isfinite(point_value) or point_value <= 0:
        raise ValueError("point_value must be greater than zero")

    if len(high_array) == 0:
        return [], []

    tolerance = float(tolerance_pips) * float(point_value)
    start_index = max(0, len(high_array) - lookback)

    high_clusters = _cluster_price_levels(
        high_array, start_index, tolerance
    )
    low_clusters = _cluster_price_levels(
        low_array, start_index, tolerance
    )

    buy_side: List[LiquidityLevel] = []
    sell_side: List[LiquidityLevel] = []

    for price, indices in high_clusters:
        if len(indices) >= 2:
            buy_side.append(
                LiquidityLevel(
                    price=price,
                    type="buy_side",
                    strength=len(indices),
                    indices=indices,
                )
            )

    for price, indices in low_clusters:
        if len(indices) >= 2:
            sell_side.append(
                LiquidityLevel(
                    price=price,
                    type="sell_side",
                    strength=len(indices),
                    indices=indices,
                )
            )

    return buy_side, sell_side


def check_liquidity_sweep(
    buy_side: List[LiquidityLevel],
    sell_side: List[LiquidityLevel],
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    current_index: int,
    pierce_buffer: float = 0.0,
    reclaim_buffer: float = 0.0,
) -> Tuple[List[LiquidityLevel], List[LiquidityLevel]]:
    """Classify the latest closed candle against liquidity levels.

    A level is consumed when the candle wick pierces it strictly:

      - buy-side  : high  > level.price + pierce_buffer
      - sell-side : low   < level.price - pierce_buffer

    An exact touch (high == level.price for buy-side, low ==
    level.price for sell-side) does NOT consume the level.

    A consumed level is classified by the same candle's close:

      - buy-side  : sweep    if close < level.price - reclaim_buffer
                    breakout if close >= level.price - reclaim_buffer
      - sell-side : sweep    if close > level.price + reclaim_buffer
                    breakout if close <= level.price + reclaim_buffer

    Only "sweep"-classified levels are eligible to feed strategy
    entry signals. "breakout"-classified levels are still excluded
    from active_levels (they have been structurally consumed) but
    do NOT trigger was_*_swept / _find_sweep_price.

    The caller is responsible for supplying closed candles only;
    the function reads closes[current_index] and will therefore
    repaint if handed a forming candle. LiquidityAnalyzer already
    enforces this by dropping the in-progress candle before calling.

    Args:
        pierce_buffer: minimum price distance beyond the level to
            count as a pierce (price units, NOT pips).
        reclaim_buffer: minimum price distance the close must return
            beyond the level to count as a reclaim (price units).
    """

    high_array = _as_float_array(highs, "highs")
    low_array = _as_float_array(lows, "lows")
    close_array = _as_float_array(closes, "closes")

    if not (
        len(high_array) == len(low_array) == len(close_array)
    ):
        raise ValueError(
            "highs, lows, and closes must have equal lengths"
        )

    if len(high_array) == 0:
        return buy_side, sell_side

    if not np.isfinite(pierce_buffer) or pierce_buffer < 0:
        raise ValueError(
            "pierce_buffer must be a non-negative finite number"
        )

    if not np.isfinite(reclaim_buffer) or reclaim_buffer < 0:
        raise ValueError(
            "reclaim_buffer must be a non-negative finite number"
        )

    current_index = _validate_index(current_index, len(high_array))

    current_high = float(high_array[current_index])
    current_low = float(low_array[current_index])
    current_close = float(close_array[current_index])

    for level in buy_side:
        if level.type != "buy_side" or level.swept:
            continue

        # Strict pierce with buffer.
        if current_high <= level.price + pierce_buffer:
            continue

        level.swept = True
        level.sweep_index = current_index

        if current_close < level.price - reclaim_buffer:
            level.sweep_kind = "sweep"
        else:
            level.sweep_kind = "breakout"

    for level in sell_side:
        if level.type != "sell_side" or level.swept:
            continue

        if current_low >= level.price - pierce_buffer:
            continue

        level.swept = True
        level.sweep_index = current_index

        if current_close > level.price + reclaim_buffer:
            level.sweep_kind = "sweep"
        else:
            level.sweep_kind = "breakout"

    return buy_side, sell_side
# ---------------------------------------------------------------------------
# Premium and discount
# ---------------------------------------------------------------------------

def calculate_premium_discount(
    high: float,
    low: float,
    current_price: float
) -> Dict[str, Any]:
    """Calculate the current price position within a dealing range."""
    values = np.asarray([high, low, current_price], dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError("Premium/discount prices must be finite numbers")

    high = float(high)
    low = float(low)
    current_price = float(current_price)

    # FIX: Handle inverted high/low gracefully to prevent backtest crashes.
    if high < low:
        high, low = low, high

    midpoint = (high + low) / 2.0
    equilibrium = midpoint

    if high == low:
        zone = "equilibrium"
        distance_from_eq = 0.0
    elif current_price > midpoint:
        zone = "premium"
        distance_from_eq = (current_price - midpoint) / (high - midpoint)
    elif current_price < midpoint:
        zone = "discount"
        distance_from_eq = (midpoint - current_price) / (midpoint - low)
    else:
        zone = "equilibrium"
        distance_from_eq = 0.0

    return {
        "zone": zone,
        "equilibrium": equilibrium,
        "high": high,
        "low": low,
        "midpoint": midpoint,
        "current_price": current_price,
        "distance_from_equilibrium": float(distance_from_eq),
    }

# ---------------------------------------------------------------------------
# Displacement
# ---------------------------------------------------------------------------

def _compute_atr_series(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
) -> np.ndarray:
    """Compute ATR[i] using true-range values from [i-period, i-1].

    ATR[i] therefore reflects volatility strictly BEFORE candle i, so
    the candidate candle never contributes to its own comparison
    baseline. Returns an array of the same length as `closes` where
    insufficient-history positions are NaN.
    """
    n = len(closes)
    atr = np.full(n, np.nan, dtype=float)

    if period < 1 or n < period + 1:
        return atr

    # TR[k] for k >= 1 (needs close[k-1]); TR[0] undefined -> NaN.
    tr = np.full(n, np.nan, dtype=float)
    for k in range(1, n):
        h = float(highs[k])
        l = float(lows[k])
        prev_c = float(closes[k - 1])
        tr[k] = max(h - l, abs(h - prev_c), abs(l - prev_c))

    # ATR[i] = mean(tr[i-period : i]) — window ends at i-1.
    for i in range(period + 1, n):
        window = tr[i - period:i]
        if np.all(np.isfinite(window)):
            atr[i] = float(np.mean(window))

    return atr


def detect_displacement(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    lookback: int = 10,
    atr_period: int = 14,
    atr_multiplier: float = 1.5,
    min_body_pips: float = 5.0,
    point_value: float = 0.01,
) -> List[Dict[str, Any]]:
    """Detect displacement candles using ATR-relative volatility.

    A candle i is classified as displacement only when BOTH hold:

      1. body_size >= atr_multiplier * ATR(atr_period) at index i
         (ATR is computed strictly from data before i, so the
         candidate candle cannot inflate its own baseline)
      2. body_size >= min_body_pips * point_value

    Condition (1) ensures the body is unusual for the prevailing
    volatility regime. An ordinary candle during a high-volatility
    burst has a large body but a proportionally large ATR and is
    therefore NOT flagged. Condition (2) is an absolute floor that
    rejects tiny bodies during ultra-quiet regimes.

    No lookahead: ATR[i] uses only candles at indices < i. The body
    comparison uses candle i, which the caller must supply as a
    closed candle.

    Returns a list of dicts with keys:
        index, direction, body_size, atr, atr_ratio,
        avg_body (legacy), ratio (legacy — body/avg_body)

    Raises ValueError for invalid parameters or non-finite inputs.
    """
    open_array = _as_float_array(opens, "opens")
    high_array = _as_float_array(highs, "highs")
    low_array = _as_float_array(lows, "lows")
    close_array = _as_float_array(closes, "closes")

    if not (
        len(open_array) == len(high_array)
        == len(low_array) == len(close_array)
    ):
        raise ValueError(
            "opens, highs, lows, and closes must have equal lengths"
        )

    if np.any(high_array < low_array):
        raise ValueError("A candle has a high below its low")

    lookback = _validate_lookback(lookback)

    if isinstance(atr_period, bool) or not isinstance(
        atr_period, (int, np.integer)
    ):
        raise ValueError("atr_period must be an integer")
    atr_period = int(atr_period)
    if atr_period < 1:
        raise ValueError("atr_period must be at least 1")

    if not np.isfinite(atr_multiplier) or atr_multiplier <= 0:
        raise ValueError("atr_multiplier must be greater than zero")

    if not np.isfinite(min_body_pips) or min_body_pips < 0:
        raise ValueError("min_body_pips must be non-negative")

    if not np.isfinite(point_value) or point_value <= 0:
        raise ValueError("point_value must be greater than zero")

    n = len(open_array)
    if n == 0:
        return []

    atr_series = _compute_atr_series(
        high_array, low_array, close_array, atr_period
    )

    min_body_price = float(min_body_pips) * float(point_value)
    atr_multiplier = float(atr_multiplier)

    displacements: List[Dict[str, Any]] = []

    start = max(0, n - lookback)
    for i in range(start, n):
        atr = float(atr_series[i])
        if not np.isfinite(atr) or atr <= 0.0:
            continue

        body_size = abs(float(close_array[i] - open_array[i]))
        if body_size <= 0.0:
            continue

        if body_size < atr_multiplier * atr:
            continue

        if body_size < min_body_price:
            continue

        direction = (
            "bullish"
            if close_array[i] > open_array[i]
            else "bearish"
        )

        # Legacy fields preserved for backwards compat with existing
        # logging and any downstream that reads them.
        baseline_start = max(0, i - 10)
        previous_bodies = np.abs(
            close_array[baseline_start:i]
            - open_array[baseline_start:i]
        )
        avg_body = (
            float(np.mean(previous_bodies))
            if len(previous_bodies) > 0
            else 0.0
        )

        displacements.append({
            "index": i,
            "direction": direction,
            "body_size": body_size,
            "atr": atr,
            "atr_ratio": body_size / atr,
            "avg_body": avg_body,
            "ratio": (
                body_size / avg_body if avg_body > 0.0 else 0.0
            ),
        })

    return displacements