
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
        len(close_array)
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

        # Require the center to be the unique highest point in the window.
        if center == np.max(window) and np.count_nonzero(window == center) == 1:
            swings.append(
                SwingPoint(
                    index=i,
                    price=float(center),
                    type="high"
                )
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

        # Require the center to be the unique lowest point in the window.
        if center == np.min(window) and np.count_nonzero(window == center) == 1:
            swings.append(
                SwingPoint(
                    index=i,
                    price=float(center),
                    type="low"
                )
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
        find_swing_lows(lows, lookback)
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
        high_array,
        low_array,
        swing_lookback
    )

    start_index = max(0, len(close_array) - lookback)

    recent_highs = [
        swing for swing in all_highs
        if swing.index >= start_index
    ]
    recent_lows = [
        swing for swing in all_lows
        if swing.index >= start_index
    ]

    # Swing lists are chronological because the detection functions scan
    # from the oldest candle to the newest candle.
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
            structure_levels={}
        )

    higher_highs = 0
    lower_highs = 0
    higher_lows = 0
    lower_lows = 0

    for previous, current in zip(recent_highs, recent_highs[1:]):
        if current.price > previous.price:
            higher_highs += 1
        elif current.price < previous.price:
            lower_highs += 1

    for previous, current in zip(recent_lows, recent_lows[1:]):
        if current.price > previous.price:
            higher_lows += 1
        elif current.price < previous.price:
            lower_lows += 1

    bullish_structure = (
        higher_highs > lower_highs
        and higher_lows > lower_lows
    )
    bearish_structure = (
        lower_highs > higher_highs
        and lower_lows > higher_lows
    )

    if bullish_structure:
        trend = "bullish"
    elif bearish_structure:
        trend = "bearish"
    else:
        trend = "ranging"

    bos_detected = False
    choch_detected = False
    last_bos_index: Optional[int] = None
    last_choch_index: Optional[int] = None
    last_bos_direction: Optional[str] = None
    last_choch_direction: Optional[str] = None

    # Use the latest confirmed swing, not the highest/lowest price from
    # the entire lookback window.
    latest_swing_high = recent_highs[-1]
    latest_swing_low = recent_lows[-1]

    # Only inspect candles after the relevant swing was formed. This avoids
    # treating a close that occurred before the swing as a break of it.
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
        structure_levels=structure_levels
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
        # Bullish FVG: current candle's low is above the high two candles ago.
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
                        bottom=float(bullish_bottom)
                    )
                )

        # Bearish FVG: current candle's high is below the low two candles ago.
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
                        bottom=float(bearish_bottom)
                    )
                )

    return fvgs


def check_fvg_mitigation(
    fvgs: List[FVG],
    highs: np.ndarray,
    lows: np.ndarray,
    current_index: int
) -> List[FVG]:
    """Update FVGs when a later candle completely fills the gap.

    A simple touch or partial retracement does not mark an FVG as fully
    mitigated. The function scans all candles after formation up to
    current_index so an earlier fill is not missed.
    """
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
    """Detect candidate order blocks before strong directional moves.

    The order block zone uses the full high-low range of the opposing
    candle. Strength is measured against the average body size of the
    five preceding candles.
    """
    open_array, high_array, low_array, close_array = _validate_ohlc(
        opens, highs, lows, closes
    )

    lookback = _validate_lookback(lookback)

    if not np.isfinite(displacement_threshold) or displacement_threshold <= 0:
        raise ValueError("displacement_threshold must be greater than zero")

    obs: List[OrderBlock] = []
    n = len(open_array)

    # Need five prior candles for the baseline and at least one later
    # candle to confirm displacement.
    start_index = max(lookback, 5)

    for i in range(start_index, n - 1):
        previous_bodies = np.abs(
            close_array[i - 5:i] - open_array[i - 5:i]
        )
        avg_size = float(np.mean(previous_bodies))

        if avg_size <= 0:
            continue

        future_end = min(i + 5, n)
        future_closes = close_array[i + 1:future_end]

        if len(future_closes) == 0:
            continue

        # Bullish OB: bearish candle followed by meaningful upward displacement.
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
                        strength=float(strength)
                    )
                )

        # Bearish OB: bullish candle followed by meaningful downward displacement.
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
                        strength=float(strength)
                    )
                )

    return obs


def check_ob_mitigation(
    obs: List[OrderBlock],
    highs: np.ndarray,
    lows: np.ndarray,
    current_index: int
) -> List[OrderBlock]:
    """Mark order blocks fully mitigated by a later candle.

    Entering an order block is not considered full mitigation. A bullish
    block is fully mitigated at/below its bottom; a bearish block is fully
    mitigated at/above its top.
    """
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
    strength: int  # number of touches
    indices: List[int]
    timestamp: Any = None
    swept: bool = False
    sweep_index: Optional[int] = None


def _cluster_price_levels(
    values: np.ndarray,
    start_index: int,
    tolerance: float
) -> List[Tuple[float, List[int]]]:
    """Cluster nearby prices and return average price plus candle indices."""
    indexed_values = [
        (float(values[i]), i)
        for i in range(start_index, len(values))
    ]
    indexed_values.sort(key=lambda item: item[0])

    clusters: List[List[Tuple[float, int]]] = []

    for price, index in indexed_values:
        if not clusters:
            clusters.append([(price, index)])
            continue

        current_cluster = clusters[-1]
        cluster_mean = float(
            np.mean([item[0] for item in current_cluster])
        )

        if abs(price - cluster_mean) <= tolerance:
            current_cluster.append((price, index))
        else:
            clusters.append([(price, index)])

    result: List[Tuple[float, List[int]]] = []

    for cluster in clusters:
        average_price = float(
            np.mean([item[0] for item in cluster])
        )
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
    """Detect buy-side and sell-side liquidity from equal highs/lows.

    Buy-side liquidity:
        Equal highs where buy stops are likely resting.

    Sell-side liquidity:
        Equal lows where sell stops are likely resting.

    Returns:
        (buy_side_levels, sell_side_levels)
    """
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

    # Equal highs are BUY-SIDE liquidity.
    high_clusters = _cluster_price_levels(
        high_array,
        start_index,
        tolerance
    )

    # Equal lows are SELL-SIDE liquidity.
    low_clusters = _cluster_price_levels(
        low_array,
        start_index,
        tolerance
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
                    indices=indices
                )
            )

    for price, indices in low_clusters:
        if len(indices) >= 2:
            sell_side.append(
                LiquidityLevel(
                    price=price,
                    type="sell_side",
                    strength=len(indices),
                    indices=indices
                )
            )

    return buy_side, sell_side


def check_liquidity_sweep(
    buy_side: List[LiquidityLevel],
    sell_side: List[LiquidityLevel],
    highs: np.ndarray,
    lows: np.ndarray,
    current_index: int
) -> Tuple[List[LiquidityLevel], List[LiquidityLevel]]:
    """Mark liquidity levels swept by the specified candle.

    A buy-side sweep occurs when price trades at/above equal highs.
    A sell-side sweep occurs when price trades at/below equal lows.

    This detects a wick through a level. It does not confirm a reversal:
    rejection/close confirmation must be handled by the strategy layer.
    """
    high_array = _as_float_array(highs, "highs")
    low_array = _as_float_array(lows, "lows")

    if len(high_array) != len(low_array):
        raise ValueError("highs and lows must have equal lengths")

    if len(high_array) == 0:
        return buy_side, sell_side

    current_index = _validate_index(current_index, len(high_array))

    current_high = float(high_array[current_index])
    current_low = float(low_array[current_index])

    for level in buy_side:
        if level.type != "buy_side" or level.swept:
            continue

        if current_high >= level.price:
            level.swept = True
            level.sweep_index = current_index

    for level in sell_side:
        if level.type != "sell_side" or level.swept:
            continue

        if current_low <= level.price:
            level.swept = True
            level.sweep_index = current_index

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

    if high < low:
        raise ValueError("high must be greater than or equal to low")

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
        "distance_from_equilibrium": float(distance_from_eq)
    }


# ---------------------------------------------------------------------------
# Displacement
# ---------------------------------------------------------------------------

def detect_displacement(
    opens: np.ndarray,
    closes: np.ndarray,
    lookback: int = 10,
    threshold: float = 1.5
) -> List[Dict[str, Any]]:
    """Detect unusually large candle bodies relative to prior candle bodies."""
    open_array = _as_float_array(opens, "opens")
    close_array = _as_float_array(closes, "closes")

    if len(open_array) != len(close_array):
        raise ValueError("opens and closes must have equal lengths")

    lookback = _validate_lookback(lookback)

    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be greater than zero")

    displacements: List[Dict[str, Any]] = []

    for i in range(max(0, len(open_array) - lookback), len(open_array)):
        body_size = abs(float(close_array[i] - open_array[i]))

        # Compare the current candle against preceding candles only.
        # This avoids including the candle being evaluated in its baseline.
        baseline_start = max(0, i - 10)
        previous_bodies = np.abs(
            close_array[baseline_start:i] - open_array[baseline_start:i]
        )

        if len(previous_bodies) == 0:
            continue

        avg_body = float(np.mean(previous_bodies))

        if avg_body <= 0:
            continue

        ratio = body_size / avg_body

        if ratio > threshold:
            direction = (
                "bullish"
                if close_array[i] > open_array[i]
                else "bearish"
            )

            # Ignore doji candles; they have no directional displacement.
            if close_array[i] == open_array[i]:
                continue

            displacements.append({
                "index": i,
                "direction": direction,
                "body_size": body_size,
                "avg_body": avg_body,
                "ratio": float(ratio)
            })

    return displacements
