
"""Order Block analysis module.

Detects bullish and bearish order blocks, tracks mitigation, and provides
price-zone queries for the SMC strategy engine.
"""

from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import numpy as np

from src.indicators import (
    detect_order_blocks,
    check_ob_mitigation,
    OrderBlock,
)
from src.market_data import market_data
from src.mt5_connector import mt5_connector, MT5Connector
from src.config_loader import config
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OBState:
    """Current order block state for a timeframe."""

    timeframe: str
    timestamp: datetime
    order_blocks: List[OrderBlock]
    active_obs: List[OrderBlock] = field(default_factory=list)
    tested_obs: List[OrderBlock] = field(default_factory=list)


class OrderBlockAnalyzer:
    """Analyzes order blocks across multiple timeframes."""

    def __init__(self):
        self.timeframes = ["h4", "h1", "m15", "m5"]
        self.ob_cache: Dict[str, OBState] = {}

        self.lookback = self._read_positive_int(
            "ob_lookback", default=20
        )
        self.displacement_threshold = self._read_positive_float(
            "displacement_threshold", default=1.5
        )

    @staticmethod
    def _read_positive_int(key: str, default: int) -> int:
        """M-11: read a positive int from config.smc."""
        try:
            value = getattr(config.smc, key, default)
        except Exception:
            return default
        if isinstance(value, bool):
            return default
        try:
            value = int(value)
        except (TypeError, ValueError):
            return default
        if value < 1:
            return default
        return value

    @staticmethod
    def _read_positive_float(key: str, default: float) -> float:
        """M-11: read a strictly positive float from config.smc."""
        try:
            value = getattr(config.smc, key, default)
        except Exception:
            return default
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(value) or value <= 0:
            return default
        return value

    @property
    def point_value(self) -> float:
        """Broker-aware pip size (price per pip). See module docstring.

        Gold-units fix: for a 2-digit gold symbol, 1 pip is 10 × point
        (0.10 on XM GOLD). The previous hardcoded 0.01 was the *point*
        value, which made every 'pips' threshold in this file 10× too
        tight.
        """
        try:
            info = mt5_connector.get_symbol_info()
        except Exception:
            info = None

        if info is not None:
            size = MT5Connector.pip_size_for(info)
            if isinstance(size, (int, float)) and size > 0:
                return float(size)

        # Fail-safe fallback for the (rare) case where the symbol info
        # is unavailable. Matches the previous default so callers do not
        # crash; live code never reaches here when MT5 is connected.
        logger.warning(
            "Order-block analyzer: symbol info unavailable; "
            "falling back to 0.01 price-per-pip."
        )
        return 0.01

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_all_timeframes(self) -> Dict[str, OBState]:
        """Analyze order blocks for every supported timeframe."""

        results: Dict[str, OBState] = {}

        for timeframe in self.timeframes:
            state = self.analyze_timeframe(timeframe)

            if state is not None:
                results[timeframe] = state

        return results

    def analyze_timeframe(
        self,
        timeframe: str
    ) -> Optional[OBState]:
        """Detect order blocks and update their mitigation status."""

        timeframe = timeframe.lower()

        if timeframe not in self.timeframes:
            logger.warning(
                "Unsupported order-block timeframe: %s",
                timeframe
            )
            return None

        try:
            # H-12: OB detection looks ahead i+1..i+5 closes to compute
            # displacement strength. If the forming candle is in the
            # array, it enters that lookahead window and the OB
            # repaints. Use closed candles only.
            candles = market_data.get_closed_candles(
                timeframe,
                self.lookback + 20
            )

            if candles is None or len(candles) < self.lookback + 10:
                logger.debug(
                    "Insufficient candles for order-block analysis: %s",
                    timeframe
                )
                return None

            opens = np.asarray(
                [float(candle.open) for candle in candles],
                dtype=float
            )

            highs = np.asarray(
                [float(candle.high) for candle in candles],
                dtype=float
            )

            lows = np.asarray(
                [float(candle.low) for candle in candles],
                dtype=float
            )

            closes = np.asarray(
                [float(candle.close) for candle in candles],
                dtype=float
            )

            arrays = (opens, highs, lows, closes)

            if any(not np.all(np.isfinite(values)) for values in arrays):
                logger.warning(
                    "Invalid candle values in %s order-block data.",
                    timeframe
                )
                return None

            if (
                np.any(opens <= 0)
                or np.any(highs <= 0)
                or np.any(lows <= 0)
                or np.any(closes <= 0)
                or np.any(highs < lows)
                or np.any(highs < opens)
                or np.any(highs < closes)
                or np.any(lows > opens)
                or np.any(lows > closes)
            ):
                logger.warning(
                    "Inconsistent OHLC values in %s order-block data.",
                    timeframe
                )
                return None

            order_blocks = detect_order_blocks(
                opens,
                highs,
                lows,
                closes,
                lookback=self.lookback,
                displacement_threshold=self.displacement_threshold
            )

            current_index = len(candles) - 1

            order_blocks = check_ob_mitigation(
                order_blocks,
                highs,
                lows,
                current_index
            )

            active = [
                ob for ob in order_blocks
                if not ob.tested
            ]

            tested = [
                ob for ob in order_blocks
                if ob.tested
            ]

            state = OBState(
                timeframe=timeframe,
                timestamp=candles[current_index].timestamp,
                order_blocks=order_blocks,
                active_obs=active,
                tested_obs=tested
            )

            self.ob_cache[timeframe] = state

            logger.debug(
                "Order-block analysis completed: timeframe=%s "
                "total=%d active=%d tested=%d",
                timeframe,
                len(order_blocks),
                len(active),
                len(tested)
            )

            return state

        except Exception:
            logger.exception(
                "Order-block analysis failed for timeframe %s.",
                timeframe
            )
            return None

    # ------------------------------------------------------------------
    # Cached state
    # ------------------------------------------------------------------

    def get_obs(self, timeframe: str) -> Optional[OBState]:
        """Return cached order-block state."""

        return self.ob_cache.get(timeframe.lower())

    def get_active_bullish_obs(
        self,
        timeframe: str
    ) -> List[OrderBlock]:
        """Return active bullish order blocks."""

        state = self.get_obs(timeframe)

        if state is None:
            return []

        return [
            ob for ob in state.active_obs
            if str(ob.type).lower() == "bullish"
        ]

    def get_active_bearish_obs(
        self,
        timeframe: str
    ) -> List[OrderBlock]:
        """Return active bearish order blocks."""

        state = self.get_obs(timeframe)

        if state is None:
            return []

        return [
            ob for ob in state.active_obs
            if str(ob.type).lower() == "bearish"
        ]

    # ------------------------------------------------------------------
    # Nearest order-block zones
    # ------------------------------------------------------------------

    @staticmethod
    def _distance_to_zone(
        price: float,
        bottom: float,
        top: float
    ) -> float:
        """Calculate distance from price to an order-block zone."""

        if bottom <= price <= top:
            return 0.0

        if price < bottom:
            return bottom - price

        return price - top

    def get_nearest_bullish_ob_below(
        self,
        timeframe: str,
        price: float
    ) -> Optional[OrderBlock]:
        """
        Return the nearest active bullish OB below or containing price.

        An OB is considered relevant if its lower boundary is below the
        current price. A zone containing price is also eligible.
        """

        if not np.isfinite(price) or price <= 0:
            return None

        obs = self.get_active_bullish_obs(timeframe)

        candidates = [
            ob for ob in obs
            if (
                float(ob.bottom) > 0
                and float(ob.top) >= float(ob.bottom)
                and float(ob.bottom) <= price
            )
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda ob: self._distance_to_zone(
                price,
                float(ob.bottom),
                float(ob.top)
            )
        )

    def get_nearest_bearish_ob_above(
        self,
        timeframe: str,
        price: float
    ) -> Optional[OrderBlock]:
        """
        Return the nearest active bearish OB above or containing price.

        An OB is considered relevant if its upper boundary is above the
        current price. A zone containing price is also eligible.
        """

        if not np.isfinite(price) or price <= 0:
            return None

        obs = self.get_active_bearish_obs(timeframe)

        candidates = [
            ob for ob in obs
            if (
                float(ob.bottom) > 0
                and float(ob.top) >= float(ob.bottom)
                and float(ob.top) >= price
            )
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda ob: self._distance_to_zone(
                price,
                float(ob.bottom),
                float(ob.top)
            )
        )

    # ------------------------------------------------------------------
    # Price interaction with order-block zones
    # ------------------------------------------------------------------

    def is_price_at_bullish_ob(
        self,
        timeframe: str,
        price: float,
        tolerance_pips: float = 5
    ) -> bool:
        """Check whether price is inside or sufficiently close to bullish OB."""

        if (
            not np.isfinite(price)
            or price <= 0
            or tolerance_pips < 0
        ):
            return False

        ob = self.get_nearest_bullish_ob_below(
            timeframe,
            price
        )

        if ob is None:
            return False

        tolerance = float(tolerance_pips) * self.point_value

        distance = self._distance_to_zone(
            price,
            float(ob.bottom),
            float(ob.top)
        )

        return distance <= tolerance

    def is_price_at_bearish_ob(
        self,
        timeframe: str,
        price: float,
        tolerance_pips: float = 5
    ) -> bool:
        """Check whether price is inside or sufficiently close to bearish OB."""

        if (
            not np.isfinite(price)
            or price <= 0
            or tolerance_pips < 0
        ):
            return False

        ob = self.get_nearest_bearish_ob_above(
            timeframe,
            price
        )

        if ob is None:
            return False

        tolerance = float(tolerance_pips) * self.point_value

        distance = self._distance_to_zone(
            price,
            float(ob.bottom),
            float(ob.top)
        )

        return distance <= tolerance

    # ------------------------------------------------------------------
    # Order-block details
    # ------------------------------------------------------------------

    def get_ob_details(
        self,
        timeframe: str
    ) -> List[Dict]:
        """Return order-block details for diagnostics and strategy analysis."""

        state = self.get_obs(timeframe)

        if state is None:
            return []

        details = []

        for ob in state.order_blocks:
            details.append({
                "timeframe": timeframe.lower(),
                "index": ob.index,
                "type": ob.type,
                "top": float(ob.top),
                "bottom": float(ob.bottom),
                "strength": ob.strength,
                "tested": ob.tested,
                "mitigation_index": ob.mitigation_index,
                "timestamp": state.timestamp
            })

        return details


# Global order-block analyzer instance
order_blocks = OrderBlockAnalyzer()
