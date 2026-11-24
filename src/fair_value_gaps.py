"""Fair Value Gap analysis module.

Detects bullish and bearish FVGs, tracks mitigation, and provides
entry-zone queries for the SMC strategy engine.
"""

from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import numpy as np

from src.indicators import (
    detect_fvgs,
    check_fvg_mitigation,
    FVG,
)
from src.market_data import market_data
from src.mt5_connector import mt5_connector, MT5Connector
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FVGState:
    """Current FVG state for a timeframe."""

    timeframe: str
    timestamp: datetime
    fvgs: List[FVG]
    active_fvgs: List[FVG] = field(default_factory=list)
    mitigated_fvgs: List[FVG] = field(default_factory=list)


class FVGAnalyzer:
    """Analyzes Fair Value Gaps across multiple timeframes."""

    def __init__(self):
        self.timeframes = ["h4", "h1", "m15", "m5"]
        self.fvg_cache: Dict[str, FVGState] = {}

        self.min_size_pips = 5.0

    @property
    def point_value(self) -> float:
        """Broker-aware pip size (price per pip). See module docstring.

        Gold-units fix: for 2-digit gold, 1 pip = 10 × point (0.10).
        The previous hardcoded 0.01 made the FVG minimum-width filter
        and the size-pips reporting 10× too permissive / wrong.
        """
        try:
            info = mt5_connector.get_symbol_info()
        except Exception:
            info = None

        if info is not None:
            size = MT5Connector.pip_size_for(info)
            if isinstance(size, (int, float)) and size > 0:
                return float(size)

        logger.warning(
            "FVG analyzer: symbol info unavailable; "
            "falling back to 0.01 price-per-pip."
        )
        return 0.01

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_all_timeframes(self) -> Dict[str, FVGState]:
        """Analyze FVGs for every supported timeframe."""

        results: Dict[str, FVGState] = {}

        for timeframe in self.timeframes:
            state = self.analyze_timeframe(timeframe)

            if state is not None:
                results[timeframe] = state

        return results

    def analyze_timeframe(
        self,
        timeframe: str
    ) -> Optional[FVGState]:
        """Detect FVGs and update their mitigation status."""

        timeframe = timeframe.lower()

        if timeframe not in self.timeframes:
            logger.warning(
                "Unsupported FVG timeframe: %s",
                timeframe
            )
            return None

        try:
            candles = market_data.get_candles(timeframe, 100)

            if candles is None or len(candles) < 20:
                logger.debug(
                    "Insufficient candles for FVG analysis: %s",
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
                    "Invalid candle data in %s FVG analysis.",
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
                    "Inconsistent OHLC data in %s FVG analysis.",
                    timeframe
                )
                return None

            # Analyze only confirmed candles — a gap that only exists on
            # the forming candle must not produce a signal.
            candles = candles[:-1]
            opens = opens[:-1]
            highs = highs[:-1]
            lows = lows[:-1]
            closes = closes[:-1]

            if len(candles) < 10:
                logger.debug(
                    "Insufficient confirmed candles for FVG: %s",
                    timeframe
                )
                return None

            fvgs = detect_fvgs(
                opens,
                highs,
                lows,
                closes,
                min_size_pips=self.min_size_pips,
                point_value=self.point_value
            )

            current_index = len(candles) - 1

            fvgs = check_fvg_mitigation(
                fvgs,
                highs,
                lows,
                current_index
            )

            active = [
                fvg for fvg in fvgs
                if not fvg.mitigated
            ]

            mitigated = [
                fvg for fvg in fvgs
                if fvg.mitigated
            ]

            state = FVGState(
                timeframe=timeframe,
                timestamp=candles[current_index].timestamp,
                fvgs=fvgs,
                active_fvgs=active,
                mitigated_fvgs=mitigated
            )

            self.fvg_cache[timeframe] = state

            logger.debug(
                "FVG analysis completed: timeframe=%s "
                "total=%d active=%d mitigated=%d",
                timeframe,
                len(fvgs),
                len(active),
                len(mitigated)
            )

            return state

        except Exception:
            logger.exception(
                "FVG analysis failed for timeframe %s.",
                timeframe
            )
            return None

    # ------------------------------------------------------------------
    # Cached state
    # ------------------------------------------------------------------

    def get_fvgs(self, timeframe: str) -> Optional[FVGState]:
        """Return cached FVG state."""

        return self.fvg_cache.get(timeframe.lower())

    def get_active_bullish_fvgs(
        self,
        timeframe: str
    ) -> List[FVG]:
        """Return active bullish FVGs."""

        state = self.get_fvgs(timeframe)

        if state is None:
            return []

        return [
            fvg for fvg in state.active_fvgs
            if str(fvg.type).lower() == "bullish"
        ]

    def get_active_bearish_fvgs(
        self,
        timeframe: str
    ) -> List[FVG]:
        """Return active bearish FVGs."""

        state = self.get_fvgs(timeframe)

        if state is None:
            return []

        return [
            fvg for fvg in state.active_fvgs
            if str(fvg.type).lower() == "bearish"
        ]

    # ------------------------------------------------------------------
    # Nearest FVG selection
    # ------------------------------------------------------------------

    @staticmethod
    def _distance_to_zone(
        price: float,
        bottom: float,
        top: float
    ) -> float:
        """Calculate distance from price to an FVG zone."""

        if bottom <= price <= top:
            return 0.0

        if price < bottom:
            return bottom - price

        return price - top

    def get_bullish_fvg_below(
        self,
        timeframe: str,
        price: float
    ) -> Optional[FVG]:
        """Get nearest active bullish FVG below or containing price."""

        if not np.isfinite(price) or price <= 0:
            return None

        fvgs = self.get_active_bullish_fvgs(timeframe)

        candidates = [
            fvg for fvg in fvgs
            if (
                float(fvg.bottom) > 0
                and float(fvg.top) >= float(fvg.bottom)
                and float(fvg.bottom) <= price
            )
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda fvg: self._distance_to_zone(
                price,
                float(fvg.bottom),
                float(fvg.top)
            )
        )

    def get_bearish_fvg_above(
        self,
        timeframe: str,
        price: float
    ) -> Optional[FVG]:
        """Get nearest active bearish FVG above or containing price."""

        if not np.isfinite(price) or price <= 0:
            return None

        fvgs = self.get_active_bearish_fvgs(timeframe)

        candidates = [
            fvg for fvg in fvgs
            if (
                float(fvg.bottom) > 0
                and float(fvg.top) >= float(fvg.bottom)
                and float(fvg.top) >= price
            )
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda fvg: self._distance_to_zone(
                price,
                float(fvg.bottom),
                float(fvg.top)
            )
        )

    # ------------------------------------------------------------------
    # Price interaction with FVG zones
    # ------------------------------------------------------------------

    def is_price_in_bullish_fvg(
        self,
        timeframe: str,
        price: float
    ) -> bool:
        """Check whether price is inside an active bullish FVG."""

        if not np.isfinite(price) or price <= 0:
            return False

        return any(
            float(fvg.bottom) <= price <= float(fvg.top)
            for fvg in self.get_active_bullish_fvgs(timeframe)
            if (
                float(fvg.bottom) > 0
                and float(fvg.top) >= float(fvg.bottom)
            )
        )

    def is_price_in_bearish_fvg(
        self,
        timeframe: str,
        price: float
    ) -> bool:
        """Check whether price is inside an active bearish FVG."""

        if not np.isfinite(price) or price <= 0:
            return False

        return any(
            float(fvg.bottom) <= price <= float(fvg.top)
            for fvg in self.get_active_bearish_fvgs(timeframe)
            if (
                float(fvg.bottom) > 0
                and float(fvg.top) >= float(fvg.bottom)
            )
        )

    # ------------------------------------------------------------------
    # FVG details
    # ------------------------------------------------------------------

    def get_fvg_details(
        self,
        timeframe: str
    ) -> List[Dict]:
        """Return FVG details for strategy analysis and diagnostics."""

        state = self.get_fvgs(timeframe)

        if state is None:
            return []

        details = []

        for fvg in state.fvgs:
            bottom = float(fvg.bottom)
            top = float(fvg.top)

            details.append({
                "timeframe": timeframe.lower(),
                "index": fvg.index,
                "type": fvg.type,
                "top": top,
                "bottom": bottom,
                "size_pips": (
                    (top - bottom) / self.point_value
                ),
                "mitigated": fvg.mitigated,
                "mitigation_index": fvg.mitigation_index,
                "timestamp": state.timestamp
            })

        return details


# Global FVG analyzer instance
fair_value_gaps = FVGAnalyzer()