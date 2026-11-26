"""Liquidity analysis module.

Detects buy-side and sell-side liquidity, identifies candidate sweeps,
and exposes liquidity targets for the SMC strategy engine.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dataclasses import dataclass, field

import numpy as np

from src.indicators import (
    detect_liquidity_levels,
    check_liquidity_sweep,
    LiquidityLevel,
)
from src.market_data import market_data
from src.mt5_connector import mt5_connector, MT5Connector
from src.config_loader import config
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LiquidityState:
    """Current liquidity state for a timeframe."""

    timeframe: str
    timestamp: datetime
    buy_side: List[LiquidityLevel]
    sell_side: List[LiquidityLevel]
    recent_sweeps: List[Dict] = field(default_factory=list)
    active_levels: Dict[str, List[LiquidityLevel]] = field(
        default_factory=dict
    )


class LiquidityAnalyzer:
    """Analyzes liquidity levels and sweeps across timeframes."""

    def __init__(self):
        self.timeframes = ["h4", "h1", "m15", "m5"]
        self.liquidity_cache: Dict[str, LiquidityState] = {}

        self.lookback = self._read_positive_int(
            "liquidity_lookback", default=100
        )
        self.tolerance_pips = self._read_non_negative_float(
            "equal_high_low_tolerance", default=2.0
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
    def _read_non_negative_float(key: str, default: float) -> float:
        """M-11: read a non-negative float from config.smc."""
        try:
            value = getattr(config.smc, key, default)
        except Exception:
            return default
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(value) or value < 0:
            return default
        return value

    @property
    def point_value(self) -> float:
        """Broker-aware pip size (price per pip). See module docstring.

        Gold-units fix: for 2-digit gold, 1 pip = 10 × point (0.10).
        The previous hardcoded 0.01 made the equal-highs/equal-lows
        clustering tolerance 10× too tight.
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
            "Liquidity analyzer: symbol info unavailable; "
            "falling back to 0.01 price-per-pip."
        )
        return 0.01

    def _get_sweep_buffers_pips(self) -> Tuple[float, float]:
        """Read M-01 sweep classification buffers from config.

        Returns (pierce_buffer_pips, reclaim_buffer_pips). Missing,
        invalid, or negative values fall back to 0.0 so the sweep
        logic remains active with zero buffers (strict pierce, any
        reclaim), preserving the safe default.
        """
        pierce = 0.0
        reclaim = 0.0

        try:
            section = getattr(config, "liquidity", None)
            if section is not None:
                pierce = self._coerce_non_negative_float(
                    getattr(
                        section,
                        "sweep_pierce_buffer_pips",
                        0.0,
                    )
                )
                reclaim = self._coerce_non_negative_float(
                    getattr(
                        section,
                        "sweep_reclaim_buffer_pips",
                        0.0,
                    )
                )
        except Exception:
            logger.exception(
                "Failed to read liquidity sweep buffer config; "
                "using 0.0 defaults."
            )
            pierce = 0.0
            reclaim = 0.0

        return pierce, reclaim

    @staticmethod
    def _coerce_non_negative_float(value: Any) -> float:
        """Return value as float if finite and >= 0, else 0.0."""
        try:
            result = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not np.isfinite(result) or result < 0:
            return 0.0

        return result

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_all_timeframes(self) -> Dict[str, LiquidityState]:
        """Analyze liquidity across all supported timeframes."""

        results: Dict[str, LiquidityState] = {}

        for timeframe in self.timeframes:
            state = self.analyze_timeframe(timeframe)

            if state is not None:
                results[timeframe] = state

        return results

    def analyze_timeframe(
        self,
        timeframe: str
    ) -> Optional[LiquidityState]:
        """Analyze liquidity pools and the latest candle for sweeps."""

        timeframe = timeframe.lower()

        if timeframe not in self.timeframes:
            logger.warning(
                "Unsupported liquidity timeframe: %s",
                timeframe
            )
            return None

        try:
            candles = market_data.get_candles(
                timeframe,
                self.lookback + 20
            )

            if candles is None or len(candles) < self.lookback + 10:
                self.liquidity_cache.pop(timeframe, None)

                logger.debug(
                    "Insufficient candles for liquidity analysis: %s",
                    timeframe
                )
                return None

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

            if (
                not np.all(np.isfinite(highs))
                or not np.all(np.isfinite(lows))
                or not np.all(np.isfinite(closes))
                or np.any(highs <= 0)
                or np.any(lows <= 0)
                or np.any(closes <= 0)
                or np.any(highs < lows)
            ):
                self.liquidity_cache.pop(timeframe, None)

                logger.warning(
                    "Invalid candle prices in %s liquidity data.",
                    timeframe
                )
                return None

            # Analyze only confirmed (closed) candles. Dropping the last
            # bar prevents repainting: a sweep that only exists on the
            # half-closed candle cannot generate a signal.
            candles = candles[:-1]
            highs = highs[:-1]
            lows = lows[:-1]
            closes = closes[:-1]

            if len(candles) < 10:
                self.liquidity_cache.pop(timeframe, None)
                return None

            current_index = len(candles) - 1

            # Do not use the candle being tested to construct its own
            # liquidity pools. Otherwise, its high/low can contribute
            # to a level that it immediately appears to sweep.
            historical_highs = highs[:current_index]
            historical_lows = lows[:current_index]

            if len(historical_highs) < 10:
                self.liquidity_cache.pop(timeframe, None)
                return None

            pip_size = self.point_value

            buy_side, sell_side = detect_liquidity_levels(
                historical_highs,
                historical_lows,
                lookback=self.lookback,
                tolerance_pips=self.tolerance_pips,
                point_value=pip_size
            )

            # M-01: classify the latest closed candle against the
            # level pools with configurable pierce/reclaim buffers.
            pierce_pips, reclaim_pips = self._get_sweep_buffers_pips()
            pierce_buffer = pierce_pips * pip_size
            reclaim_buffer = reclaim_pips * pip_size

            buy_side, sell_side = check_liquidity_sweep(
                buy_side,
                sell_side,
                highs,
                lows,
                closes,
                current_index,
                pierce_buffer=pierce_buffer,
                reclaim_buffer=reclaim_buffer,
            )

            current_candle = candles[current_index]

            recent_sweeps = self._collect_recent_sweeps(
                buy_side,
                sell_side,
                current_candle,
                current_index
            )

            active_buy = [
                level for level in buy_side
                if not level.swept
            ]

            active_sell = [
                level for level in sell_side
                if not level.swept
            ]

            state = LiquidityState(
                timeframe=timeframe,
                timestamp=current_candle.timestamp,
                buy_side=buy_side,
                sell_side=sell_side,
                recent_sweeps=recent_sweeps,
                active_levels={
                    "buy_side": active_buy,
                    "sell_side": active_sell
                }
            )

            self.liquidity_cache[timeframe] = state

            for sweep in recent_sweeps:
                logger.debug(
                    "Liquidity sweep: timeframe=%s type=%s "
                    "price=%.3f strength=%s",
                    timeframe,
                    sweep["type"],
                    sweep["price"],
                    sweep["strength"]
                )

            return state

        except Exception:
            self.liquidity_cache.pop(timeframe, None)

            logger.exception(
                "Liquidity analysis failed for timeframe %s.",
                timeframe
            )
            return None       


        except Exception:
            self.liquidity_cache.pop(timeframe, None)

            logger.exception(
                "Liquidity analysis failed for timeframe %s.",
                timeframe
            )
            return None

    def _collect_recent_sweeps(
        self,
        buy_side: List[LiquidityLevel],
        sell_side: List[LiquidityLevel],
        current_candle,
        current_index: int
    ) -> List[Dict]:
        """Collect genuine sweeps on the latest analyzed candle.

        M-01: a "breakout" (pierced the level without reclaiming it)
        is intentionally NOT included here. Breakouts are consumed
        (removed from active_levels) but MUST NOT trigger the
        was_*_swept helpers or _find_sweep_price.
        """

        recent_sweeps = []

        for level in buy_side:
            if (
                level.swept
                and level.sweep_kind == "sweep"
                and level.sweep_index == current_index
            ):
                recent_sweeps.append({
                    "type": "buy_side_sweep",
                    "price": float(level.price),
                    "strength": level.strength,
                    "timestamp": current_candle.timestamp
                })

        for level in sell_side:
            if (
                level.swept
                and level.sweep_kind == "sweep"
                and level.sweep_index == current_index
            ):
                recent_sweeps.append({
                    "type": "sell_side_sweep",
                    "price": float(level.price),
                    "strength": level.strength,
                    "timestamp": current_candle.timestamp
                })

        return recent_sweeps
    # ------------------------------------------------------------------
    # Cached state
    # ------------------------------------------------------------------

    def get_liquidity(
        self,
        timeframe: str
    ) -> Optional[LiquidityState]:
        """Return the most recently analyzed liquidity state."""

        return self.liquidity_cache.get(timeframe.lower())

    def get_buy_side_liquidity(
        self,
        timeframe: str
    ) -> List[LiquidityLevel]:
        """Return active buy-side liquidity above price."""

        state = self.get_liquidity(timeframe)

        if state is None:
            return []

        return state.active_levels.get("buy_side", [])

    def get_sell_side_liquidity(
        self,
        timeframe: str
    ) -> List[LiquidityLevel]:
        """Return active sell-side liquidity below price."""

        state = self.get_liquidity(timeframe)

        if state is None:
            return []

        return state.active_levels.get("sell_side", [])

    # ------------------------------------------------------------------
    # Sweep detection
    # ------------------------------------------------------------------

    def was_buy_side_swept(self, timeframe: str) -> bool:
        """Return whether buy-side liquidity was swept on the latest candle."""

        state = self.get_liquidity(timeframe)

        if state is None:
            return False

        return any(
            sweep["type"] == "buy_side_sweep"
            for sweep in state.recent_sweeps
        )

    def was_sell_side_swept(self, timeframe: str) -> bool:
        """Return whether sell-side liquidity was swept on the latest candle."""

        state = self.get_liquidity(timeframe)

        if state is None:
            return False

        return any(
            sweep["type"] == "sell_side_sweep"
            for sweep in state.recent_sweeps
        )

    def get_recent_sweep_details(
        self,
        timeframe: str
    ) -> List[Dict]:
        """Return details of sweeps detected on the latest candle."""

        state = self.get_liquidity(timeframe)

        if state is None:
            return []

        return list(state.recent_sweeps)

    # ------------------------------------------------------------------
    # Nearest liquidity
    # ------------------------------------------------------------------

    def get_nearest_buy_side_above(
        self,
        timeframe: str,
        price: float
    ) -> Optional[LiquidityLevel]:
        """Find the nearest active buy-side liquidity above price."""

        try:
            price = float(price)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(price) or price <= 0:
            return None

        above = [
            level
            for level in self.get_buy_side_liquidity(timeframe)
            if float(level.price) > price
        ]

        return min(
            above,
            key=lambda level: float(level.price),
            default=None
        )

    def get_nearest_sell_side_below(
        self,
        timeframe: str,
        price: float
    ) -> Optional[LiquidityLevel]:
        """Find the nearest active sell-side liquidity below price."""

        try:
            price = float(price)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(price) or price <= 0:
            return None

        below = [
            level
            for level in self.get_sell_side_liquidity(timeframe)
            if float(level.price) < price
        ]

        return max(
            below,
            key=lambda level: float(level.price),
            default=None
        )

    # ------------------------------------------------------------------
    # Directional targets
    # ------------------------------------------------------------------

    def get_liquidity_for_direction(
        self,
        timeframe: str,
        direction: str
    ) -> List[LiquidityLevel]:
        """Return liquidity pools in the intended trade direction.

        BUY  -> Buy-side liquidity (equal highs).
        SELL -> Sell-side liquidity (equal lows).
        """

        direction = direction.upper()

        if direction == "BUY":
            return self.get_buy_side_liquidity(timeframe)

        if direction == "SELL":
            return self.get_sell_side_liquidity(timeframe)

        logger.warning(
            "Unsupported liquidity target direction: %s",
            direction
        )

        return []


# Shared liquidity analyzer instance.
liquidity = LiquidityAnalyzer()