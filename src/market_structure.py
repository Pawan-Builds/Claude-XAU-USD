
"""Multi-timeframe market structure analysis."""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from src.indicators import (
    analyze_market_structure,
    MarketStructure,
    SwingPoint,
)
from src.market_data import market_data
from src.config_loader import config
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StructureState:
    """Market structure state for one timeframe."""

    timeframe: str
    timestamp: datetime
    structure: MarketStructure
    swing_highs: List[SwingPoint]
    swing_lows: List[SwingPoint]
    bias: str
    key_levels: Dict[str, float]


class MarketStructureAnalyzer:
    """Analyzes H4, H1, M15 and M5 market structure."""

    _SUPPORTED_TIMEFRAMES = frozenset({"h4", "h1", "m15", "m5"})

    def __init__(self):
        self.timeframes = ["h4", "h1", "m15", "m5"]
        self.structure_cache: Dict[str, StructureState] = {}
        # M-11: the small pivot-confirmation window (5) is an
        # internal constant of the structure analyser; it is not
        # intended to be operator-tunable.
        self.swing_lookback = 5
        self.structure_lookback = self._read_structure_lookback()
        self.structure_event_lookback = self._read_event_lookback()
        # M-06: configurable bias / structure timeframes.
        self._bias_tf = self._read_timeframe_key("bias_tf", "h4")
        self._structure_tf = self._read_timeframe_key(
            "structure_tf", "h1"
        )

    @staticmethod
    def _read_structure_lookback() -> int:
        """M-11: read the structure-analysis window from config.

        The config key `smc.swing_lookback` (default 50) carries the
        outer structure-analysis window used by
        `analyze_market_structure(lookback=...)`. The pre-M-11
        hardcoded value was also 50, so behaviour is unchanged when
        the config is present. Invalid or non-positive values fall
        back to 50.
        """
        try:
            value = getattr(config.smc, "swing_lookback", 50)
        except Exception:
            return 50

        if isinstance(value, bool):
            return 50

        try:
            value = int(value)
        except (TypeError, ValueError):
            return 50

        if value < 1:
            return 50

        return value

    def _read_timeframe_key(self, key: str, default: str) -> str:
        """M-06: read a timeframe config value safely.

        Missing, empty, non-string, or unsupported values fall back
        to `default`. Supported values are restricted to the set of
        timeframes this analyzer actually caches (h4/h1/m15/m5); any
        other value is rejected with a warning because the analyzer
        would otherwise return "neutral" forever.
        """
        try:
            value = getattr(config.timeframes, key, default)
        except Exception:
            return default

        if not isinstance(value, str):
            return default

        value = value.strip().lower()
        if value not in self._SUPPORTED_TIMEFRAMES:
            logger.warning(
                "Unsupported config.timeframes.%s=%r; falling back "
                "to %r.", key, value, default,
            )
            return default

        return value

    @staticmethod
    def _read_event_lookback() -> int:
        """M-03: read structure event lookback from config.smc.

        Missing, non-int, or out-of-range values fall back to 10
        (the pre-M-03 hardcoded value). Zero and negative values are
        rejected because they would produce an empty event window.
        """
        try:
            value = getattr(config.smc, "structure_event_lookback", 10)
        except Exception:
            return 10

        if isinstance(value, bool):
            return 10

        try:
            value = int(value)
        except (TypeError, ValueError):
            return 10

        if value < 1:
            return 10

        return value

    def analyze_all_timeframes(self) -> Dict[str, StructureState]:
        """Refresh market structure for all configured timeframes."""

        results: Dict[str, StructureState] = {}

        for timeframe in self.timeframes:
            try:
                state = self.analyze_timeframe(timeframe)
            except Exception:
                logger.exception(
                    "Market structure analysis failed for %s.",
                    timeframe,
                )
                continue

            if state is not None:
                results[timeframe] = state
                self.structure_cache[timeframe] = state

        return results

    def analyze_timeframe(
        self,
        timeframe: str,
    ) -> Optional[StructureState]:
        """Analyze candles and calculate market structure."""

        if timeframe not in self.timeframes:
            logger.warning("Unsupported structure timeframe: %s", timeframe)
            return None

        candles = market_data.get_closed_candles(
            timeframe,
            self.structure_lookback + 20,
        )

        minimum_candles = self.structure_lookback + 10

        if candles is None or len(candles) < minimum_candles:
            logger.debug(
                "Insufficient candles for %s structure analysis: %s/%s.",
                timeframe,
                len(candles) if candles else 0,
                minimum_candles,
            )
            return None

        highs = np.asarray([c.high for c in candles], dtype=float)
        lows = np.asarray([c.low for c in candles], dtype=float)
        closes = np.asarray([c.close for c in candles], dtype=float)

        if not (
            np.all(np.isfinite(highs))
            and np.all(np.isfinite(lows))
            and np.all(np.isfinite(closes))
        ):
            logger.warning(
                "Invalid OHLC data detected for %s. Structure not updated.",
                timeframe,
            )
            return None

        if np.any(highs <= 0) or np.any(lows <= 0) or np.any(closes <= 0):
            logger.warning(
                "Non-positive market prices detected for %s.",
                timeframe,
            )
            return None

        if np.any(highs < lows):
            logger.warning(
                "Invalid candle high/low relationship for %s.",
                timeframe,
            )
            return None

        structure = analyze_market_structure(
            highs,
            lows,
            closes,
            lookback=self.structure_lookback,
            swing_lookback=self.swing_lookback,
            event_lookback=self.structure_event_lookback,
        )

        if structure is None:
            logger.warning(
                "Structure calculation returned no result for %s.",
                timeframe,
            )
            return None

        trend = str(structure.trend).lower()

        if trend == "bullish":
            bias = "bullish"
        elif trend == "bearish":
            bias = "bearish"
        else:
            bias = "neutral"

        key_levels: Dict[str, float] = {}

        if isinstance(structure.structure_levels, dict):
            for key, value in structure.structure_levels.items():
                try:
                    price = float(value)
                except (TypeError, ValueError):
                    continue

                if np.isfinite(price) and price > 0:
                    key_levels[key] = price

        swing_highs = list(structure.swing_highs or [])
        swing_lows = list(structure.swing_lows or [])

        if swing_highs:
            latest_high = self._last_swing(swing_highs)
            if latest_high is not None:
                key_levels["latest_swing_high"] = float(latest_high.price)

        if swing_lows:
            latest_low = self._last_swing(swing_lows)
            if latest_low is not None:
                key_levels["latest_swing_low"] = float(latest_low.price)

        timestamp = getattr(candles[-1], "timestamp", None)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return StructureState(
            timeframe=timeframe,
            timestamp=timestamp,
            structure=structure,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            bias=bias,
            key_levels=key_levels,
        )

    @staticmethod
    def _last_swing(swings: List[SwingPoint]) -> Optional[SwingPoint]:
        """Return the most recently detected swing.

        The analyzer's swing lists are expected to be in candle order.
        If an index or timestamp attribute exists, use it to establish order.
        """

        if not swings:
            return None

        if all(hasattr(item, "index") for item in swings):
            return max(swings, key=lambda item: item.index)

        if all(hasattr(item, "timestamp") for item in swings):
            return max(swings, key=lambda item: item.timestamp)

        return swings[-1]

    def get_structure(
        self,
        timeframe: str,
    ) -> Optional[StructureState]:
        """Return the most recently cached structure."""

        return self.structure_cache.get(timeframe)

    def get_higher_tf_bias(self) -> str:
        """Return bias for the configured bias timeframe.

        M-06: reads `config.timeframes.bias_tf` (default "h4").
        """
        state = self.get_structure(self._bias_tf)
        return state.bias if state else "neutral"

    def get_structure_tf_bias(self) -> str:
        """Return bias for the configured structure timeframe.

        M-06: reads `config.timeframes.structure_tf` (default "h1").
        """
        state = self.get_structure(self._structure_tf)
        return state.bias if state else "neutral"

    def get_bias_timeframe(self) -> str:
        """Return the configured bias timeframe name (M-06)."""
        return self._bias_tf

    def get_structure_timeframe(self) -> str:
        """Return the configured structure timeframe name (M-06)."""
        return self._structure_tf

    def is_bos_bullish(self, timeframe: str) -> bool:
        state = self.get_structure(timeframe)

        return bool(
            state
            and state.structure.bos_detected
            and state.structure.last_bos_direction == "bullish"
        )

    def is_bos_bearish(self, timeframe: str) -> bool:
        state = self.get_structure(timeframe)

        return bool(
            state
            and state.structure.bos_detected
            and state.structure.last_bos_direction == "bearish"
        )

    def is_choch_bullish(self, timeframe: str) -> bool:
        state = self.get_structure(timeframe)

        return bool(
            state
            and state.structure.choch_detected
            and state.structure.last_choch_direction == "bullish"
        )

    def is_choch_bearish(self, timeframe: str) -> bool:
        state = self.get_structure(timeframe)

        return bool(
            state
            and state.structure.choch_detected
            and state.structure.last_choch_direction == "bearish"
        )

    def get_recent_swing_high(
        self,
        timeframe: str,
    ) -> Optional[float]:
        state = self.get_structure(timeframe)

        if not state:
            return None

        swing = self._last_swing(state.swing_highs)
        return float(swing.price) if swing else None

    def get_recent_swing_low(
        self,
        timeframe: str,
    ) -> Optional[float]:
        state = self.get_structure(timeframe)

        if not state:
            return None

        swing = self._last_swing(state.swing_lows)
        return float(swing.price) if swing else None

    def get_previous_swing_high(
        self,
        timeframe: str,
    ) -> Optional[float]:
        state = self.get_structure(timeframe)

        if not state or len(state.swing_highs) < 2:
            return None

        swings = state.swing_highs

        if all(hasattr(item, "index") for item in swings):
            swings = sorted(swings, key=lambda item: item.index)
        elif all(hasattr(item, "timestamp") for item in swings):
            swings = sorted(swings, key=lambda item: item.timestamp)

        return float(swings[-2].price)

    def get_previous_swing_low(
        self,
        timeframe: str,
    ) -> Optional[float]:
        state = self.get_structure(timeframe)

        if not state or len(state.swing_lows) < 2:
            return None

        swings = state.swing_lows

        if all(hasattr(item, "index") for item in swings):
            swings = sorted(swings, key=lambda item: item.index)
        elif all(hasattr(item, "timestamp") for item in swings):
            swings = sorted(swings, key=lambda item: item.timestamp)

        return float(swings[-2].price)


# Shared analyzer instance.
market_structure = MarketStructureAnalyzer()
