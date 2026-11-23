"""Premium/Discount zone analysis module."""
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from src.indicators import calculate_premium_discount
from src.market_structure import market_structure
from src.market_data import market_data
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PremiumDiscountState:
    """Current premium/discount state for a timeframe."""
    timeframe: str
    timestamp: datetime
    zone: str  # 'premium', 'discount', 'equilibrium'
    equilibrium: float
    high: float
    low: float
    midpoint: float
    current_price: float
    distance_from_equilibrium: float


class PremiumDiscountAnalyzer:
    """Analyzes premium/discount zones across timeframes."""

    def __init__(self):
        self.timeframes = ["h4", "h1", "m15", "m5"]
        self.pd_cache: Dict[str, PremiumDiscountState] = {}
        self.lookback = 50

    def analyze_all_timeframes(self) -> Dict[str, PremiumDiscountState]:
        """Analyze premium/discount for all configured timeframes."""
        results = {}

        for tf in self.timeframes:
            state = self.analyze_timeframe(tf)
            if state:
                results[tf] = state
                self.pd_cache[tf] = state

        return results

    def analyze_timeframe(self, timeframe: str) -> Optional[PremiumDiscountState]:
        """Analyze premium/discount for a specific timeframe."""
        candles = market_data.get_candles(timeframe, self.lookback)
        if len(candles) < 20:
            return None

        # Use swing high/low from market structure for range
        structure_state = market_structure.get_structure(timeframe)
        if not structure_state or not structure_state.key_levels:
            # Fallback to recent high/low
            highs = np.array([c.high for c in candles])
            lows = np.array([c.low for c in candles])
            high = np.max(highs)
            low = np.min(lows)
        else:
            high = structure_state.key_levels.get("recent_high", np.max([c.high for c in candles]))
            low = structure_state.key_levels.get("recent_low", np.min([c.low for c in candles]))

        current_price = candles[-1].close

        # Calculate premium/discount
        pd = calculate_premium_discount(high, low, current_price)

        state = PremiumDiscountState(
            timeframe=timeframe,
            timestamp=candles[-1].timestamp,
            zone=pd["zone"],
            equilibrium=pd["equilibrium"],
            high=pd["high"],
            low=pd["low"],
            midpoint=pd["midpoint"],
            current_price=pd["current_price"],
            distance_from_equilibrium=pd["distance_from_equilibrium"]
        )

        return state

    def get_pd_state(self, timeframe: str) -> Optional[PremiumDiscountState]:
        """Get cached premium/discount state."""
        return self.pd_cache.get(timeframe)

    def get_zone(self, timeframe: str) -> str:
        """Get current zone for timeframe."""
        state = self.get_pd_state(timeframe)
        return state.zone if state else "unknown"

    def is_premium(self, timeframe: str) -> bool:
        """Check if price is in premium zone."""
        return self.get_zone(timeframe) == "premium"

    def is_discount(self, timeframe: str) -> bool:
        """Check if price is in discount zone."""
        return self.get_zone(timeframe) == "discount"

    def is_equilibrium(self, timeframe: str) -> bool:
        """Check if price is at equilibrium."""
        return self.get_zone(timeframe) == "equilibrium"

    def get_equilibrium_price(self, timeframe: str) -> Optional[float]:
        """Get equilibrium price for timeframe."""
        state = self.get_pd_state(timeframe)
        return state.equilibrium if state else None

    def get_distance_from_equilibrium(self, timeframe: str) -> float:
        """Get distance from equilibrium (0-1)."""
        state = self.get_pd_state(timeframe)
        return state.distance_from_equilibrium if state else 0.0

    def should_buy_in_discount(self, timeframe: str) -> bool:
        """Check if conditions favor buying in discount."""
        return self.is_discount(timeframe)

    def should_sell_in_premium(self, timeframe: str) -> bool:
        """Check if conditions favor selling in premium."""
        return self.is_premium(timeframe)


# Global premium/discount analyzer instance
premium_discount = PremiumDiscountAnalyzer()