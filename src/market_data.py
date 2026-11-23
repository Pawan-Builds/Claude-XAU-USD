"""Market data engine for historical and live data management."""
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import deque
import threading

from src.config_loader import config
from src.mt5_connector import mt5_connector
from src.database_manager import db_manager, HistoricalCandle
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float = 0.0


class MarketDataEngine:
    """Manages market data retrieval, caching, and synchronization."""

    def __init__(self):
        self.timeframes = config.timeframes
        self.smc_config = config.smc
        self._candle_cache: Dict[str, deque] = {}  # timeframe -> deque of candles
        self._last_candle_time: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        self._callbacks: List[callable] = []
        self._running = False

    def start(self):
        """Start the market data engine."""
        self._running = True
        logger.info("Market data engine started")

    def stop(self):
        """Stop the market data engine."""
        self._running = False
        logger.info("Market data engine stopped")

    def register_callback(self, callback: callable):
        """Register a callback for new completed candles."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, timeframe: str, candle: Candle):
        """Notify all registered callbacks of a new completed candle."""
        for callback in self._callbacks:
            try:
                callback(timeframe, candle)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def initialize_cache(self, count: int = 500) -> bool:
        """Initialize candle cache for all timeframes."""
        logger.info("Initializing market data cache...")

        tf_map = {
            "m5": 5,
            "m15": 15,
            "h1": 60,
            "h4": 240
        }

        for tf_name, tf_minutes in tf_map.items():
            rates = mt5_connector.get_rates(tf_minutes, count)
            if rates is None:
                logger.error(f"Failed to initialize cache for {tf_name}")
                return False

            candles = []
            for rate in rates:
                candle = Candle(
                    timestamp=rate["time"],
                    open=rate["open"],
                    high=rate["high"],
                    low=rate["low"],
                    close=rate["close"],
                    volume=rate["tick_volume"],
                    spread=rate["spread"]
                )
                candles.append(candle)

            with self._lock:
                self._candle_cache[tf_name] = deque(candles, maxlen=1000)
                self._last_candle_time[tf_name] = candles[-1].timestamp

            logger.info(f"Cached {len(candles)} candles for {tf_name}")

        return True

    def get_candles(self, timeframe: str, count: int = 100) -> List[Candle]:
        """Get cached candles for a timeframe."""
        with self._lock:
            if timeframe not in self._candle_cache:
                return []
            return list(self._candle_cache[timeframe])[-count:]

    def get_latest_candle(self, timeframe: str) -> Optional[Candle]:
        """Get the latest completed candle."""
        with self._lock:
            if timeframe not in self._candle_cache or not self._candle_cache[timeframe]:
                return None
            return self._candle_cache[timeframe][-1]

    def get_current_tick(self) -> Optional[Dict]:
        """Get current tick data."""
        return mt5_connector.get_tick()

    def update_live_data(self) -> bool:
        """Update cache with latest live data."""
        tick = mt5_connector.get_tick()
        if not tick:
            return False

        current_time = tick["time"]
        bid = tick["bid"]
        ask = tick["ask"]
        spread = tick["spread"]

        tf_map = {
            "m5": 5,
            "m15": 15,
            "h1": 60,
            "h4": 240
        }

        updated = False
        for tf_name, tf_minutes in tf_map.items():
            with self._lock:
                if tf_name not in self._candle_cache or not self._candle_cache[tf_name]:
                    continue

                last_candle = self._candle_cache[tf_name][-1]
                candle_start = last_candle.timestamp
                candle_end = candle_start + timedelta(minutes=tf_minutes)

                # Check if current tick belongs to a new candle
                if current_time >= candle_end:
                    # Previous candle is complete, create new one
                    completed_candle = Candle(
                        timestamp=last_candle.timestamp,
                        open=last_candle.open,
                        high=last_candle.high,
                        low=last_candle.low,
                        close=last_candle.close,
                        volume=last_candle.volume,
                        spread=last_candle.spread
                    )

                    # Start new candle
                    new_candle = Candle(
                        timestamp=candle_end,
                        open=bid,
                        high=bid,
                        low=bid,
                        close=bid,
                        volume=0,
                        spread=spread
                    )

                    self._candle_cache[tf_name].append(new_candle)
                    self._last_candle_time[tf_name] = candle_end

                    # Notify callbacks of completed candle
                    self._notify_callbacks(tf_name, completed_candle)
                    updated = True
                else:
                    # Update current forming candle
                    last_candle.high = max(last_candle.high, bid)
                    last_candle.low = min(last_candle.low, bid)
                    last_candle.close = bid
                    last_candle.volume += 1
                    last_candle.spread = spread

        return updated

    def sync_with_mt5(self, timeframe: str, count: int = 100) -> bool:
        """Synchronize cache with MT5 data."""
        tf_map = {"m5": 5, "m15": 15, "h1": 60, "h4": 240}
        tf_minutes = tf_map.get(timeframe)
        if not tf_minutes:
            return False

        rates = mt5_connector.get_rates(tf_minutes, count)
        if rates is None:
            return False

        candles = []
        for rate in rates:
            candles.append(Candle(
                timestamp=rate["time"],
                open=rate["open"],
                high=rate["high"],
                low=rate["low"],
                close=rate["close"],
                volume=rate["tick_volume"],
                spread=rate["spread"]
            ))

        with self._lock:
            self._candle_cache[timeframe] = deque(candles, maxlen=1000)
            self._last_candle_time[timeframe] = candles[-1].timestamp

        return True

    def store_candles_to_db(self, timeframe: str, candles: List[Candle]) -> int:
        """Store candles to database."""
        symbol = mt5_connector.get_symbol_name() or config.trading.symbol
        stored = 0

        with db_manager.get_session() as session:
            for candle in candles:
                existing = session.query(HistoricalCandle).filter(
                    HistoricalCandle.symbol == symbol,
                    HistoricalCandle.timeframe == timeframe,
                    HistoricalCandle.timestamp == candle.timestamp
                ).first()

                if not existing:
                    db_candle = HistoricalCandle(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=candle.timestamp,
                        open=candle.open,
                        high=candle.high,
                        low=candle.low,
                        close=candle.close,
                        volume=candle.volume,
                        spread=candle.spread
                    )
                    session.add(db_candle)
                    stored += 1

        if stored > 0:
            logger.info(f"Stored {stored} new candles for {timeframe}")

        return stored

    def load_candles_from_db(self, timeframe: str, count: int = 500) -> List[Candle]:
        """Load candles from database."""
        symbol = mt5_connector.get_symbol_name() or config.trading.symbol
        db_candles = db_manager.get_latest_candles(symbol, timeframe, count)

        candles = []
        for db_candle in reversed(db_candles):
            candles.append(Candle(
                timestamp=db_candle.timestamp,
                open=db_candle.open,
                high=db_candle.high,
                low=db_candle.low,
                close=db_candle.close,
                volume=db_candle.volume,
                spread=db_candle.spread
            ))

        return candles

    def get_candles_as_dataframe(self, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get candles as pandas DataFrame."""
        candles = self.get_candles(timeframe, count)
        if not candles:
            return pd.DataFrame()

        data = {
            "timestamp": [c.timestamp for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
            "spread": [c.spread for c in candles]
        }

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def calculate_atr(self, timeframe: str, period: int = 14) -> Optional[float]:
        """Calculate Average True Range."""
        df = self.get_candles_as_dataframe(timeframe, period + 10)
        if len(df) < period + 1:
            return None

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])

        true_range = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = np.mean(true_range[-period:])

        return float(atr)

    def get_spread_pips(self) -> float:
        """Get current spread in pips."""
        tick = mt5_connector.get_tick()
        if not tick:
            return 0.0

        symbol_info = mt5_connector.get_symbol_info()
        if not symbol_info:
            return 0.0

        # Convert spread to pips (1 pip = 0.01 for XAUUSD typically)
        point = symbol_info.point
        if point == 0:
            return 0.0

        spread_pips = tick["spread"] / (point * 10)
        return spread_pips

    def is_high_spread(self) -> bool:
        """Check if spread exceeds maximum allowed."""
        max_spread = config.trade_management.max_spread_pips
        current_spread = self.get_spread_pips()
        return current_spread > max_spread

    def get_session_info(self) -> Dict[str, Any]:
        """Get current trading session information."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        minute = now.minute
        current_time = hour * 60 + minute

        sessions = config.sessions
        session_info = {
            "current_time_utc": now.strftime("%H:%M"),
            "london": False,
            "new_york": False,
            "asian": False,
            "overlap": False,
            "active_session": "closed"
        }

        # London: 08:00-17:00 UTC
        london_open = 8 * 60
        london_close = 17 * 60
        if london_open <= current_time < london_close:
            session_info["london"] = True
            session_info["active_session"] = "london"

        # New York: 13:00-22:00 UTC
        ny_open = 13 * 60
        ny_close = 22 * 60
        if ny_open <= current_time < ny_close:
            session_info["new_york"] = True
            if session_info["active_session"] == "london":
                session_info["overlap"] = True
                session_info["active_session"] = "london_ny_overlap"
            elif session_info["active_session"] == "closed":
                session_info["active_session"] = "new_york"

        # Asian: 00:00-09:00 UTC
        asian_open = 0
        asian_close = 9 * 60
        if asian_open <= current_time < asian_close:
            session_info["asian"] = True
            if session_info["active_session"] == "closed":
                session_info["active_session"] = "asian"

        return session_info

    def is_trading_session(self) -> bool:
        """Check if current time is within allowed trading sessions."""
        session_info = self.get_session_info()
        sessions = config.sessions

        if sessions.london.get("enabled", True) and session_info["london"]:
            return True
        if sessions.new_york.get("enabled", True) and session_info["new_york"]:
            return True
        if sessions.asian.get("enabled", False) and session_info["asian"]:
            return True

        return False

    def is_weekend_protection_active(self) -> bool:
        """Check if weekend protection should be active."""
        if not config.sessions.weekend_protection:
            return False

        now = datetime.now(timezone.utc)
        # Friday close_before_weekend_hours before market close (22:00 UTC)
        if now.weekday() == 4:  # Friday
            hours_until_close = 22 - now.hour - now.minute / 60
            if hours_until_close <= config.sessions.close_before_weekend_hours:
                return True

        # Saturday/Sunday
        if now.weekday() >= 5:
            return True

        return False


# Global market data engine instance
market_data = MarketDataEngine()