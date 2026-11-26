"""Market data engine for historical and live data management.

M-05 additions:
- is_timeframe_fresh(tf) -> (ok, reason)
- is_market_data_reliable() -> (ok, reason)   [entry-gate API]
- get_spread_status() -> (spread_pips, status)
- is_high_spread() fails closed on stale/unavailable data
- get_current_tick() now returns only fresh ticks (get_fresh_tick)
- update_live_data() resyncs from MT5 on gaps instead of
  synthesizing missing candles
- initialize_cache() and sync_with_mt5() validate candles for
  duplicates, out-of-order, invalid prices, and log gap counts
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import deque
import threading
from contextvars import ContextVar

from src.config_loader import config
from src.mt5_connector import mt5_connector
from src.database_manager import db_manager, HistoricalCandle
from src.logger import get_logger

logger = get_logger(__name__)

# N-H04: context-local backtest data provider. Thread-local by
# default; the live trading thread always sees None and therefore
# reads real MT5-cached data. A backtest thread sets a provider
# for the duration of its run only.
_backtest_provider_ctx: ContextVar[Optional[Any]] = ContextVar(
    "market_data_backtest_provider", default=None
)


def set_backtest_data_provider(provider: Any) -> Any:
    """N-H04: install a backtest data provider for the current
    thread/task. Returns a token for reset_backtest_data_provider."""
    return _backtest_provider_ctx.set(provider)


def reset_backtest_data_provider(token: Any) -> None:
    """N-H04: restore the previous data provider."""
    _backtest_provider_ctx.reset(token)

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

    # Single source of truth for timeframe minutes.
    _TF_MINUTES: Dict[str, int] = {
        "m5": 5, "m15": 15, "h1": 60, "h4": 240,
    }

    # M-05 D16: gap threshold multiplier. A tick that would require
    # more than this many synthetic candles triggers a broker resync.
    _GAP_RESYNC_MULTIPLIER: float = 1.5

    # M-05 D16: number of candles to fetch on resync.
    _GAP_RESYNC_COUNT: int = 500

    def __init__(self):
        self.timeframes = config.timeframes
        self.smc_config = config.smc
        self._candle_cache: Dict[str, deque] = {}
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

    # ------------------------------------------------------------------
    # M-05: candle-list validation
    # ------------------------------------------------------------------

    def _validate_candles(
        self,
        candles: List[Candle],
        tf_minutes: int,
    ) -> Tuple[bool, str, int]:
        """Validate a candle list for duplicates, ordering, and prices.

        Returns (ok, reason, gap_count).
        - Fails on: empty list, duplicates, out-of-order timestamps,
          non-positive OHLC, high < low.
        - Accepts but counts gaps (weekend/maintenance gaps are
          legitimate; the caller decides whether to warn).
        """
        if not candles:
            return (False, "empty candle list", 0)

        expected_delta = timedelta(minutes=tf_minutes)
        prev_ts: Optional[datetime] = None
        seen = set()
        gaps = 0

        for c in candles:
            if c.timestamp in seen:
                return (
                    False,
                    f"duplicate timestamp {c.timestamp}",
                    gaps,
                )
            seen.add(c.timestamp)

            try:
                o = float(c.open)
                h = float(c.high)
                l = float(c.low)
                cl = float(c.close)
            except (TypeError, ValueError):
                return (
                    False,
                    f"non-numeric OHLC at {c.timestamp}",
                    gaps,
                )

            if not all(
                np.isfinite(v) for v in (o, h, l, cl)
            ):
                return (
                    False,
                    f"non-finite OHLC at {c.timestamp}",
                    gaps,
                )

            if o <= 0 or h <= 0 or l <= 0 or cl <= 0:
                return (
                    False,
                    f"non-positive price at {c.timestamp}",
                    gaps,
                )

            if h < l:
                return (
                    False,
                    f"high below low at {c.timestamp}",
                    gaps,
                )

            if prev_ts is not None:
                if c.timestamp <= prev_ts:
                    return (
                        False,
                        f"out-of-order timestamp at {c.timestamp}",
                        gaps,
                    )
                if (c.timestamp - prev_ts) != expected_delta:
                    gaps += 1
            prev_ts = c.timestamp

        return (True, "ok", gaps)

    # ------------------------------------------------------------------
    # M-05: freshness configuration
    # ------------------------------------------------------------------

    def _get_max_candle_age_bars(self) -> int:
        """Read config.data.max_candle_age_bars; safe default 2."""
        try:
            value = getattr(config.data, "max_candle_age_bars", 2)
        except Exception:
            return 2

        if isinstance(value, bool):
            return 2

        try:
            value = int(value)
        except (TypeError, ValueError):
            return 2

        if value < 1:
            return 2

        return value

    def is_timeframe_fresh(
        self, timeframe: str
    ) -> Tuple[bool, str]:
        """Return (ok, reason) for a timeframe's closed-candle freshness.

        Fresh means: the close-time of the most recent *closed*
        candle is within max_candle_age_bars × timeframe minutes of
        now. Empty or single-candle caches return False (fail-closed).
        """
        timeframe = timeframe.lower()
        tf_minutes = self._TF_MINUTES.get(timeframe)
        if tf_minutes is None:
            return (False, f"unknown timeframe {timeframe!r}")

        closed = self.get_closed_candles(timeframe, count=1)
        if not closed:
            return (
                False,
                f"{timeframe} has no closed candles in cache",
            )

        last_closed = closed[-1]
        last_closed_end = (
            last_closed.timestamp + timedelta(minutes=tf_minutes)
        )
        if last_closed_end.tzinfo is None:
            last_closed_end = last_closed_end.replace(
                tzinfo=timezone.utc
            )

        max_bars = self._get_max_candle_age_bars()
        max_age = timedelta(minutes=max_bars * tf_minutes)

        age = datetime.now(timezone.utc) - last_closed_end
        if age > max_age:
            return (
                False,
                (
                    f"{timeframe} stale: last closed candle ended "
                    f"{age.total_seconds() / 60:.1f}min ago "
                    f"(max {max_age.total_seconds() / 60:.1f}min)"
                ),
            )

        return (True, "ok")

    def is_market_data_reliable(self) -> Tuple[bool, str]:
        """Single fail-closed API for the entry gate (M-10 consumer).

        Returns (ok, reason).
        Checks: MT5 connectivity, tick freshness, and closed-candle
        freshness for all required timeframes (h4, h1, m15, m5).
        First failure short-circuits with its reason.
        """
        try:
            if not mt5_connector.is_connected():
                return (False, "MT5 not connected")
        except Exception:
            logger.exception(
                "MT5 connectivity check raised during reliability gate."
            )
            return (False, "MT5 connectivity check failed")

        try:
            if not mt5_connector.is_tick_fresh():
                return (False, "tick is stale or unavailable")
        except Exception:
            logger.exception("Tick freshness check raised.")
            return (False, "tick freshness check failed")

        for tf in ("h4", "h1", "m15", "m5"):
            try:
                ok, reason = self.is_timeframe_fresh(tf)
            except Exception:
                logger.exception(
                    "Timeframe freshness check raised for %s.", tf
                )
                return (False, f"{tf} freshness check failed")
            if not ok:
                return (False, reason)

        return (True, "ok")

    # ------------------------------------------------------------------
    # Cache initialization and sync (with M-05 validation)
    # ------------------------------------------------------------------

    def initialize_cache(self, count: int = 500) -> bool:
        """Initialize candle cache for all timeframes atomically.

        FIX #23: if any timeframe fails, clear the entire cache so a
        partial load cannot leave the strategy engine reading mixed
        fresh/stale candle series from different timeframes.

        M-05: each timeframe's candles are validated for duplicates,
        ordering, and price sanity before insertion. Validation
        failures abort the whole initialization (fail-closed). Gaps
        are logged but accepted (legitimate weekend gaps).
        """
        logger.info("Initializing market data cache...")

        loaded: Dict[str, List[Candle]] = {}

        for tf_name, tf_minutes in self._TF_MINUTES.items():
            rates = mt5_connector.get_rates(tf_minutes, count)

            if rates is None:
                logger.error(
                    "Failed to initialize cache for %s; aborting "
                    "cache load",
                    tf_name,
                )
                with self._lock:
                    self._candle_cache.clear()
                    self._last_candle_time.clear()
                return False

            if len(rates) == 0:
                logger.error(
                    "Empty candle list for %s; aborting cache load",
                    tf_name,
                )
                with self._lock:
                    self._candle_cache.clear()
                    self._last_candle_time.clear()
                return False

            candles = [
                Candle(
                    timestamp=rate["time"],
                    open=rate["open"],
                    high=rate["high"],
                    low=rate["low"],
                    close=rate["close"],
                    volume=rate["tick_volume"],
                    spread=rate["spread"],
                )
                for rate in rates
            ]

            ok, reason, gaps = self._validate_candles(
                candles, tf_minutes
            )
            if not ok:
                logger.error(
                    "Rejecting MT5 initialization for %s: %s. "
                    "Aborting cache load.",
                    tf_name, reason,
                )
                with self._lock:
                    self._candle_cache.clear()
                    self._last_candle_time.clear()
                return False

            if gaps > 0:
                logger.info(
                    "Initialization for %s contains %d historical "
                    "gap(s); accepting (expected for weekends and "
                    "broker maintenance).",
                    tf_name, gaps,
                )

            loaded[tf_name] = candles

        with self._lock:
            for tf_name, candles in loaded.items():
                self._candle_cache[tf_name] = deque(
                    candles, maxlen=1000
                )
                self._last_candle_time[tf_name] = (
                    candles[-1].timestamp
                )

        for tf_name, candles in loaded.items():
            logger.info(
                "Cached %d candles for %s", len(candles), tf_name
            )

        return True

    def get_candles(
        self, timeframe: str, count: int = 100
    ) -> List[Candle]:
        """Get cached candles for a timeframe."""
        _bp = _backtest_provider_ctx.get()
        if _bp is not None:
            return _bp.get_candles(timeframe, count)
        with self._lock:
            if timeframe not in self._candle_cache:
                return []
            return list(self._candle_cache[timeframe])[-count:]

    def get_closed_candles(
        self,
        timeframe: str,
        count: int = 100,
    ) -> List[Candle]:
        """Return only *closed* candles; excludes the forming candle."""
        _bp = _backtest_provider_ctx.get()
        if _bp is not None:
            return _bp.get_closed_candles(timeframe, count)
        if count <= 0:
            return []
        with self._lock:
            if timeframe not in self._candle_cache:
                return []
            candles = list(self._candle_cache[timeframe])
        if len(candles) <= 1:
            return []
        closed = candles[:-1]
        return closed[-count:]

    def get_latest_closed_candle(
        self,
        timeframe: str,
    ) -> Optional[Candle]:
        """Return the most recent *closed* candle, or None."""
        closed = self.get_closed_candles(timeframe, count=1)
        return closed[0] if closed else None

    def get_current_tick(self) -> Optional[Dict]:
        """Return the current tick only if fresh (M-05 fail-closed).

        Behaviour change vs. pre-M-05: returns None when the latest
        tick is older than config.data.max_tick_age_seconds or when
        MT5 supplies no tick. Entry-path callers MUST treat None as
        "cannot act on market data right now". Position management
        uses MT5 tick reads directly and is unaffected.
        """
        return mt5_connector.get_fresh_tick()

    def update_live_data(self) -> bool:
        """Update cache with latest live data.

        M-05 D16: when a tick would require more than 1.5× the
        timeframe of synthetic candles to bridge the gap to the
        last closed candle, the cache for that timeframe is NOT
        patched with synthetic bars. Instead, sync_with_mt5() is
        called to refetch genuine history from MT5. If resync
        fails, the cache is left unchanged; callers must fail
        closed via is_timeframe_fresh().
        """
        tick = mt5_connector.get_tick()
        if not tick:
            return False

        current_time = tick["time"]
        bid = tick["bid"]
        ask = tick["ask"]
        spread = tick["spread"]
        mid = (bid + ask) / 2.0

        updated = False
        resync_needed: List[str] = []

        for tf_name, tf_minutes in self._TF_MINUTES.items():
            with self._lock:
                if (
                    tf_name not in self._candle_cache
                    or not self._candle_cache[tf_name]
                ):
                    continue

                last_candle = self._candle_cache[tf_name][-1]
                candle_start = last_candle.timestamp
                candle_end = candle_start + timedelta(
                    minutes=tf_minutes
                )

                if current_time >= candle_end:
                    gap_minutes = (
                        current_time - candle_end
                    ).total_seconds() / 60.0

                    if (
                        gap_minutes
                        > self._GAP_RESYNC_MULTIPLIER * tf_minutes
                    ):
                        # Gap too large for a single rollover --
                        # do not synthesize missing candles.
                        resync_needed.append(tf_name)
                        continue

                    completed_candle = Candle(
                        timestamp=last_candle.timestamp,
                        open=last_candle.open,
                        high=last_candle.high,
                        low=last_candle.low,
                        close=last_candle.close,
                        volume=last_candle.volume,
                        spread=last_candle.spread,
                    )

                    new_candle = Candle(
                        timestamp=candle_end,
                        open=mid,
                        high=ask,
                        low=bid,
                        close=mid,
                        volume=0,
                        spread=spread,
                    )

                    self._candle_cache[tf_name].append(new_candle)
                    self._last_candle_time[tf_name] = candle_end

                    self._notify_callbacks(
                        tf_name, completed_candle
                    )
                    updated = True
                else:
                    last_candle.high = max(last_candle.high, ask)
                    last_candle.low = min(last_candle.low, bid)
                    last_candle.close = mid
                    last_candle.volume += 1
                    last_candle.spread = spread

        # M-05: resync gapped timeframes outside the lock.
        for tf_name in resync_needed:
            logger.warning(
                "Detected data gap for %s; resyncing from MT5 "
                "(count=%d).",
                tf_name, self._GAP_RESYNC_COUNT,
            )
            if not self.sync_with_mt5(
                tf_name, count=self._GAP_RESYNC_COUNT
            ):
                logger.error(
                    "Gap resync failed for %s; leaving cache "
                    "unchanged. Callers must fail closed via "
                    "is_timeframe_fresh().",
                    tf_name,
                )

        return updated

    def sync_with_mt5(
        self, timeframe: str, count: int = 100
    ) -> bool:
        """Synchronize cache with MT5 data.

        M-05: returned candles are validated before replacing the
        cache. Duplicates, out-of-order timestamps, and invalid
        prices cause the sync to fail (cache unchanged, fail-closed).
        Gaps are counted and logged but accepted.
        """
        tf_minutes = self._TF_MINUTES.get(timeframe)
        if not tf_minutes:
            return False

        rates = mt5_connector.get_rates(tf_minutes, count)
        if rates is None:
            return False

        candles: List[Candle] = []
        for rate in rates:
            candles.append(Candle(
                timestamp=rate["time"],
                open=rate["open"],
                high=rate["high"],
                low=rate["low"],
                close=rate["close"],
                volume=rate["tick_volume"],
                spread=rate["spread"],
            ))

        if not candles:
            return False

        ok, reason, gaps = self._validate_candles(
            candles, tf_minutes
        )
        if not ok:
            logger.error(
                "Rejecting MT5 sync for %s: %s", timeframe, reason
            )
            return False

        if gaps > 0:
            logger.info(
                "MT5 sync for %s contains %d gap(s); accepting "
                "(weekend/maintenance).",
                timeframe, gaps,
            )

        with self._lock:
            self._candle_cache[timeframe] = deque(
                candles, maxlen=1000
            )
            self._last_candle_time[timeframe] = (
                candles[-1].timestamp
            )

        return True

    def store_candles_to_db(
        self, timeframe: str, candles: List[Candle]
    ) -> int:
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
            logger.info(
                "Stored %d new candles for %s", stored, timeframe
            )

        return stored

    def load_candles_from_db(
        self, timeframe: str, count: int = 500
    ) -> List[Candle]:
        """Load candles from database."""
        symbol = (
            mt5_connector.get_symbol_name() or config.trading.symbol
        )
        db_candles = db_manager.get_latest_candles(
            symbol, timeframe, count
        )

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

    def get_candles_as_dataframe(
        self, timeframe: str, count: int = 100
    ) -> pd.DataFrame:
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

    def calculate_atr(
        self, timeframe: str, period: int = 14
    ) -> Optional[float]:
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

    # ------------------------------------------------------------------
    # M-05: Spread status and fail-closed entry check
    # ------------------------------------------------------------------

    def get_spread_status(self) -> Tuple[Optional[float], str]:
        """Return (spread_pips, status).

        Status values:
          - "ok"          : valid spread (may be exactly 0.0 if the
                            broker is genuinely at zero, but only if
                            the tick is fresh and symbol info is
                            readable)
          - "stale"       : tick is older than configured limit
          - "unavailable" : no tick, or symbol info / pip size
                            unreadable

        On non-"ok" status, the first tuple element is None. This
        API is the disambiguator that separates a genuine zero
        spread from missing data; M-10 will wire it into the entry
        gate.
        """
        if not mt5_connector.is_tick_fresh():
            return (None, "stale")

        tick = mt5_connector.get_tick()
        if not tick:
            return (None, "unavailable")

        symbol_info = mt5_connector.get_symbol_info()
        if not symbol_info:
            return (None, "unavailable")

        pip_size = mt5_connector.pip_size_for(symbol_info)
        if pip_size <= 0:
            return (None, "unavailable")

        try:
            spread = float(tick["spread"]) / pip_size
        except (TypeError, ValueError, KeyError):
            return (None, "unavailable")

        if not np.isfinite(spread):
            return (None, "unavailable")

        return (spread, "ok")

    def get_spread_pips(self) -> float:
        """Get current spread in pips.

        Pre-M-05 behavior: derived spread from whatever MT5 gave,
        with a 0.0 fallback on missing tick.

        M-05: uses get_spread_status() internally. Returns 0.0 when
        status is not "ok" (preserving the historical return type
        for non-entry-gate consumers).

        NOTE: 0.0 is ambiguous (valid zero spread vs unavailable).
        Safety-critical callers MUST use get_spread_status() or
        is_high_spread(), not this method.
        """
        spread, status = self.get_spread_status()
        if status != "ok" or spread is None:
            return 0.0
        return spread

    def is_high_spread(self) -> bool:
        """Return True if spread is high OR spread data is unavailable.

        M-05 D17: fail-closed on stale/unavailable data — the entry
        gate consumes this method and MUST block new entries when
        spread data cannot be trusted. Confirmed scope: only the
        entry path calls this; management and emergency-close paths
        bypass spread checks entirely.
        """
        spread, status = self.get_spread_status()
        if status != "ok" or spread is None:
            return True
        return spread > config.trade_management.max_spread_pips

    def get_session_info(self) -> Dict[str, Any]:
        """Get current trading session information."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        minute = now.minute
        current_time = hour * 60 + minute

        session_info = {
            "current_time_utc": now.strftime("%H:%M"),
            "london": False,
            "new_york": False,
            "asian": False,
            "overlap": False,
            "active_session": "closed"
        }

        london_open = 8 * 60
        london_close = 17 * 60
        if london_open <= current_time < london_close:
            session_info["london"] = True
            session_info["active_session"] = "london"

        ny_open = 13 * 60
        ny_close = 22 * 60
        if ny_open <= current_time < ny_close:
            session_info["new_york"] = True
            if session_info["active_session"] == "london":
                session_info["overlap"] = True
                session_info["active_session"] = (
                    "london_ny_overlap"
                )
            elif session_info["active_session"] == "closed":
                session_info["active_session"] = "new_york"

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
        if now.weekday() == 4:  # Friday
            hours_until_close = 22 - now.hour - now.minute / 60
            if hours_until_close <= config.sessions.close_before_weekend_hours:
                return True

        if now.weekday() >= 5:
            return True

        return False


# Global market data engine instance
market_data = MarketDataEngine()