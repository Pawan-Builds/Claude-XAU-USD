"""M-05 tests: tick/candle freshness, gap resync, spread status,
candle validation, and the reliability API.

Run with: pytest tests/test_market_data_freshness.py -v
"""

from datetime import datetime, timezone, timedelta
from collections import deque
from unittest.mock import Mock, patch

import pytest

from src.mt5_connector import MT5Connector
from src.market_data import MarketDataEngine, Candle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tick(age_s: float = 0.0, bid: float = 2000.0, ask: float = 2000.5):
    return {
        "symbol": "GOLD",
        "bid": bid,
        "ask": ask,
        "spread": ask - bid,
        "time": datetime.now(timezone.utc) - timedelta(seconds=age_s),
    }


def _candle(ts, o=100.0, h=101.0, l=99.0, c=100.0, v=1.0):
    return Candle(
        timestamp=ts, open=o, high=h, low=l, close=c, volume=v,
    )


def _make_engine_with(tf: str, candles):
    engine = MarketDataEngine()
    engine._candle_cache[tf] = deque(candles, maxlen=1000)
    engine._last_candle_time[tf] = candles[-1].timestamp
    return engine


# ---------------------------------------------------------------------------
# Tick freshness (MT5Connector)
# ---------------------------------------------------------------------------

class TestTickFreshness:

    def test_fresh_tick(self):
        conn = MT5Connector()
        with patch.object(conn, "get_tick", return_value=_tick(10)):
            with patch.object(
                conn, "_get_max_tick_age_seconds", return_value=60.0
            ):
                assert conn.is_tick_fresh() is True

    def test_stale_tick(self):
        conn = MT5Connector()
        with patch.object(conn, "get_tick", return_value=_tick(120)):
            with patch.object(
                conn, "_get_max_tick_age_seconds", return_value=60.0
            ):
                assert conn.is_tick_fresh() is False

    def test_no_tick(self):
        conn = MT5Connector()
        with patch.object(conn, "get_tick", return_value=None):
            with patch.object(
                conn, "_get_max_tick_age_seconds", return_value=60.0
            ):
                assert conn.is_tick_fresh() is False

    def test_missing_time_field(self):
        conn = MT5Connector()
        t = _tick(10)
        del t["time"]
        with patch.object(conn, "get_tick", return_value=t):
            with patch.object(
                conn, "_get_max_tick_age_seconds", return_value=60.0
            ):
                assert conn.is_tick_fresh() is False

    def test_get_fresh_tick_returns_tick_when_fresh(self):
        conn = MT5Connector()
        t = _tick(5)
        with patch.object(conn, "get_tick", return_value=t):
            with patch.object(
                conn, "_get_max_tick_age_seconds", return_value=60.0
            ):
                assert conn.get_fresh_tick() is t

    def test_get_fresh_tick_returns_none_when_stale(self):
        conn = MT5Connector()
        with patch.object(conn, "get_tick", return_value=_tick(120)):
            with patch.object(
                conn, "_get_max_tick_age_seconds", return_value=60.0
            ):
                assert conn.get_fresh_tick() is None

    def test_get_tick_raw_returns_underlying_tick(self):
        conn = MT5Connector()
        t = _tick(999)  # stale
        with patch.object(conn, "get_tick", return_value=t):
            assert conn.get_tick_raw() is t

    def test_config_default_used_when_no_arg(self):
        conn = MT5Connector()
        with patch.object(
            conn, "_get_max_tick_age_seconds", return_value=30.0
        ):
            with patch.object(
                conn, "get_tick", return_value=_tick(45)
            ):
                assert conn.is_tick_fresh() is False
            with patch.object(
                conn, "get_tick", return_value=_tick(15)
            ):
                assert conn.is_tick_fresh() is True


# ---------------------------------------------------------------------------
# Candle validation
# ---------------------------------------------------------------------------

class TestCandleValidation:

    def _ts(self, minute):
        return datetime(
            2024, 1, 1, 0, minute, tzinfo=timezone.utc
        )

    def test_valid_candles(self):
        engine = MarketDataEngine()
        candles = [
            _candle(self._ts(0)),
            _candle(self._ts(5)),
            _candle(self._ts(10)),
        ]
        ok, reason, gaps = engine._validate_candles(candles, 5)
        assert ok is True and gaps == 0

    def test_duplicate_rejected(self):
        engine = MarketDataEngine()
        ts = self._ts(0)
        candles = [_candle(ts), _candle(ts)]
        ok, reason, _ = engine._validate_candles(candles, 5)
        assert ok is False
        assert "duplicate" in reason.lower()

    def test_out_of_order_rejected(self):
        engine = MarketDataEngine()
        candles = [
            _candle(self._ts(10)),
            _candle(self._ts(5)),
        ]
        ok, reason, _ = engine._validate_candles(candles, 5)
        assert ok is False
        assert "out-of-order" in reason.lower()

    def test_gap_counted_accepted(self):
        engine = MarketDataEngine()
        candles = [
            _candle(self._ts(0)),
            _candle(self._ts(15)),
        ]
        ok, reason, gaps = engine._validate_candles(candles, 5)
        assert ok is True
        assert gaps == 1

    def test_non_positive_price_rejected(self):
        engine = MarketDataEngine()
        candles = [
            _candle(self._ts(0), o=0.0),
        ]
        ok, reason, _ = engine._validate_candles(candles, 5)
        assert ok is False
        assert "non-positive" in reason.lower()

    def test_high_below_low_rejected(self):
        engine = MarketDataEngine()
        candles = [
            _candle(self._ts(0), h=99.0, l=101.0),
        ]
        ok, reason, _ = engine._validate_candles(candles, 5)
        assert ok is False
        assert "high below low" in reason.lower()

    def test_empty_rejected(self):
        engine = MarketDataEngine()
        ok, reason, _ = engine._validate_candles([], 5)
        assert ok is False


# ---------------------------------------------------------------------------
# Timeframe freshness
# ---------------------------------------------------------------------------

class TestTimeframeFreshness:

    def test_fresh_timeframe(self):
        now = datetime.now(timezone.utc)
        # Forming candle at now, last closed candle ended at now.
        candles = [
            _candle(now - timedelta(minutes=5)),
            _candle(now),
        ]
        engine = _make_engine_with("m5", candles)
        with patch.object(
            engine, "_get_max_candle_age_bars", return_value=2
        ):
            ok, reason = engine.is_timeframe_fresh("m5")
        assert ok is True, reason

    def test_stale_timeframe(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(minutes=30)
        candles = [
            _candle(old - timedelta(minutes=5)),
            _candle(old),
        ]
        engine = _make_engine_with("m5", candles)
        with patch.object(
            engine, "_get_max_candle_age_bars", return_value=2
        ):
            ok, reason = engine.is_timeframe_fresh("m5")
        assert ok is False
        assert "stale" in reason.lower()

    def test_empty_cache_is_stale(self):
        engine = MarketDataEngine()
        ok, _ = engine.is_timeframe_fresh("m5")
        assert ok is False

    def test_unknown_timeframe_is_stale(self):
        engine = MarketDataEngine()
        ok, reason = engine.is_timeframe_fresh("wat")
        assert ok is False
        assert "unknown" in reason.lower()

    def test_single_candle_cache_is_stale(self):
        engine = _make_engine_with(
            "m5", [_candle(datetime.now(timezone.utc))]
        )
        ok, _ = engine.is_timeframe_fresh("m5")
        assert ok is False


# ---------------------------------------------------------------------------
# Reliability API (single entry-gate hook)
# ---------------------------------------------------------------------------

class TestMarketDataReliable:

    def test_all_fresh(self):
        engine = MarketDataEngine()
        with patch("src.market_data.mt5_connector") as mock_conn:
            mock_conn.is_connected.return_value = True
            mock_conn.is_tick_fresh.return_value = True
            with patch.object(
                engine, "is_timeframe_fresh",
                return_value=(True, "ok"),
            ):
                ok, reason = engine.is_market_data_reliable()
        assert ok is True, reason

    def test_disconnected(self):
        engine = MarketDataEngine()
        with patch("src.market_data.mt5_connector") as mock_conn:
            mock_conn.is_connected.return_value = False
            ok, reason = engine.is_market_data_reliable()
        assert ok is False
        assert "not connected" in reason.lower()

    def test_stale_tick(self):
        engine = MarketDataEngine()
        with patch("src.market_data.mt5_connector") as mock_conn:
            mock_conn.is_connected.return_value = True
            mock_conn.is_tick_fresh.return_value = False
            ok, reason = engine.is_market_data_reliable()
        assert ok is False
        assert "tick" in reason.lower()

    def test_one_stale_timeframe(self):
        engine = MarketDataEngine()

        def _fresh(tf):
            return (tf != "h1", f"{tf} stale")

        with patch("src.market_data.mt5_connector") as mock_conn:
            mock_conn.is_connected.return_value = True
            mock_conn.is_tick_fresh.return_value = True
            with patch.object(
                engine, "is_timeframe_fresh",
                side_effect=_fresh,
            ):
                ok, reason = engine.is_market_data_reliable()
        assert ok is False
        assert "h1" in reason.lower()


# ---------------------------------------------------------------------------
# Spread status and is_high_spread
# ---------------------------------------------------------------------------

class TestSpreadStatus:

    def test_ok_status(self):
        engine = MarketDataEngine()
        t = _tick(2)
        info = Mock(point=0.01, digits=2, name="GOLD")
        with patch("src.market_data.mt5_connector") as mock_conn:
            mock_conn.is_tick_fresh.return_value = True
            mock_conn.get_tick.return_value = t
            mock_conn.get_symbol_info.return_value = info
            mock_conn.pip_size_for.return_value = 0.1
            spread, status = engine.get_spread_status()
        assert status == "ok"
        assert spread == pytest.approx(5.0)

    def test_stale_status(self):
        engine = MarketDataEngine()
        with patch("src.market_data.mt5_connector") as mock_conn:
            mock_conn.is_tick_fresh.return_value = False
            spread, status = engine.get_spread_status()
        assert status == "stale"
        assert spread is None

    def test_unavailable_status_missing_tick(self):
        engine = MarketDataEngine()
        with patch("src.market_data.mt5_connector") as mock_conn:
            mock_conn.is_tick_fresh.return_value = True
            mock_conn.get_tick.return_value = None
            spread, status = engine.get_spread_status()
        assert status == "unavailable"
        assert spread is None

    def test_unavailable_status_missing_symbol_info(self):
        engine = MarketDataEngine()
        with patch("src.market_data.mt5_connector") as mock_conn:
            mock_conn.is_tick_fresh.return_value = True
            mock_conn.get_tick.return_value = _tick(2)
            mock_conn.get_symbol_info.return_value = None
            spread, status = engine.get_spread_status()
        assert status == "unavailable"
        assert spread is None

    def test_get_spread_pips_returns_zero_on_stale(self):
        engine = MarketDataEngine()
        with patch.object(
            engine, "get_spread_status",
            return_value=(None, "stale"),
        ):
            assert engine.get_spread_pips() == 0.0

    def test_get_spread_pips_returns_value_on_ok(self):
        engine = MarketDataEngine()
        with patch.object(
            engine, "get_spread_status",
            return_value=(2.5, "ok"),
        ):
            assert engine.get_spread_pips() == 2.5


class TestIsHighSpread:

    def test_stale_returns_true(self):
        engine = MarketDataEngine()
        with patch.object(
            engine, "get_spread_status",
            return_value=(None, "stale"),
        ):
            assert engine.is_high_spread() is True

    def test_unavailable_returns_true(self):
        engine = MarketDataEngine()
        with patch.object(
            engine, "get_spread_status",
            return_value=(None, "unavailable"),
        ):
            assert engine.is_high_spread() is True

    def test_high_spread_returns_true(self):
        from src.config_loader import config
        engine = MarketDataEngine()
        with patch.object(
            engine, "get_spread_status",
            return_value=(10.0, "ok"),
        ):
            with patch.object(
                config.trade_management, "max_spread_pips", 5
            ):
                assert engine.is_high_spread() is True

    def test_low_spread_returns_false(self):
        from src.config_loader import config
        engine = MarketDataEngine()
        with patch.object(
            engine, "get_spread_status",
            return_value=(2.0, "ok"),
        ):
            with patch.object(
                config.trade_management, "max_spread_pips", 5
            ):
                assert engine.is_high_spread() is False


# ---------------------------------------------------------------------------
# Gap resync in update_live_data
# ---------------------------------------------------------------------------

class TestGapResync:

    def test_large_gap_triggers_resync_not_synthesis(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(minutes=30)
        candles = [_candle(old)]
        engine = _make_engine_with("m5", candles)

        tick = {
            "bid": 2000.0, "ask": 2000.5,
            "spread": 0.5, "time": now,
        }
        with patch("src.market_data.mt5_connector") as mock_conn:
            mock_conn.get_tick.return_value = tick
            with patch.object(
                engine, "sync_with_mt5", return_value=True
            ) as mock_sync:
                engine.update_live_data()
                mock_sync.assert_called_once_with(
                    "m5", count=engine._GAP_RESYNC_COUNT,
                )
        # Cache is not patched with synthetic candles.
        assert len(engine._candle_cache["m5"]) == 1

    def test_resync_failure_leaves_cache_unchanged(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(minutes=30)
        engine = _make_engine_with("m5", [_candle(old)])
        original_len = len(engine._candle_cache["m5"])

        tick = {
            "bid": 2000.0, "ask": 2000.5,
            "spread": 0.5, "time": now,
        }
        with patch("src.market_data.mt5_connector") as mock_conn:
            mock_conn.get_tick.return_value = tick
            with patch.object(
                engine, "sync_with_mt5", return_value=False
            ):
                engine.update_live_data()
        assert len(engine._candle_cache["m5"]) == original_len

    def test_normal_rollover_still_synthesizes_one_candle(self):
        now = datetime.now(timezone.utc)
        # Last candle started 5 min ago; tick is just past its end.
        last_start = now - timedelta(minutes=5)
        engine = _make_engine_with("m5", [_candle(last_start)])

        tick = {
            "bid": 2000.0, "ask": 2000.5,
            "spread": 0.5, "time": now,
        }
        with patch("src.market_data.mt5_connector") as mock_conn:
            mock_conn.get_tick.return_value = tick
            with patch.object(
                engine, "sync_with_mt5"
            ) as mock_sync:
                engine.update_live_data()
                mock_sync.assert_not_called()
        # One new candle added (rollover, not gap).
        assert len(engine._candle_cache["m5"]) == 2


# ---------------------------------------------------------------------------
# sync_with_mt5 validation
# ---------------------------------------------------------------------------

class TestSyncValidation:

    def _ts(self, m):
        return datetime(
            2024, 1, 1, 0, m, tzinfo=timezone.utc
        )

    def _rate(self, ts, o=100.0, h=101.0, l=99.0, c=100.0):
        return {
            "time": ts, "open": o, "high": h, "low": l,
            "close": c, "tick_volume": 1, "spread": 0,
        }

    def test_valid_rates_accepted(self):
        engine = MarketDataEngine()
        rates = [
            self._rate(self._ts(0)),
            self._rate(self._ts(5)),
        ]
        with patch("src.market_data.mt5_connector") as mock_conn:
            mock_conn.get_rates.return_value = rates
            assert engine.sync_with_mt5("m5") is True
        assert len(engine._candle_cache["m5"]) == 2

    def test_duplicate_rates_rejected(self):
        engine = MarketDataEngine()
        ts = self._ts(0)
        rates = [self._rate(ts), self._rate(ts)]
        with patch("src.market_data.mt5_connector") as mock_conn:
            mock_conn.get_rates.return_value = rates
            assert engine.sync_with_mt5("m5") is False
        assert "m5" not in engine._candle_cache

    def test_out_of_order_rates_rejected(self):
        engine = MarketDataEngine()
        rates = [
            self._rate(self._ts(10)),
            self._rate(self._ts(5)),
        ]
        with patch("src.market_data.mt5_connector") as mock_conn:
            mock_conn.get_rates.return_value = rates
            assert engine.sync_with_mt5("m5") is False

    def test_gaps_accepted(self):
        engine = MarketDataEngine()
        rates = [
            self._rate(self._ts(0)),
            self._rate(self._ts(15)),  # 15 min gap
        ]
        with patch("src.market_data.mt5_connector") as mock_conn:
            mock_conn.get_rates.return_value = rates
            assert engine.sync_with_mt5("m5") is True