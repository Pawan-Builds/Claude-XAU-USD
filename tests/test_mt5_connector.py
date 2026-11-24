"""Tests for MT5 connector, risk manager, indicators, and signal validator.

All external dependencies (MT5, config, market_data, DB) are mocked.
"""

from pathlib import Path
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import numpy as np
import tempfile
from src.config_loader import config
from src.mt5_connector import AccountInfo, MT5Connector, OrderResult
from src.risk_manager import RiskManager
from src.signal_validator import SignalValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_mt5_constants(mock_mt5):
    """Copy MetaTrader5 integer constants onto the mocked module.

    Without this, comparisons like `trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO`
    compare an int to a Mock and silently evaluate False.
    """
    try:
        from src.mt5_connector import mt5 as real_mt5
        for attr in dir(real_mt5):
            if attr.isupper() and not attr.startswith("_"):
                try:
                    setattr(mock_mt5, attr, getattr(real_mt5, attr))
                except Exception:
                    pass
    except Exception:
        # Fallback: hardcoded MetaTrader5 constant values
        for name, value in {
            "ACCOUNT_TRADE_MODE_DEMO": 0,
            "ACCOUNT_TRADE_MODE_CONTEST": 1,
            "ACCOUNT_TRADE_MODE_REAL": 2,
            "ACCOUNT_MARGIN_MODE_RETAIL_NETTING": 0,
            "ACCOUNT_MARGIN_MODE_EXCHANGE": 1,
            "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING": 2,
            "SYMBOL_TRADE_MODE_DISABLED": 0,
            "SYMBOL_TRADE_MODE_LONGONLY": 1,
            "SYMBOL_TRADE_MODE_SHORTONLY": 2,
            "SYMBOL_TRADE_MODE_CLOSEONLY": 3,
            "SYMBOL_TRADE_MODE_FULL": 4,
            "TRADE_RETCODE_DONE": 10009,
            "TRADE_RETCODE_DONE_PARTIAL": 10010,
            "TRADE_RETCODE_PLACED": 10008,
            "TRADE_ACTION_DEAL": 1,
            "TRADE_ACTION_PENDING": 5,
            "TRADE_ACTION_SLTP": 6,
            "TRADE_ACTION_REMOVE": 8,
            "ORDER_TYPE_BUY": 0,
            "ORDER_TYPE_SELL": 1,
            "ORDER_TYPE_BUY_LIMIT": 2,
            "ORDER_TYPE_SELL_LIMIT": 3,
            "ORDER_FILLING_IOC": 1,
            "ORDER_FILLING_FOK": 0,
            "ORDER_FILLING_RETURN": 2,
            "ORDER_TIME_GTC": 0,
            "SYMBOL_FILLING_FOK": 1,
            "SYMBOL_FILLING_IOC": 2,
            "POSITION_TYPE_BUY": 0,
            "POSITION_TYPE_SELL": 1,
            "DEAL_TYPE_BUY": 0,
            "DEAL_TYPE_SELL": 1,
            "DEAL_TYPE_BALANCE": 2,
            "DEAL_TYPE_CREDIT": 3,
        }.items():
            setattr(mock_mt5, name, value)


def _make_symbol_mock(
    name="XAUUSD",
    trade_mode=4,           # SYMBOL_TRADE_MODE_FULL
    point=0.01,
    digits=2,
):
    m = Mock()
    m.name = name
    m.trade_mode = trade_mode
    m.point = point
    m.digits = digits
    m.trade_tick_size = point
    m.trade_tick_value = 1.0
    m.trade_contract_size = 100.0
    m.volume_min = 0.01
    m.volume_max = 100.0
    m.volume_step = 0.01
    m.swap_long = -1.0
    m.swap_short = -1.0
    m.margin_initial = 1000.0
    m.margin_maintenance = 500.0
    m.session_deals = 0
    m.session_buy_orders = 0
    m.session_sell_orders = 0
    m.volume_limit = 0
    m.filling_mode = 2  # IOC flag
    return m


def _make_tick_mock(bid=2000.0, ask=2000.5):
    t = Mock()
    t.bid = bid
    t.ask = ask
    t.last = ask
    t.volume = 100
    t.time = int(datetime.now(timezone.utc).timestamp())
    t.time_msc = t.time * 1000
    t.flags = 0
    return t


def _make_account_mock(
    login=12345,
    server="XMGlobal-MT5 5",
    trade_mode=0,          # ACCOUNT_TRADE_MODE_DEMO
    equity=10000.0,
    balance=10000.0,
    trade_allowed=True,
    trade_expert=True,
):
    return Mock(
        login=login,
        server=server,
        name="Test Account",
        company="XM",
        currency="USD",
        balance=balance,
        equity=equity,
        margin=0.0,
        margin_free=equity,
        margin_level=0.0,
        leverage=100,
        trade_mode=trade_mode,
        trade_allowed=trade_allowed,
        trade_expert=trade_expert,
        margin_mode=0,
    )


# ---------------------------------------------------------------------------
# MT5 Connector
# ---------------------------------------------------------------------------

class TestMT5Connector(unittest.TestCase):

    def setUp(self):
        self.connector = MT5Connector()

    # --- initialization ---

    @patch("src.mt5_connector.mt5")
    @patch("os.path.exists", return_value=True)
    def test_initialize_success(self, mock_os_path, mock_mt5):
        _apply_mt5_constants(mock_mt5)

        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = Mock()

        demo_login = int(config.mt5.login) if config.mt5.login else 12345
        demo_server = config.mt5.server or "XMGlobal-MT5 5"

        mock_mt5.account_info.return_value = _make_account_mock(
            login=demo_login,
            server=demo_server,
            trade_mode=mock_mt5.ACCOUNT_TRADE_MODE_DEMO,
        )

        symbol = _make_symbol_mock(
            name="XAUUSD",
            trade_mode=mock_mt5.SYMBOL_TRADE_MODE_FULL,
        )
        mock_mt5.symbols_get.return_value = [symbol]
        mock_mt5.symbol_info.return_value = symbol
        mock_mt5.symbol_info_tick.return_value = _make_tick_mock()
        mock_mt5.symbol_select.return_value = True

        result = self.connector.initialize()

        self.assertTrue(result)

    @patch("src.mt5_connector.mt5")
    @patch("os.path.exists", return_value=False)
    def test_initialize_terminal_missing(self, mock_os_path, mock_mt5):
        _apply_mt5_constants(mock_mt5)

        result = self.connector.initialize()

        self.assertFalse(result)
        self.assertIn("not found", self.connector.get_last_error() or "")

    @patch("src.mt5_connector.mt5")
    def test_verify_connection_not_initialized(self, mock_mt5):
        _apply_mt5_constants(mock_mt5)

        result = self.connector.verify_connection()

        self.assertFalse(result)

    # --- account-type safety ---

    @patch("src.mt5_connector.mt5")
    def test_real_account_rejected_when_demo_only(self, mock_mt5):
        if not config.safety.demo_only:
            self.skipTest("demo_only is disabled in config")

        _apply_mt5_constants(mock_mt5)

        real_login = int(config.mt5.login) if config.mt5.login else 12345
        real_server = config.mt5.server or "XMGlobal-MT5 5"

        self.connector._initialized = True
        mock_mt5.terminal_info.return_value = Mock()
        mock_mt5.account_info.return_value = _make_account_mock(
            login=real_login,
            server=real_server,
            trade_mode=mock_mt5.ACCOUNT_TRADE_MODE_REAL,
        )

        result = self.connector.verify_connection()

        self.assertFalse(result)
        err = (self.connector.get_last_error() or "").lower()
    # The new mode<->account-type gate fires before demo_only, so accept
    # either message; both correctly refuse a real account in DEMO mode.
        self.assertTrue(
            "real" in err or "mismatch" in err,
            f"Unexpected rejection reason: {err!r}",
        )

    @patch("src.mt5_connector.mt5")
    def test_live_mode_rejects_demo_account(self, mock_mt5):
        """LIVE mode must never accept a DEMO account."""
        _apply_mt5_constants(mock_mt5)

        real_login = int(config.mt5.login) if config.mt5.login else 12345
        real_server = config.mt5.server or "XMGlobal-MT5 5"

        self.connector._initialized = True
        mock_mt5.terminal_info.return_value = Mock()
        mock_mt5.account_info.return_value = _make_account_mock(
            login=real_login,
            server=real_server,
            trade_mode=mock_mt5.ACCOUNT_TRADE_MODE_DEMO,
        )

        original_mode = config.trading.mode

        with patch.object(
            type(config), "trading_mode", return_value="LIVE"
        ), patch.object(
            type(config),
            "trading",
            new_callable=lambda: Mock(
                mode="LIVE", symbol="XAUUSD",
                risk_percent=1.0, min_risk_reward=2.0,
                magic_number=123456789, max_positions=3,
                max_daily_trades=20, max_daily_loss_percent=5.0,
                max_drawdown_percent=10.0, max_consecutive_losses=5,
                max_holding_hours=48, entry_order_type="limit",
                pending_expiry_hours=4, max_pending_orders=5,
            ),
        ):
            result = self.connector.verify_connection()

        self.assertFalse(result)
        err = (self.connector.get_last_error() or "").lower()
        self.assertTrue(
            "live" in err and "demo" in err,
            f"LIVE mode should reject a DEMO account, got: {err!r}",
        )

    # --- order submission ---

    def _wire_connected_connector(self, mock_mt5):
        """Set up connector + mock_mt5 so _trading_precheck passes."""
        _apply_mt5_constants(mock_mt5)

    # Use the actual configured credentials so the connector's
    # new account-identity checks (login + server + mode) pass.
        real_login = int(config.mt5.login) if config.mt5.login else 12345
        real_server = config.mt5.server or "XMGlobal-MT5 5"

        self.connector._initialized = True
        self.connector._symbol_name = "XAUUSD"
        self.connector._account_info = AccountInfo(
            login=real_login,
            server=real_server,
            name="Test",
            company="XM",
            currency="USD",
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            free_margin=10000.0,
            margin_level=0.0,
            leverage=100,
            trade_mode="DEMO",
            trade_allowed=True,
            trade_expert=True,
            margin_mode="RETAIL_NETTING",
            is_demo=True,
        )

        mock_mt5.terminal_info.return_value = Mock()
        mock_mt5.account_info.return_value = _make_account_mock(
            login=real_login,
            server=real_server,
            trade_mode=mock_mt5.ACCOUNT_TRADE_MODE_DEMO,
        )
        mock_mt5.symbol_info.return_value = _make_symbol_mock()
        mock_mt5.symbol_info_tick.return_value = _make_tick_mock()

    @patch("src.mt5_connector.mt5")
    def test_send_order_success(self, mock_mt5):
        self._wire_connected_connector(mock_mt5)

        mock_result = Mock()
        mock_result.retcode = mock_mt5.TRADE_RETCODE_DONE
        mock_result.order = 11111
        mock_result.deal = 22222
        mock_result.volume = 0.1
        mock_result.price = 2000.5
        mock_result.comment = "Done"
        mock_mt5.order_send.return_value = mock_result

        request = {
            "action": mock_mt5.TRADE_ACTION_DEAL,
            "symbol": "XAUUSD",
            "type": mock_mt5.ORDER_TYPE_BUY,
            "volume": 0.1,
            "price": 2000.5,
            "sl": 1990.0,
            "tp": 2020.0,
            "magic": 123456789,
            "comment": "TEST",
        }

        result = self.connector.send_order(request)

        self.assertTrue(result.success)
        self.assertEqual(result.ticket, 11111)

    @patch("src.mt5_connector.mt5")
    def test_send_order_rejected(self, mock_mt5):
        self._wire_connected_connector(mock_mt5)

        mock_result = Mock()
        mock_result.retcode = 10015
        mock_result.order = None
        mock_result.deal = None
        mock_result.volume = 0.1
        mock_result.price = 0.0
        mock_result.comment = "Insufficient funds"
        mock_mt5.order_send.return_value = mock_result

        request = {
            "action": mock_mt5.TRADE_ACTION_DEAL,
            "symbol": "XAUUSD",
            "type": mock_mt5.ORDER_TYPE_BUY,
            "volume": 0.1,
            "price": 2000.5,
            "magic": 123456789,
        }

        result = self.connector.send_order(request)

        self.assertFalse(result.success)
        self.assertEqual(result.retcode, 10015)
        self.assertEqual(result.error_code, 10015)


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class TestRiskManager(unittest.TestCase):

    def setUp(self):
        self.risk_manager = RiskManager()

    @staticmethod
    def _symbol_info(volume_min=0.01, volume_max=100.0, volume_step=0.01):
        return Mock(
            point=0.01,
            digits=2,
            tick_size=0.01,
            tick_value=1.0,
            contract_size=100.0,
            volume_min=volume_min,
            volume_max=volume_max,
            volume_step=volume_step,
            margin_initial=0.0,
        )

    @patch("src.risk_manager.mt5_connector")
    def test_calculate_position_size_buy(self, mock_mt5):
        mock_mt5.is_connected.return_value = True
        mock_mt5.get_account_info.return_value = Mock(
            equity=10000.0,
            free_margin=10000.0,
            leverage=100
        )
        mock_mt5.get_symbol_info.return_value = self._symbol_info()
        mock_mt5.get_positions.return_value = []

        result = self.risk_manager.calculate_position_size(
            entry_price=2000.0,
            sl_price=1990.0,
            direction="BUY",
            risk_percent=2.0,
        )

        self.assertGreater(result.lot_size, 0)
        self.assertGreater(result.risk_amount, 0)
        self.assertAlmostEqual(result.lot_size, 0.2)

    @patch("src.risk_manager.mt5_connector")
    def test_calculate_position_size_sell(self, mock_mt5):
        mock_mt5.is_connected.return_value = True
        mock_mt5.get_account_info.return_value = Mock(
            equity=10000.0,
            free_margin=10000.0,
            leverage=100,
        )
        mock_mt5.get_symbol_info.return_value = self._symbol_info()
        mock_mt5.get_positions.return_value = []

        result = self.risk_manager.calculate_position_size(
            entry_price=2000.0,
            sl_price=2010.0,
            direction="SELL",
            risk_percent=2.0,
        )

        self.assertGreater(result.lot_size, 0)
        self.assertGreater(result.risk_amount, 0)

    @patch("src.risk_manager.mt5_connector")
    def test_min_lot_exceeds_risk(self, mock_mt5):
        mock_mt5.is_connected.return_value = True
        mock_mt5.get_account_info.return_value = Mock(equity=100.0)
        mock_mt5.get_symbol_info.return_value = self._symbol_info(volume_min=1.0)
        mock_mt5.get_positions.return_value = []

        result = self.risk_manager.calculate_position_size(
            entry_price=2000.0,
            sl_price=1990.0,
            direction="BUY",
            risk_percent=2.0,
        )

        self.assertEqual(result.lot_size, 0.0)
        self.assertIn("below broker minimum", result.reason.lower())


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

class TestIndicators(unittest.TestCase):

    def test_detect_fvgs(self):
        from src.indicators import detect_fvgs

        # Valid OHLC where a bullish FVG exists at index 3
        # (lows[3]=104 > highs[1]=103).
        opens = np.array([100.0, 102.0, 101.0, 105.0, 106.0])
        highs = np.array([101.5, 103.0, 102.5, 106.0, 107.0])
        lows = np.array([ 99.0, 100.5,  99.5, 104.0, 105.0])
        closes = np.array([100.5, 102.5, 101.5, 105.5, 106.5])

        fvgs = detect_fvgs(
            opens, highs, lows, closes,
            min_size_pips=1, point_value=0.01,
        )

        self.assertGreaterEqual(len(fvgs), 1)
        self.assertTrue(any(f.type == "bullish" for f in fvgs))

    def test_detect_order_blocks(self):
        from src.indicators import detect_order_blocks

        opens = np.array([100.0, 102.0, 101.0, 99.0, 105.0, 107.0, 110.0])
        highs = np.array([101.0, 103.0, 102.0, 100.0, 106.0, 108.0, 111.0])
        lows = np.array([ 99.0, 101.0, 100.0,  98.0, 104.0, 106.0, 109.0])
        closes = np.array([100.5, 102.5, 100.0, 98.5, 105.5, 107.5, 110.5])

        obs = detect_order_blocks(
            opens, highs, lows, closes,
            lookback=2, displacement_threshold=1.0,
        )

        bullish_obs = [ob for ob in obs if ob.type == "bullish"]
        self.assertGreaterEqual(len(bullish_obs), 1)

    def test_analyze_market_structure_bullish(self):
        from src.indicators import analyze_market_structure

        # Swing highs at indices 2 (105) and 5 (108) → higher highs
        # Swing lows  at indices 3 (97)  and 6 (98)  → higher lows
        highs = np.array([100.0, 101.0, 105.0, 102.0, 103.0, 108.0, 104.0, 105.0, 112.0, 110.0])
        lows = np.array([ 99.0,  98.0, 100.0,  97.0,  99.0, 101.0,  98.0, 100.0, 103.0, 102.0])
        closes = np.array([ 99.5,  99.0, 104.0,  98.0, 100.0, 107.0, 100.0, 103.0, 111.0, 108.0])

        structure = analyze_market_structure(
            highs, lows, closes, lookback=10, swing_lookback=2,
        )

        self.assertEqual(structure.trend, "bullish")

    def test_analyze_market_structure_bearish(self):
        from src.indicators import analyze_market_structure

        highs = np.array([
            110.0, 109.0, 112.0, 108.0, 106.0,
            109.0, 107.0, 105.0, 103.0, 102.0
        ])

        lows = np.array([
            108.0, 107.0, 109.0, 104.0, 105.0,
            107.0, 100.0, 101.0, 102.0, 99.0
        ])

        closes = np.array([
            109.0, 108.0, 110.0, 105.0, 105.5,
            108.0, 104.0, 102.0, 102.5, 100.0
        ])

        structure = analyze_market_structure(
            highs, lows, closes, lookback=10, swing_lookback=2,
        )

        self.assertEqual(structure.trend, "bearish")

    def test_calculate_premium_discount(self):
        from src.indicators import calculate_premium_discount

        self.assertEqual(
            calculate_premium_discount(2000.0, 1900.0, 1980.0)["zone"],
            "premium",
        )
        self.assertEqual(
            calculate_premium_discount(2000.0, 1900.0, 1920.0)["zone"],
            "discount",
        )
        self.assertEqual(
            calculate_premium_discount(2000.0, 1900.0, 1950.0)["zone"],
            "equilibrium",
        )


# ---------------------------------------------------------------------------
# Signal Validator
# ---------------------------------------------------------------------------

class TestSignalValidator(unittest.TestCase):

    @patch("src.signal_validator.mt5_connector")
    @patch("src.signal_validator.market_data")
    def test_validate_signal_success(self, mock_market, mock_mt5):
        from src.smc_strategy import (
            Confirmation,
            SignalDirection,
            StrategyType,
            TradeSignal,
        )

        # ---- Configure mocks BEFORE creating validator so that
        # SignalValidator.__init__ captures the mocked config values. ----
        mock_mt5.is_connected.return_value = True
        mock_mt5.get_account_info.return_value = Mock(
            is_demo=True,
            trade_allowed=True,
            trade_expert=True,
            equity=10000.0,
            balance=10000.0,
        )
        mock_mt5.get_positions.return_value = []
        mock_mt5.get_symbol_info.return_value = Mock(
            point=0.01,
            digits=2,
            tick_size=0.01,
            tick_value=1.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
        )
        mock_market.is_high_spread.return_value = False
        mock_market.is_trading_session.return_value = True
        mock_market.is_weekend_protection_active.return_value = False
        mock_market.get_current_tick.return_value = {"bid": 2000.0, "ask": 2000.1}

        validator = SignalValidator()

        # entry=1999.5 (below ask=2000.1 → valid BUY_LIMIT),
        # sl=1990.0 (risk=9.5), tp=2020.5 (reward=21.0), RR=2.21 > 2.0
        signal = TradeSignal(
            timestamp=datetime.now(timezone.utc),
            symbol="XAUUSD",
            direction=SignalDirection.BUY,
            entry_price=1999.5,
            sl_price=1990.0,
            tp_price=2020.5,
            lot_size=0.1,
            risk_percent=1.0,
            risk_reward=2.21,
            strategy=StrategyType.LIQUIDITY_SWEEP_FVG,
            confirmations=[Confirmation("Test", True, "OK")],
            timeframe_analysis={},
            entry_tf="m5",
            setup_tf="m15",
            structure_tf="h1",
            bias_tf="h4",
        )

        # Patch every check that hits the database or the real config.
        with patch.object(validator, "_check_daily_trade_limit", return_value=True), \
             patch.object(validator, "_check_daily_loss_limit", return_value=True), \
             patch.object(validator, "_check_max_drawdown", return_value=True), \
             patch.object(validator, "_check_consecutive_losses", return_value=True), \
             patch.object(validator, "_check_total_open_risk", return_value=True), \
             patch.object(validator, "_check_news_filter", return_value=True):
            result = validator.validate_signal(signal)

        if not result.valid:
            print(f"Validation failed: {result.reason}")
            print(f"Checks: {result.checks}")

        self.assertTrue(result.valid, f"Validation failed: {result.reason}")

    # ---------------------------------------------------------------------------
# Database migration
# ---------------------------------------------------------------------------

class TestDatabaseMigration(unittest.TestCase):

    def test_create_tables_is_idempotent(self):
        """FIX #27: create_tables must run twice on the same DB."""
        import tempfile
        from pathlib import Path
        from src.database_manager import DatabaseManager

        with tempfile.TemporaryDirectory() as tmp:
            db_url = f"sqlite:///{Path(tmp) / 'test.db'}"

            manager = DatabaseManager(database_url=db_url)

            # First call: full create + migrate.
            manager.create_tables()

            # Second call: must be a no-op, not ResourceClosedError.
            manager.create_tables()

            # Third call, just to be sure.
            manager.create_tables()

            manager.close()


if __name__ == "__main__":
    unittest.main()