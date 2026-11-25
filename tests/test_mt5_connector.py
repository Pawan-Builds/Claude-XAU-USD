"""Tests for MT5 connector, risk manager, indicators, and signal validator.

All external dependencies (MT5, config, market_data, DB) are mocked.
"""

from pathlib import Path
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import json
from unittest.mock import Mock, patch
from src.database_manager import DatabaseManager, AppSettings
from src.trade_manager import TradeManager
import os
import tempfile
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from sqlalchemy import text
from src.database_manager import DatabaseManager
from unittest.mock import Mock, patch
from src.database_manager import DatabaseManager, TradePlan
from src.smc_strategy import SMCEngine
import numpy as np
import tempfile
from src import gate_token
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
        gate_token.reset_for_tests()

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

        # C-03 residual: opening actions require a gate token. Tests that
        # call the connector directly must issue one, exactly as
        # OrderExecutor.execute_signal would.
        result = self.connector.send_order(
            request, gate_token=gate_token.issue()
        )

        self.assertTrue(result.success)
        self.assertEqual(result.ticket, 11111)

    @patch("src.mt5_connector.mt5")
    def test_send_order_rejected(self, mock_mt5):
        self._wire_connected_connector(mock_mt5)
        gate_token.reset_for_tests()

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

        result = self.connector.send_order(
            request, gate_token=gate_token.issue()
        )

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

# ---------------------------------------------------------------------------
# H-01: Account-wide exposure
# ---------------------------------------------------------------------------

class TestH01ConnectorAccountWide(unittest.TestCase):

    @patch("src.mt5_connector.mt5")
    def test_get_all_positions_includes_manual_and_other_symbols(
        self, mock_mt5
    ):
        _apply_mt5_constants(mock_mt5)
        connector = MT5Connector()

        manual = Mock(
            ticket=1, magic=0, symbol="EURUSD",
            type=mock_mt5.POSITION_TYPE_BUY,
            volume=1.0, price_open=1.1, price_current=1.1,
            sl=1.09, tp=1.2, swap=0.0, profit=10.0,
            commission=0.0, comment="manual",
            time=1_700_000_000, time_update=1_700_000_000,
        )
        other_ea = Mock(
            ticket=2, magic=999999, symbol="GBPUSD",
            type=mock_mt5.POSITION_TYPE_SELL,
            volume=0.5, price_open=1.3, price_current=1.3,
            sl=1.31, tp=1.28, swap=0.0, profit=-5.0,
            commission=0.0, comment="other-ea",
            time=1_700_000_000, time_update=1_700_000_000,
        )
        mock_mt5.positions_get.return_value = (manual, other_ea)

        result = connector.get_all_positions()

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(
            {p["symbol"] for p in result}, {"EURUSD", "GBPUSD"}
        )
        # Bot-scoped get_positions must remain bot-only:
        connector._symbol_name = "XAUUSD"
        bot_scoped = connector.get_positions()
        self.assertEqual(bot_scoped, [])

    @patch("src.mt5_connector.mt5")
    def test_get_all_orders_includes_manual(self, mock_mt5):
        _apply_mt5_constants(mock_mt5)
        connector = MT5Connector()
        manual = Mock(
            ticket=10, magic=0, symbol="XAUUSD",
            type=mock_mt5.ORDER_TYPE_BUY_LIMIT,
            volume_initial=0.1, price_open=1990.0,
            sl=1980.0, tp=2010.0, comment="manual",
            time_setup=1_700_000_000, time_expiration=0,
        )
        mock_mt5.orders_get.return_value = (manual,)

        result = connector.get_all_orders()

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["magic"], 0)

    @patch("src.mt5_connector.mt5")
    def test_get_all_positions_returns_none_on_query_failure(
        self, mock_mt5
    ):
        _apply_mt5_constants(mock_mt5)
        connector = MT5Connector()
        mock_mt5.positions_get.return_value = None
        mock_mt5.last_error.return_value = (10004, "No connection")

        self.assertIsNone(connector.get_all_positions())

    @patch("src.mt5_connector.mt5")
    def test_get_all_orders_returns_none_on_query_failure(
        self, mock_mt5
    ):
        _apply_mt5_constants(mock_mt5)
        connector = MT5Connector()
        mock_mt5.orders_get.return_value = None
        mock_mt5.last_error.return_value = (10004, "No connection")

        self.assertIsNone(connector.get_all_orders())

    @patch("src.mt5_connector.mt5")
    def test_get_symbol_specs_returns_none_for_unknown_symbol(
        self, mock_mt5
    ):
        _apply_mt5_constants(mock_mt5)
        connector = MT5Connector()
        mock_mt5.symbol_select.return_value = True
        mock_mt5.symbol_info.return_value = None
        mock_mt5.last_error.return_value = (4301, "Unknown symbol")

        self.assertIsNone(connector.get_symbol_specs("NOPE"))

    @patch("src.mt5_connector.mt5")
    def test_get_symbol_specs_returns_none_for_invalid_tick_value(
        self, mock_mt5
    ):
        _apply_mt5_constants(mock_mt5)
        connector = MT5Connector()
        info = Mock(
            trade_tick_size=0.01, trade_tick_value=0.0,
            point=0.01, trade_contract_size=100.0,
        )
        mock_mt5.symbol_select.return_value = True
        mock_mt5.symbol_info.return_value = info

        self.assertIsNone(connector.get_symbol_specs("XAUUSD"))


class TestH01RiskAccountWide(unittest.TestCase):

    def setUp(self):
        self.risk_manager = RiskManager()

    @staticmethod
    def _specs(symbol, tick_size, tick_value):
        return {
            "symbol": symbol,
            "point": tick_size,
            "tick_size": tick_size,
            "tick_value": tick_value,
            "contract_size": 100.0,
        }

    @patch("src.risk_manager.mt5_connector")
    def test_check_total_risk_includes_manual_other_symbol(
        self, mock_mt5
    ):
        """A manual EURUSD BUY 1.0 with SL 100 pips away and
        tick_value=10 contributes $1000 risk. With equity=10,000 and
        the default limit (assumed 5%), the check must return False,
        proving the manual position was counted."""
        mock_mt5.get_all_positions.return_value = [{
            "ticket": 1, "magic": 0, "symbol": "EURUSD",
            "type": "BUY", "volume": 1.0,
            "price_open": 1.1000, "sl": 1.0900,
        }]
        mock_mt5.get_all_orders.return_value = []
        mock_mt5.get_symbol_specs.side_effect = (
            lambda s: self._specs("EURUSD", 0.0001, 10.0)
        )
        self.assertFalse(
            self.risk_manager._check_total_risk(10_000.0, [
                {"ticket": 1, "magic": 0, "symbol": "EURUSD",
                 "type": "BUY", "volume": 1.0,
                 "price_open": 1.1000, "sl": 1.0900},
            ])
        )

    @patch("src.risk_manager.mt5_connector")
    def test_check_total_risk_fails_closed_on_malformed_position(
        self, mock_mt5
    ):
        """A position record with non-numeric volume must make
        _check_total_risk fail closed, not raise and not pass."""
        mock_mt5.get_all_orders.return_value = []
        mock_mt5.get_symbol_specs.side_effect = (
            lambda s: self._specs("EURUSD", 0.0001, 10.0)
        )
        positions = [{
            "ticket": 1, "symbol": "EURUSD", "type": "BUY",
            "volume": "not-a-number",
            "price_open": 1.1, "sl": 1.09,
        }]
        self.assertFalse(
            self.risk_manager._check_total_risk(10_000.0, positions)
        )

    @patch("src.risk_manager.mt5_connector")
    def test_check_total_risk_fails_closed_on_order_query_failure(
        self, mock_mt5
    ):
        mock_mt5.get_all_positions.return_value = []
        mock_mt5.get_all_orders.return_value = None
        self.assertFalse(
            self.risk_manager._check_total_risk(10_000.0, [])
        )

    @patch("src.risk_manager.mt5_connector")
    def test_check_total_risk_fails_closed_on_missing_symbol_specs(
        self, mock_mt5
    ):
        mock_mt5.get_symbol_specs.return_value = None
        mock_mt5.get_all_orders.return_value = []
        positions = [{
            "ticket": 1, "symbol": "ZZZ", "type": "BUY",
            "volume": 1.0, "price_open": 1.0, "sl": 0.9,
        }]
        self.assertFalse(
            self.risk_manager._check_total_risk(10_000.0, positions)
        )

    @patch("src.risk_manager.mt5_connector")
    def test_check_total_risk_fails_closed_on_position_without_sl(
        self, mock_mt5
    ):
        mock_mt5.get_all_orders.return_value = []
        mock_mt5.get_symbol_specs.side_effect = (
            lambda s: self._specs("EURUSD", 0.0001, 10.0)
        )
        positions = [{
            "ticket": 1, "symbol": "EURUSD", "type": "BUY",
            "volume": 1.0, "price_open": 1.1, "sl": 0.0,
        }]
        self.assertFalse(
            self.risk_manager._check_total_risk(10_000.0, positions)
        )

    @patch("src.risk_manager.mt5_connector")
    def test_check_risk_limits_uses_account_wide_positions(
        self, mock_mt5
    ):
        mock_mt5.is_connected.return_value = True
        mock_mt5.get_account_info.return_value = Mock(
            equity=10_000.0, balance=10_000.0
        )
        mock_mt5.get_all_positions.return_value = [{"ticket": 1}]
        mock_mt5.get_all_orders.return_value = []
        mock_mt5.get_symbol_specs.return_value = None

        # The account-wide call is the one that must be exercised.
        self.risk_manager.check_risk_limits()
        mock_mt5.get_all_positions.assert_called_once()

    @patch("src.risk_manager.mt5_connector")
    def test_check_risk_limits_fails_closed_when_positions_unavailable(
        self, mock_mt5
    ):
        mock_mt5.is_connected.return_value = True
        mock_mt5.get_account_info.return_value = Mock(
            equity=10_000.0, balance=10_000.0
        )
        mock_mt5.get_all_positions.return_value = None

        result = self.risk_manager.check_risk_limits()
        self.assertFalse(result["ok"])
        self.assertIn("account-wide", result["reason"].lower())

class TestH02DailyBaseline(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".db"
        )
        self._tmp.close()
        self.db = DatabaseManager(
            database_url=f"sqlite:///{self._tmp.name}"
        )
        self.db.create_tables()
        self._db_patch = patch(
            "src.risk_manager.db_manager", self.db
        )
        self._db_patch.start()
        self.rm = RiskManager()

    def tearDown(self):
        self._db_patch.stop()
        try:
            self.db.close()
        except Exception:
            pass
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    @staticmethod
    def _deal(profit=0.0, swap=0.0, commission=0.0):
        return Mock(profit=profit, swap=swap, commission=commission)

    def test_broker_day_start_default_is_utc_midnight(self):
        start = self.rm._get_broker_day_start_utc()
        self.assertIsNotNone(start)
        self.assertEqual(start.hour, 0)
        self.assertEqual(start.minute, 0)
        self.assertEqual(start.tzinfo, timezone.utc)

    def test_broker_day_start_respects_rollover_hour(self):
        patched_cfg = Mock(broker_day_rollover_hour_utc=22,
                           daily_loss_limit_percent=5.0)
        with patch.object(self.rm, "risk_config", patched_cfg):
            start = self.rm._get_broker_day_start_utc()
        self.assertIsNotNone(start)
        self.assertEqual(start.hour, 22)

    def test_broker_day_start_rejects_bad_config(self):
        bad = Mock(broker_day_rollover_hour_utc=99)
        with patch.object(self.rm, "risk_config", bad):
            self.assertIsNone(self.rm._get_broker_day_start_utc())

    @patch("src.mt5_connector.mt5")
    @patch("src.risk_manager.mt5_connector")
    def test_reconstruct_no_deals_returns_current_balance(
        self, mock_conn, mock_mt5
    ):
        mock_conn.get_account_info.return_value = Mock(
            balance=10_000.0
        )
        mock_mt5.history_deals_get.return_value = ()
        day_start = datetime(2025, 11, 25, tzinfo=timezone.utc)
        result = self.rm._reconstruct_opening_balance_from_history(
            day_start
        )
        self.assertEqual(result, 10_000.0)

    @patch("src.mt5_connector.mt5")
    @patch("src.risk_manager.mt5_connector")
    def test_reconstruct_offline_losses(self, mock_conn, mock_mt5):
        # Bot was offline 08:00-15:00; three SL hits, $900 lost.
        mock_conn.get_account_info.return_value = Mock(
            balance=9_100.0
        )
        mock_mt5.history_deals_get.return_value = (
            self._deal(profit=-300.0),
            self._deal(profit=-300.0),
            self._deal(profit=-300.0),
        )
        day_start = datetime(2025, 11, 25, tzinfo=timezone.utc)
        result = self.rm._reconstruct_opening_balance_from_history(
            day_start
        )
        self.assertEqual(result, 10_000.0)

    @patch("src.mt5_connector.mt5")
    @patch("src.risk_manager.mt5_connector")
    def test_reconstruct_fails_closed_on_history_none(
        self, mock_conn, mock_mt5
    ):
        mock_conn.get_account_info.return_value = Mock(
            balance=10_000.0
        )
        mock_mt5.history_deals_get.return_value = None
        day_start = datetime(2025, 11, 25, tzinfo=timezone.utc)
        self.assertIsNone(
            self.rm._reconstruct_opening_balance_from_history(
                day_start
            )
        )

    @patch("src.mt5_connector.mt5")
    @patch("src.risk_manager.mt5_connector")
    def test_reconstruct_fails_closed_on_bad_deal_field(
        self, mock_conn, mock_mt5
    ):
        mock_conn.get_account_info.return_value = Mock(
            balance=10_000.0
        )
        bad_deal = Mock()
        bad_deal.profit = "garbage"
        bad_deal.swap = 0.0
        bad_deal.commission = 0.0
        mock_mt5.history_deals_get.return_value = (bad_deal,)
        day_start = datetime(2025, 11, 25, tzinfo=timezone.utc)
        self.assertIsNone(
            self.rm._reconstruct_opening_balance_from_history(
                day_start
            )
        )

    @patch("src.mt5_connector.mt5")
    @patch("src.risk_manager.mt5_connector")
    def test_baseline_persisted_and_reused_across_restart(
        self, mock_conn, mock_mt5
    ):
        mock_conn.get_account_info.return_value = Mock(
            balance=9_100.0
        )
        mock_mt5.history_deals_get.return_value = (
            self._deal(profit=-900.0),
        )
        day_start = datetime(2025, 11, 25, tzinfo=timezone.utc)

        first = self.rm._get_or_establish_daily_baseline(day_start)
        self.assertEqual(first, 10_000.0)

        # Simulate restart: fresh RiskManager, same db_manager.
        rm2 = RiskManager()
        mock_mt5.history_deals_get.reset_mock()
        second = rm2._get_or_establish_daily_baseline(day_start)
        self.assertEqual(second, 10_000.0)
        mock_mt5.history_deals_get.assert_not_called()

    @patch("src.mt5_connector.mt5")
    @patch("src.risk_manager.mt5_connector")
    def test_baseline_new_day_recomputes(
        self, mock_conn, mock_mt5
    ):
        # Day 1: establish 10,000 baseline.
        mock_conn.get_account_info.return_value = Mock(
            balance=10_000.0
        )
        mock_mt5.history_deals_get.return_value = ()
        day1 = datetime(2025, 11, 25, tzinfo=timezone.utc)
        self.rm._get_or_establish_daily_baseline(day1)

        # Day 2: baseline must be rebuilt, not reused.
        mock_conn.get_account_info.return_value = Mock(
            balance=9_500.0
        )
        mock_mt5.history_deals_get.return_value = (
            self._deal(profit=-500.0),
        )
        day2 = datetime(2025, 11, 26, tzinfo=timezone.utc)
        result = self.rm._get_or_establish_daily_baseline(day2)
        self.assertEqual(result, 10_000.0)

    def test_check_daily_loss_uses_new_baseline(self):
        cfg = Mock(daily_loss_limit_percent=5.0)
        with (
            patch.object(
                self.rm, "_get_broker_day_start_utc",
                return_value=datetime(2025, 11, 25, tzinfo=timezone.utc),
            ),
            patch.object(
                self.rm, "_get_or_establish_daily_baseline",
                return_value=10_000.0,
            ),
            patch.object(self.rm, "risk_config", cfg),
        ):
            self.assertTrue(self.rm._check_daily_loss(9_900.0))
            self.assertFalse(self.rm._check_daily_loss(9_000.0))

    def test_check_daily_loss_fails_closed_on_baseline_failure(self):
        cfg = Mock(daily_loss_limit_percent=5.0)
        with (
            patch.object(
                self.rm, "_get_broker_day_start_utc",
                return_value=datetime(2025, 11, 25, tzinfo=timezone.utc),
            ),
            patch.object(
                self.rm, "_get_or_establish_daily_baseline",
                return_value=None,
            ),
            patch.object(self.rm, "risk_config", cfg),
        ):
            self.assertFalse(self.rm._check_daily_loss(10_000.0))

    def test_check_daily_loss_fails_closed_on_bad_limit(self):
        cfg = Mock(daily_loss_limit_percent=0)
        with (
            patch.object(
                self.rm, "_get_broker_day_start_utc",
                return_value=datetime(2025, 11, 25, tzinfo=timezone.utc),
            ),
            patch.object(
                self.rm, "_get_or_establish_daily_baseline",
                return_value=10_000.0,
            ),
            patch.object(self.rm, "risk_config", cfg),
        ):
            self.assertFalse(self.rm._check_daily_loss(10_000.0))



# ---------------------------------------------------------------------------
# H-03: structured setup_state persistence and full reconfirmation
# ---------------------------------------------------------------------------


class TestH03Reconfirmation(unittest.TestCase):

    def setUp(self):
        self.engine = SMCEngine()

    def _plan(self, strategy, direction, entry, sl, tp, setup_state):
        return TradePlan(
            timestamp=datetime.now(timezone.utc),
            symbol="XAUUSD",
            direction=direction,
            strategy=strategy,
            entry_price=entry,
            sl_price=sl,
            tp_price=tp,
            risk_reward=2.0,
            status="WAITING",
            expiry_time=datetime.now(timezone.utc),
            environment="demo",
            setup_state=setup_state,
        )

    # --- common precondition: pre-H-03 plan ---

    def test_no_setup_state_fails_closed(self):
        plan = TradePlan(
            timestamp=datetime.now(timezone.utc),
            symbol="XAUUSD",
            direction="BUY",
            strategy="liquidity_sweep_fvg",
            entry_price=2000.0, sl_price=1990.0, tp_price=2020.0,
            risk_reward=2.0, status="WAITING",
            expiry_time=datetime.now(timezone.utc),
            environment="demo", setup_state=None,
        )
        ok, reason = self.engine.reconfirm_plan(plan)
        self.assertFalse(ok)
        self.assertIn("setup_state", reason.lower())

    def test_empty_setup_state_fails_closed(self):
        plan = self._plan(
            "liquidity_sweep_fvg", "BUY",
            2000.0, 1990.0, 2020.0, {},
        )
        ok, reason = self.engine.reconfirm_plan(plan)
        self.assertFalse(ok)

    # --- H4 bias gate ---

    @patch("src.smc_strategy.market_structure")
    def test_h4_bias_flip_fails(self, mock_ms):
        mock_ms.get_higher_tf_bias.return_value = "bearish"
        mock_ms.get_structure_tf_bias.return_value = "bearish"
        plan = self._plan(
            "liquidity_sweep_fvg", "BUY",
            2000.0, 1990.0, 2020.0,
            {"fvg": {"type": "bullish", "top": 2001.0, "bottom": 2000.0}},
        )
        ok, reason = self.engine.reconfirm_plan(plan)
        self.assertFalse(ok)
        self.assertIn("h4", reason.lower())

    # --- liquidity_sweep_fvg ---

    @patch("src.smc_strategy.liquidity")
    @patch("src.smc_strategy.fair_value_gaps")
    @patch("src.smc_strategy.market_structure")
    def test_sweep_fvg_all_valid(self, mock_ms, mock_fvg, mock_liq):
        mock_ms.get_higher_tf_bias.return_value = "bullish"
        mock_ms.get_structure_tf_bias.return_value = "bullish"

        fvg = Mock(top=2001.0, bottom=2000.0)
        mock_fvg.get_active_bullish_fvgs.return_value = [fvg]

        level = Mock(price=1995.0)
        liq_state = Mock(sell_side=[level], buy_side=[])
        mock_liq.get_liquidity.return_value = liq_state

        plan = self._plan(
            "liquidity_sweep_fvg", "BUY",
            2000.0, 1990.0, 2020.0,
            {
                "fvg": {"type": "bullish", "top": 2001.0, "bottom": 2000.0},
                "sweep": {"side": "sell_side", "price": 1995.0},
            },
        )
        ok, reason = self.engine.reconfirm_plan(plan)
        self.assertTrue(ok, reason)

    @patch("src.smc_strategy.liquidity")
    @patch("src.smc_strategy.fair_value_gaps")
    @patch("src.smc_strategy.market_structure")
    def test_sweep_fvg_identity_mismatch_fails(
        self, mock_ms, mock_fvg, mock_liq
    ):
        mock_ms.get_higher_tf_bias.return_value = "bullish"
        mock_ms.get_structure_tf_bias.return_value = "bullish"

        # A DIFFERENT FVG is now present (price range shifted)
        different_fvg = Mock(top=2005.0, bottom=2004.0)
        mock_fvg.get_active_bullish_fvgs.return_value = [different_fvg]

        level = Mock(price=1995.0)
        mock_liq.get_liquidity.return_value = Mock(sell_side=[level])

        plan = self._plan(
            "liquidity_sweep_fvg", "BUY",
            2000.0, 1990.0, 2020.0,
            {
                "fvg": {"type": "bullish", "top": 2001.0, "bottom": 2000.0},
                "sweep": {"side": "sell_side", "price": 1995.0},
            },
        )
        ok, reason = self.engine.reconfirm_plan(plan)
        self.assertFalse(ok)
        self.assertIn("fvg", reason.lower())

    @patch("src.smc_strategy.liquidity")
    @patch("src.smc_strategy.fair_value_gaps")
    @patch("src.smc_strategy.market_structure")
    def test_sweep_fvg_missing_sweep_level_fails(
        self, mock_ms, mock_fvg, mock_liq
    ):
        mock_ms.get_higher_tf_bias.return_value = "bullish"
        mock_ms.get_structure_tf_bias.return_value = "bullish"

        fvg = Mock(top=2001.0, bottom=2000.0)
        mock_fvg.get_active_bullish_fvgs.return_value = [fvg]

        # No sell-side level near 1995 anymore
        mock_liq.get_liquidity.return_value = Mock(sell_side=[])

        plan = self._plan(
            "liquidity_sweep_fvg", "BUY",
            2000.0, 1990.0, 2020.0,
            {
                "fvg": {"type": "bullish", "top": 2001.0, "bottom": 2000.0},
                "sweep": {"side": "sell_side", "price": 1995.0},
            },
        )
        ok, reason = self.engine.reconfirm_plan(plan)
        self.assertFalse(ok)
        self.assertIn("liquidity level", reason.lower())

    # --- liquidity_sweep_ob_fvg: OR -> AND fix ---

    @patch("src.smc_strategy.liquidity")
    @patch("src.smc_strategy.order_blocks")
    @patch("src.smc_strategy.fair_value_gaps")
    @patch("src.smc_strategy.market_structure")
    def test_sweep_ob_fvg_ob_present_fvg_absent_fails(
        self, mock_ms, mock_fvg, mock_ob, mock_liq
    ):
        """H-03 regression: pre-H-03 used OR; a plan with only OB
        (FVG gone) must now be INVALIDATED."""
        mock_ms.get_higher_tf_bias.return_value = "bullish"
        mock_ms.get_structure_tf_bias.return_value = "bullish"

        ob = Mock(top=2001.0, bottom=2000.0)
        mock_ob.get_active_bullish_obs.return_value = [ob]
        mock_fvg.get_active_bullish_fvgs.return_value = []  # FVG gone

        mock_liq.get_liquidity.return_value = Mock(
            sell_side=[Mock(price=1995.0)]
        )

        plan = self._plan(
            "liquidity_sweep_ob_fvg", "BUY",
            2000.0, 1990.0, 2020.0,
            {
                "ob":  {"type": "bullish", "top": 2001.0, "bottom": 2000.0},
                "fvg": {"type": "bullish", "top": 2002.0, "bottom": 2000.0},
                "sweep": {"side": "sell_side", "price": 1995.0},
            },
        )
        ok, reason = self.engine.reconfirm_plan(plan)
        self.assertFalse(ok)
        self.assertIn("fvg", reason.lower())

    @patch("src.smc_strategy.liquidity")
    @patch("src.smc_strategy.order_blocks")
    @patch("src.smc_strategy.fair_value_gaps")
    @patch("src.smc_strategy.market_structure")
    def test_sweep_ob_fvg_both_present_passes(
        self, mock_ms, mock_fvg, mock_ob, mock_liq
    ):
        mock_ms.get_higher_tf_bias.return_value = "bullish"
        mock_ms.get_structure_tf_bias.return_value = "bullish"

        ob = Mock(top=2001.0, bottom=2000.0)
        fvg = Mock(top=2002.0, bottom=2000.0)
        mock_ob.get_active_bullish_obs.return_value = [ob]
        mock_fvg.get_active_bullish_fvgs.return_value = [fvg]

        mock_liq.get_liquidity.return_value = Mock(
            sell_side=[Mock(price=1995.0)]
        )

        plan = self._plan(
            "liquidity_sweep_ob_fvg", "BUY",
            2000.0, 1990.0, 2020.0,
            {
                "ob":  {"type": "bullish", "top": 2001.0, "bottom": 2000.0},
                "fvg": {"type": "bullish", "top": 2002.0, "bottom": 2000.0},
                "sweep": {"side": "sell_side", "price": 1995.0},
            },
        )
        ok, reason = self.engine.reconfirm_plan(plan)
        self.assertTrue(ok, reason)

    # --- bos_order_block ---

    @patch("src.smc_strategy.order_blocks")
    @patch("src.smc_strategy.market_structure")
    def test_bos_ob_bos_flipped_fails(self, mock_ms, mock_ob):
        mock_ms.get_higher_tf_bias.return_value = "bullish"
        mock_ms.get_structure_tf_bias.return_value = "bullish"
        mock_ms.is_bos_bullish.return_value = False

        plan = self._plan(
            "bos_order_block", "BUY",
            2000.0, 1990.0, 2020.0,
            {
                "ob":  {"type": "bullish", "top": 2001.0, "bottom": 2000.0},
                "bos": {"timeframe": "h1", "direction": "bullish"},
            },
        )
        ok, reason = self.engine.reconfirm_plan(plan)
        self.assertFalse(ok)
        self.assertIn("bos", reason.lower())

    @patch("src.smc_strategy.order_blocks")
    @patch("src.smc_strategy.market_structure")
    def test_bos_ob_valid(self, mock_ms, mock_ob):
        mock_ms.get_higher_tf_bias.return_value = "bullish"
        mock_ms.get_structure_tf_bias.return_value = "bullish"
        mock_ms.is_bos_bullish.return_value = True

        ob = Mock(top=2001.0, bottom=2000.0)
        mock_ob.get_active_bullish_obs.return_value = [ob]

        plan = self._plan(
            "bos_order_block", "BUY",
            2000.0, 1990.0, 2020.0,
            {
                "ob":  {"type": "bullish", "top": 2001.0, "bottom": 2000.0},
                "bos": {"timeframe": "h1", "direction": "bullish"},
            },
        )
        ok, reason = self.engine.reconfirm_plan(plan)
        self.assertTrue(ok, reason)

    # --- choch_order_block ---

    @patch("src.smc_strategy.order_blocks")
    @patch("src.smc_strategy.market_structure")
    def test_choch_ob_choch_flipped_fails(self, mock_ms, mock_ob):
        mock_ms.get_higher_tf_bias.return_value = "bullish"
        mock_ms.get_structure_tf_bias.return_value = "bullish"
        mock_ms.is_choch_bullish.return_value = False

        plan = self._plan(
            "choch_order_block", "BUY",
            2000.0, 1990.0, 2020.0,
            {
                "ob":    {"type": "bullish", "top": 2001.0, "bottom": 2000.0},
                "choch": {"timeframe": "m15", "direction": "bullish"},
            },
        )
        ok, reason = self.engine.reconfirm_plan(plan)
        self.assertFalse(ok)

    # --- displacement_fvg ---

    @patch("src.smc_strategy.market_data")
    @patch("src.smc_strategy.fair_value_gaps")
    @patch("src.smc_strategy.market_structure")
    def test_displacement_flipped_fails(
        self, mock_ms, mock_fvg, mock_md
    ):
        mock_ms.get_higher_tf_bias.return_value = "bullish"
        mock_ms.get_structure_tf_bias.return_value = "bullish"

        # Provide 20 closed candles so detect_displacement can run.
        candles = [Mock(open=100.0, close=100.0) for _ in range(30)]
        mock_md.get_closed_candles.return_value = candles

        plan = self._plan(
            "displacement_fvg", "BUY",
            2000.0, 1990.0, 2020.0,
            {
                "fvg": {"type": "bullish", "top": 2001.0, "bottom": 2000.0},
                "displacement": {"direction": "bullish", "ratio_at_creation": 2.0},
            },
        )
        # No displacement will be detected on flat candles
        ok, reason = self.engine.reconfirm_plan(plan)
        self.assertFalse(ok)
        self.assertIn("displacement", reason.lower())

    # --- migration + persistence round-trip ---

    def test_setup_state_persisted_round_trip(self):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.close()
        try:
            db = DatabaseManager(database_url=f"sqlite:///{tmp.name}")
            db.create_tables()

            state = {
                "fvg": {"type": "bullish", "top": 2001.0, "bottom": 2000.0},
                "sweep": {"side": "sell_side", "price": 1995.0},
            }
            plan = TradePlan(
                timestamp=datetime.now(timezone.utc),
                symbol="XAUUSD", direction="BUY",
                strategy="liquidity_sweep_fvg",
                entry_price=2000.0, sl_price=1990.0, tp_price=2020.0,
                risk_reward=2.0, status="WAITING",
                expiry_time=datetime.now(timezone.utc),
                environment="demo",
                setup_state=state,
            )
            with db.get_session() as session:
                session.add(plan)
                session.flush()
                plan_id = plan.id

            fetched = db.get(TradePlan, plan_id)
            self.assertEqual(fetched.setup_state, state)
            db.close()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def test_migration_invalidates_pre_h03_waiting_plans(self):
        """Insert a WAITING plan with setup_state=NULL and confirm
        create_tables (which runs the migration) marks it INVALIDATED."""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.close()
        try:
            db = DatabaseManager(database_url=f"sqlite:///{tmp.name}")
            db.create_tables()

            # Simulate a pre-H-03 row.
            with db.get_session() as session:
                session.execute(text("""
                    INSERT INTO trade_plans
                        (timestamp, symbol, direction, strategy,
                         entry_price, sl_price, tp_price, risk_reward,
                         status, expiry_time, environment)
                    VALUES (CURRENT_TIMESTAMP, 'XAUUSD', 'BUY',
                            'liquidity_sweep_fvg',
                            2000.0, 1990.0, 2020.0, 2.0,
                            'WAITING', CURRENT_TIMESTAMP, 'demo')
                """))

            # Re-run migration: should invalidate the WAITING row.
            db.create_tables()

            with db.get_session() as session:
                row = session.query(TradePlan).filter_by(
                    status="WAITING"
                ).first()
                self.assertIsNone(row)

                invalidated = session.query(TradePlan).filter_by(
                    status="INVALIDATED"
                ).first()
                self.assertIsNotNone(invalidated)
                self.assertIn(
                    "pre-H-03", invalidated.invalidated_reason or ""
                )
            db.close()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


class TestH04ReconcileQueryFailures(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self._tmp.close()
        self._db = DatabaseManager(
            database_url=f"sqlite:///{self._tmp.name}"
        )
        self._db.create_tables()

        # Patch module-level db_manager (used inside trade_manager).
        self._db_patch = patch(
            "src.trade_manager.db_manager", self._db
        )
        self._db_patch.start()

        # Patch module-level mt5_connector (used inside trade_manager).
        self._mt5_patch = patch("src.trade_manager.mt5_connector")
        self._mt5 = self._mt5_patch.start()
        self._mt5.get_last_error.return_value = (10004, "no connection")
        self._mt5.get_history_deals.return_value = []

        self._tm = TradeManager()

    def tearDown(self):
        self._mt5_patch.stop()
        self._db_patch.stop()
        try:
            self._db.close()
        except Exception:
            pass
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def _unresolved_marker(self):
        return self._db.query_one(
            AppSettings,
            key=f"unresolved_recon_{self._tm.environment}",
        )

    # --- the primary H-04 regression ---

    def test_db_open_positions_failure_returns_false_with_no_unresolved(self):
        """The exact defect: MT5 reports no live positions, the DB list
        query fails, and the old code still returned True because the
        unresolved list was empty."""
        self._mt5.get_positions.return_value = []

        with patch.object(
            self._db,
            "get_open_positions",
            side_effect=RuntimeError("sqlite locked"),
        ):
            result = self._tm.reconcile_with_broker()

        self.assertFalse(result)

    def test_db_open_positions_failure_persists_diagnostic(self):
        self._mt5.get_positions.return_value = []

        with patch.object(
            self._db,
            "get_open_positions",
            side_effect=RuntimeError("sqlite locked"),
        ):
            self._tm.reconcile_with_broker()

        marker = self._unresolved_marker()
        self.assertIsNotNone(marker)
        payload = json.loads(marker.value)
        self.assertIn("sqlite locked", payload.get("errors", ""))

    # --- happy path still returns True ---

    def test_no_positions_no_failures_returns_true(self):
        self._mt5.get_positions.return_value = []
        result = self._tm.reconcile_with_broker()
        self.assertTrue(result)

    # --- unresolved tickets still fail ---

    def test_live_position_with_no_reconstruction_returns_false(self):
        self._mt5.get_positions.return_value = [{
            "ticket": 42, "type": "BUY", "price_open": 2000.0,
            "price_current": 2000.0, "sl": 1990.0, "tp": 2020.0,
            "volume": 0.1, "swap": 0.0, "profit": 0.0,
            "magic": 1, "symbol": "XAUUSD",
        }]
        # No deal history for this ticket → unresolved.
        self._mt5.get_history_deals.return_value = []

        result = self._tm.reconcile_with_broker()
        self.assertFalse(result)

    # --- H-04 diagnostic improvement: deals query failure vs no deals ---

    def test_deals_for_ticket_returns_none_on_history_failure(self):
        self._mt5.get_history_deals.side_effect = RuntimeError(
            "history endpoint down"
        )
        self.assertIsNone(self._tm._deals_for_ticket(123))

    def test_deals_for_ticket_returns_empty_when_no_deals(self):
        self._mt5.get_history_deals.return_value = []
        self.assertEqual(self._tm._deals_for_ticket(123), [])

    def test_reconstruct_closed_trade_returns_false_on_history_failure(self):
        self._mt5.get_history_deals.side_effect = RuntimeError(
            "history endpoint down"
        )
        self.assertFalse(self._tm.reconstruct_closed_trade(123))

if __name__ == "__main__":
    unittest.main()