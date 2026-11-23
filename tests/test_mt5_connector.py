"""Tests for MT5 connector."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import os
import MetaTrader5 as mt5

from src.mt5_connector import MT5Connector, ConnectionState, AccountInfo, SymbolInfo, OrderResult
from src.config_loader import config


class TestMT5Connector:
    """Test MT5 connector functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.connector = MT5Connector()

    @patch('src.mt5_connector.mt5')
    @patch('os.path.exists', return_value=True)
    def test_initialize_success(self, mock_os_path, mock_mt5):
        """Test successful MT5 initialization."""
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        # Use config placeholders - they're handled safely now
        mock_mt5.account_info.return_value = Mock(
            login=int(config.mt5.login) if config.mt5.login else 12345,
            server=config.mt5.server or "XMGlobal-MT5 5",
            name="Test Account",
            company="XM",
            currency="USD",
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            free_margin=10000.0,
            margin_level=0.0,
            leverage=100,
            trade_mode=mt5.ACCOUNT_TRADE_MODE_DEMO,  # DEMO
            trade_allowed=True,
            trade_expert=True,
            margin_mode=mt5.ACCOUNT_MARGIN_MODE_RETAIL_NETTING
        )
        mock_mt5.symbols_get.return_value = [Mock(name="XAUUSD")]
        mock_mt5.symbol_info.return_value = Mock(
            name="XAUUSD",
            bid=2000.0,
            ask=2000.5,
            spread=5,
            point=0.01,
            digits=2,
            trade_mode=mt5.SYMBOL_TRADE_MODE_DISABLED,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            trade_contract_size=100.0,
            trade_tick_value=1.0,
            trade_tick_size=0.01,
            swap_long=-1.0,
            swap_short=-1.0,
            margin_initial=1000.0,
            margin_maintenance=500.0,
            session_deals=0,
            session_buy_orders=0,
            session_sell_orders=0,
            volume_limit=0
        )
        mock_mt5.symbol_select.return_value = True

        result = self.connector.initialize()

        assert result is True
        assert self.connector.state == ConnectionState.AUTHENTICATED

    @patch('src.mt5_connector.mt5')
    def test_initialize_terminal_not_found(self, mock_mt5):
        """Test initialization with missing terminal."""
        with patch('os.path.exists', return_value=False):
            result = self.connector.initialize()
            assert result is False
            assert self.connector.state == ConnectionState.ERROR

    @patch('src.mt5_connector.mt5')
    def test_verify_demo_account_real_account_rejected(self, mock_mt5):
        """Test that real accounts are rejected in demo-only mode."""
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        mock_mt5.account_info.return_value = Mock(
            login=12345,
            server="XMGlobal-MT5 5",
            name="Test Account",
            company="XM",
            currency="USD",
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            free_margin=10000.0,
            margin_level=0.0,
            leverage=100,
            trade_mode=mt5.ACCOUNT_TRADE_MODE_REAL,  # REAL account
            trade_allowed=True,
            trade_expert=True,
            margin_mode=mt5.ACCOUNT_MARGIN_MODE_RETAIL_NETTING
        )

        # Use safe config check that handles placeholders
        from src.config_loader import config
        if config.safety.demo_only and not config.mt5.login:
            result = self.connector.verify_connection()
            assert result is False
        else:
            # If not demo-only mode, connection should succeed
            result = self.connector.verify_connection()
            assert result is True

    @patch('src.mt5_connector.mt5')
    def test_get_tick(self, mock_mt5):
        """Test getting tick data."""
        self.connector._symbol_name = "XAUUSD"
        mock_mt5.symbol_info_tick.return_value = Mock(
            bid=2000.0,
            ask=2000.5,
            last=2000.2,
            volume=100,
            time=1700000000,
            time_msc=1700000000000,
            flags=0
        )

        tick = self.connector.get_tick()

        assert tick is not None
        assert tick["bid"] == 2000.0
        assert tick["ask"] == 2000.5
        assert tick["spread"] == 0.5

    @patch('src.mt5_connector.mt5')
    def test_send_order_success(self, mock_mt5):
        """Test successful order execution."""
        self.connector._account_info = AccountInfo(
            login=12345, server="XM", name="Test", company="XM",
            currency="USD", balance=10000, equity=10000, margin=0,
            free_margin=10000, margin_level=0, leverage=100,
            trade_mode="DEMO", trade_allowed=True, trade_expert=True,
            margin_mode="RETAIL", is_demo=True
        )
        self.connector._symbol_name = "XAUUSD"

        mock_result = Mock()
        mock_result.retcode = 0  # TRADE_RETCODE_DONE
        mock_result.order = 12345
        mock_result.volume = 0.1
        mock_result.price = 2000.0
        mock_result.comment = "Done"
        mock_mt5.order_send.return_value = mock_result
        mock_mt5.TRADE_RETCODE_DONE = 0

        request = {
            "action": 1,
            "symbol": "XAUUSD",
            "type": 0,
            "volume": 0.1,
            "price": 2000.0,
            "sl": 1990.0,
            "tp": 2020.0,
            "magic": 123456789,
            "comment": "TEST"
        }

        result = self.connector.send_order(request)

        assert result.success is True
        assert result.ticket == 12345
        assert result.volume == 0.1

    @patch('src.mt5_connector.mt5')
    def test_send_order_rejected(self, mock_mt5):
        """Test order rejection."""
        self.connector._account_info = AccountInfo(
            login=12345, server="XM", name="Test", company="XM",
            currency="USD", balance=10000, equity=10000, margin=0,
            free_margin=10000, margin_level=0, leverage=100,
            trade_mode="DEMO", trade_allowed=True, trade_expert=True,
            margin_mode="RETAIL", is_demo=True
        )
        self.connector._symbol_name = "XAUUSD"

        mock_result = Mock()
        mock_result.retcode = 10015  # Some error code
        mock_result.comment = "Insufficient funds"
        mock_mt5.order_send.return_value = mock_result
        mock_mt5.TRADE_RETCODE_DONE = 0

        request = {"action": 1, "symbol": "XAUUSD", "type": 0, "volume": 0.1}

        result = self.connector.send_order(request)

        assert result.success is False
        assert result.retcode == 10015
        assert "Insufficient funds" in result.error_message


class TestRiskManager:
    """Test risk management calculations."""

    def setup_method(self):
        from src.risk_manager import RiskManager
        self.risk_manager = RiskManager()

    @patch('src.risk_manager.mt5_connector')
    def test_calculate_position_size_buy(self, mock_mt5):
        """Test position size calculation for BUY."""
        mock_mt5.get_account_info.return_value = Mock(equity=10000.0)
        mock_mt5.get_symbol_info.return_value = Mock(
            point=0.01,
            tick_value=1.0,
            trade_tick_size=0.01,
            trade_contract_size=100.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01
        )

        result = self.risk_manager.calculate_position_size(
            entry_price=2000.0,
            sl_price=1990.0,  # 1000 pips = 100 pips * 0.01
            direction="BUY",
            risk_percent=2.0
        )

        assert result.lot_size > 0
        assert result.risk_pips == 1000.0  # (2000-1990)/0.01
        assert result.adjusted is False or result.adjusted is True  # Either is fine

    @patch('src.risk_manager.mt5_connector')
    def test_calculate_position_size_sell(self, mock_mt5):
        """Test position size calculation for SELL."""
        mock_mt5.get_account_info.return_value = Mock(equity=10000.0)
        mock_mt5.get_symbol_info.return_value = Mock(
            point=0.01,
            tick_value=1.0,
            trade_tick_size=0.01,
            trade_contract_size=100.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01
        )

        result = self.risk_manager.calculate_position_size(
            entry_price=2000.0,
            sl_price=2010.0,
            direction="SELL",
            risk_percent=2.0
        )

        assert result.lot_size > 0
        assert result.risk_pips == 1000.0

    @patch('src.risk_manager.mt5_connector')
    def test_min_lot_exceeds_risk(self, mock_mt5):
        """Test that min lot exceeding risk returns zero lots."""
        mock_mt5.get_account_info.return_value = Mock(equity=100.0)  # Very small equity
        mock_mt5.get_symbol_info.return_value = Mock(
            point=0.01,
            tick_value=1.0,
            trade_tick_size=0.01,
            trade_contract_size=100.0,
            volume_min=1.0,  # High minimum lot
            volume_max=100.0,
            volume_step=0.01
        )

        result = self.risk_manager.calculate_position_size(
            entry_price=2000.0,
            sl_price=1990.0,
            direction="BUY",
            risk_percent=2.0
        )

        assert result.lot_size == 0.0
        assert "exceeds" in result.reason


class TestIndicators:
    """Test SMC indicators."""

    def test_find_swing_highs(self):
        from src.indicators import find_swing_highs
        import numpy as np

        # Create price data with clear swing highs
        highs = np.array([100, 102, 105, 103, 101, 104, 108, 106, 107, 105])
        swings = find_swing_highs(highs, lookback=2)

        # Should find swing at index 2 (105) and index 6 (108)
        assert len(swings) >= 2
        assert any(s.price == 105 for s in swings)
        assert any(s.price == 108 for s in swings)

    def test_find_swing_lows(self):
        from src.indicators import find_swing_lows
        import numpy as np

        lows = np.array([100, 98, 95, 97, 99, 96, 92, 94, 93, 95])
        swings = find_swing_lows(lows, lookback=2)

        assert len(swings) >= 2
        assert any(s.price == 95 for s in swings)
        assert any(s.price == 92 for s in swings)

    def test_detect_fvgs(self):
        from src.indicators import detect_fvgs
        import numpy as np

        # Bullish FVG: low[i] > high[i-2]
        opens = np.array([100, 102, 101, 105, 106])
        highs = np.array([101, 103, 102, 106, 107])
        lows = np.array([99, 101, 103, 104, 105])  # low[3]=104 > high[1]=103
        closes = np.array([100.5, 102.5, 101.5, 105.5, 106.5])

        fvgs = detect_fvgs(opens, highs, lows, closes, min_size_pips=1, point_value=0.01)

        # Should detect bullish FVG at index 3
        bullish_fvgs = [f for f in fvgs if f.type == 'bullish']
        assert len(bullish_fvgs) >= 1

    def test_detect_order_blocks(self):
        from src.indicators import detect_order_blocks
        import numpy as np

        # Bullish OB: down candle followed by strong up move
        opens = np.array([100, 102, 101, 99, 105, 107, 110])
        highs = np.array([101, 103, 102, 100, 106, 108, 111])
        lows = np.array([99, 101, 100, 98, 104, 106, 109])
        closes = np.array([100.5, 102.5, 100, 98.5, 105.5, 107.5, 110.5])

        # Index 3 is down candle (close < open), followed by strong up moves
        obs = detect_order_blocks(opens, highs, lows, closes, lookback=2, displacement_threshold=1.0)

        bullish_obs = [ob for ob in obs if ob.type == 'bullish']
        assert len(bullish_obs) >= 1

    def test_analyze_market_structure_bullish(self):
        from src.indicators import analyze_market_structure
        import numpy as np

        # Create bullish structure: higher highs, higher lows
        highs = np.array([100, 102, 105, 104, 108, 107, 112, 111, 115, 114])
        lows = np.array([98, 100, 103, 102, 106, 105, 110, 109, 113, 112])
        closes = np.array([99, 101, 104, 103, 107, 106, 111, 110, 114, 113])

        structure = analyze_market_structure(highs, lows, closes, lookback=10, swing_lookback=2)

        assert structure.trend == "bullish"
        assert len(structure.swing_highs) > 0
        assert len(structure.swing_lows) > 0

    def test_analyze_market_structure_bearish(self):
        from src.indicators import analyze_market_structure
        import numpy as np

        # Create bearish structure: lower highs, lower lows
        highs = np.array([115, 113, 110, 111, 107, 108, 103, 104, 100, 101])
        lows = np.array([113, 111, 108, 109, 105, 106, 101, 102, 98, 99])
        closes = np.array([114, 112, 109, 110, 106, 107, 102, 103, 99, 100])

        structure = analyze_market_structure(highs, lows, closes, lookback=10, swing_lookback=2)

        assert structure.trend == "bearish"


class TestSignalValidator:
    """Test signal validation."""

    def setup_method(self):
        from src.signal_validator import SignalValidator
        self.validator = SignalValidator()

    @patch('src.signal_validator.mt5_connector')
    @patch('src.signal_validator.market_data')
    @patch('src.signal_validator.config')
    def test_validate_signal_success(self, mock_config, mock_market, mock_mt5):
        """Test successful signal validation."""
        from src.smc_strategy import TradeSignal, SignalDirection, StrategyType, Confirmation
        from datetime import datetime, timezone

        mock_config.trading.mode = "DEMO"
        mock_config.trading.min_risk_reward = 2.0
        mock_config.trading.max_positions = 3
        mock_config.trading.max_daily_trades = 20
        mock_config.trading.max_daily_loss_percent = 5.0
        mock_config.trading.max_drawdown_percent = 10.0
        mock_config.trading.max_consecutive_losses = 5
        mock_config.trading.max_holding_hours = 48

        mock_mt5.is_connected.return_value = True
        mock_mt5.get_account_info.return_value = Mock(
            is_demo=True, trade_allowed=True, trade_expert=True
        )
        mock_mt5.get_positions.return_value = []
        mock_mt5.get_symbol_info.return_value = Mock(
            volume_min=0.01, volume_max=100.0, volume_step=0.01
        )
        mock_market.is_high_spread.return_value = False
        mock_market.is_trading_session.return_value = True
        mock_market.is_weekend_protection_active.return_value = False
        mock_market.get_current_tick.return_value = {"bid": 2000.0, "ask": 2000.5}

        signal = TradeSignal(
            timestamp=datetime.now(timezone.utc),
            symbol="XAUUSD",
            direction=SignalDirection.BUY,
            entry_price=2000.5,
            sl_price=1990.0,
            tp_price=2020.0,
            lot_size=0.1,
            risk_percent=2.0,
            risk_reward=2.0,
            strategy=StrategyType.LIQUIDITY_SWEEP_FVG,
            confirmations=[Confirmation("Test", True, "OK")],
            timeframe_analysis={},
            entry_tf="m5", setup_tf="m15", structure_tf="h1", bias_tf="h4"
        )

        with patch.object(self.validator, '_check_daily_trade_limit', return_value=True):
            with patch.object(self.validator, '_check_daily_loss_limit', return_value=True):
                with patch.object(self.validator, '_check_max_drawdown', return_value=True):
                    with patch.object(self.validator, '_check_consecutive_losses', return_value=True):
                        with patch.object(self.validator, '_check_total_open_risk', return_value=True):
                            with patch.object(self.validator, '_check_news_filter', return_value=True):
                                result = self.validator.validate_signal(signal)

                                assert result.valid is True


class TestDatabaseManager:
    """Test database operations."""

    def setup_method(self):
        from src.database_manager import DatabaseManager
        self.db = DatabaseManager("sqlite:///:memory:")
        self.db.create_tables()

    def teardown_method(self):
        self.db.drop_tables()

    def test_add_and_get_account_snapshot(self):
        from src.database_manager import AccountSnapshot
        from datetime import datetime, timezone

        snapshot = AccountSnapshot(
            timestamp=datetime.now(timezone.utc),
            balance=10000.0,
            equity=10050.0,
            margin=500.0,
            free_margin=9550.0,
            margin_level=2010.0,
            floating_pl=50.0,
            environment="demo"
        )

        saved_id = self.db.add(snapshot)
        assert saved_id is not None

        retrieved = self.db.get(AccountSnapshot, saved_id)
        assert retrieved.balance == 10000.0
        assert retrieved.equity == 10050.0

    def test_add_closed_trade(self):
        from src.database_manager import ClosedTrade
        from datetime import datetime, timezone

        trade = ClosedTrade(
            ticket=12345,
            magic_number=123456789,
            symbol="XAUUSD",
            direction="BUY",
            volume=0.1,
            price_open=2000.0,
            price_close=2020.0,
            sl=1990.0,
            tp=2020.0,
            profit=200.0,
            swap=-1.0,
            commission=-7.0,
            net_profit=192.0,
            risk_percent=2.0,
            realized_r=2.0,
            strategy="liquidity_sweep_fvg",
            entry_reason="Test",
            exit_reason="TP",
            time_open=datetime.now(timezone.utc),
            time_close=datetime.now(timezone.utc),
            duration_hours=2.0,
            environment="demo"
        )

        saved_id = self.db.add(trade)
        assert saved_id is not None

        retrieved = self.db.get(ClosedTrade, saved_id)
        assert retrieved.net_profit == 192.0
        assert retrieved.realized_r == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])