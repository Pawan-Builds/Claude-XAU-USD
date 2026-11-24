"""Genuine backtesting engine for SMC strategies using real MT5 data."""
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from src.config_loader import config
from src.database_manager import db_manager, BacktestConfig, BacktestReport, BacktestTrade
from src.mt5_connector import mt5_connector
from src.market_data import market_data, Candle
from src.indicators import (
    analyze_market_structure, detect_fvgs, detect_order_blocks,
    detect_liquidity_levels, check_liquidity_sweep, check_fvg_mitigation,
    check_ob_mitigation, calculate_premium_discount, detect_displacement,
)
from src.smc_strategy import SMCEngine, SignalDirection, StrategyType, Confirmation, TradeSignal
from src.logger import get_logger

logger = get_logger(__name__)


class BacktestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BacktestPosition:
    ticket: int
    direction: str
    entry_price: float
    sl: float
    tp: float
    volume: float
    entry_time: datetime
    entry_index: int
    strategy: str
    risk_percent: float
    realized_r: float = 0.0
    closed: bool = False
    close_price: float = 0.0
    close_time: datetime = None
    close_reason: str = ""
    gross_profit: float = 0.0
    net_profit: float = 0.0
    spread_cost: float = 0.0


@dataclass
class BacktestState:
    current_index: int
    current_time: datetime
    equity: float
    balance: float
    open_positions: List[BacktestPosition]
    closed_trades: List[BacktestTrade]
    daily_pl: float
    max_equity: float
    max_drawdown: float
    max_drawdown_pct: float


class BacktestEngine:
    """Main backtesting engine."""

    def __init__(self, config_id: Optional[int] = None):
        self.config_id = config_id
        self.backtest_config = config.backtesting
        self.trading_config = config.trading
        self.risk_config = config.risk
        
        self.candles: Dict[str, pd.DataFrame] = {}
        self.current_state: Optional[BacktestState] = None
        self.status = BacktestStatus.PENDING
        self.progress = 0.0
        self.report: Optional[BacktestReport] = None
        self.trades: List[BacktestTrade] = []
        self._ticket_counter = 1000000
        
        self.point_value = 0.01
        self.pip_value = 0.1  # 10 points

    def load_data(self, symbol: str, timeframes: List[str],
                  start_date: datetime, end_date: datetime) -> bool:
        """Load genuine historical data from MT5 and resample."""
        logger.info(f"Loading historical data for {symbol} from {start_date} to {end_date}")

        if not mt5_connector.is_connected():
            if not mt5_connector.initialize():
                logger.error("Cannot load backtest data: MT5 not connected.")
                return False

        tf_minutes = 5
        rates = mt5_connector.get_rates_range(tf_minutes, start_date, end_date)
        if not rates or len(rates) < 100:
            logger.error("Insufficient historical data returned from MT5.")
            return False

        df = pd.DataFrame(rates)
        df.set_index("time", inplace=True)
        
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
            
        logger.info(f"Loaded {len(df)} base M5 candles.")

        resample_rules = {
            "m5": "5min",
            "m15": "15min",
            "h1": "1h",
            "h4": "4h"
        }

        for tf in timeframes:
            rule = resample_rules.get(tf)
            if not rule:
                continue
            
            if tf == "m5":
                self.candles[tf] = df.copy()
            else:
                resampled = df.resample(rule).agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "tick_volume": "sum"
                }).dropna()
                self.candles[tf] = resampled
            
            logger.info(f"Resampled {tf}: {len(self.candles[tf])} candles.")

        return True

    def run_backtest(self, config: BacktestConfig) -> BacktestReport:
        self.status = BacktestStatus.RUNNING
        self.progress = 0.0

        logger.info(f"Starting backtest: {config.name}")

        try:
            self.current_state = BacktestState(
                current_index=0,
                current_time=config.start_date,
                equity=config.initial_balance,
                balance=config.initial_balance,
                open_positions=[],
                closed_trades=[],
                daily_pl=0.0,
                max_equity=config.initial_balance,
                max_drawdown=0.0,
                max_drawdown_pct=0.0
            )

            primary_tf = "m5"
            df = self.candles.get(primary_tf)
            if df is None or df.empty:
                logger.error("No candle data for backtest")
                self.status = BacktestStatus.FAILED
                return self._create_failed_report(config)

            total_candles = len(df)
            lookback = 100

            smc_engine = SMCEngine()

            for i in range(lookback, total_candles):
                current_time = df.index[i]
                current_candle = df.iloc[i]
                
                self.current_state.current_index = i
                self.current_state.current_time = current_time
                self.progress = (i - lookback) / (total_candles - lookback) * 100

                self._check_exits(current_candle, current_time)
                self._update_equity(current_candle)

                if i % 5 == 0 and len(self.current_state.open_positions) < self.trading_config.max_positions:
                    self._check_signals(df, i, smc_engine, config)

                self._update_drawdown()

                if i > 0 and current_time.date() != df.index[i-1].date():
                    self.current_state.daily_pl = 0.0

            self._close_all_positions(df.iloc[-1], df.index[-1])
            self.report = self._generate_report(config)
            self.status = BacktestStatus.COMPLETED
            self.progress = 100.0

            logger.info(f"Backtest completed: {self.report.net_profit:.2f} profit, {self.report.total_trades} trades")
            return self.report

        except Exception as e:
            logger.exception(f"Backtest failed: {e}")
            self.status = BacktestStatus.FAILED
            return self._create_failed_report(config)

    def _update_equity(self, current_candle: pd.Series):
        spread = 0.20
        bid = current_candle["close"] - (spread / 2)
        
        floating_pl = 0.0
        for pos in self.current_state.open_positions:
            if pos.direction == "BUY":
                pl_pips = (bid - pos.entry_price) / self.pip_value
            else:
                pl_pips = (pos.entry_price - bid) / self.pip_value

            tick_value = 10.0 
            floating_pl += pl_pips * tick_value * pos.volume

        self.current_state.equity = self.current_state.balance + floating_pl

    def _check_exits(self, current_candle: pd.Series, current_time: datetime):
        high = current_candle["high"]
        low = current_candle["low"]
        
        spread = 0.20
        ask_high = high + (spread / 2)
        ask_low = low + (spread / 2)
        bid_high = high - (spread / 2)
        bid_low = low - (spread / 2)

        remaining = []
        for pos in self.current_state.open_positions:
            closed = False

            if pos.direction == "BUY":
                if bid_low <= pos.sl:
                    pos.close_price = pos.sl
                    pos.close_time = current_time
                    pos.close_reason = "SL"
                    pos.realized_r = -1.0
                    closed = True
                elif bid_high >= pos.tp:
                    pos.close_price = pos.tp
                    pos.close_time = current_time
                    pos.close_reason = "TP"
                    risk = pos.entry_price - pos.sl
                    reward = pos.tp - pos.entry_price
                    pos.realized_r = reward / risk if risk > 0 else 0
                    closed = True
            else:
                if ask_high >= pos.sl:
                    pos.close_price = pos.sl
                    pos.close_time = current_time
                    pos.close_reason = "SL"
                    pos.realized_r = -1.0
                    closed = True
                elif ask_low <= pos.tp:
                    pos.close_price = pos.tp
                    pos.close_time = current_time
                    pos.close_reason = "TP"
                    risk = pos.sl - pos.entry_price
                    reward = pos.entry_price - pos.tp
                    pos.realized_r = reward / risk if risk > 0 else 0
                    closed = True

            if closed:
                if pos.direction == "BUY":
                    profit_pips = (pos.close_price - pos.entry_price) / self.pip_value
                else:
                    profit_pips = (pos.entry_price - pos.close_price) / self.pip_value

                gross = profit_pips * 10.0 * pos.volume
                spread_cost = 10.0 * pos.volume
                commission = config.backtesting.commission_per_lot * pos.volume
                net = gross - commission - spread_cost

                pos.gross_profit = gross
                pos.net_profit = net
                pos.spread_cost = spread_cost

                self.current_state.balance += net
                self.current_state.daily_pl += net
                self.current_state.closed_trades.append(self._position_to_trade(pos))
                self.trades.append(self._position_to_trade(pos))
            else:
                remaining.append(pos)

        self.current_state.open_positions = remaining

    def _check_signals(self, df_m5: pd.DataFrame, current_index: int, smc_engine: SMCEngine, config: BacktestConfig):
        current_time = df_m5.index[current_index]
        
        # Use getattr to safely capture original methods, even if they don't exist
        orig_get_candles = getattr(market_data, "get_candles", None)
        orig_get_closed = getattr(market_data, "get_closed_candles", None)
        orig_latest = getattr(market_data, "get_latest_candle", None)
        orig_calculate_atr = getattr(market_data, "calculate_atr", None)

        def mock_get_candles(tf: str, count: int):
            df = self.candles.get(tf)
            if df is None: return []
            sliced = df.loc[:current_time].tail(count)
            return [Candle(timestamp=r.name, open=r["open"], high=r["high"], low=r["low"], close=r["close"], volume=r["tick_volume"]) for _, r in sliced.iterrows()]

        def mock_get_closed_candles(tf: str, count: int):
            df = self.candles.get(tf)
            if df is None: return []
            sliced = df.loc[:current_time]
            if len(sliced) > 1:
                sliced = sliced.iloc[:-1].tail(count)
            else:
                return []
            return [Candle(timestamp=r.name, open=r["open"], high=r["high"], low=r["low"], close=r["close"], volume=r["tick_volume"]) for _, r in sliced.iterrows()]
            
        def mock_get_latest(tf: str):
            df = self.candles.get(tf)
            if df is None: return None
            sliced = df.loc[:current_time]
            if sliced.empty: return None
            r = sliced.iloc[-1]
            return Candle(timestamp=sliced.index[-1], open=r["open"], high=r["high"], low=r["low"], close=r["close"], volume=r["tick_volume"])

        def mock_calculate_atr(tf: str, period: int = 14):
            df = self.candles.get(tf)
            if df is None: return 0.0
            sliced = df.loc[:current_time]
            if len(sliced) > 1:
                sliced = sliced.iloc[:-1]
            else:
                return 0.0
            if len(sliced) < period + 1: return 0.0
            
            high = sliced["high"].values
            low = sliced["low"].values
            close = sliced["close"].values
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            true_range = np.maximum(np.maximum(tr1, tr2), tr3)
            return float(np.mean(true_range[-period:]))

        # Apply mocks
        market_data.get_candles = mock_get_candles
        market_data.get_closed_candles = mock_get_closed_candles
        market_data.get_latest_candle = mock_get_latest
        market_data.calculate_atr = mock_calculate_atr

        try:
            signals = smc_engine.analyze_all_strategies()
            
            for sig in signals:
                current_candle = df_m5.iloc[current_index]
                # Ensure the entry is within the current candle's range (simulating waiting for price)
                if current_candle["low"] <= sig.entry_price <= current_candle["high"]:
                    self._execute_signal(sig, current_candle, df_m5.index[current_index], config)
                        
        finally:
            # Restore originals safely
            if orig_get_candles: market_data.get_candles = orig_get_candles
            if orig_get_closed: market_data.get_closed_candles = orig_get_closed
            if orig_latest: market_data.get_latest_candle = orig_latest
            if orig_calculate_atr: market_data.calculate_atr = orig_calculate_atr

        def mock_get_candles(tf: str, count: int):
            df = self.candles.get(tf)
            if df is None: return []
            sliced = df.loc[:current_time].tail(count)
            return [Candle(timestamp=r.name, open=r["open"], high=r["high"], low=r["low"], close=r["close"], volume=r["tick_volume"]) for _, r in sliced.iterrows()]

        def mock_get_closed_candles(tf: str, count: int):
            df = self.candles.get(tf)
            if df is None: return []
            sliced = df.loc[:current_time]
            if len(sliced) > 1:
                sliced = sliced.iloc[:-1].tail(count)
            else:
                return []
            return [Candle(timestamp=r.name, open=r["open"], high=r["high"], low=r["low"], close=r["close"], volume=r["tick_volume"]) for _, r in sliced.iterrows()]
            
        def mock_get_latest(tf: str):
            df = self.candles.get(tf)
            if df is None: return None
            sliced = df.loc[:current_time]
            if sliced.empty: return None
            r = sliced.iloc[-1]
            return Candle(timestamp=sliced.index[-1], open=r["open"], high=r["high"], low=r["low"], close=r["close"], volume=r["tick_volume"])

        def mock_calculate_atr(tf: str, period: int = 14):
            df = self.candles.get(tf)
            if df is None: return 0.0
            sliced = df.loc[:current_time]
            if len(sliced) > 1:
                sliced = sliced.iloc[:-1]
            else:
                return 0.0
            if len(sliced) < period + 1: return 0.0
            
            high = sliced["high"].values
            low = sliced["low"].values
            close = sliced["close"].values
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            true_range = np.maximum(np.maximum(tr1, tr2), tr3)
            return float(np.mean(true_range[-period:]))

        market_data.get_candles = mock_get_candles
        market_data.get_closed_candles = mock_get_closed_candles
        market_data.get_latest_candle = mock_get_latest
        market_data.calculate_atr = mock_calculate_atr

        try:
            signals = smc_engine.analyze_all_strategies()
            
            for sig in signals:
                current_candle = df_m5.iloc[current_index]
                if current_candle["low"] <= sig.entry_price <= current_candle["high"]:
                    self._execute_signal(sig, current_candle, df_m5.index[current_index], config)
                        
        finally:
            market_data.get_candles = orig_get_candles
            market_data.get_closed_candles = orig_get_closed
            market_data.get_latest_candle = orig_latest
            market_data.calculate_atr = orig_calculate_atr

    def _execute_signal(self, sig: TradeSignal, current_candle: pd.Series, current_time: datetime, config: BacktestConfig):
        equity = self.current_state.equity
        risk_amount = equity * (config.risk_percent / 100)

        if sig.direction == SignalDirection.BUY:
            sl_distance = sig.entry_price - sig.sl_price
        else:
            sl_distance = sig.sl_price - sig.entry_price

        if sl_distance <= 0: return
        sl_pips = sl_distance / self.pip_value
        if sl_pips <= 0: return
            
        tick_value = 10.0
        volume = risk_amount / (sl_pips * tick_value)
        volume = round(volume / 0.01) * 0.01
        volume = max(0.01, min(50.0, volume))

        pos = BacktestPosition(
            ticket=self._ticket_counter,
            direction=sig.direction.value,
            entry_price=sig.entry_price,
            sl=sig.sl_price,
            tp=sig.tp_price,
            volume=volume,
            entry_time=current_time,
            entry_index=self.current_state.current_index,
            strategy=sig.strategy.value,
            risk_percent=config.risk_percent
        )
        self._ticket_counter += 1
        self.current_state.open_positions.append(pos)

    def _close_all_positions(self, last_candle: pd.Series, last_time: datetime):
        spread = 0.20
        bid = last_candle["close"] - (spread / 2)
        ask = last_candle["close"] + (spread / 2)

        for pos in self.current_state.open_positions:
            pos.close_price = bid if pos.direction == "BUY" else ask
            pos.close_time = last_time
            pos.close_reason = "END_OF_BACKTEST"

            if pos.direction == "BUY":
                profit_pips = (pos.close_price - pos.entry_price) / self.pip_value
            else:
                profit_pips = (pos.entry_price - pos.close_price) / self.pip_value

            gross = profit_pips * 10.0 * pos.volume
            spread_cost = 10.0 * pos.volume
            commission = config.backtesting.commission_per_lot * pos.volume
            net = gross - commission - spread_cost

            pos.gross_profit = gross
            pos.net_profit = net
            pos.realized_r = net / (pos.volume * abs(pos.entry_price - pos.sl) / self.pip_value * 10.0) if pos.sl != pos.entry_price else 0

            self.current_state.balance += net
            self.current_state.closed_trades.append(self._position_to_trade(pos))
            self.trades.append(self._position_to_trade(pos))

        self.current_state.open_positions = []

    def _position_to_trade(self, pos: BacktestPosition) -> BacktestTrade:
        return BacktestTrade(
            report_id=0,
            ticket=pos.ticket,
            direction=pos.direction,
            entry_time=pos.entry_time,
            exit_time=pos.close_time or pos.entry_time,
            entry_price=pos.entry_price,
            exit_price=pos.close_price or pos.entry_price,
            sl=pos.sl,
            tp=pos.tp,
            volume=pos.volume,
            risk_percent=pos.risk_percent,
            realized_r=pos.realized_r,
            gross_profit=pos.gross_profit,
            net_profit=pos.net_profit,
            strategy=pos.strategy,
            exit_reason=pos.close_reason
        )

    def _update_drawdown(self):
        if self.current_state.equity > self.current_state.max_equity:
            self.current_state.max_equity = self.current_state.equity

        drawdown = self.current_state.max_equity - self.current_state.equity
        if drawdown > self.current_state.max_drawdown:
            self.current_state.max_drawdown = drawdown

        if self.current_state.max_equity > 0:
            drawdown_pct = drawdown / self.current_state.max_equity * 100
            if drawdown_pct > self.current_state.max_drawdown_pct:
                self.current_state.max_drawdown_pct = drawdown_pct

    def _generate_report(self, config: BacktestConfig) -> BacktestReport:
        trades = self.current_state.closed_trades
        if not trades:
            return self._create_failed_report(config)

        winners = [t for t in trades if t.net_profit > 0]
        losers = [t for t in trades if t.net_profit < 0]

        gross_profit = sum(t.gross_profit for t in winners)
        gross_loss = abs(sum(t.gross_profit for t in losers))
        net_profit = sum(t.net_profit for t in trades)
        net_return = (self.current_state.equity - config.initial_balance) / config.initial_balance * 100

        win_rate = len(winners) / len(trades) * 100
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = gross_profit / len(winners) if winners else 0
        avg_loss = gross_loss / len(losers) if losers else 0
        expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)

        avg_r = np.mean([t.realized_r for t in trades])
        avg_rr = np.mean([abs(t.realized_r) for t in trades if t.realized_r != 0]) if any(t.realized_r != 0 for t in trades) else 0

        returns = [t.net_profit / config.initial_balance for t in trades]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        downside_returns = [r for r in returns if r < 0]
        sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252) if downside_returns and np.std(downside_returns) > 0 else 0

        recovery = net_profit / self.current_state.max_drawdown if self.current_state.max_drawdown > 0 else 0

        max_consec_wins = max_consec_losses = 0
        current_streak = 0
        for t in trades:
            if t.net_profit > 0:
                if current_streak >= 0: current_streak += 1
                else: current_streak = 1
                max_consec_wins = max(max_consec_wins, current_streak)
            else:
                if current_streak <= 0: current_streak -= 1
                else: current_streak = -1
                max_consec_losses = max(max_consec_losses, abs(current_streak))

        long_trades = [t for t in trades if t.direction == "BUY"]
        short_trades = [t for t in trades if t.direction == "SELL"]
        long_wins = [t for t in long_trades if t.net_profit > 0]
        short_wins = [t for t in short_trades if t.net_profit > 0]

        report = BacktestReport(
            config_id=config.id,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_balance,
            final_capital=self.current_state.equity,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=net_profit,
            net_return_pct=net_return,
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=self.current_state.max_drawdown,
            max_drawdown_pct=self.current_state.max_drawdown_pct,
            avg_risk_reward=avg_rr,
            avg_realized_r=avg_r,
            expectancy=expectancy,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            recovery_factor=recovery,
            largest_win=max(t.net_profit for t in trades),
            largest_loss=min(t.net_profit for t in trades),
            avg_holding_hours=np.mean([(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]),
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            long_trades=len(long_trades),
            long_win_rate=len(long_wins) / len(long_trades) * 100 if long_trades else 0,
            long_profit=sum(t.net_profit for t in long_trades),
            short_trades=len(short_trades),
            short_win_rate=len(short_wins) / len(short_trades) * 100 if short_trades else 0,
            short_profit=sum(t.net_profit for t in short_trades),
            environment="backtest"
        )

        saved_report_id = db_manager.add(report)
        for trade in self.trades:
            trade.report_id = saved_report_id
            db_manager.add(trade)

        return report

    def _create_failed_report(self, config: BacktestConfig) -> BacktestReport:
        return BacktestReport(
            config_id=config.id,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_balance,
            final_capital=config.initial_balance,
            net_profit=0.0,
            total_trades=0,
            environment="backtest"
        )

    def get_progress(self) -> float:
        return self.progress

    def get_status(self) -> BacktestStatus:
        return self.status


def run_backtest_async(config: BacktestConfig) -> BacktestReport:
    engine = BacktestEngine()
    return engine.run_backtest(config)