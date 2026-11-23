"""Backtesting engine for SMC strategies."""
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import uuid

from src.config_loader import config
from src.database_manager import db_manager, BacktestConfig, BacktestReport, BacktestTrade
from src.indicators import (
    analyze_market_structure, detect_fvgs, detect_order_blocks,
    detect_liquidity_levels, check_liquidity_sweep, check_fvg_mitigation,
    check_ob_mitigation, calculate_premium_discount, detect_displacement,
    MarketStructure, FVG, OrderBlock, LiquidityLevel
)
from src.logger import get_logger

logger = get_logger(__name__)


class BacktestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BacktestPosition:
    """Position during backtest."""
    ticket: int
    direction: str  # 'BUY' or 'SELL'
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


@dataclass
class BacktestState:
    """Current state during backtest."""
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
        self.tm_config = config.trade_management
        self.smc_config = config.smc

        # Data
        self.candles: Dict[str, List[Dict]] = {}
        self.current_state: Optional[BacktestState] = None
        self.status = BacktestStatus.PENDING
        self.progress = 0.0
        self.report: Optional[BacktestReport] = None
        self.trades: List[BacktestTrade] = []

        # Ticket counter
        self._ticket_counter = 1000000

    def load_data(self, symbol: str, timeframes: List[str],
                  start_date: datetime, end_date: datetime) -> bool:
        """Load historical data for backtesting."""
        logger.info(f"Loading historical data for {symbol} from {start_date} to {end_date}")

        tf_map = {"m5": 5, "m15": 15, "h1": 60, "h4": 240}

        for tf in timeframes:
            tf_minutes = tf_map.get(tf)
            if not tf_minutes:
                continue

            # In real implementation, this would fetch from MT5 or data files
            # For now, generate synthetic data structure
            self.candles[tf] = []

        return True

    def run_backtest(self, config: BacktestConfig) -> BacktestReport:
        """Run a complete backtest."""
        self.status = BacktestStatus.RUNNING
        self.progress = 0.0

        logger.info(f"Starting backtest: {config.name}")

        try:
            # Initialize state
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

            # Get primary timeframe (M5 for execution)
            primary_tf = "m5"
            candles = self.candles.get(primary_tf, [])

            if not candles:
                logger.error("No candle data for backtest")
                self.status = BacktestStatus.FAILED
                return self._create_failed_report(config)

            total_candles = len(candles)
            lookback = 100

            # Main backtest loop
            for i in range(lookback, total_candles):
                self.current_state.current_index = i
                self.current_state.current_time = candles[i]["time"]
                self.progress = (i - lookback) / (total_candles - lookback) * 100

                # Update equity with open positions
                self._update_equity(candles[i])

                # Check for position exits (SL/TP)
                self._check_exits(candles[i])

                # Check for new signals (every 5 candles = 25 min on M5)
                if i % 5 == 0:
                    self._check_signals(candles, i, config)

                # Update drawdown
                self._update_drawdown()

                # Daily P/L reset
                if i > 0 and candles[i]["time"].date() != candles[i-1]["time"].date():
                    self.current_state.daily_pl = 0.0

            # Close any remaining positions at end
            self._close_all_positions(candles[-1])

            # Generate report
            self.report = self._generate_report(config)
            self.status = BacktestStatus.COMPLETED
            self.progress = 100.0

            logger.info(f"Backtest completed: {self.report.net_profit:.2f} profit, {self.report.total_trades} trades")

            return self.report

        except Exception as e:
            logger.exception(f"Backtest failed: {e}")
            self.status = BacktestStatus.FAILED
            return self._create_failed_report(config)

    def _update_equity(self, current_candle: Dict):
        """Update equity with floating P/L."""
        floating_pl = 0.0
        for pos in self.current_state.open_positions:
            if pos.direction == "BUY":
                pl_pips = (current_candle["close"] - pos.entry_price) / 0.01
            else:
                pl_pips = (pos.entry_price - current_candle["close"]) / 0.01

            # Simplified P/L calculation
            tick_value = 1.0  # $1 per pip per lot for XAUUSD (approximate)
            floating_pl += pl_pips * tick_value * pos.volume

        self.current_state.equity = self.current_state.balance + floating_pl

    def _check_exits(self, current_candle: Dict):
        """Check if any positions hit SL or TP."""
        high = current_candle["high"]
        low = current_candle["low"]
        close = current_candle["close"]
        current_time = current_candle["time"]

        remaining = []
        for pos in self.current_state.open_positions:
            closed = False

            if pos.direction == "BUY":
                # Check SL
                if low <= pos.sl:
                    pos.close_price = pos.sl
                    pos.close_time = current_time
                    pos.close_reason = "SL"
                    pos.realized_r = -1.0
                    closed = True
                # Check TP
                elif high >= pos.tp:
                    pos.close_price = pos.tp
                    pos.close_time = current_time
                    pos.close_reason = "TP"
                    risk = pos.entry_price - pos.sl
                    reward = pos.tp - pos.entry_price
                    pos.realized_r = reward / risk if risk > 0 else 0
                    closed = True
            else:
                # Check SL
                if high >= pos.sl:
                    pos.close_price = pos.sl
                    pos.close_time = current_time
                    pos.close_reason = "SL"
                    pos.realized_r = -1.0
                    closed = True
                # Check TP
                elif low <= pos.tp:
                    pos.close_price = pos.tp
                    pos.close_time = current_time
                    pos.close_reason = "TP"
                    risk = pos.sl - pos.entry_price
                    reward = pos.entry_price - pos.tp
                    pos.realized_r = reward / risk if risk > 0 else 0
                    closed = True

            if closed:
                # Calculate profit
                if pos.direction == "BUY":
                    profit_pips = (pos.close_price - pos.entry_price) / 0.01
                else:
                    profit_pips = (pos.entry_price - pos.close_price) / 0.01

                tick_value = 1.0
                gross = profit_pips * tick_value * pos.volume
                commission = config.backtesting.commission_per_lot * pos.volume
                swap = 0  # Simplified
                net = gross - commission + swap

                pos.gross_profit = gross
                pos.net_profit = net

                self.current_state.balance += net
                self.current_state.daily_pl += net
                self.current_state.closed_trades.append(self._position_to_trade(pos))
                self.trades.append(self._position_to_trade(pos))
            else:
                remaining.append(pos)

        self.current_state.open_positions = remaining

    def _check_signals(self, all_candles: List[Dict], current_index: int, config: BacktestConfig):
        """Check for new trade signals."""
        # Skip if max positions reached
        if len(self.current_state.open_positions) >= self.trading_config.max_positions:
            return

        # Skip if daily trade limit reached
        if len([t for t in self.current_state.closed_trades
                if t.entry_time.date() == all_candles[current_index]["time"].date()]) >= self.trading_config.max_daily_trades:
            return

        # Skip if daily loss limit reached
        if self.current_state.daily_pl < 0:
            daily_loss_pct = abs(self.current_state.daily_pl) / self.current_state.equity * 100
            if daily_loss_pct >= self.risk_config.daily_loss_limit_percent:
                return

        # Analyze market structure on H4, H1, M15
        h4_structure = self._analyze_structure("h4", all_candles, current_index)
        h1_structure = self._analyze_structure("h1", all_candles, current_index)
        m15_structure = self._analyze_structure("m15", all_candles, current_index)

        if not h4_structure or not h1_structure or not m15_structure:
            return

        # Get bias
        h4_bias = h4_structure.trend
        if h4_bias == "ranging":
            return

        # Check strategies
        signal = self._check_strategies(h4_bias, h1_structure, m15_structure, all_candles, current_index)

        if signal:
            self._execute_signal(signal, all_candles[current_index], config)

    def _analyze_structure(self, timeframe: str, all_candles: List[Dict], current_index: int) -> Optional[MarketStructure]:
        """Analyze market structure for a timeframe at a specific index."""
        # In real implementation, would use resampled data for each timeframe
        # For now, use primary timeframe data as approximation
        lookback = 50
        start = max(0, current_index - lookback)
        candles_slice = all_candles[start:current_index + 1]

        if len(candles_slice) < 20:
            return None

        highs = np.array([c["high"] for c in candles_slice])
        lows = np.array([c["low"] for c in candles_slice])
        closes = np.array([c["close"] for c in candles_slice])

        return analyze_market_structure(highs, lows, closes, lookback=min(lookback, len(candles_slice)), swing_lookback=5)

    def _check_strategies(self, h4_bias: str, h1_structure: MarketStructure, m15_structure: MarketStructure,
                          all_candles: List[Dict], current_index: int) -> Optional[Dict]:
        """Check all SMC strategies for a signal."""
        # This is a simplified version - full implementation would be more complex
        # Check liquidity sweep + FVG
        current_candle = all_candles[current_index]
        entry_price = current_candle["close"]

        # For demo, generate a simple signal based on bias
        if h4_bias == "bullish" and h1_structure.trend == "bullish":
            # Look for bullish setup
            sl = entry_price - 30 * 0.01
            tp = entry_price + 60 * 0.01
            return {
                "direction": "BUY",
                "entry": entry_price,
                "sl": sl,
                "tp": tp,
                "strategy": "liquidity_sweep_fvg",
                "risk_reward": 2.0
            }
        elif h4_bias == "bearish" and h1_structure.trend == "bearish":
            sl = entry_price + 30 * 0.01
            tp = entry_price - 60 * 0.01
            return {
                "direction": "SELL",
                "entry": entry_price,
                "sl": sl,
                "tp": tp,
                "strategy": "liquidity_sweep_fvg",
                "risk_reward": 2.0
            }

        return None

    def _execute_signal(self, signal: Dict, current_candle: Dict, config: BacktestConfig):
        """Execute a signal in backtest."""
        # Calculate position size
        equity = self.current_state.equity
        risk_amount = equity * (config.risk_percent / 100)

        if signal["direction"] == "BUY":
            sl_distance = signal["entry"] - signal["sl"]
        else:
            sl_distance = signal["sl"] - signal["entry"]

        sl_pips = sl_distance / 0.01
        tick_value = 1.0
        volume = risk_amount / (sl_pips * tick_value)

        # Round to step
        volume = round(volume / 0.01) * 0.01
        volume = max(0.01, min(100, volume))

        if volume < 0.01:
            return

        # Create position
        pos = BacktestPosition(
            ticket=self._ticket_counter,
            direction=signal["direction"],
            entry_price=signal["entry"],
            sl=signal["sl"],
            tp=signal["tp"],
            volume=volume,
            entry_time=current_candle["time"],
            entry_index=self.current_state.current_index,
            strategy=signal["strategy"],
            risk_percent=config.risk_percent
        )
        self._ticket_counter += 1

        self.current_state.open_positions.append(pos)

    def _close_all_positions(self, last_candle: Dict):
        """Close all remaining positions at end of backtest."""
        for pos in self.current_state.open_positions:
            pos.close_price = last_candle["close"]
            pos.close_time = last_candle["time"]
            pos.close_reason = "END_OF_BACKTEST"

            if pos.direction == "BUY":
                profit_pips = (pos.close_price - pos.entry_price) / 0.01
            else:
                profit_pips = (pos.entry_price - pos.close_price) / 0.01

            tick_value = 1.0
            gross = profit_pips * tick_value * pos.volume
            commission = config.backtesting.commission_per_lot * pos.volume
            net = gross - commission

            pos.gross_profit = gross
            pos.net_profit = net
            pos.realized_r = net / (pos.volume * abs(pos.entry_price - pos.sl) / 0.01 * tick_value) if pos.sl != pos.entry_price else 0

            self.current_state.balance += net
            self.current_state.closed_trades.append(self._position_to_trade(pos))
            self.trades.append(self._position_to_trade(pos))

        self.current_state.open_positions = []

    def _position_to_trade(self, pos: BacktestPosition) -> BacktestTrade:
        """Convert position to BacktestTrade record."""
        return BacktestTrade(
            report_id=0,  # Will be set later
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
        """Update drawdown metrics."""
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
        """Generate backtest report."""
        trades = self.current_state.closed_trades

        if not trades:
            return BacktestReport(
                config_id=config.id,
                start_date=config.start_date,
                end_date=config.end_date,
                initial_capital=config.initial_balance,
                final_capital=self.current_state.equity,
                environment="backtest"
            )

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

        # Sharpe ratio (simplified)
        returns = [t.net_profit / config.initial_balance for t in trades]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Sortino ratio
        downside_returns = [r for r in returns if r < 0]
        sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252) if downside_returns and np.std(downside_returns) > 0 else 0

        # Recovery factor
        recovery = net_profit / self.current_state.max_drawdown if self.current_state.max_drawdown > 0 else 0

        # Consecutive wins/losses
        max_consec_wins = max_consec_losses = 0
        current_streak = 0
        for t in trades:
            if t.net_profit > 0:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_consec_wins = max(max_consec_wins, current_streak)
            else:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_consec_losses = max(max_consec_losses, abs(current_streak))

        # Long/Short performance
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

        # Save trades with report_id
        for trade in self.trades:
            trade.report_id = report.id
            db_manager.add(trade)

        db_manager.add(report)

        return report

    def _create_failed_report(self, config: BacktestConfig) -> BacktestReport:
        """Create a failed report."""
        return BacktestReport(
            config_id=config.id,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_balance,
            final_capital=config.initial_balance,
            environment="backtest"
        )

    def get_progress(self) -> float:
        return self.progress

    def get_status(self) -> BacktestStatus:
        return self.status


def run_backtest_async(config: BacktestConfig) -> BacktestReport:
    """Run backtest synchronously."""
    engine = BacktestEngine()
    return engine.run_backtest(config)