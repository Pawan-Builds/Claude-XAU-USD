"""Backtesting performance metrics."""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TradeMetrics:
    """Metrics for a single trade."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    volume: float
    gross_profit: float
    net_profit: float
    realized_r: float
    holding_hours: float


def calculate_metrics(trades: List[TradeMetrics], initial_capital: float) -> Dict[str, Any]:
    """Calculate comprehensive backtest metrics."""
    if not trades:
        return _empty_metrics()

    # Basic counts
    total_trades = len(trades)
    winners = [t for t in trades if t.net_profit > 0]
    losers = [t for t in trades if t.net_profit < 0]
    winning_trades = len(winners)
    losing_trades = len(losers)
    win_rate = winning_trades / total_trades * 100

    # Profit/Loss
    gross_profit = sum(t.gross_profit for t in winners)
    gross_loss = abs(sum(t.gross_profit for t in losers))
    net_profit = sum(t.net_profit for t in trades)
    net_return_pct = net_profit / initial_capital * 100

    # Profit factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Average win/loss
    avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
    avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0

    # Expectancy
    expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)

    # R-metrics
    realized_rs = [t.realized_r for t in trades]
    avg_realized_r = np.mean(realized_rs)
    median_realized_r = np.median(realized_rs)

    # Risk/Reward
    risk_rewards = [abs(t.realized_r) for t in trades if t.realized_r != 0]
    avg_risk_reward = np.mean(risk_rewards) if risk_rewards else 0

    # Drawdown
    equity_curve = _calculate_equity_curve(trades, initial_capital)
    max_drawdown, max_drawdown_pct = _calculate_max_drawdown(equity_curve, initial_capital)

    # Sharpe ratio
    returns = _calculate_returns(trades, initial_capital)
    sharpe_ratio = _calculate_sharpe(returns)
    sortino_ratio = _calculate_sortino(returns)

    # Recovery factor
    recovery_factor = net_profit / max_drawdown if max_drawdown > 0 else 0

    # Largest trades
    largest_win = max(t.net_profit for t in trades)
    largest_loss = min(t.net_profit for t in trades)

    # Holding duration
    avg_holding = np.mean([t.holding_hours for t in trades])

    # Consecutive wins/losses
    max_consec_wins, max_consec_losses = _calculate_consecutive(trades)

    # Long/Short breakdown
    long_trades = [t for t in trades if t.direction == "BUY"]
    short_trades = [t for t in trades if t.direction == "SELL"]

    long_wins = [t for t in long_trades if t.net_profit > 0]
    short_wins = [t for t in short_trades if t.net_profit > 0]

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": net_profit,
        "net_return_pct": net_return_pct,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "avg_realized_r": avg_realized_r,
        "median_realized_r": median_realized_r,
        "avg_risk_reward": avg_risk_reward,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "recovery_factor": recovery_factor,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "avg_holding_hours": avg_holding,
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "long_trades": len(long_trades),
        "long_win_rate": len(long_wins) / len(long_trades) * 100 if long_trades else 0,
        "long_profit": sum(t.net_profit for t in long_trades),
        "short_trades": len(short_trades),
        "short_win_rate": len(short_wins) / len(short_trades) * 100 if short_trades else 0,
        "short_profit": sum(t.net_profit for t in short_trades),
        "equity_curve": equity_curve,
        "monthly_returns": _calculate_monthly_returns(trades, initial_capital),
        "daily_returns": _calculate_daily_returns(trades, initial_capital),
    }


def _empty_metrics() -> Dict[str, Any]:
    return {
        "total_trades": 0, "winning_trades": 0, "losing_trades": 0, "win_rate": 0,
        "gross_profit": 0, "gross_loss": 0, "net_profit": 0, "net_return_pct": 0,
        "profit_factor": 0, "avg_win": 0, "avg_loss": 0, "expectancy": 0,
        "avg_realized_r": 0, "median_realized_r": 0, "avg_risk_reward": 0,
        "max_drawdown": 0, "max_drawdown_pct": 0, "sharpe_ratio": 0, "sortino_ratio": 0,
        "recovery_factor": 0, "largest_win": 0, "largest_loss": 0, "avg_holding_hours": 0,
        "max_consecutive_wins": 0, "max_consecutive_losses": 0,
        "long_trades": 0, "long_win_rate": 0, "long_profit": 0,
        "short_trades": 0, "short_win_rate": 0, "short_profit": 0,
        "equity_curve": [], "monthly_returns": {}, "daily_returns": {}
    }


def _calculate_equity_curve(trades: List[TradeMetrics], initial_capital: float) -> List[Dict]:
    """Calculate equity curve points."""
    curve = [{"date": trades[0].entry_time.isoformat() if trades else datetime.now().isoformat(),
              "equity": initial_capital, "balance": initial_capital, "drawdown": 0, "drawdown_pct": 0}]

    equity = initial_capital
    peak = initial_capital

    for trade in sorted(trades, key=lambda x: x.exit_time):
        equity += trade.net_profit
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        drawdown_pct = drawdown / peak * 100 if peak > 0 else 0

        curve.append({
            "date": trade.exit_time.isoformat(),
            "equity": equity,
            "balance": equity,
            "drawdown": drawdown,
            "drawdown_pct": drawdown_pct
        })

    return curve


def _calculate_max_drawdown(equity_curve: List[Dict], initial_capital: float) -> Tuple[float, float]:
    """Calculate maximum drawdown."""
    max_dd = 0
    max_dd_pct = 0
    peak = initial_capital

    for point in equity_curve:
        equity = point["equity"]
        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_pct = dd / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    return max_dd, max_dd_pct


def _calculate_returns(trades: List[TradeMetrics], initial_capital: float) -> List[float]:
    """Calculate period returns."""
    returns = []
    equity = initial_capital

    for trade in sorted(trades, key=lambda x: x.exit_time):
        ret = trade.net_profit / equity
        returns.append(ret)
        equity += trade.net_profit

    return returns


def _calculate_sharpe(returns: List[float], risk_free: float = 0.02) -> float:
    """Calculate Sharpe ratio (annualized)."""
    if not returns or len(returns) < 2:
        return 0

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    if std_ret == 0:
        return 0

    # Annualize (assuming daily returns, 252 trading days)
    sharpe = (mean_ret * 252 - risk_free) / (std_ret * np.sqrt(252))
    return sharpe


def _calculate_sortino(returns: List[float], risk_free: float = 0.02) -> float:
    """Calculate Sortino ratio (annualized)."""
    if not returns or len(returns) < 2:
        return 0

    mean_ret = np.mean(returns)
    downside_returns = [r for r in returns if r < 0]

    if not downside_returns:
        return float('inf')

    downside_std = np.std(downside_returns, ddof=1)

    if downside_std == 0:
        return 0

    sortino = (mean_ret * 252 - risk_free) / (downside_std * np.sqrt(252))
    return sortino


def _calculate_consecutive(trades: List[TradeMetrics]) -> Tuple[int, int]:
    """Calculate max consecutive wins/losses."""
    max_wins = max_losses = 0
    current = 0

    for trade in sorted(trades, key=lambda x: x.exit_time):
        if trade.net_profit > 0:
            if current >= 0:
                current += 1
            else:
                current = 1
            max_wins = max(max_wins, current)
        else:
            if current <= 0:
                current -= 1
            else:
                current = -1
            max_losses = max(max_losses, abs(current))

    return max_wins, max_losses


def _calculate_monthly_returns(trades: List[TradeMetrics], initial_capital: float) -> Dict[str, float]:
    """Calculate monthly returns."""
    from collections import defaultdict
    monthly = defaultdict(float)
    equity = initial_capital

    for trade in sorted(trades, key=lambda x: x.exit_time):
        key = trade.exit_time.strftime("%Y-%m")
        ret = trade.net_profit / equity * 100
        monthly[key] += ret
        equity += trade.net_profit

    return dict(monthly)


def _calculate_daily_returns(trades: List[TradeMetrics], initial_capital: float) -> Dict[str, float]:
    """Calculate daily returns."""
    from collections import defaultdict
    daily = defaultdict(float)
    equity = initial_capital

    for trade in sorted(trades, key=lambda x: x.exit_time):
        key = trade.exit_time.strftime("%Y-%m-%d")
        ret = trade.net_profit / equity * 100
        daily[key] += ret
        equity += trade.net_profit

    return dict(daily)


def calculate_strategy_metrics(trades: List[TradeMetrics], initial_capital: float) -> Dict[str, Dict]:
    """Calculate metrics per strategy."""
    from collections import defaultdict

    strategy_trades = defaultdict(list)
    for trade in trades:
        strategy_trades[trade.strategy].append(trade)

    results = {}
    for strategy, strat_trades in strategy_trades.items():
        results[strategy] = calculate_metrics(strat_trades, initial_capital)

    return results


def calculate_session_metrics(trades: List[TradeMetrics], initial_capital: float) -> Dict[str, Dict]:
    """Calculate metrics per trading session."""
    from collections import defaultdict

    session_trades = defaultdict(list)
    for trade in trades:
        hour = trade.entry_time.hour
        if 8 <= hour < 17:
            session = "london"
        elif 13 <= hour < 22:
            session = "new_york"
        elif 0 <= hour < 9:
            session = "asian"
        else:
            session = "other"
        session_trades[session].append(trade)

    results = {}
    for session, sess_trades in session_trades.items():
        results[session] = calculate_metrics(sess_trades, initial_capital)

    return results