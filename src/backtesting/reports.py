"""Backtest report generation and export."""
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import json
import csv
from pathlib import Path
from io import StringIO, BytesIO

from src.database_manager import db_manager, BacktestReport, BacktestTrade, BacktestConfig
from src.backtesting.metrics import calculate_metrics, TradeMetrics
from src.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generates and exports backtest reports."""

    def __init__(self):
        self.reports_dir = Path("data/historical/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, report: BacktestReport, trades: List[BacktestTrade],
                        config: BacktestConfig) -> Dict[str, Any]:
        """Generate comprehensive report dictionary."""
        # Convert trades to metrics format
        trade_metrics = []
        for t in trades:
            trade_metrics.append(TradeMetrics(
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                direction=t.direction,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                volume=t.volume,
                gross_profit=t.gross_profit,
                net_profit=t.net_profit,
                realized_r=t.realized_r,
                holding_hours=t.holding_hours if hasattr(t, 'holding_hours') else 0
            ))

        # Calculate metrics
        metrics = calculate_metrics(trade_metrics, config.initial_balance)

        return {
            "config": {
                "name": config.name,
                "start_date": config.start_date.isoformat(),
                "end_date": config.end_date.isoformat(),
                "initial_balance": config.initial_balance,
                "risk_percent": config.risk_percent,
                "min_risk_reward": config.min_risk_reward,
                "timeframes": config.timeframes,
                "sessions": config.sessions,
                "strategies": config.strategies,
                "spread_pips": config.spread_pips,
                "slippage_pips": config.slippage_pips,
                "commission_per_lot": config.commission_per_lot,
                "swap_enabled": config.swap_enabled
            },
            "summary": {
                "start_date": report.start_date.isoformat(),
                "end_date": report.end_date.isoformat(),
                "initial_capital": report.initial_capital,
                "final_capital": report.final_capital,
                "net_profit": report.net_profit,
                "net_return_pct": report.net_return_pct,
                "total_trades": report.total_trades,
                "win_rate": report.win_rate,
                "profit_factor": report.profit_factor,
                "max_drawdown": report.max_drawdown,
                "max_drawdown_pct": report.max_drawdown_pct,
                "expectancy": report.expectancy,
                "sharpe_ratio": report.sharpe_ratio,
                "sortino_ratio": report.sortino_ratio,
                "recovery_factor": report.recovery_factor
            },
            "detailed_metrics": metrics,
            "trades": [self._trade_to_dict(t) for t in trades],
            "equity_curve": metrics.get("equity_curve", []),
            "monthly_returns": metrics.get("monthly_returns", {}),
            "daily_returns": metrics.get("daily_returns", {}),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    def _trade_to_dict(self, trade: BacktestTrade) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "ticket": trade.ticket,
            "direction": trade.direction,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat(),
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "sl": trade.sl,
            "tp": trade.tp,
            "volume": trade.volume,
            "risk_percent": trade.risk_percent,
            "realized_r": trade.realized_r,
            "gross_profit": trade.gross_profit,
            "net_profit": trade.net_profit,
            "strategy": trade.strategy,
            "exit_reason": trade.exit_reason,
            "bos_confirmed": trade.bos_confirmed,
            "choch_confirmed": trade.choch_confirmed,
            "liquidity_sweep": trade.liquidity_sweep,
            "ob_confirmed": trade.ob_confirmed,
            "fvg_confirmed": trade.fvg_confirmed,
            "session": trade.session
        }

    def export_json(self, report_data: Dict[str, Any], filename: str = None) -> str:
        """Export report as JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{timestamp}.json"

        filepath = self.reports_dir / filename
        with open(filepath, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"Report exported to {filepath}")
        return str(filepath)

    def export_csv(self, trades: List[BacktestTrade], filename: str = None) -> str:
        """Export trades as CSV."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_trades_{timestamp}.csv"

        filepath = self.reports_dir / filename

        fieldnames = [
            "ticket", "direction", "entry_time", "exit_time",
            "entry_price", "exit_price", "sl", "tp", "volume",
            "risk_percent", "realized_r", "gross_profit", "net_profit",
            "strategy", "exit_reason", "bos_confirmed", "choch_confirmed",
            "liquidity_sweep", "ob_confirmed", "fvg_confirmed", "session"
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for trade in trades:
                writer.writerow(self._trade_to_dict(trade))

        logger.info(f"Trades exported to {filepath}")
        return str(filepath)

    def export_pdf(self, report_data: Dict[str, Any], filename: str = None) -> str:
        """Export report as PDF (placeholder - requires fpdf2 or reportlab)."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{timestamp}.pdf"

        filepath = self.reports_dir / filename

        # For now, create a text-based report that can be converted to PDF
        text_content = self._generate_text_report(report_data)
        text_file = filepath.with_suffix(".txt")
        with open(text_file, "w") as f:
            f.write(text_content)

        logger.info(f"Text report exported to {text_file} (PDF generation requires additional library)")
        return str(text_file)

    def _generate_text_report(self, report_data: Dict[str, Any]) -> str:
        """Generate text report."""
        lines = []
        lines.append("=" * 60)
        lines.append("XAU/USD SMC BACKTEST REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {report_data['generated_at']}")
        lines.append("")

        # Config
        cfg = report_data["config"]
        lines.append("CONFIGURATION")
        lines.append("-" * 40)
        lines.append(f"Name: {cfg['name']}")
        lines.append(f"Period: {cfg['start_date']} to {cfg['end_date']}")
        lines.append(f"Initial Balance: ${cfg['initial_balance']:,.2f}")
        lines.append(f"Risk per Trade: {cfg['risk_percent']}%")
        lines.append(f"Min R:R: 1:{cfg['min_risk_reward']}")
        lines.append(f"Spread: {cfg['spread_pips']} pips")
        lines.append(f"Slippage: {cfg['slippage_pips']} pips")
        lines.append(f"Commission: ${cfg['commission_per_lot']}/lot")
        lines.append("")

        # Summary
        s = report_data["summary"]
        lines.append("PERFORMANCE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Initial Capital: ${s['initial_capital']:,.2f}")
        lines.append(f"Final Capital: ${s['final_capital']:,.2f}")
        lines.append(f"Net Profit: ${s['net_profit']:+,.2f}")
        lines.append(f"Net Return: {s['net_return_pct']:+.2f}%")
        lines.append(f"Total Trades: {s['total_trades']}")
        lines.append(f"Win Rate: {s['win_rate']:.1f}%")
        lines.append(f"Profit Factor: {s['profit_factor']:.2f}")
        lines.append(f"Max Drawdown: ${s['max_drawdown']:,.2f} ({s['max_drawdown_pct']:.2f}%)")
        lines.append(f"Expectancy: ${s['expectancy']:,.2f}")
        lines.append(f"Sharpe Ratio: {s['sharpe_ratio']:.2f}")
        lines.append(f"Sortino Ratio: {s['sortino_ratio']:.2f}")
        lines.append(f"Recovery Factor: {s['recovery_factor']:.2f}")
        lines.append("")

        # Detailed metrics
        m = report_data["detailed_metrics"]
        lines.append("DETAILED METRICS")
        lines.append("-" * 40)
        lines.append(f"Gross Profit: ${m['gross_profit']:,.2f}")
        lines.append(f"Gross Loss: ${m['gross_loss']:,.2f}")
        lines.append(f"Avg Win: ${m['avg_win']:,.2f}")
        lines.append(f"Avg Loss: ${m['avg_loss']:,.2f}")
        lines.append(f"Avg Realized R: {m['avg_realized_r']:.2f}")
        lines.append(f"Avg Risk/Reward: {m['avg_risk_reward']:.2f}")
        lines.append(f"Largest Win: ${m['largest_win']:,.2f}")
        lines.append(f"Largest Loss: ${m['largest_loss']:,.2f}")
        lines.append(f"Avg Holding: {m['avg_holding_hours']:.1f} hours")
        lines.append(f"Max Consecutive Wins: {m['max_consecutive_wins']}")
        lines.append(f"Max Consecutive Losses: {m['max_consecutive_losses']}")
        lines.append("")

        # Long/Short
        lines.append("LONG/SHORT BREAKDOWN")
        lines.append("-" * 40)
        lines.append(f"Long Trades: {m['long_trades']} (WR: {m['long_win_rate']:.1f}%, P/L: ${m['long_profit']:+,.2f})")
        lines.append(f"Short Trades: {m['short_trades']} (WR: {m['short_win_rate']:.1f}%, P/L: ${m['short_profit']:+,.2f})")
        lines.append("")

        # Monthly returns
        lines.append("MONTHLY RETURNS")
        lines.append("-" * 40)
        for month, ret in report_data["monthly_returns"].items():
            lines.append(f"  {month}: {ret:+.2f}%")
        lines.append("")

        # Trades
        lines.append("TRADE LOG")
        lines.append("-" * 40)
        for trade in report_data["trades"][:20]:  # First 20
            lines.append(f"{trade['entry_time'][:10]} {trade['direction']} "
                         f"Entry:{trade['entry_price']:.2f} Exit:{trade['exit_price']:.2f} "
                         f"R:{trade['realized_r']:+.2f} P/L:${trade['net_profit']:+,.2f} "
                         f"{trade['strategy']} {trade['exit_reason']}")

        if len(report_data["trades"]) > 20:
            lines.append(f"... and {len(report_data['trades']) - 20} more trades")

        return "\n".join(lines)


def get_report_summary(report_id: int) -> Optional[Dict[str, Any]]:
    """Get summary of a backtest report."""
    report = db_manager.get(BacktestReport, report_id)
    if not report:
        return None

    config = db_manager.get(BacktestConfig, report.config_id) if report.config_id else None
    trades = db_manager.get_backtest_trades(report_id)

    generator = ReportGenerator()
    return generator.generate_report(report, trades, config) if config else None


def list_reports(limit: int = 50) -> List[Dict[str, Any]]:
    """List recent backtest reports."""
    reports = db_manager.get_backtest_reports(limit)
    return [
        {
            "id": r.id,
            "config_id": r.config_id,
            "start_date": r.start_date.isoformat(),
            "end_date": r.end_date.isoformat(),
            "initial_capital": r.initial_capital,
            "final_capital": r.final_capital,
            "net_profit": r.net_profit,
            "net_return_pct": r.net_return_pct,
            "total_trades": r.total_trades,
            "win_rate": r.win_rate,
            "profit_factor": r.profit_factor,
            "max_drawdown_pct": r.max_drawdown_pct,
            "completed_at": r.completed_at.isoformat()
        }
        for r in reports
    ]


def compare_reports(report_ids: List[int]) -> Dict[str, Any]:
    """Compare multiple backtest reports."""
    reports = []
    for rid in report_ids:
        summary = get_report_summary(rid)
        if summary:
            reports.append(summary)

    if not reports:
        return {"error": "No valid reports"}

    comparison = {
        "reports": [],
        "best_net_profit": None,
        "best_win_rate": None,
        "best_profit_factor": None,
        "best_sharpe": None,
        "lowest_drawdown": None
    }

    for r in reports:
        summary = r["summary"]
        comparison["reports"].append({
            "id": r["config"]["name"] if "config" in r else "unknown",
            "net_profit": summary["net_profit"],
            "net_return_pct": summary["net_return_pct"],
            "total_trades": summary["total_trades"],
            "win_rate": summary["win_rate"],
            "profit_factor": summary["profit_factor"],
            "max_drawdown_pct": summary["max_drawdown_pct"],
            "sharpe_ratio": summary["sharpe_ratio"],
            "expectancy": summary["expectancy"]
        })

    # Find best
    valid = [r for r in comparison["reports"] if r["net_profit"] is not None]
    if valid:
        comparison["best_net_profit"] = max(valid, key=lambda x: x["net_profit"])
        comparison["best_win_rate"] = max(valid, key=lambda x: x["win_rate"])
        comparison["best_profit_factor"] = max(valid, key=lambda x: x["profit_factor"] if x["profit_factor"] != float('inf') else 0)
        comparison["best_sharpe"] = max(valid, key=lambda x: x["sharpe_ratio"])
        comparison["lowest_drawdown"] = min(valid, key=lambda x: x["max_drawdown_pct"])

    return comparison