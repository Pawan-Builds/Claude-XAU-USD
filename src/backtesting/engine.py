"""Genuine backtesting engine for SMC strategies using real MT5 data.

M-04 execution-cost model (see `assumptions_applied` at runtime):
- Spread: config.backtesting.spread_pips, converted to price via
  broker pip_size, applied as half-spread on each side of every fill.
- Slippage: config.backtesting.slippage_pips, applied adversely on
  entry and on stop-loss exits. Take-profit exits are assumed to be
  limit fills with no slippage.
- Tick economics: broker tick_size / tick_value from MT5 symbol info.
  P&L = price_delta * (tick_value / tick_size) * volume.
- Volume: risk-based, rounded DOWN to broker volume_step. Below
  volume_min -> trade skipped (fail-closed). Above volume_max ->
  capped at volume_max.
- Commission: config.backtesting.commission_per_lot, per side if
  commission_per_side, else once at close.
- Swap: applied at close only, as configured swap_long/short *
  volume * integer_days_held (usd_per_lot_per_day model).

Candle prices are treated as MID prices (bid = mid - half_spread,
ask = mid + half_spread). This matches the pre-existing convention.
"""
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from src.config_loader import config
from src.database_manager import db_manager, BacktestConfig, BacktestReport, BacktestTrade
from src.mt5_connector import mt5_connector, MT5Connector
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

        # M-04: broker-aware economics; populated at run_backtest start.
        # None here means "not yet loaded"; methods guard defensively.
        self._broker_spec: Optional[Dict[str, Any]] = None
        self._cost_config: Optional[Dict[str, Any]] = None

        # M-04: what the backtest actually modelled. Read via
        # get_assumptions_applied() after run_backtest completes.
        self.assumptions_applied: Dict[str, Any] = {}

        # Legacy fields retained for compatibility. Updated from broker
        # spec at run_backtest start; pre-M-04 fallbacks used only
        # before run_backtest loads spec (methods guard against this).
        self.point_value = 0.01
        self.pip_value = 0.1

    # ------------------------------------------------------------------
    # M-04: broker spec and cost configuration
    # ------------------------------------------------------------------

    def _load_broker_spec(self) -> Optional[Dict[str, Any]]:
        """Load and validate broker symbol spec. None = fail closed."""
        try:
            info = mt5_connector.get_symbol_info()
        except Exception:
            logger.exception(
                "Broker spec lookup raised; backtest cannot proceed."
            )
            return None

        if info is None:
            logger.error(
                "Broker symbol spec unavailable; refusing to run "
                "backtest with hardcoded fallback economics."
            )
            return None

        try:
            tick_size = float(info.tick_size)
            tick_value = float(info.tick_value)
            contract_size = float(info.contract_size)
            volume_min = float(info.volume_min)
            volume_max = float(info.volume_max)
            volume_step = float(info.volume_step)
            point = float(info.point)
            swap_long = float(info.swap_long)
            swap_short = float(info.swap_short)
            pip_size = float(MT5Connector.pip_size_for(info))
        except (TypeError, ValueError, AttributeError) as exc:
            logger.error(
                "Broker spec has non-numeric fields: %s", exc
            )
            return None

        if tick_size <= 0 or tick_value <= 0 or contract_size <= 0:
            logger.error(
                "Broker spec invalid: tick_size=%s tick_value=%s "
                "contract_size=%s", tick_size, tick_value, contract_size,
            )
            return None

        if volume_min <= 0 or volume_step <= 0 or volume_max < volume_min:
            logger.error(
                "Broker spec invalid: volume_min=%s volume_step=%s "
                "volume_max=%s", volume_min, volume_step, volume_max,
            )
            return None

        if pip_size <= 0 or point <= 0:
            logger.error(
                "Broker spec invalid: pip_size=%s point=%s",
                pip_size, point,
            )
            return None

        return {
            "tick_size": tick_size,
            "tick_value": tick_value,
            "contract_size": contract_size,
            "volume_min": volume_min,
            "volume_max": volume_max,
            "volume_step": volume_step,
            "point": point,
            "pip_size": pip_size,
            "swap_long": swap_long,
            "swap_short": swap_short,
        }

    def _load_cost_config(self, bt_config: BacktestConfig) -> Dict[str, Any]:
        """Read cost params from the per-backtest DB row, falling back
        to config.backtesting defaults from yaml."""
        def _pick(name: str, default: Any, numeric: bool = True) -> Any:
            value = getattr(bt_config, name, None)
            if value is None:
                value = getattr(self.backtest_config, name, default)
            if numeric:
                return self._coerce_non_negative(value, default)
            return value if value is not None else default

        commission_per_side_raw = getattr(
            bt_config, "commission_per_side", None
        )
        if commission_per_side_raw is None:
            commission_per_side_raw = getattr(
                self.backtest_config, "commission_per_side", True
            )

        swap_enabled_raw = getattr(bt_config, "swap_enabled", None)
        if swap_enabled_raw is None:
            swap_enabled_raw = getattr(
                self.backtest_config, "swap_enabled", True
            )

        swap_unit_raw = getattr(bt_config, "swap_unit", None)
        if swap_unit_raw is None:
            swap_unit_raw = getattr(
                self.backtest_config, "swap_unit", "usd_per_lot_per_day"
            )
        swap_unit = str(swap_unit_raw).strip().lower()
        if swap_unit not in ("none", "usd_per_lot_per_day"):
            logger.warning(
                "Unknown swap_unit=%r; falling back to "
                "'usd_per_lot_per_day'.", swap_unit_raw,
            )
            swap_unit = "usd_per_lot_per_day"

        return {
            "spread_pips": _pick("spread_pips", 2.0),
            "slippage_pips": _pick("slippage_pips", 1.0),
            "commission_per_lot": _pick("commission_per_lot", 7.0),
            "commission_per_side": bool(commission_per_side_raw),
            "swap_enabled": bool(swap_enabled_raw),
            "swap_unit": swap_unit,
        }

    @staticmethod
    def _coerce_non_negative(value: Any, default: float) -> float:
        """Return finite non-negative float, else default."""
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(result) or result < 0:
            return float(default)
        return result

    # ------------------------------------------------------------------
    # M-04: P&L, volume, entry/exit costs, swap
    # ------------------------------------------------------------------

    def _pl_per_lot_per_price(self) -> float:
        """USD P&L per 1.0 lot per 1.0 price unit, from broker spec."""
        spec = self._broker_spec
        return spec["tick_value"] / spec["tick_size"]

    def _calculate_gross_pl(
        self,
        direction: str,
        entry_price: float,
        exit_price: float,
        volume: float,
    ) -> float:
        """Gross P&L using broker tick economics."""
        delta = (
            exit_price - entry_price
            if direction == "BUY"
            else entry_price - exit_price
        )
        return delta * self._pl_per_lot_per_price() * volume

    def _calculate_volume(
        self, risk_amount: float, sl_distance: float
    ) -> Optional[float]:
        """Risk-based volume, floor-rounded to broker volume_step.

        Returns None when the risk budget cannot afford volume_min
        (fail closed). Caps at volume_max when raw_volume would exceed
        the broker limit.
        """
        spec = self._broker_spec
        if spec is None:
            return None

        if sl_distance <= 0 or risk_amount <= 0:
            return None

        pl_per_lot_at_sl = sl_distance * self._pl_per_lot_per_price()
        if pl_per_lot_at_sl <= 0:
            return None

        raw_volume = risk_amount / pl_per_lot_at_sl

        try:
            raw_dec = Decimal(str(raw_volume))
            min_dec = Decimal(str(spec["volume_min"]))
            max_dec = Decimal(str(spec["volume_max"]))
            step_dec = Decimal(str(spec["volume_step"]))
        except Exception:
            return None

        if raw_dec < min_dec:
            # Risk budget cannot cover minimum lot -> skip trade.
            return None

        if raw_dec > max_dec:
            return float(max_dec)

        steps = ((raw_dec - min_dec) / step_dec).to_integral_value(
            rounding=ROUND_DOWN
        )
        volume = min_dec + steps * step_dec

        if volume < min_dec or volume > max_dec:
            return None

        return float(volume)

    def _apply_entry_costs(self, side: str, signal_price: float) -> float:
        """Adverse entry fill: half-spread + slippage."""
        spec = self._broker_spec
        cost = self._cost_config
        pip_size = spec["pip_size"]

        half_spread = cost["spread_pips"] * pip_size / 2.0
        slippage = cost["slippage_pips"] * pip_size

        if side == "BUY":
            return signal_price + half_spread + slippage
        return signal_price - half_spread - slippage

    def _apply_exit_costs(
        self, side: str, exit_price: float, is_limit: bool
    ) -> float:
        """Adverse exit fill. Limit exits (TP) get no slippage."""
        spec = self._broker_spec
        cost = self._cost_config
        pip_size = spec["pip_size"]

        slippage = 0.0 if is_limit else cost["slippage_pips"] * pip_size

        if side == "BUY":
            return exit_price - slippage
        return exit_price + slippage

    def _apply_swap(
        self,
        direction: str,
        volume: float,
        entry_time: datetime,
        close_time: datetime,
    ) -> float:
        """Swap P&L (USD). Zero when disabled or model unsupported."""
        cost = self._cost_config
        spec = self._broker_spec

        if not cost["swap_enabled"] or cost["swap_unit"] == "none":
            return 0.0

        if entry_time is None or close_time is None:
            return 0.0

        days_held = max(0, (close_time - entry_time).days)
        if days_held <= 0:
            return 0.0

        rate = (
            spec["swap_long"]
            if direction == "BUY"
            else spec["swap_short"]
        )
        return float(rate) * volume * days_held

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, symbol: str, timeframes: List[str],
                  start_date: datetime, end_date: datetime) -> bool:
        """Load genuine historical data from MT5 and resample."""
        logger.info(
            "Loading historical data for %s from %s to %s",
            symbol, start_date, end_date,
        )

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

        logger.info("Loaded %d base M5 candles.", len(df))

        resample_rules = {
            "m5": "5min",
            "m15": "15min",
            "h1": "1h",
            "h4": "4h",
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

            logger.info("Resampled %s: %d candles.", tf, len(self.candles[tf]))

        return True

    # ------------------------------------------------------------------
    # Backtest loop
    # ------------------------------------------------------------------

    def run_backtest(self, config: BacktestConfig) -> BacktestReport:
        self.status = BacktestStatus.RUNNING
        self.progress = 0.0

        # M-04: load broker spec BEFORE anything else. Fail closed if
        # unavailable -- never fall back to hardcoded economics.
        self._broker_spec = self._load_broker_spec()
        if self._broker_spec is None:
            logger.error(
                "Backtest aborted: broker symbol spec unavailable. "
                "M-04 refuses to model costs with hardcoded fallbacks."
            )
            self.status = BacktestStatus.FAILED
            return self._create_failed_report(config)

        # Reflect broker pip/point onto legacy fields for compatibility.
        self.pip_value = self._broker_spec["pip_size"]
        self.point_value = self._broker_spec["point"]

        self._cost_config = self._load_cost_config(config)

        self.assumptions_applied = {
            "spread_model": "half_per_side",
            "spread_pips": self._cost_config["spread_pips"],
            "slippage_model": "entry_and_sl_exit",
            "slippage_pips": self._cost_config["slippage_pips"],
            "commission_per_lot": self._cost_config["commission_per_lot"],
            "commission_per_side": self._cost_config["commission_per_side"],
            "volume_rounding": "floor_to_step",
            "volume_below_min": "skip_trade",
            "volume_above_max": "cap_at_max",
            "swap_model": (
                self._cost_config["swap_unit"]
                if self._cost_config["swap_enabled"]
                else "none"
            ),
            "candle_basis": "mid",
            "pip_size": self.pip_value,
            "point": self.point_value,
            "tick_size": self._broker_spec["tick_size"],
            "tick_value": self._broker_spec["tick_value"],
            "contract_size": self._broker_spec["contract_size"],
        }
        logger.info("Backtest assumptions: %s", self.assumptions_applied)

        logger.info("Starting backtest: %s", config.name)

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

                self._check_exits(current_candle, current_time, config)
                self._update_equity(current_candle)

                if i % 5 == 0 and len(self.current_state.open_positions) < self.trading_config.max_positions:
                    self._check_signals(df, i, smc_engine, config)

                self._update_drawdown()

                if i > 0 and current_time.date() != df.index[i-1].date():
                    self.current_state.daily_pl = 0.0

            self._close_all_positions(df.iloc[-1], df.index[-1], config)
            self.report = self._generate_report(config)
            self.status = BacktestStatus.COMPLETED
            self.progress = 100.0

            logger.info(
                "Backtest completed: %.2f profit, %d trades",
                self.report.net_profit, self.report.total_trades,
            )
            return self.report

        except Exception as e:
            logger.exception("Backtest failed: %s", e)
            self.status = BacktestStatus.FAILED
            return self._create_failed_report(config)

    # ------------------------------------------------------------------
    # Equity, exits, entries
    # ------------------------------------------------------------------

    def _update_equity(self, current_candle: pd.Series):
        spec = self._broker_spec
        cost = self._cost_config
        if spec is None or cost is None:
            return

        pip_size = spec["pip_size"]
        half_spread = cost["spread_pips"] * pip_size / 2.0
        pl_per_lot = self._pl_per_lot_per_price()

        mid = float(current_candle["close"])
        bid = mid - half_spread
        ask = mid + half_spread

        floating_pl = 0.0
        for pos in self.current_state.open_positions:
            if pos.direction == "BUY":
                delta = bid - pos.entry_price
            else:
                delta = pos.entry_price - ask
            floating_pl += delta * pl_per_lot * pos.volume

        self.current_state.equity = self.current_state.balance + floating_pl

    def _check_exits(
        self,
        current_candle: pd.Series,
        current_time: datetime,
        config: BacktestConfig,
    ):
        spec = self._broker_spec
        cost = self._cost_config
        if spec is None or cost is None:
            return

        pip_size = spec["pip_size"]
        half_spread = cost["spread_pips"] * pip_size / 2.0
        pl_per_lot = self._pl_per_lot_per_price()

        high = float(current_candle["high"])
        low = float(current_candle["low"])

        # Candle treated as mid; derive bid/ask quotes from mid.
        bid_high = high - half_spread
        bid_low = low - half_spread
        ask_high = high + half_spread
        ask_low = low + half_spread

        commission_multiplier = (
            2.0 if cost["commission_per_side"] else 1.0
        )

        remaining = []
        for pos in self.current_state.open_positions:
            exit_fill: Optional[float] = None
            exit_reason = ""

            if pos.direction == "BUY":
                if bid_low <= pos.sl:
                    exit_fill = self._apply_exit_costs(
                        "BUY", pos.sl, is_limit=False,
                    )
                    exit_reason = "SL"
                elif bid_high >= pos.tp:
                    exit_fill = self._apply_exit_costs(
                        "BUY", pos.tp, is_limit=True,
                    )
                    exit_reason = "TP"
            else:
                if ask_high >= pos.sl:
                    exit_fill = self._apply_exit_costs(
                        "SELL", pos.sl, is_limit=False,
                    )
                    exit_reason = "SL"
                elif ask_low <= pos.tp:
                    exit_fill = self._apply_exit_costs(
                        "SELL", pos.tp, is_limit=True,
                    )
                    exit_reason = "TP"

            if exit_fill is None:
                remaining.append(pos)
                continue

            pos.close_price = exit_fill
            pos.close_time = current_time
            pos.close_reason = exit_reason

            gross = self._calculate_gross_pl(
                pos.direction, pos.entry_price, exit_fill, pos.volume
            )

            risk = abs(pos.entry_price - pos.sl)
            if risk > 0 and exit_fill != pos.entry_price:
                reward = abs(exit_fill - pos.entry_price)
                sign = 1.0 if gross >= 0 else -1.0
                pos.realized_r = sign * reward / risk
            else:
                pos.realized_r = 0.0

            commission = (
                cost["commission_per_lot"]
                * commission_multiplier
                * pos.volume
            )
            swap = self._apply_swap(
                pos.direction, pos.volume, pos.entry_time, current_time
            )

            net = gross - commission + swap

            pos.gross_profit = gross
            pos.net_profit = net
            pos.spread_cost = (
                cost["spread_pips"] * pip_size * pl_per_lot * pos.volume
            )

            self.current_state.balance += net
            self.current_state.daily_pl += net
            trade = self._position_to_trade(pos)
            self.current_state.closed_trades.append(trade)
            self.trades.append(trade)

        self.current_state.open_positions = remaining

    def _check_signals(
        self,
        df_m5: pd.DataFrame,
        current_index: int,
        smc_engine: SMCEngine,
        config: BacktestConfig,
    ):
        """Run SMC analysis on a sliced view of the data, then execute.

        The mock functions replace market_data's candle/ATR accessors
        for the duration of one analysis call. Restoration is guarded
        with try/finally and defensive ``is not None`` checks so that
        an exception during analysis cannot leave market_data in a
        mocked state (Phase 4A: fix duplicate body + restore leak).
        """
        current_time = df_m5.index[current_index]

        orig_get_candles = getattr(market_data, "get_candles", None)
        orig_get_closed = getattr(market_data, "get_closed_candles", None)
        orig_latest = getattr(market_data, "get_latest_candle", None)
        orig_calculate_atr = getattr(market_data, "calculate_atr", None)

        def mock_get_candles(tf: str, count: int):
            df = self.candles.get(tf)
            if df is None:
                return []
            sliced = df.loc[:current_time].tail(count)
            return [
                Candle(
                    timestamp=r.name,
                    open=r["open"], high=r["high"],
                    low=r["low"], close=r["close"],
                    volume=r["tick_volume"],
                )
                for _, r in sliced.iterrows()
            ]

        def mock_get_closed_candles(tf: str, count: int):
            df = self.candles.get(tf)
            if df is None:
                return []
            sliced = df.loc[:current_time]
            if len(sliced) <= 1:
                return []
            sliced = sliced.iloc[:-1].tail(count)
            return [
                Candle(
                    timestamp=r.name,
                    open=r["open"], high=r["high"],
                    low=r["low"], close=r["close"],
                    volume=r["tick_volume"],
                )
                for _, r in sliced.iterrows()
            ]

        def mock_get_latest(tf: str):
            df = self.candles.get(tf)
            if df is None:
                return None
            sliced = df.loc[:current_time]
            if sliced.empty:
                return None
            r = sliced.iloc[-1]
            return Candle(
                timestamp=sliced.index[-1],
                open=r["open"], high=r["high"],
                low=r["low"], close=r["close"],
                volume=r["tick_volume"],
            )

        def mock_calculate_atr(tf: str, period: int = 14):
            df = self.candles.get(tf)
            if df is None:
                return 0.0
            sliced = df.loc[:current_time]
            if len(sliced) > 1:
                sliced = sliced.iloc[:-1]
            else:
                return 0.0
            if len(sliced) < period + 1:
                return 0.0

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
                    self._execute_signal(
                        sig, current_candle, df_m5.index[current_index],
                        config,
                    )
        finally:
            if orig_get_candles is not None:
                market_data.get_candles = orig_get_candles
            if orig_get_closed is not None:
                market_data.get_closed_candles = orig_get_closed
            if orig_latest is not None:
                market_data.get_latest_candle = orig_latest
            if orig_calculate_atr is not None:
                market_data.calculate_atr = orig_calculate_atr

    def _execute_signal(
        self,
        sig: TradeSignal,
        current_candle: pd.Series,
        current_time: datetime,
        config: BacktestConfig,
    ):
        spec = self._broker_spec
        cost = self._cost_config
        if spec is None or cost is None:
            logger.error(
                "Cannot execute signal: broker spec/cost config not "
                "loaded. Signal %s %s dropped.",
                sig.direction.value, sig.strategy.value,
            )
            return

        max_pos = int(self.trading_config.max_positions)
        if len(self.current_state.open_positions) >= max_pos:
            logger.debug(
                "Backtest: skipping signal %s %s; max_positions=%d "
                "already reached.",
                sig.direction.value, sig.strategy.value, max_pos,
            )
            return

        equity = self.current_state.equity
        risk_amount = equity * (config.risk_percent / 100)

        # M-04: fill price includes spread half + slippage.
        fill_price = self._apply_entry_costs(
            sig.direction.value, sig.entry_price
        )

        # Actual risk distance from fill to signal SL.
        if sig.direction == SignalDirection.BUY:
            sl_distance = fill_price - sig.sl_price
        else:
            sl_distance = sig.sl_price - fill_price

        if sl_distance <= 0:
            logger.debug(
                "Backtest: signal %s %s rejected; fill vs SL geometry "
                "invalid (fill=%.5f sl=%.5f).",
                sig.direction.value, sig.strategy.value,
                fill_price, sig.sl_price,
            )
            return

        volume = self._calculate_volume(risk_amount, sl_distance)
        if volume is None:
            logger.debug(
                "Backtest: signal %s %s skipped; risk budget %.2f "
                "cannot afford broker volume_min=%.4f at sl_distance="
                "%.5f.",
                sig.direction.value, sig.strategy.value,
                risk_amount, spec["volume_min"], sl_distance,
            )
            return

        pos = BacktestPosition(
            ticket=self._ticket_counter,
            direction=sig.direction.value,
            entry_price=fill_price,
            sl=sig.sl_price,
            tp=sig.tp_price,
            volume=volume,
            entry_time=current_time,
            entry_index=self.current_state.current_index,
            strategy=sig.strategy.value,
            risk_percent=config.risk_percent,
        )
        self._ticket_counter += 1
        self.current_state.open_positions.append(pos)

    def _close_all_positions(
        self,
        last_candle: pd.Series,
        last_time: datetime,
        config: BacktestConfig,
    ):
        spec = self._broker_spec
        cost = self._cost_config
        if spec is None or cost is None:
            return

        pip_size = spec["pip_size"]
        half_spread = cost["spread_pips"] * pip_size / 2.0
        pl_per_lot = self._pl_per_lot_per_price()

        mid = float(last_candle["close"])
        bid = mid - half_spread
        ask = mid + half_spread

        commission_multiplier = (
            2.0 if cost["commission_per_side"] else 1.0
        )

        for pos in self.current_state.open_positions:
            if pos.direction == "BUY":
                exit_fill = self._apply_exit_costs(
                    "BUY", bid, is_limit=False,
                )
            else:
                exit_fill = self._apply_exit_costs(
                    "SELL", ask, is_limit=False,
                )

            pos.close_price = exit_fill
            pos.close_time = last_time
            pos.close_reason = "END_OF_BACKTEST"

            gross = self._calculate_gross_pl(
                pos.direction, pos.entry_price, exit_fill, pos.volume
            )

            risk = abs(pos.entry_price - pos.sl)
            if risk > 0 and exit_fill != pos.entry_price:
                reward = abs(exit_fill - pos.entry_price)
                sign = 1.0 if gross >= 0 else -1.0
                pos.realized_r = sign * reward / risk
            else:
                pos.realized_r = 0.0

            commission = (
                cost["commission_per_lot"]
                * commission_multiplier
                * pos.volume
            )
            swap = self._apply_swap(
                pos.direction, pos.volume, pos.entry_time, last_time
            )

            net = gross - commission + swap

            pos.gross_profit = gross
            pos.net_profit = net
            pos.spread_cost = (
                cost["spread_pips"] * pip_size * pl_per_lot * pos.volume
            )

            self.current_state.balance += net
            trade = self._position_to_trade(pos)
            self.current_state.closed_trades.append(trade)
            self.trades.append(trade)

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
            exit_reason=pos.close_reason,
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
        net_return = (
            (self.current_state.equity - config.initial_balance)
            / config.initial_balance * 100
        )

        win_rate = len(winners) / len(trades) * 100
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float('inf')
        )

        avg_win = gross_profit / len(winners) if winners else 0
        avg_loss = gross_loss / len(losers) if losers else 0
        expectancy = (
            (win_rate / 100 * avg_win)
            - ((100 - win_rate) / 100 * avg_loss)
        )

        avg_r = np.mean([t.realized_r for t in trades])
        avg_rr = (
            np.mean([abs(t.realized_r) for t in trades if t.realized_r != 0])
            if any(t.realized_r != 0 for t in trades) else 0
        )

        returns = [t.net_profit / config.initial_balance for t in trades]
        sharpe = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if np.std(returns) > 0 else 0
        )

        downside_returns = [r for r in returns if r < 0]
        sortino = (
            np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
            if downside_returns and np.std(downside_returns) > 0 else 0
        )

        recovery = (
            net_profit / self.current_state.max_drawdown
            if self.current_state.max_drawdown > 0 else 0
        )

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
                max_consec_losses = max(
                    max_consec_losses, abs(current_streak)
                )

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
            avg_holding_hours=np.mean([
                (t.exit_time - t.entry_time).total_seconds() / 3600
                for t in trades
            ]),
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            long_trades=len(long_trades),
            long_win_rate=(
                len(long_wins) / len(long_trades) * 100
                if long_trades else 0
            ),
            long_profit=sum(t.net_profit for t in long_trades),
            short_trades=len(short_trades),
            short_win_rate=(
                len(short_wins) / len(short_trades) * 100
                if short_trades else 0
            ),
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

    def get_assumptions_applied(self) -> Dict[str, Any]:
        """M-04: return a copy of the execution assumptions modelled."""
        return dict(self.assumptions_applied)


async def run_backtest_async(config_id: int) -> BacktestReport:
    """Run a backtest asynchronously."""
    import asyncio
    import threading

    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    def _worker() -> None:
        try:
            bt_config = db_manager.get(BacktestConfig, int(config_id))
            if bt_config is None:
                raise ValueError(
                    f"BacktestConfig id {config_id} not found."
                )
            engine = BacktestEngine(config_id=int(config_id))
            ok = engine.load_data(
                config.trading.symbol,
                bt_config.timeframes,
                bt_config.start_date,
                bt_config.end_date,
            )
            if not ok:
                raise RuntimeError(
                    "Backtest data load failed (MT5 unavailable or "
                    "insufficient history for the requested range)."
                )
            report = engine.run_backtest(bt_config)
            loop.call_soon_threadsafe(future.set_result, report)
        except Exception as exc:
            logger.exception("Backtest worker thread failed.")
            loop.call_soon_threadsafe(future.set_exception, exc)

    threading.Thread(target=_worker, daemon=True).start()
    return await future