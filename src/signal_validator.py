
"""Signal validation and safety filtering."""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from src.smc_strategy import TradeSignal, SignalDirection
from src.market_data import market_data
from src.mt5_connector import mt5_connector
from src.config_loader import config
from src.database_manager import db_manager
from src.logger import get_logger

logger = get_logger(__name__)


class ValidationResult:
    """Result of validating a trading signal."""

    def __init__(
        self,
        valid: bool,
        reason: str,
        checks: Dict[str, bool]
    ):
        self.valid = bool(valid)
        self.reason = str(reason)
        self.checks = checks


class SignalValidator:
    """Validates trading signals before they reach the order executor."""

    ENVIRONMENT = "demo"

    def __init__(self):
        self.trading_config = config.trading
        self.risk_config = config.risk
        self.tm_config = config.trade_management

    # ------------------------------------------------------------------
    # Main validation
    # ------------------------------------------------------------------

    def validate_signal(self, signal: TradeSignal) -> ValidationResult:
        checks: Dict[str, bool] = {}
        reasons: List[str] = []

        def record(name: str, passed: bool, failure_reason: str):
            checks[name] = bool(passed)
            if not passed:
                reasons.append(failure_reason)

        # 1. Trading mode
        record(
            "trading_mode",
            str(self.trading_config.mode).upper() == "DEMO",
            "Trading mode is not DEMO."
        )

        # 2. MT5 connection
        connected = mt5_connector.is_connected()
        record(
            "mt5_connected",
            connected,
            "MT5 is not connected."
        )

        if not connected:
            return self._result(checks, reasons)

        # 3. Account information
        account = mt5_connector.get_account_info()

        record(
            "account_available",
            account is not None,
            "Could not retrieve MT5 account information."
        )

        if account is None:
            return self._result(checks, reasons)

        # 4. Demo account
        record(
            "demo_account",
            bool(account.is_demo),
            "Connected MT5 account is not a demo account."
        )

        # 5. Trading permissions
        record(
            "trading_allowed",
            bool(account.trade_allowed and account.trade_expert),
            "Trading is not permitted on this account or terminal."
        )

        # 6. Signal data
        signal_data_valid = self._check_signal_data(signal)
        record(
            "signal_data",
            signal_data_valid,
            "Signal contains invalid prices, direction, or risk parameters."
        )

        if not signal_data_valid:
            return self._result(checks, reasons)

        # 7. Open positions
        positions = mt5_connector.get_positions()

        record(
            "positions_available",
            positions is not None,
            "Could not retrieve open MT5 positions."
        )

        if positions is None:
            return self._result(checks, reasons)

        record(
            "max_positions",
            len(positions) < int(self.trading_config.max_positions),
            f"Maximum open positions ({self.trading_config.max_positions}) reached."
        )

        # 8. Daily trade limit
        daily_trades_ok = self._check_daily_trade_limit()
        record(
            "daily_trades",
            daily_trades_ok,
            "Daily trade limit reached or trade history unavailable."
        )

        # 9. Daily loss limit
        daily_loss_ok = self._check_daily_loss_limit()
        record(
            "daily_loss",
            daily_loss_ok,
            "Daily loss limit reached or daily loss data unavailable."
        )

        # 10. Maximum drawdown
        drawdown_ok = self._check_max_drawdown()
        record(
            "max_drawdown",
            drawdown_ok,
            "Maximum drawdown reached or account snapshot history unavailable."
        )

        # 11. Consecutive losses
        consecutive_ok = self._check_consecutive_losses()
        record(
            "consecutive_losses",
            consecutive_ok,
            "Maximum consecutive losses reached or trade history unavailable."
        )

        # 12. Spread
        try:
            spread_ok = not market_data.is_high_spread()
        except Exception:
            logger.exception("Spread check failed.")
            spread_ok = False

        record(
            "spread",
            spread_ok,
            "Spread is too high or spread information is unavailable."
        )

        # 13. Trading session
        try:
            session_ok = bool(market_data.is_trading_session())
        except Exception:
            logger.exception("Trading session check failed.")
            session_ok = False

        record(
            "trading_session",
            session_ok,
            "Outside configured trading session or session data unavailable."
        )

        # 14. Weekend protection
        try:
            weekend_ok = not market_data.is_weekend_protection_active()
        except Exception:
            logger.exception("Weekend protection check failed.")
            weekend_ok = False

        record(
            "weekend",
            weekend_ok,
            "Weekend protection is active or could not be checked."
        )

        # 15. Risk/reward, recalculated from prices
        rr_ok, actual_rr = self._check_risk_reward(signal)

        record(
            "risk_reward",
            rr_ok,
            (
                f"Risk/reward {actual_rr:.2f} is below the configured "
                f"minimum ({self.trading_config.min_risk_reward})."
            )
        )

        # 16. Current entry price
        entry_ok = self._check_entry_price(signal)
        record(
            "entry_price",
            entry_ok,
            "Entry price is too far from the current executable market price."
        )

        # 17. Stop-loss and take-profit distances/direction
        distance_ok, distance_reason = self._check_sl_tp_distance(signal)

        record(
            "sl_tp_distance",
            distance_ok,
            distance_reason
        )

        # 18-20. Broker lot constraints.
        # Initial validation occurs before the executor calculates volume.
        # A positive lot size means the signal is already sized and can be
        # checked here. A zero lot size is not accepted as an executable
        # order; the executor must run validation again after sizing.
        try:
            lot_size = float(signal.lot_size or 0.0)
        except (TypeError, ValueError):
            lot_size = -1.0

        if lot_size > 0:
            lot_checks = self._check_lot_constraints(lot_size)
            for name, passed in lot_checks.items():
                record(
                    name,
                    passed,
                    f"Position volume failed broker constraint: {name}."
                )
        elif lot_size < 0:
            record(
                "lot_size",
                False,
                "Signal lot size is invalid."
            )
        else:
            checks["lot_size_pending"] = True

        # 21. Total open risk
        total_risk_ok = self._check_total_open_risk(
            signal,
            positions,
            account
        )

        record(
            "total_risk",
            total_risk_ok,
            "Total open risk would exceed the configured limit or could not be calculated."
        )

        # 22. News filter
        # This is explicitly a placeholder. No economic calendar is connected.
        news_ok = self._check_news_filter()

        record(
            "news",
            news_ok,
            "News filter blocked the signal."
        )

        return self._result(checks, reasons)

    def _result(
        self,
        checks: Dict[str, bool],
        reasons: List[str]
    ) -> ValidationResult:
        valid = bool(checks) and all(checks.values())

        return ValidationResult(
            valid=valid,
            reason="; ".join(reasons) if reasons else "All validation checks passed.",
            checks=checks
        )

    # ------------------------------------------------------------------
    # Basic signal checks
    # ------------------------------------------------------------------

    def _check_signal_data(self, signal: TradeSignal) -> bool:
        try:
            direction = signal.direction
            entry = float(signal.entry_price)
            stop = float(signal.sl_price)
            target = float(signal.tp_price)
            risk_percent = float(signal.risk_percent)

            if direction not in (SignalDirection.BUY, SignalDirection.SELL):
                return False

            if min(entry, stop, target) <= 0:
                return False

            if risk_percent <= 0:
                return False

            if not signal.symbol or not signal.strategy:
                return False

            if direction == SignalDirection.BUY:
                return stop < entry < target

            return target < entry < stop

        except (AttributeError, TypeError, ValueError):
            logger.exception("Signal data validation failed.")
            return False

    def _check_risk_reward(self, signal: TradeSignal):
        entry = float(signal.entry_price)
        stop = float(signal.sl_price)
        target = float(signal.tp_price)

        risk_distance = abs(entry - stop)
        reward_distance = abs(target - entry)

        if risk_distance <= 0:
            return False, 0.0

        actual_rr = reward_distance / risk_distance
        minimum_rr = float(self.trading_config.min_risk_reward)

        return actual_rr >= minimum_rr, actual_rr

    def _check_entry_price(self, signal: TradeSignal) -> bool:
        """Check signal entry against current Ask/Bid using broker points."""

        try:
            tick = market_data.get_current_tick()
            symbol_info = mt5_connector.get_symbol_info()

            if tick is None or symbol_info is None:
                return False

            point = float(symbol_info.point)
            if point <= 0:
                return False

            direction = signal.direction
            signal_entry = float(signal.entry_price)

            if direction == SignalDirection.BUY:
                current_price = float(tick["ask"])
            else:
                current_price = float(tick["bid"])

            if current_price <= 0:
                return False

            # 50 broker points is a tolerance, not a universal pip value.
            max_deviation_points = 50
            allowed_deviation = max_deviation_points * point

            return abs(signal_entry - current_price) <= allowed_deviation

        except Exception:
            logger.exception("Entry-price validation failed.")
            return False

    def _check_sl_tp_distance(self, signal: TradeSignal):
        """Validate stop and target placement using broker metadata."""

        try:
            symbol_info = mt5_connector.get_symbol_info()
            if symbol_info is None:
                return False, "Broker symbol information unavailable."

            point = float(symbol_info.point)
            digits = int(symbol_info.digits)

            if point <= 0:
                return False, "Invalid broker point size."

            pip_size = point * 10 if digits in (3, 5) else point

            entry = float(signal.entry_price)
            stop = float(signal.sl_price)
            target = float(signal.tp_price)

            if signal.direction == SignalDirection.BUY:
                stop_distance = entry - stop
                target_distance = target - entry
            else:
                stop_distance = stop - entry
                target_distance = entry - target

            if stop_distance <= 0 or target_distance <= 0:
                return False, "Stop-loss or take-profit is on the wrong side."

            minimum_pips = 10.0
            stop_pips = stop_distance / pip_size
            target_pips = target_distance / pip_size

            if stop_pips < minimum_pips:
                return (
                    False,
                    f"Stop-loss distance {stop_pips:.2f} pips is below "
                    f"the minimum {minimum_pips:.2f} pips."
                )

            if target_pips < minimum_pips:
                return (
                    False,
                    f"Take-profit distance {target_pips:.2f} pips is below "
                    f"the minimum {minimum_pips:.2f} pips."
                )

            return True, "OK"

        except Exception:
            logger.exception("SL/TP distance validation failed.")
            return False, "Could not validate stop-loss/take-profit distances."

    # ------------------------------------------------------------------
    # Volume validation
    # ------------------------------------------------------------------

    def _check_lot_constraints(self, lot_size: float) -> Dict[str, bool]:
        try:
            symbol_info = mt5_connector.get_symbol_info()
            if symbol_info is None:
                return {
                    "symbol_info_available": False,
                    "min_lot": False,
                    "max_lot": False,
                    "lot_step": False
                }

            minimum = float(symbol_info.volume_min)
            maximum = float(symbol_info.volume_max)
            step = float(symbol_info.volume_step)

            if minimum <= 0 or maximum < minimum or step <= 0:
                return {
                    "symbol_info_available": False,
                    "min_lot": False,
                    "max_lot": False,
                    "lot_step": False
                }

            step_units = (lot_size - minimum) / step

            return {
                "symbol_info_available": True,
                "min_lot": lot_size >= minimum - 1e-8,
                "max_lot": lot_size <= maximum + 1e-8,
                "lot_step": (
                    lot_size >= minimum - 1e-8
                    and abs(step_units - round(step_units)) < 1e-6
                )
            }

        except Exception:
            logger.exception("Broker volume validation failed.")
            return {
                "symbol_info_available": False,
                "min_lot": False,
                "max_lot": False,
                "lot_step": False
            }

    # ------------------------------------------------------------------
    # Database-based limits
    # ------------------------------------------------------------------

    def _check_daily_trade_limit(self) -> bool:
        """Count today's executed/closed trades without double counting."""

        from src.database_manager import ClosedTrade, ExecutedOrder

        try:
            today_start = datetime.now(timezone.utc).replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0
            )

            with db_manager.get_session() as session:
                closed_count = session.query(ClosedTrade).filter(
                    ClosedTrade.environment == self.ENVIRONMENT,
                    ClosedTrade.time_close >= today_start
                ).count()

                open_count = session.query(ExecutedOrder).filter(
                    ExecutedOrder.environment == self.ENVIRONMENT,
                    ExecutedOrder.time_close.is_(None),
                    ExecutedOrder.time_open >= today_start
                ).count()

            total_trades = closed_count + open_count

            return total_trades < int(self.trading_config.max_daily_trades)

        except Exception:
            logger.exception("Daily trade limit check failed.")
            return False

    def _check_daily_loss_limit(self) -> bool:
        from src.database_manager import ClosedTrade

        try:
            account = mt5_connector.get_account_info()
            if account is None or float(account.equity) <= 0:
                return False

            today_start = datetime.now(timezone.utc).replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0
            )

            with db_manager.get_session() as session:
                trades = session.query(ClosedTrade).filter(
                    ClosedTrade.environment == self.ENVIRONMENT,
                    ClosedTrade.time_close >= today_start
                ).all()

            daily_pl = sum(
                float(trade.net_profit or 0)
                for trade in trades
            )

            if daily_pl >= 0:
                return True

            daily_loss_pct = abs(daily_pl) / float(account.equity) * 100.0

            return daily_loss_pct < float(
                self.risk_config.daily_loss_limit_percent
            )

        except Exception:
            logger.exception("Daily loss limit check failed.")
            return False

    def _get_peak_equity(self) -> Optional[float]:
        from src.database_manager import AccountSnapshot

        try:
            with db_manager.get_session() as session:
                snapshots = session.query(AccountSnapshot).filter(
                    AccountSnapshot.environment == self.ENVIRONMENT
                ).order_by(
                    AccountSnapshot.timestamp.desc()
                ).limit(1000).all()

            if not snapshots:
                logger.error(
                    "No account snapshots available for drawdown calculation."
                )
                return None

            peak = max(float(snapshot.equity) for snapshot in snapshots)

            return peak if peak > 0 else None

        except Exception:
            logger.exception("Account snapshot retrieval failed.")
            return None

    def _check_max_drawdown(self) -> bool:
        account = mt5_connector.get_account_info()

        if account is None or float(account.equity) <= 0:
            return False

        peak_equity = self._get_peak_equity()
        if peak_equity is None:
            return False

        drawdown_pct = max(
            0.0,
            (peak_equity - float(account.equity)) / peak_equity * 100.0
        )

        return drawdown_pct < float(
            self.risk_config.max_account_drawdown_percent
        )

    def _check_consecutive_losses(self) -> bool:
        from src.database_manager import ClosedTrade

        try:
            with db_manager.get_session() as session:
                trades = session.query(ClosedTrade).filter(
                    ClosedTrade.environment == self.ENVIRONMENT
                ).order_by(
                    ClosedTrade.time_close.desc()
                ).limit(20).all()

            consecutive = 0

            for trade in trades:
                if float(trade.net_profit or 0) < 0:
                    consecutive += 1
                else:
                    break

            return consecutive < int(
                self.risk_config.max_consecutive_losses
            )

        except Exception:
            logger.exception("Consecutive-loss check failed.")
            return False

    # ------------------------------------------------------------------
    # Total open risk
    # ------------------------------------------------------------------

    def _check_total_open_risk(
        self,
        signal: TradeSignal,
        positions: List[Dict[str, Any]],
        account
    ) -> bool:
        """Calculate existing position risk plus proposed signal risk."""

        try:
            equity = float(account.equity)
            if equity <= 0:
                return False

            symbol_info = mt5_connector.get_symbol_info()
            if symbol_info is None:
                return False

            tick_size = float(symbol_info.tick_size or 0)
            tick_value = float(symbol_info.tick_value or 0)

            if tick_size <= 0 or tick_value <= 0:
                return False

            total_risk = 0.0

            for position in positions:
                direction = str(
                    position.get("type", position.get("direction", ""))
                ).upper()

                entry = float(position["price_open"])
                stop = float(position.get("sl") or 0)
                volume = float(position["volume"])

                # An open position without a stop-loss is not safe.
                if stop <= 0 or volume <= 0:
                    logger.error(
                        "Position %s has missing SL or invalid volume.",
                        position.get("ticket")
                    )
                    return False

                if direction == "BUY":
                    distance = entry - stop
                elif direction == "SELL":
                    distance = stop - entry
                else:
                    return False

                if distance <= 0:
                    return False

                position_risk = (
                    distance / tick_size
                ) * tick_value * volume

                total_risk += position_risk

            # Estimate proposed trade risk.
            entry = float(signal.entry_price)
            stop = float(signal.sl_price)
            direction = signal.direction

            if direction == SignalDirection.BUY:
                proposed_distance = entry - stop
            else:
                proposed_distance = stop - entry

            if proposed_distance <= 0:
                return False

            proposed_volume = float(signal.lot_size or 0)

            if proposed_volume > 0:
                proposed_risk = (
                    proposed_distance / tick_size
                ) * tick_value * proposed_volume
            else:
                # During pre-sizing validation, use the configured
                # percentage of equity as the proposed risk estimate.
                risk_percent = float(signal.risk_percent)
                if risk_percent <= 0:
                    return False

                proposed_risk = equity * risk_percent / 100.0

            total_risk += proposed_risk

            maximum_risk = (
                float(self.risk_config.max_total_open_risk_percent)
                / 100.0
                * equity
            )

            return total_risk <= maximum_risk

        except Exception:
            logger.exception("Total open-risk validation failed.")
            return False

    # ------------------------------------------------------------------
    # News filter
    # ------------------------------------------------------------------

    def _check_news_filter(self) -> bool:
        """
        Placeholder only.

        No economic calendar or news provider is currently connected.
        This method does NOT protect the strategy from news volatility.
        """
        return True


# Global signal validator instance
signal_validator = SignalValidator()
