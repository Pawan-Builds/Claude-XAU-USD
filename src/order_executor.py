
"""Safe order execution, position reconciliation, and trade persistence."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from src.mt5_connector import mt5_connector, OrderResult, mt5
from src.risk_manager import risk_manager, PositionSizeResult
from src.signal_validator import signal_validator, ValidationResult
from src.trade_manager import trade_manager
from src.config_loader import config
from src.database_manager import (
    db_manager,
    OrderRequest,
    ExecutedOrder,
    OpenPosition,
    TradingSignal,
)
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    signal_id: Optional[int] = None
    order_result: Optional[OrderResult] = None
    position_size: Optional[PositionSizeResult] = None
    validation: Optional[ValidationResult] = None
    error: Optional[str] = None


class OrderExecutor:
    """Executes validated signals and reconciles MT5 positions."""

    ENVIRONMENT = "demo"

    def __init__(self):
        self.trading_config = config.trading
        self.magic_number = self.trading_config.magic_number

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute_signal(self, signal) -> ExecutionResult:
        """Validate, size, submit, reconcile, and persist a trade."""

        # This executor is deliberately restricted to DEMO trading.
        if not mt5_connector.is_connected():
            return ExecutionResult(
                success=False,
                error="MT5 is not connected."
            )

        try:
            validation = signal_validator.validate_signal(signal)
        except Exception as exc:
            logger.exception("Signal validation raised an exception.")
            return ExecutionResult(
                success=False,
                error=f"Signal validation error: {exc}"
            )

        if not validation.valid:
            logger.warning("Signal rejected: %s", validation.reason)
            return ExecutionResult(
                success=False,
                validation=validation,
                error=validation.reason
            )

        try:
            position_size = risk_manager.calculate_position_size(
                signal.entry_price,
                signal.sl_price,
                signal.direction.value,
                signal.risk_percent
            )
        except Exception as exc:
            logger.exception("Position sizing failed.")
            return ExecutionResult(
                success=False,
                validation=validation,
                error=f"Position sizing error: {exc}"
            )

        if position_size is None or position_size.lot_size <= 0:
            reason = (
                position_size.reason
                if position_size is not None
                else "Risk manager returned no position-size result."
            )
            logger.warning("Position sizing rejected: %s", reason)
            return ExecutionResult(
                success=False,
                position_size=position_size,
                validation=validation,
                error=reason
            )

        signal.lot_size = position_size.lot_size

        # Persist the signal before submitting anything to the broker.
        try:
            signal_id = self._save_signal(
                signal,
                position_size,
                validation
            )
        except Exception as exc:
            logger.exception("Could not persist signal; order not submitted.")
            return ExecutionResult(
                success=False,
                position_size=position_size,
                validation=validation,
                error=f"Could not save signal. Order was not submitted: {exc}"
            )

        try:
            order_request = self._prepare_order_request(
                signal,
                position_size
            )
            request_id = self._save_order_request(
                signal_id,
                order_request
            )
        except Exception as exc:
            logger.exception("Could not prepare or persist order request.")
            return ExecutionResult(
                success=False,
                signal_id=signal_id,
                position_size=position_size,
                validation=validation,
                error=f"Could not prepare order request: {exc}"
            )

        # Submit exactly once. Do not automatically retry an uncertain
        # broker response: the first request may already have been filled.
        try:
            order_result = mt5_connector.send_order(order_request)
        except Exception as exc:
            logger.exception("MT5 order submission raised an exception.")
            self._update_order_request_error(
                request_id,
                f"Submission exception: {exc}"
            )
            return ExecutionResult(
                success=False,
                signal_id=signal_id,
                position_size=position_size,
                validation=validation,
                error=f"MT5 submission exception: {exc}"
            )

        try:
            self._update_order_request(request_id, order_result)
        except Exception:
            logger.exception(
                "Could not update order-request record after MT5 response."
            )

        if not order_result.success:
            error = (
                order_result.error_message
                or order_result.comment
                or "MT5 rejected the order."
            )
            logger.error("Order rejected: %s", error)
            return ExecutionResult(
                success=False,
                signal_id=signal_id,
                order_result=order_result,
                position_size=position_size,
                validation=validation,
                error=error
            )

        # The broker accepted the order. Now locate the actual position.
        position = self._find_position_for_result(
            order_result,
            signal,
            position_size
        )

        if position is None:
            message = (
                "MT5 accepted the order, but the resulting open position "
                "could not be uniquely identified. Do not submit this signal "
                "again. Reconcile the position in MT5 before continuing."
            )
            logger.critical(message)
            logger.critical(
                "Accepted order details: order=%s ticket=%s deal=%s symbol=%s",
                getattr(order_result, "order_id", None),
                getattr(order_result, "ticket", None),
                getattr(order_result, "deal_id", None),
                signal.symbol
            )

            return ExecutionResult(
                success=True,
                signal_id=signal_id,
                order_result=order_result,
                position_size=position_size,
                validation=validation,
                error=message
            )

        ticket = int(position["ticket"])
        actual_volume = float(position["volume"])
        actual_entry = float(position["price_open"])
        actual_sl = float(position.get("sl") or signal.sl_price)
        actual_tp = float(position.get("tp") or signal.tp_price)

        # Persist the actual broker position before initializing management.
        try:
            self._save_executed_order(position)
            self._save_open_position(position, signal)
        except Exception as exc:
            logger.exception(
                "CRITICAL: broker position %s is open but database "
                "persistence failed.",
                ticket
            )
            return ExecutionResult(
                success=True,
                signal_id=signal_id,
                order_result=order_result,
                position_size=position_size,
                validation=validation,
                error=(
                    f"Order was filled and position {ticket} is open, "
                    f"but database persistence failed: {exc}. "
                    "Reconcile this position before allowing further trades."
                )
            )

        try:
            trade_manager.initialize_position_tracking(
                ticket,
                position["type"],
                actual_entry,
                actual_sl,
                actual_tp,
                actual_volume,
                signal.strategy.value
            )
        except Exception as exc:
            logger.exception(
                "CRITICAL: position %s was saved but trade tracking "
                "initialization failed.",
                ticket
            )
            return ExecutionResult(
                success=True,
                signal_id=signal_id,
                order_result=order_result,
                position_size=position_size,
                validation=validation,
                error=(
                    f"Position {ticket} is open and saved, but trade "
                    f"management initialization failed: {exc}"
                )
            )

        logger.info(
            "Trade filled and tracking initialized: ticket=%s direction=%s "
            "volume=%s entry=%s SL=%s TP=%s",
            ticket,
            position["type"],
            actual_volume,
            actual_entry,
            actual_sl,
            actual_tp
        )

        return ExecutionResult(
            success=True,
            signal_id=signal_id,
            order_result=order_result,
            position_size=position_size,
            validation=validation
        )

    # ------------------------------------------------------------------
    # Order request preparation
    # ------------------------------------------------------------------

    def _prepare_order_request(
        self,
        signal,
        position_size: PositionSizeResult
    ) -> Dict[str, Any]:
        """Build a market order using the latest available broker quote."""

        symbol = signal.symbol
        direction = signal.direction.value.upper()

        if direction not in ("BUY", "SELL"):
            raise ValueError(f"Unsupported trade direction: {direction}")

        symbol_info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)

        if symbol_info is None or tick is None:
            raise RuntimeError(
                f"Could not obtain broker symbol information or quote for {symbol}."
            )

        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            current_price = float(tick.ask)
        else:
            order_type = mt5.ORDER_TYPE_SELL
            current_price = float(tick.bid)

        if current_price <= 0:
            raise RuntimeError(
                f"Invalid current market price for {symbol}: {current_price}"
            )

        point = float(symbol_info.point)
        digits = int(symbol_info.digits)

        if point <= 0:
            raise RuntimeError(f"Invalid point size for {symbol}.")

        # A pip is normally 10 points on 3/5-digit symbols and one point
        # on 2/4-digit symbols. For GOLD with 2 digits, this is 0.01.
        pip_size = point * 10 if digits in (3, 5) else point

        configured_slippage = getattr(
            self.trading_config,
            "max_slippage_pips",
            3
        )
        deviation = max(
            1,
            int(round(float(configured_slippage) * pip_size / point))
        )

        # The connector chooses the broker-supported filling mode.
        # Do not force IOC here.
        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "type": order_type,
            "volume": float(position_size.lot_size),
            "price": round(current_price, digits),
            "sl": round(float(signal.sl_price), digits),
            "tp": round(float(signal.tp_price), digits),
            "deviation": deviation,
            "magic": self.magic_number,
            "comment": f"SMC_{signal.strategy.value}"[:31],
            "type_time": mt5.ORDER_TIME_GTC,
        }

    # ------------------------------------------------------------------
    # Position reconciliation
    # ------------------------------------------------------------------

    def _find_position_for_result(
        self,
        result: OrderResult,
        signal,
        position_size: PositionSizeResult
    ) -> Optional[Dict[str, Any]]:
        """
        Match the broker response to one open position.

        Never select an arbitrary position when multiple candidates exist.
        """

        positions = mt5_connector.get_positions()

        if positions is None:
            logger.error("MT5 position query returned None.")
            return None

        if not positions:
            logger.error(
                "No open positions returned after successful order submission."
            )
            return None

        result_ids = {
            int(value)
            for value in (
                getattr(result, "ticket", None),
                getattr(result, "order_id", None),
                getattr(result, "deal_id", None),
            )
            if value not in (None, 0)
        }

        # First try exact ticket/order identifiers.
        exact_matches = [
            position
            for position in positions
            if int(position.get("ticket", 0)) in result_ids
            and self._position_matches_signal(position, signal)
        ]

        if len(exact_matches) == 1:
            return exact_matches[0]

        if len(exact_matches) > 1:
            logger.error("Multiple exact position matches found; refusing to guess.")
            return None

        # Fallback: only accept one unambiguous candidate with matching
        # symbol, direction, magic number, and filled volume.
        expected_volume = float(position_size.lot_size)
        volume_tolerance = 1e-8

        candidates = []
        for position in positions:
            if not self._position_matches_signal(position, signal):
                continue

            actual_volume = float(position.get("volume", 0))
            if abs(actual_volume - expected_volume) <= volume_tolerance:
                candidates.append(position)

        if len(candidates) == 1:
            logger.warning(
                "Position matched by symbol/direction/magic/volume rather "
                "than an exact order identifier. Ticket=%s",
                candidates[0].get("ticket")
            )
            return candidates[0]

        if len(candidates) > 1:
            logger.error(
                "Multiple positions match the submitted order. "
                "Refusing to guess which position belongs to this execution."
            )
        else:
            logger.error(
                "No matching position found. Order ticket=%s order_id=%s",
                getattr(result, "ticket", None),
                getattr(result, "order_id", None)
            )

        return None

    def _position_matches_signal(self, position, signal) -> bool:
        """Check that a position belongs to this bot and signal."""

        position_magic = position.get(
            "magic",
            position.get("magic_number")
        )

        if position_magic is not None:
            if int(position_magic) != int(self.magic_number):
                return False

        if position.get("symbol") != signal.symbol:
            return False

        if str(position.get("type", "")).upper() != signal.direction.value.upper():
            return False

        return True

    # ------------------------------------------------------------------
    # Database persistence
    # ------------------------------------------------------------------

    def _save_signal(
        self,
        signal,
        position_size: PositionSizeResult,
        validation: ValidationResult
    ) -> int:
        db_signal = TradingSignal(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            timeframe=signal.entry_tf,
            direction=signal.direction.value,
            entry_price=signal.entry_price,
            sl_price=signal.sl_price,
            tp_price=signal.tp_price,
            lot_size=position_size.lot_size,
            risk_percent=signal.risk_percent,
            risk_reward=signal.risk_reward,
            strategy=signal.strategy.value,
            confirmations=[
                {
                    "name": confirmation.name,
                    "passed": confirmation.passed,
                    "details": confirmation.details
                }
                for confirmation in signal.confirmations
            ],
            status="validated",
            environment=self.ENVIRONMENT
        )

        saved = db_manager.add(db_signal)
        if saved is None or not getattr(saved, "id", None):
            raise RuntimeError("Database did not return a saved signal ID.")

        return int(saved.id)

    def _save_order_request(
        self,
        signal_id: int,
        request: Dict[str, Any]
    ) -> int:
        db_request = OrderRequest(
            signal_id=signal_id,
            magic_number=self.magic_number,
            symbol=request["symbol"],
            order_type=request["type"],
            volume=request["volume"],
            price=request.get("price"),
            sl=request.get("sl"),
            tp=request.get("tp"),
            comment=request.get("comment"),
            request_data=request,
            environment=self.ENVIRONMENT
        )

        saved = db_manager.add(db_request)
        if saved is None or not getattr(saved, "id", None):
            raise RuntimeError("Database did not return an order-request ID.")

        return int(saved.id)

    def _update_order_request(
        self,
        request_id: int,
        result: OrderResult
    ) -> None:
        with db_manager.get_session() as session:
            request = session.query(OrderRequest).filter(
                OrderRequest.id == request_id
            ).first()

            if request is None:
                logger.error(
                    "Could not find order request database row %s.",
                    request_id
                )
                return

            request.response_data = {
                "success": result.success,
                "ticket": getattr(result, "ticket", None),
                "order_id": getattr(result, "order_id", None),
                "deal_id": getattr(result, "deal_id", None),
                "volume": result.volume,
                "price": result.price,
                "retcode": result.retcode,
                "comment": result.comment,
                "error_code": result.error_code,
                "error_message": result.error_message,
            }
            request.status = "filled" if result.success else "rejected"
            request.error_code = result.error_code
            request.error_message = result.error_message

    def _update_order_request_error(
        self,
        request_id: int,
        message: str
    ) -> None:
        try:
            with db_manager.get_session() as session:
                request = session.query(OrderRequest).filter(
                    OrderRequest.id == request_id
                ).first()

                if request is not None:
                    request.status = "submission_error"
                    request.error_message = message
        except Exception:
            logger.exception(
                "Could not record order submission exception for request %s.",
                request_id
            )

    def _save_executed_order(self, position: Dict[str, Any]) -> None:
        ticket = int(position["ticket"])

        existing = db_manager.query_one(
            ExecutedOrder,
            ticket=ticket,
            environment=self.ENVIRONMENT
        )
        if existing is not None:
            logger.warning(
                "Executed order record already exists for ticket %s.",
                ticket
            )
            return

        db_order = ExecutedOrder(
            ticket=ticket,
            magic_number=position.get("magic", self.magic_number),
            symbol=position["symbol"],
            order_type=position["type"],
            volume=position["volume"],
            price_open=position["price_open"],
            price_current=position["price_current"],
            sl=position["sl"],
            tp=position["tp"],
            swap=position["swap"],
            profit=position["profit"],
            commission=position["commission"],
            comment=position.get("comment", ""),
            time_open=position["time_open"],
            environment=self.ENVIRONMENT
        )

        db_manager.add(db_order)

    def _save_open_position(
        self,
        position: Dict[str, Any],
        signal
    ) -> None:
        ticket = int(position["ticket"])

        existing = db_manager.query_one(
            OpenPosition,
            ticket=ticket,
            environment=self.ENVIRONMENT
        )
        if existing is not None:
            logger.warning(
                "Open-position record already exists for ticket %s.",
                ticket
            )
            return

        entry = float(position["price_open"])
        stop_loss = float(position.get("sl") or signal.sl_price)
        take_profit = float(position.get("tp") or signal.tp_price)
        initial_risk = abs(entry - stop_loss)

        db_position = OpenPosition(
            ticket=ticket,
            magic_number=position.get("magic", self.magic_number),
            symbol=position["symbol"],
            direction=position["type"],
            volume=position["volume"],
            price_open=entry,
            price_current=position["price_current"],
            sl=stop_loss,
            tp=take_profit,
            swap=position["swap"],
            profit=position["profit"],
            commission=position["commission"],
            strategy=signal.strategy.value,
            time_open=position["time_open"],
            environment=self.ENVIRONMENT,
            initial_sl=stop_loss,
            initial_tp=take_profit,
            initial_risk_distance=initial_risk,
            remaining_volume=position["volume"],
            highest_r=0.0,
            lowest_r=0.0,
            break_even_activated=False,
            trailing_active=False,
            partial_tp_taken=False,
            management_state={}
        )

        db_manager.add(db_position)

    # ------------------------------------------------------------------
    # Closing and modifying positions
    # ------------------------------------------------------------------

    def close_position(
        self,
        ticket: int,
        reason: str = "MANUAL"
    ) -> OrderResult:
        logger.info("Closing position %s - reason: %s", ticket, reason)

        result = mt5_connector.close_position(ticket)

        if result.success:
            try:
                self._update_position_on_close(ticket, reason, result)
                trade_manager.remove_position(ticket)
            except Exception:
                logger.exception(
                    "Position %s was closed at MT5 but database cleanup failed.",
                    ticket
                )

        return result

    def close_all_positions(
        self,
        reason: str = "EMERGENCY"
    ) -> List[OrderResult]:
        """Close positions returned by the connector (bot-owned positions)."""

        positions = mt5_connector.get_positions()

        if positions is None:
            logger.error(
                "Cannot close all positions: MT5 position query failed."
            )
            return []

        results = []
        for position in positions:
            ticket = int(position["ticket"])
            results.append(self.close_position(ticket, reason))

        return results

    def modify_position_sl_tp(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> OrderResult:
        result = mt5_connector.modify_position(ticket, sl, tp)

        if result.success:
            self._update_position_sl_tp(ticket, sl, tp)

        return result

    # ------------------------------------------------------------------
    # Database close/update helpers
    # ------------------------------------------------------------------

    def _update_position_on_close(
        self,
        ticket: int,
        reason: str,
        close_result: OrderResult
    ) -> None:
        """
        Mark the position closed and create a closed-trade record.

        Note: final realized P/L should ultimately be reconciled from MT5
        deal history. The last stored position P/L is not guaranteed to be
        the exact final realized amount.
        """

        with db_manager.get_session() as session:
            order = session.query(ExecutedOrder).filter(
                ExecutedOrder.ticket == ticket,
                ExecutedOrder.environment == self.ENVIRONMENT
            ).first()

            open_position = session.query(OpenPosition).filter(
                OpenPosition.ticket == ticket,
                OpenPosition.environment == self.ENVIRONMENT
            ).first()

            if order is not None:
                close_time = datetime.now(timezone.utc)
                close_price = (
                    float(close_result.price)
                    if getattr(close_result, "price", None)
                    else float(order.price_current)
                )

                order.time_close = close_time
                order.reason = reason
                order.price_current = close_price

                duration = max(
                    0.0,
                    (close_time - order.time_open).total_seconds() / 3600
                )

                initial_risk = (
                    float(open_position.initial_risk_distance)
                    if open_position is not None
                    and open_position.initial_risk_distance
                    else abs(float(order.price_open) - float(order.sl or 0))
                )

                if initial_risk > 0:
                    if str(order.order_type).upper() == "BUY":
                        realized_r = (
                            close_price - float(order.price_open)
                        ) / initial_risk
                    else:
                        realized_r = (
                            float(order.price_open) - close_price
                        ) / initial_risk
                else:
                    realized_r = 0.0

                net_profit = (
                    float(order.profit or 0)
                    + float(order.swap or 0)
                    + float(order.commission or 0)
                )

                closed_trade = {
                    "ticket": order.ticket,
                    "magic_number": order.magic_number,
                    "symbol": order.symbol,
                    "direction": order.order_type,
                    "volume": order.volume,
                    "price_open": order.price_open,
                    "price_close": close_price,
                    "sl": order.sl,
                    "tp": order.tp,
                    "profit": order.profit,
                    "swap": order.swap,
                    "commission": order.commission,
                    "net_profit": net_profit,
                    "risk_percent": 0.0,
                    "realized_r": realized_r,
                    "strategy": (
                        order.comment.replace("SMC_", "")
                        if order.comment else "UNKNOWN"
                    ),
                    "exit_reason": reason,
                    "time_open": order.time_open,
                    "time_close": close_time,
                    "duration_hours": duration,
                    "environment": self.ENVIRONMENT
                }

                from src.database_manager import ClosedTrade
                session.add(ClosedTrade(**closed_trade))

            session.query(OpenPosition).filter(
                OpenPosition.ticket == ticket,
                OpenPosition.environment == self.ENVIRONMENT
            ).delete()

    def _update_position_sl_tp(
        self,
        ticket: int,
        sl: Optional[float],
        tp: Optional[float]
    ) -> None:
        with db_manager.get_session() as session:
            order = session.query(ExecutedOrder).filter(
                ExecutedOrder.ticket == ticket,
                ExecutedOrder.environment == self.ENVIRONMENT
            ).first()

            if order is not None:
                if sl is not None:
                    order.sl = sl
                if tp is not None:
                    order.tp = tp

            position = session.query(OpenPosition).filter(
                OpenPosition.ticket == ticket,
                OpenPosition.environment == self.ENVIRONMENT
            ).first()

            if position is not None:
                if sl is not None:
                    position.sl = sl
                if tp is not None:
                    position.tp = tp


# Global executor instance
order_executor = OrderExecutor()
