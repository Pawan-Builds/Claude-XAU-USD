"""Safe order execution with pending (limit) order lifecycle management.

The bot places BUY_LIMIT / SELL_LIMIT orders at signal.entry_price. Orders
sit in MT5 until price retraces to the planned level, then fill at the
planned entry — preserving the strategy's R:R. If price never returns,
orders expire or are cancelled. This module reconciles MT5 pending order
state with the database every cycle.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
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
    # NEW: distinguishes an order sitting at the limit vs an immediate fill
    pending: bool = False
    pending_order_ticket: Optional[int] = None


class OrderExecutor:
    """Places pending limit orders and reconciles their fill state."""

    def __init__(self):
        self.trading_config = config.trading
        self.magic_number = self.trading_config.magic_number
        # derive environment from config so LIVE trades are recorded as LIVE
        self.ENVIRONMENT = config.trading.mode.lower()

    # ==================================================================
    # 1. Public entry point
    # ==================================================================

    def execute_signal(self, signal, market_entry: bool = False) -> ExecutionResult:
        """Validate, size, place a pending limit order or market order, and persist it."""

        if not mt5_connector.is_connected():
            return ExecutionResult(success=False, error="MT5 is not connected.")

        if self.ENVIRONMENT == "live" and not config.live_trading_enabled:
            return ExecutionResult(
                success=False,
                error="LIVE trading is disabled (LIVE_TRADING_ENABLED=false).",
            )

        try:
            validation = signal_validator.validate_signal(signal, market_entry=market_entry)
        except Exception as exc:
            logger.exception("Signal validation raised an exception.")
            return ExecutionResult(
                success=False, error=f"Signal validation error: {exc}"
            )

        if not validation.valid:
            logger.warning("Signal rejected: %s", validation.reason)
            return ExecutionResult(
                success=False, validation=validation, error=validation.reason
            )

        try:
            position_size = risk_manager.calculate_position_size(
                signal.entry_price,
                signal.sl_price,
                signal.direction.value,
                signal.risk_percent,
            )
        except Exception as exc:
            logger.exception("Position sizing failed.")
            return ExecutionResult(
                success=False,
                validation=validation,
                error=f"Position sizing error: {exc}",
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
                error=reason,
            )

        signal.lot_size = position_size.lot_size

        fingerprint = self._generate_fingerprint(signal)
        signal.fingerprint = fingerprint

        if self._is_duplicate(fingerprint):
            logger.warning("Duplicate signal rejected: fingerprint %s", fingerprint)
            return ExecutionResult(
                success=False,
                validation=validation,
                error="Duplicate signal already active or recently executed.",
            )

        try:
            signal_id = self._save_signal(signal, position_size, validation)
        except Exception as exc:
            logger.exception("Could not persist signal; order not submitted.")
            return ExecutionResult(
                success=False,
                position_size=position_size,
                validation=validation,
                error=f"Could not save signal. Order was not submitted: {exc}",
            )

        try:
            order_request = self._prepare_order_request(signal, position_size, market_entry=market_entry)
            request_id = self._save_order_request(signal_id, order_request)
        except Exception as exc:
            logger.exception("Could not prepare or persist order request.")
            return ExecutionResult(
                success=False,
                signal_id=signal_id,
                position_size=position_size,
                validation=validation,
                error=f"Could not prepare order request: {exc}",
            )

        try:
            order_result = mt5_connector.send_order(order_request)
        except Exception as exc:
            logger.exception("MT5 order submission raised an exception.")
            self._update_order_request_error(request_id, f"Submission exception: {exc}")
            return ExecutionResult(
                success=False,
                signal_id=signal_id,
                position_size=position_size,
                validation=validation,
                error=f"MT5 submission exception: {exc}",
            )

        try:
            self._update_order_request(request_id, order_result)
        except Exception:
            logger.exception("Could not update order-request record after MT5 response.")

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
                error=error,
            )

        pending_ticket = (
            getattr(order_result, "order_id", None)
            or getattr(order_result, "ticket", None)
        )

        is_pending = not market_entry
        position_ticket = pending_ticket or 0

        logger.info(
            "%s | ticket=%s | %s %s @ %.5f | vol=%.2f | SL=%.5f | TP=%.5f",
            "PENDING ORDER PLACED" if is_pending else "MARKET ORDER FILLED",
            position_ticket,
            signal.direction.value,
            signal.symbol,
            signal.entry_price,
            position_size.lot_size,
            signal.sl_price,
            signal.tp_price,
        )

        if not is_pending:
            # Immediate fill bookkeeping
            positions = mt5_connector.get_positions()
            if positions:
                for pos in positions:
                    if pos["ticket"] == position_ticket or (pos["symbol"] == signal.symbol and pos["magic"] == self.magic_number):
                        fake_request = Mock()
                        fake_request.id = signal_id
                        fake_request.symbol = signal.symbol
                        fake_request.comment = f"SMC_{signal.strategy.value}"
                        self._handle_filled_pending(fake_request, pos)
                        break

        return ExecutionResult(
            success=True,
            signal_id=signal_id,
            order_result=order_result,
            position_size=position_size,
            validation=validation,
            pending=is_pending,
            pending_order_ticket=(int(pending_ticket) if (is_pending and pending_ticket) else None),
        )

    def _generate_fingerprint(self, signal) -> str:
        """Create a durable hash of the setup to prevent duplicate orders."""
        # Quantize prices to 5 decimal places to avoid float noise
        parts = [
            signal.symbol,
            signal.direction.value,
            signal.strategy.value,
            f"{signal.entry_price:.5f}",
            f"{signal.sl_price:.5f}",
            f"{signal.tp_price:.5f}",
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def _is_duplicate(self, fingerprint: str) -> bool:
        """Check if this exact setup is already active or recently filled."""
        if not fingerprint:
            return False

        try:
            with db_manager.get_session() as session:
                # 1. Check for active pending orders with this fingerprint
                active_request = session.query(OrderRequest).filter(
                    OrderRequest.environment == self.ENVIRONMENT,
                    OrderRequest.fingerprint == fingerprint,
                    OrderRequest.status.in_(["placed", "pending", "sent"])
                ).first()

                if active_request:
                    logger.info("Duplicate check: active request found for fingerprint %s", fingerprint)
                    return True

                # 2. Check for recent filled orders (within 24h) to avoid re-entering
                cutoff = datetime.utcnow() - timedelta(hours=24)
                recent_filled = session.query(OrderRequest).filter(
                    OrderRequest.environment == self.ENVIRONMENT,
                    OrderRequest.fingerprint == fingerprint,
                    OrderRequest.status == "filled",
                    OrderRequest.timestamp >= cutoff
                ).first()

                if recent_filled:
                    logger.info("Duplicate check: recent filled order found for fingerprint %s", fingerprint)
                    return True

        except Exception:
            logger.exception("Duplicate check failed; failing closed.")
            return True  # Fail closed! If DB is down, do not trade.

        return False

    # ==================================================================
    # 2. Pending-order reconciliation (called every cycle from main)
    # ==================================================================

    def reconcile_pending_orders(self) -> None:
        """
        Align MT5 pending-order state with the database.

        For each DB OrderRequest with status='placed':
          - Ticket still in MT5's order list  -> leave alone.
          - Ticket gone from MT5              -> decide fill vs cancel.
              * If a matching position exists with no ExecutedOrder row,
                treat as FILLED and complete the bookkeeping.
              * Otherwise treat as CANCELLED (broker expired or rejected).
        """

        try:
            live_orders = mt5_connector.get_orders()
        except Exception:
            logger.exception("Could not fetch pending orders.")
            return

        if live_orders is None:
            logger.error(
                "MT5 pending-order query returned None; skipping reconciliation."
            )
            return

        live_tickets = {int(o["ticket"]) for o in live_orders}

        pending_requests = self._pending_db_requests()
        if pending_requests is None:
            return

        for req in pending_requests:
            ticket = self._request_ticket(req)
            if ticket is None:
                continue
            if ticket in live_tickets:
                continue  # still waiting

            # Ticket disappeared from MT5 — was it filled or cancelled?
            self._finalise_pending_request(req, ticket)

        self._cancel_expired_orders(live_orders)
        self._enforce_pending_order_cap(live_orders)

    # ---------- reconciliation helpers ----------

    def _pending_db_requests(self) -> Optional[List[OrderRequest]]:
        try:
            with db_manager.get_session() as session:
                return (
                    session.query(OrderRequest)
                    .filter(
                        OrderRequest.environment == self.ENVIRONMENT,
                        OrderRequest.status == "placed",
                    )
                    .all()
                )
        except Exception:
            logger.exception("Could not load pending order requests from DB.")
            return None

    @staticmethod
    def _request_ticket(req: OrderRequest) -> Optional[int]:
        response = req.response_data or {}
        ticket = (
            response.get("order_id")
            or response.get("ticket")
            or response.get("pending_ticket")
        )
        try:
            ticket = int(ticket)
        except (TypeError, ValueError):
            return None
        return ticket if ticket > 0 else None

    def _finalise_pending_request(
        self, req: OrderRequest, ticket: int
    ) -> None:
        try:
            positions = mt5_connector.get_positions()
        except Exception:
            logger.exception("Could not fetch positions for reconciliation.")
            positions = None

        if positions is None:
            logger.error(
                "Position query failed while finalising pending order %s; "
                "will retry next cycle.",
                ticket,
            )
            return

        fill_position = None
        for pos in positions:
            if int(pos.get("ticket", 0)) == ticket:
                fill_position = pos
                break

        if fill_position is None:
            fill_position = self._find_unrecorded_position(req, positions)

        if fill_position is not None:
            self._handle_filled_pending(req, fill_position)
        else:
            self._handle_cancelled_pending(req)

    def _find_unrecorded_position(
        self, req: OrderRequest, positions: List[Dict]
    ) -> Optional[Dict]:
        """Return the single position that this bot has not yet recorded."""
        candidates = []
        for pos in positions:
            if int(pos.get("magic", 0)) != int(self.magic_number):
                continue
            if pos.get("symbol") != req.symbol:
                continue
            existing = db_manager.query_one(
                ExecutedOrder,
                ticket=int(pos["ticket"]),
                environment=self.ENVIRONMENT,
            )
            if existing is not None:
                continue
            candidates.append(pos)

        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            logger.error(
                "Ambiguous fill: %d unrecorded positions for req %s; "
                "manual reconciliation required.",
                len(candidates),
                req.id,
            )
        return None

    def _handle_filled_pending(
        self, req: OrderRequest, position: Dict[str, Any]
    ) -> None:
        ticket = int(position["ticket"])
        logger.info(
            "PENDING ORDER FILLED | req=%s ticket=%s | %s @ %.5f",
            req.id,
            ticket,
            position["symbol"],
            float(position["price_open"]),
        )

        try:
            self._save_executed_order(position)
            self._save_open_position_from_request(position, req)

            trade_manager.initialize_position_tracking(
                ticket,
                position["type"],
                float(position["price_open"]),
                float(position.get("sl") or 0),
                float(position.get("tp") or 0),
                float(position["volume"]),
                (req.comment or "SMC_UNKNOWN").replace("SMC_", ""),
            )

            self._mark_request_status(req.id, "filled")

        except Exception:
            logger.exception(
                "Position %s is filled but post-fill bookkeeping failed.",
                ticket,
            )

    def _handle_cancelled_pending(self, req: OrderRequest) -> None:
        logger.info(
            "PENDING ORDER GONE | req=%s | marked cancelled/expired", req.id
        )
        self._mark_request_status(req.id, "cancelled")

    def _mark_request_status(self, request_id: int, status: str) -> None:
        try:
            with db_manager.get_session() as session:
                request = (
                    session.query(OrderRequest)
                    .filter(OrderRequest.id == request_id)
                    .first()
                )
                if request is not None:
                    request.status = status
        except Exception:
            logger.exception(
                "Could not update request %s status.", request_id
            )

    def _cancel_expired_orders(self, live_orders: List[Dict]) -> None:
        if not live_orders:
            return

        max_age = timedelta(hours=config.trading.pending_expiry_hours)
        now = datetime.now(timezone.utc)

        for order in live_orders:
            time_setup = order.get("time_setup")
            if time_setup is None:
                continue
            if time_setup.tzinfo is None:
                time_setup = time_setup.replace(tzinfo=timezone.utc)
            if now - time_setup > max_age:
                ticket = int(order["ticket"])
                logger.info("Cancelling expired pending order %s", ticket)
                try:
                    mt5_connector.delete_order(ticket)
                except Exception:
                    logger.exception(
                        "Failed to cancel expired pending order %s", ticket
                    )

    def _enforce_pending_order_cap(self, live_orders: List[Dict]) -> None:
        cap = int(config.trading.max_pending_orders)
        if cap <= 0 or len(live_orders) <= cap:
            return

        ordered = sorted(
            live_orders,
            key=lambda o: (
                o.get("time_setup")
                or datetime.min.replace(tzinfo=timezone.utc)
            ),
        )
        to_cancel = ordered[: len(ordered) - cap]

        for order in to_cancel:
            ticket = int(order["ticket"])
            logger.info(
                "Cancelling oldest pending order %s (cap=%s, active=%s)",
                ticket,
                cap,
                len(live_orders),
            )
            try:
                mt5_connector.delete_order(ticket)
            except Exception:
                logger.exception(
                    "Failed to cancel pending order %s", ticket
                )

    # ==================================================================
    # 3. Manual cancellation
    # ==================================================================

    def cancel_pending_order(
        self, ticket: int, reason: str = "MANUAL"
    ) -> OrderResult:
        logger.info(
            "Cancelling pending order %s - reason: %s", ticket, reason
        )
        result = mt5_connector.delete_order(ticket)
        if result.success:
            for req in self._pending_db_requests() or []:
                if self._request_ticket(req) == int(ticket):
                    self._mark_request_status(req.id, "cancelled")
                    break
        return result

    def cancel_all_pending(self, reason: str = "SHUTDOWN") -> List[OrderResult]:
        results: List[OrderResult] = []
        for order in mt5_connector.get_orders() or []:
            results.append(
                self.cancel_pending_order(int(order["ticket"]), reason)
            )
        return results

    # ==================================================================
    # 4. Order request preparation
    # ==================================================================

    def _prepare_order_request(
        self,
        signal,
        position_size: PositionSizeResult,
        market_entry: bool = False,
    ) -> Dict[str, Any]:
        """Build a market or pending limit order."""

        symbol = signal.symbol
        direction = signal.direction.value.upper()

        if direction not in ("BUY", "SELL"):
            raise ValueError(f"Unsupported trade direction: {direction}")

        symbol_info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)

        if symbol_info is None or tick is None:
            raise RuntimeError(f"Could not obtain broker symbol info or quote for {symbol}.")

        point = float(symbol_info.point)
        digits = int(symbol_info.digits)
        if point <= 0:
            raise RuntimeError(f"Invalid point size for {symbol}.")

        entry = round(float(signal.entry_price), digits)
        sl = round(float(signal.sl_price), digits)
        tp = round(float(signal.tp_price), digits)

        symbol_info_dc = mt5_connector.get_symbol_info()
        if symbol_info_dc is None:
            raise RuntimeError(f"Could not obtain broker symbol info for {symbol}.")
        pip_size = mt5_connector.pip_size_for(symbol_info_dc)
        if pip_size <= 0:
            raise RuntimeError(f"Broker pip size for {symbol} is invalid.")
            
        configured_slippage = config.trade_management.max_slippage_pips
        deviation = max(1, int(round(float(configured_slippage) * pip_size / point)))

        if market_entry:
            if direction == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
            else:
                order_type = mt5.ORDER_TYPE_SELL
                
            return {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "type": order_type,
                "volume": float(position_size.lot_size),
                "sl": sl,
                "tp": tp,
                "deviation": deviation,
                "magic": self.magic_number,
                "comment": f"SMC_{signal.strategy.value}"[:31],
                "fingerprint": signal.fingerprint,
            }
        
        else:
            if direction == "BUY":
                order_type = mt5.ORDER_TYPE_BUY_LIMIT
                current_price = float(tick.ask)
                if entry >= current_price:
                    raise RuntimeError(f"BUY_LIMIT entry {entry} is at/above current ask {current_price}; setup invalidated before placement.")
            else:
                order_type = mt5.ORDER_TYPE_SELL_LIMIT
                current_price = float(tick.bid)
                if entry <= current_price:
                    raise RuntimeError(f"SELL_LIMIT entry {entry} is at/below current bid {current_price}; setup invalidated before placement.")

            return {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "type": order_type,
                "volume": float(position_size.lot_size),
                "price": entry,
                "sl": sl,
                "tp": tp,
                "deviation": deviation,
                "magic": self.magic_number,
                "comment": f"SMC_{signal.strategy.value}"[:31],
                "type_time": mt5.ORDER_TIME_GTC,
                "fingerprint": signal.fingerprint,
            }
    # ==================================================================
    # 5. Position reconciliation (unchanged, kept for filled positions)
    # ==================================================================

    def _find_position_for_result(
        self,
        result: OrderResult,
        signal,
        position_size: PositionSizeResult,
    ) -> Optional[Dict[str, Any]]:
        """Kept for backwards compatibility with immediate-fill paths."""
        positions = mt5_connector.get_positions()
        if positions is None:
            logger.error("MT5 position query returned None.")
            return None
        if not positions:
            return None

        result_ids = {
            int(v)
            for v in (
                getattr(result, "ticket", None),
                getattr(result, "order_id", None),
                getattr(result, "deal_id", None),
            )
            if v not in (None, 0)
        }

        exact_matches = [
            p for p in positions
            if int(p.get("ticket", 0)) in result_ids
            and self._position_matches_signal(p, signal)
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]

        expected_volume = float(position_size.lot_size)
        candidates = [
            p for p in positions
            if self._position_matches_signal(p, signal)
            and abs(float(p.get("volume", 0)) - expected_volume) <= 1e-8
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _position_matches_signal(self, position, signal) -> bool:
        position_magic = position.get(
            "magic", position.get("magic_number")
        )
        if position_magic is None:
            return False  # reject positions without a magic number
        if int(position_magic) != int(self.magic_number):
            return False
        if position.get("symbol") != signal.symbol:
            return False
        if (
            str(position.get("type", "")).upper()
            != signal.direction.value.upper()
        ):
            return False
        return True

    # ==================================================================
    # 6. Database persistence (fix the db_manager.add int bug)
    # ==================================================================

    def _save_signal(
        self,
        signal,
        position_size: PositionSizeResult,
        validation: ValidationResult,
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
                    "name": c.name,
                    "passed": c.passed,
                    "details": c.details,
                }
                for c in signal.confirmations
            ],
            status="validated",
            environment=self.ENVIRONMENT,
            fingerprint=signal.fingerprint,
        )

        saved_id = db_manager.add(db_signal)
        if not saved_id or int(saved_id) <= 0:
            raise RuntimeError(
                "Database did not return a saved signal ID."
            )
        return int(saved_id)

    def _save_order_request(
        self, signal_id: int, request: Dict[str, Any]
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
            environment=self.ENVIRONMENT,
            fingerprint=request.get("fingerprint"),
        )

        saved_id = db_manager.add(db_request)
        if not saved_id or int(saved_id) <= 0:
            raise RuntimeError(
                "Database did not return an order-request ID."
            )
        return int(saved_id)

    def _update_order_request(
        self, request_id: int, result: OrderResult
    ) -> None:
        with db_manager.get_session() as session:
            request = (
                session.query(OrderRequest)
                .filter(OrderRequest.id == request_id)
                .first()
            )
            if request is None:
                logger.error(
                    "Could not find order-request row %s.", request_id
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
            # 'placed' signals a live pending order not yet filled.
            request.status = "placed" if result.success else "rejected"
            request.error_code = result.error_code
            request.error_message = result.error_message

    def _update_order_request_error(
        self, request_id: int, message: str
    ) -> None:
        try:
            with db_manager.get_session() as session:
                request = (
                    session.query(OrderRequest)
                    .filter(OrderRequest.id == request_id)
                    .first()
                )
                if request is not None:
                    request.status = "submission_error"
                    request.error_message = message
        except Exception:
            logger.exception(
                "Could not record submission exception for %s.", request_id
            )

    def _save_executed_order(self, position: Dict[str, Any]) -> None:
        ticket = int(position["ticket"])

        existing = db_manager.query_one(
            ExecutedOrder, ticket=ticket, environment=self.ENVIRONMENT
        )
        if existing is not None:
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
            environment=self.ENVIRONMENT,
        )
        db_manager.add(db_order)

    def _save_open_position_from_request(
        self, position: Dict[str, Any], request: OrderRequest
    ) -> None:
        """Create OpenPosition row from a filled pending order."""
        ticket = int(position["ticket"])

        existing = db_manager.query_one(
            OpenPosition, ticket=ticket, environment=self.ENVIRONMENT
        )
        if existing is not None:
            return

        entry = float(position["price_open"])
        stop_loss = float(position.get("sl") or 0)
        take_profit = float(position.get("tp") or 0)
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
            strategy=(request.comment or "SMC_UNKNOWN").replace("SMC_", ""),
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
            partial_tp_taken=[],  # JSON column — must be a list, not False
            management_state={},
        )
        db_manager.add(db_position)

    # ==================================================================
    # 7. Close / modify
    # ==================================================================

    def close_position(
        self, ticket: int, reason: str = "MANUAL"
    ) -> OrderResult:
        logger.info("Closing position %s - reason: %s", ticket, reason)
        result = mt5_connector.close_position(ticket)
        if result.success:
            try:
                self._update_position_on_close(ticket, reason, result)
                trade_manager.remove_position(ticket)
            except Exception:
                logger.exception(
                    "Position %s closed at MT5 but DB cleanup failed.", ticket
                )
        return result

    def close_all_positions(
        self, reason: str = "EMERGENCY"
    ) -> List[OrderResult]:
        positions = mt5_connector.get_positions()
        if positions is None:
            logger.error("Cannot close all: position query failed.")
            return []
        return [
            self.close_position(int(p["ticket"]), reason) for p in positions
        ]

    def modify_position_sl_tp(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> OrderResult:
        result = mt5_connector.modify_position(ticket, sl, tp)
        if result.success:
            self._update_position_sl_tp(ticket, sl, tp)
        return result

    def _update_position_on_close(
        self, ticket: int, reason: str, close_result: OrderResult
    ) -> None:
        with db_manager.get_session() as session:
            order = (
                session.query(ExecutedOrder)
                .filter(
                    ExecutedOrder.ticket == ticket,
                    ExecutedOrder.environment == self.ENVIRONMENT,
                )
                .first()
            )
            open_position = (
                session.query(OpenPosition)
                .filter(
                    OpenPosition.ticket == ticket,
                    OpenPosition.environment == self.ENVIRONMENT,
                )
                .first()
            )

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
                    (close_time - order.time_open).total_seconds() / 3600,
                )

                initial_risk = (
                    float(open_position.initial_risk_distance)
                    if open_position is not None
                    and open_position.initial_risk_distance
                    else abs(
                        float(order.price_open) - float(order.sl or 0)
                    )
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

                from src.database_manager import ClosedTrade

                session.add(
                    ClosedTrade(
                        ticket=order.ticket,
                        magic_number=order.magic_number,
                        symbol=order.symbol,
                        direction=order.order_type,
                        volume=order.volume,
                        price_open=order.price_open,
                        price_close=close_price,
                        sl=order.sl,
                        tp=order.tp,
                        profit=order.profit,
                        swap=order.swap,
                        commission=order.commission,
                        net_profit=net_profit,
                        risk_percent=0.0,
                        realized_r=realized_r,
                        strategy=(
                            order.comment.replace("SMC_", "")
                            if order.comment
                            else "UNKNOWN"
                        ),
                        exit_reason=reason,
                        time_open=order.time_open,
                        time_close=close_time,
                        duration_hours=duration,
                        environment=self.ENVIRONMENT,
                    )
                )

            session.query(OpenPosition).filter(
                OpenPosition.ticket == ticket,
                OpenPosition.environment == self.ENVIRONMENT,
            ).delete()

    def _update_position_sl_tp(
        self,
        ticket: int,
        sl: Optional[float],
        tp: Optional[float],
    ) -> None:
        with db_manager.get_session() as session:
            order = (
                session.query(ExecutedOrder)
                .filter(
                    ExecutedOrder.ticket == ticket,
                    ExecutedOrder.environment == self.ENVIRONMENT,
                )
                .first()
            )
            if order is not None:
                if sl is not None:
                    order.sl = sl
                if tp is not None:
                    order.tp = tp

            position = (
                session.query(OpenPosition)
                .filter(
                    OpenPosition.ticket == ticket,
                    OpenPosition.environment == self.ENVIRONMENT,
                )
                .first()
            )
            if position is not None:
                if sl is not None:
                    position.sl = sl
                if tp is not None:
                    position.tp = tp


# Global executor instance
order_executor = OrderExecutor()