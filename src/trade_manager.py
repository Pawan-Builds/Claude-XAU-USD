"""Trade management - persistent SL, TP, break-even and trailing stops."""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from src.mt5_connector import mt5_connector
from src.config_loader import config
from src.database_manager import db_manager, TradeManagementEvent, OpenPosition
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ManagementAction:
    action: str
    ticket: int
    old_value: Optional[float]
    new_value: Optional[float]
    success: bool
    reason: str


class TradeManager:
    """Manages bot-owned positions and persists their management state."""

    def __init__(self):
        self.tm_config = config.trade_management
        self.risk_config = config.risk
        self.trading_config = config.trading
        self._managed_positions: Dict[int, Dict[str, Any]] = {}
        self._pip_size: Optional[float] = None

    # ---------------------------------------------------------
    # Environment and unit helpers
    # ---------------------------------------------------------

    @property
    def environment(self) -> str:
        """FIX #11: derive from the runtime mode, not a hardcoded string."""
        return config.trading.mode.lower()

    def _get_pip_size(self) -> float:
        """FIX #10: derive pip size from broker specs, not a constant."""
        if self._pip_size is None:
            info = mt5_connector.get_symbol_info()

            if info is None:
                logger.error(
                    "Cannot determine pip size: symbol info unavailable."
                )
                return 0.0

            self._pip_size = mt5_connector.pip_size_for(info)

            logger.info(
                "Resolved pip size for %s: %s",
                info.name,
                self._pip_size,
            )

        return self._pip_size

    # ---------------------------------------------------------
    # Basic helpers
    # ---------------------------------------------------------

    @staticmethod
    def _direction(value: Any) -> str:
        """Normalize MT5 position direction."""
        value = str(value).upper()

        if value in ("BUY", "0"):
            return "BUY"

        if value in ("SELL", "1"):
            return "SELL"

        return value

    @staticmethod
    def _calculate_r(
        direction: str,
        entry: float,
        current_price: float,
        initial_risk: float,
    ) -> float:
        if initial_risk <= 0:
            return 0.0

        if direction == "BUY":
            return (current_price - entry) / initial_risk

        return (entry - current_price) / initial_risk

    @staticmethod
    def _normalize_percent(value: float) -> float:
        value = float(value)

        if value > 1:
            value /= 100.0

        return value

    def _get_mt5_position(self, ticket: int) -> Optional[Dict]:
        """FIX #1: handle None from get_positions() without crashing."""
        try:
            positions = mt5_connector.get_positions()
        except Exception as exc:
            logger.error(
                "Could not retrieve MT5 position %s: %s", ticket, exc
            )
            return None

        if positions is None:
            logger.error(
                "MT5 position query failed while looking up ticket %s",
                ticket,
            )
            return None

        for position in positions:
            if int(position["ticket"]) == int(ticket):
                return position

        return None

    def _get_database_position(self, ticket: int):
        """FIX #15: scope the DB lookup to the current environment."""
        try:
            return db_manager.query_one(
                OpenPosition,
                ticket=ticket,
                environment=self.environment,
            )
        except Exception as exc:
            logger.error(
                "Database lookup failed for position %s: %s", ticket, exc
            )
            return None

    def _persist_state(self, ticket: int) -> bool:
        state = self._managed_positions.get(ticket)

        if not state:
            return False

        try:
            record = self._get_database_position(ticket)

            if record is None:
                logger.warning(
                    "Cannot persist position %s: no OpenPosition row for "
                    "environment=%s",
                    ticket,
                    self.environment,
                )
                return False

            management_state = {
                "direction": state["direction"],
                "entry_price": state["entry_price"],
                "initial_sl": state["initial_sl"],
                "initial_tp": state["initial_tp"],
                "current_sl": state["current_sl"],
                "current_tp": state["current_tp"],
                "initial_risk_distance": state["initial_risk_distance"],
                "volume": state["volume"],
                "remaining_volume": state["remaining_volume"],
                "strategy": state["strategy"],
                "break_even_activated": state["break_even_activated"],
                "trailing_active": state["trailing_active"],
                "partial_tps_taken": state["partial_tps_taken"],
                "highest_r": state["highest_r"],
                "lowest_r": state["lowest_r"],
            }

            db_manager.update(
                OpenPosition,
                record.id,
                sl=state["current_sl"],
                tp=state["current_tp"],
                break_even_activated=state["break_even_activated"],
                trailing_active=state["trailing_active"],
                partial_tp_taken=bool(state["partial_tps_taken"]),
                initial_sl=state["initial_sl"],
                initial_tp=state["initial_tp"],
                initial_risk_distance=state["initial_risk_distance"],
                remaining_volume=state["remaining_volume"],
                highest_r=state["highest_r"],
                lowest_r=state["lowest_r"],
                management_state=management_state,
            )

            return True

        except Exception as exc:
            logger.exception(
                "Failed to persist management state for %s: %s",
                ticket,
                exc,
            )
            return False

    # ---------------------------------------------------------
    # Position initialization and restoration
    # ---------------------------------------------------------

    def initialize_position_tracking(
        self,
        ticket: int,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
        volume: float,
        strategy: str,
    ):
        direction = self._direction(direction)
        record = self._get_database_position(ticket)

        initial_sl = sl
        initial_tp = tp
        initial_risk = abs(entry_price - initial_sl)

        remaining_volume = volume
        break_even_activated = False
        trailing_active = False
        partial_tps_taken = []
        highest_r = 0.0
        lowest_r = 0.0
        current_sl = sl
        current_tp = tp

        if record is not None:
            initial_sl = record.initial_sl or sl
            initial_tp = record.initial_tp or tp
            initial_risk = (
                record.initial_risk_distance
                or abs(entry_price - initial_sl)
            )

            remaining_volume = (
                record.remaining_volume
                if record.remaining_volume is not None
                else volume
            )

            current_sl = record.sl or sl
            current_tp = record.tp or tp

            break_even_activated = bool(record.break_even_activated)
            trailing_active = bool(record.trailing_active)
            highest_r = record.highest_r or 0.0
            lowest_r = record.lowest_r or 0.0

            saved_state = record.management_state or {}

            if isinstance(saved_state, dict):
                partial_tps_taken = saved_state.get(
                    "partial_tps_taken", []
                )

                current_sl = saved_state.get("current_sl", current_sl)
                current_tp = saved_state.get("current_tp", current_tp)

                break_even_activated = saved_state.get(
                    "break_even_activated",
                    break_even_activated,
                )

                trailing_active = saved_state.get(
                    "trailing_active",
                    trailing_active,
                )

                highest_r = saved_state.get("highest_r", highest_r)
                lowest_r = saved_state.get("lowest_r", lowest_r)

        if initial_risk <= 0:
            logger.error(
                "Position %s has invalid initial risk distance. "
                "Management will not be initialized.",
                ticket,
            )
            return False

        self._managed_positions[ticket] = {
            "direction": direction,
            "entry_price": float(entry_price),
            "initial_sl": float(initial_sl),
            "current_sl": float(current_sl),
            "initial_tp": float(initial_tp),
            "current_tp": float(current_tp),
            "initial_risk_distance": float(initial_risk),
            "volume": float(volume),
            "remaining_volume": float(remaining_volume),
            "strategy": strategy,
            "break_even_activated": break_even_activated,
            "trailing_active": trailing_active,
            "partial_tps_taken": partial_tps_taken,
            "highest_r": float(highest_r),
            "lowest_r": float(lowest_r),
        }

        logger.info(
            "Initialized tracking for %s | %s | entry=%.2f | "
            "initial risk=%.2f | remaining volume=%.3f",
            ticket,
            direction,
            entry_price,
            initial_risk,
            remaining_volume,
        )

        if record is not None:
            self._persist_state(ticket)

        return True

    # ---------------------------------------------------------
    # Market updates and R calculations
    # ---------------------------------------------------------

    def update_position(
        self,
        ticket: int,
        current_price: float,
        profit: float,
        swap: float = 0,
    ):
        pos = self._managed_positions.get(ticket)

        if not pos:
            return

        r_multiple = self._calculate_r(
            pos["direction"],
            pos["entry_price"],
            current_price,
            pos["initial_risk_distance"],
        )

        pos["highest_r"] = max(pos["highest_r"], r_multiple)
        pos["lowest_r"] = min(pos["lowest_r"], r_multiple)

        self._persist_state(ticket)

    def _current_r(self, pos: Dict, current_price: float) -> float:
        return self._calculate_r(
            pos["direction"],
            pos["entry_price"],
            current_price,
            pos["initial_risk_distance"],
        )

    # ---------------------------------------------------------
    # Break-even
    # ---------------------------------------------------------

    def check_break_even(
        self,
        ticket: int,
        current_price: float,
    ) -> Optional[ManagementAction]:

        if not self.tm_config.break_even_enabled:
            return None

        pos = self._managed_positions.get(ticket)

        if not pos or pos["break_even_activated"]:
            return None

        r_multiple = self._current_r(pos, current_price)

        if r_multiple < self.tm_config.break_even_trigger_r:
            return None

        pip_size = self._get_pip_size()

        if pip_size <= 0:
            logger.error(
                "Cannot apply break-even for %s: pip size unavailable.",
                ticket,
            )
            return None

        buffer = self.tm_config.break_even_buffer_pips * pip_size

        if pos["direction"] == "BUY":
            new_sl = pos["entry_price"] + buffer
        else:
            new_sl = pos["entry_price"] - buffer

        action = self._modify_sl(ticket, new_sl, "break_even")

        if action.success:
            pos["break_even_activated"] = True
            self._persist_state(ticket)

            self._log_event(
                ticket,
                "BREAK_EVEN_ACTIVATED",
                pos["initial_sl"],
                new_sl,
                {"r_multiple": r_multiple},
            )

        return action

    # ---------------------------------------------------------
    # Trailing stop
    # ---------------------------------------------------------

    def check_trailing_stop(
        self,
        ticket: int,
        current_price: float,
    ) -> Optional[ManagementAction]:

        if not self.tm_config.trailing_stop_enabled:
            return None

        pos = self._managed_positions.get(ticket)

        if not pos:
            return None

        r_multiple = self._current_r(pos, current_price)
        start_r = self.tm_config.trailing_stop_start_r

        if r_multiple < start_r:
            return None

        pip_size = self._get_pip_size()

        if pip_size <= 0:
            logger.error(
                "Cannot apply trailing stop for %s: pip size unavailable.",
                ticket,
            )
            return None

        if not pos["trailing_active"]:
            pos["trailing_active"] = True

            logger.info(
                "Trailing stop activated for %s at %.2fR",
                ticket,
                r_multiple,
            )

            self._persist_state(ticket)

        trailing_distance = (
            self.tm_config.trailing_stop_step_pips * pip_size
        )

        if pos["direction"] == "BUY":
            new_sl = current_price - trailing_distance

            if pos["current_sl"] > 0 and new_sl <= pos["current_sl"]:
                return None

        else:
            new_sl = current_price + trailing_distance

            if pos["current_sl"] > 0 and new_sl >= pos["current_sl"]:
                return None

        return self._modify_sl(ticket, new_sl, "trailing_stop")

    # ---------------------------------------------------------
    # Partial take profit
    # ---------------------------------------------------------

    def check_partial_tp(
        self,
        ticket: int,
        current_price: float,
    ) -> List[ManagementAction]:

        if not self.tm_config.partial_tp_enabled:
            return []

        pos = self._managed_positions.get(ticket)

        if not pos:
            return []

        actions = []
        r_multiple = self._current_r(pos, current_price)

        for tp_level in self.tm_config.partial_tp_levels:
            target_r = float(tp_level["r"])
            percent = self._normalize_percent(tp_level["percent"])

            if not 0 < percent < 1:
                logger.error(
                    "Invalid partial TP percentage for ticket %s: %s",
                    ticket,
                    tp_level["percent"],
                )
                continue

            already_taken = any(
                abs(float(item["r"]) - target_r) < 0.001
                for item in pos["partial_tps_taken"]
            )

            if already_taken or r_multiple < target_r:
                continue

            requested_volume = pos["volume"] * percent

            requested_volume = min(
                requested_volume,
                pos["remaining_volume"],
            )

            if requested_volume <= 0:
                continue

            action = self._partial_close(
                ticket,
                requested_volume,
                f"partial_tp_r{target_r}",
            )

            if action.success:
                pos["partial_tps_taken"].append({
                    "r": target_r,
                    "percent": percent,
                    "time": datetime.now(timezone.utc).isoformat(),
                })

                self._refresh_remaining_volume(ticket)
                self._persist_state(ticket)

                self._log_event(
                    ticket,
                    "PARTIAL_TP",
                    None,
                    requested_volume,
                    {
                        "target_r": target_r,
                        "actual_r": r_multiple,
                        "requested_percent": percent,
                    },
                )

                actions.append(action)

        return actions

    def _refresh_remaining_volume(self, ticket: int):
        pos = self._managed_positions.get(ticket)

        if not pos:
            return

        live_position = self._get_mt5_position(ticket)

        if live_position is not None:
            pos["remaining_volume"] = float(live_position["volume"])
        else:
            pos["remaining_volume"] = 0.0

    # ---------------------------------------------------------
    # Maximum holding time
    # ---------------------------------------------------------

    def check_max_holding_time(
        self,
        ticket: int,
        open_time: datetime,
    ) -> Optional[ManagementAction]:
        """FIX #12: do NOT close here. Return a signal for main to
        route the close through order_executor so ClosedTrade is
        recorded exactly once."""

        max_hours = self.trading_config.max_holding_hours

        if max_hours is None or max_hours <= 0:
            return None

        if open_time.tzinfo is None:
            open_time = open_time.replace(tzinfo=timezone.utc)

        elapsed = (
            datetime.now(timezone.utc) - open_time
        ).total_seconds() / 3600

        if elapsed < max_hours:
            return None

        logger.warning(
            "Position %s exceeded maximum holding time: %.2f hours",
            ticket,
            elapsed,
        )

        return ManagementAction(
            "time_close",
            ticket,
            None,
            None,
            True,
            f"Maximum holding time ({max_hours}h) exceeded "
            f"(elapsed {elapsed:.2f}h)",
        )

    # ---------------------------------------------------------
    # Stop-loss modification
    # ---------------------------------------------------------

    def _modify_sl(
        self,
        ticket: int,
        new_sl: float,
        reason: str,
    ) -> ManagementAction:

        pos = self._managed_positions.get(ticket)

        if not pos:
            return ManagementAction(
                "sl_modified", ticket, None, new_sl,
                False, "Position not tracked",
            )

        old_sl = pos["current_sl"]
        direction = pos["direction"]

        live_position = self._get_mt5_position(ticket)

        if live_position is None:
            return ManagementAction(
                "sl_modified", ticket, old_sl, new_sl,
                False, "Position no longer exists in MT5",
            )

        market_price = float(live_position["price_current"])

        if old_sl > 0:
            if direction == "BUY" and new_sl <= old_sl:
                return ManagementAction(
                    "sl_modified", ticket, old_sl, new_sl,
                    False,
                    "New BUY stop would loosen the existing stop",
                )

            if direction == "SELL" and new_sl >= old_sl:
                return ManagementAction(
                    "sl_modified", ticket, old_sl, new_sl,
                    False,
                    "New SELL stop would loosen the existing stop",
                )

        if direction == "BUY" and new_sl >= market_price:
            return ManagementAction(
                "sl_modified", ticket, old_sl, new_sl,
                False,
                "BUY stop must be below the current market price",
            )

        if direction == "SELL" and new_sl <= market_price:
            return ManagementAction(
                "sl_modified", ticket, old_sl, new_sl,
                False,
                "SELL stop must be above the current market price",
            )

        result = mt5_connector.modify_position(ticket, sl=new_sl)

        if not result.success:
            error = (
                result.error_message
                or "Unknown MT5 modification error"
            )

            logger.error(
                "Failed to modify SL for %s: %s", ticket, error
            )

            return ManagementAction(
                "sl_modified", ticket, old_sl, new_sl, False, error
            )

        pos["current_sl"] = float(new_sl)

        self._persist_state(ticket)

        self._log_event(
            ticket,
            "SL_MODIFIED",
            old_sl,
            new_sl,
            {"reason": reason},
        )

        logger.info(
            "SL modified for %s: %.2f -> %.2f (%s)",
            ticket,
            old_sl,
            new_sl,
            reason,
        )

        return ManagementAction(
            "sl_modified", ticket, old_sl, new_sl, True, reason
        )

    # ---------------------------------------------------------
    # Partial close
    # ---------------------------------------------------------

    def _partial_close(
        self,
        ticket: int,
        volume: float,
        reason: str,
    ) -> ManagementAction:

        pos = self._managed_positions.get(ticket)

        if not pos:
            return ManagementAction(
                "partial_tp", ticket, volume, None,
                False, "Position not tracked",
            )

        if volume <= 0 or volume >= pos["remaining_volume"]:
            return ManagementAction(
                "partial_tp", ticket, volume, None,
                False, "Invalid partial-close volume",
            )

        result = mt5_connector.close_position(ticket, volume)

        if not result.success:
            error = result.error_message or "Unknown MT5 close error"

            logger.error(
                "Partial close failed for %s: %s", ticket, error
            )

            return ManagementAction(
                "partial_tp", ticket, volume, None, False, error
            )

        self._refresh_remaining_volume(ticket)

        logger.info(
            "Partial close accepted for %s: requested %.3f lots; "
            "remaining %.3f",
            ticket,
            volume,
            pos["remaining_volume"],
        )

        return ManagementAction(
            "partial_tp",
            ticket,
            volume,
            pos["remaining_volume"],
            True,
            reason,
        )

    # ---------------------------------------------------------
    # Database event logging
    # ---------------------------------------------------------

    def _log_event(
        self,
        ticket: int,
        event_type: str,
        old_value: Optional[float],
        new_value: Optional[float],
        details: Dict,
    ):
        """FIX #11: use runtime environment, not a hardcoded 'demo'."""
        try:
            event = TradeManagementEvent(
                ticket=ticket,
                event_type=event_type,
                old_value=old_value,
                new_value=new_value,
                details=details,
                environment=self.environment,
            )

            db_manager.add(event)

        except Exception as exc:
            logger.exception(
                "Failed to log management event for %s: %s",
                ticket,
                exc,
            )

    # ---------------------------------------------------------
    # State access and cleanup
    # ---------------------------------------------------------

    def get_position_state(self, ticket: int) -> Optional[Dict]:
        return self._managed_positions.get(ticket)

    def remove_position(self, ticket: int):
        if ticket in self._managed_positions:
            del self._managed_positions[ticket]
            logger.info(
                "Removed position %s from memory tracking", ticket
            )

    # ---------------------------------------------------------
    # MT5 synchronization and recovery
    # ---------------------------------------------------------

    def sync_with_mt5(self) -> bool:
        """FIX #1 / #17: return True on success, False on failure."""

        try:
            mt5_positions = mt5_connector.get_positions()
        except Exception as exc:
            logger.exception("Could not synchronize with MT5: %s", exc)
            return False

        if mt5_positions is None:
            logger.error(
                "MT5 returned no position data; synchronization skipped"
            )
            return False

        live_tickets = {
            int(position["ticket"]) for position in mt5_positions
        }

        for ticket in list(self._managed_positions.keys()):
            if ticket not in live_tickets:
                self.remove_position(ticket)

        for position in mt5_positions:
            ticket = int(position["ticket"])

            if ticket in self._managed_positions:
                self._refresh_remaining_volume(ticket)
                continue

            record = self._get_database_position(ticket)

            if record is None:
                logger.warning(
                    "Ignoring unrecognized MT5 position %s. "
                    "No matching OpenPosition row for environment=%s.",
                    ticket,
                    self.environment,
                )
                continue

            direction = self._direction(position["type"])

            self.initialize_position_tracking(
                ticket=ticket,
                direction=direction,
                entry_price=position["price_open"],
                sl=position["sl"],
                tp=position["tp"],
                volume=record.volume,
                strategy=record.strategy or "UNKNOWN",
            )

            if ticket in self._managed_positions:
                self._refresh_remaining_volume(ticket)

        logger.info(
            "MT5 synchronization completed. Live positions: %s | "
            "Managed positions: %s",
            len(mt5_positions),
            len(self._managed_positions),
        )

        return True


# Global trade manager instance
trade_manager = TradeManager()