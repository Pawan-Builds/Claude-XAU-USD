"""Trade management - persistent SL, TP, break-even and trailing stops.

F-05/F-06 additions:
- reconcile_with_broker(): recovers orphan fills and reconstructs missing
  closures from MT5 deal history.
- record_closed_trade_from_history(): single source of truth for ClosedTrade
  rows, populated from actual realised profit/swap/commission.
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from src.mt5_connector import mt5_connector
from src.config_loader import config
from src.database_manager import (
    db_manager,
    TradeManagementEvent,
    OpenPosition,
    ExecutedOrder,
    ClosedTrade,
    OrderRequest,
    AppSettings,
)
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
    """Manages bot-owned positions and reconciles them with the broker."""

    def __init__(self):
        self.tm_config = config.trade_management
        self.risk_config = config.risk
        self.trading_config = config.trading
        self._managed_positions: Dict[int, Dict[str, Any]] = {}
        self._pip_size: Optional[float] = None

    # ---------------------------------------------------------
    # Environment and unit helpers (unchanged)
    # ---------------------------------------------------------

    @property
    def environment(self) -> str:
        return config.trading.mode.lower()

    def _get_pip_size(self) -> float:
        if self._pip_size is None:
            info = mt5_connector.get_symbol_info()
            if info is None:
                logger.error(
                    "Cannot determine pip size: symbol info unavailable."
                )
                return 0.0
            self._pip_size = mt5_connector.pip_size_for(info)
            logger.info(
                "Resolved pip size for %s: %s", info.name, self._pip_size,
            )
        return self._pip_size

    # ---------------------------------------------------------
    # Basic helpers (unchanged)
    # ---------------------------------------------------------

    @staticmethod
    def _direction(value: Any) -> str:
        value = str(value).upper()
        if value in ("BUY", "0"):
            return "BUY"
        if value in ("SELL", "1"):
            return "SELL"
        return value

    @staticmethod
    def _calculate_r(
        direction: str, entry: float, current_price: float, initial_risk: float,
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
        try:
            positions = mt5_connector.get_positions()
        except Exception as exc:
            logger.error(
                "Could not retrieve MT5 position %s: %s", ticket, exc,
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
        try:
            return db_manager.query_one(
                OpenPosition, ticket=ticket, environment=self.environment,
            )
        except Exception as exc:
            logger.error(
                "Database lookup failed for position %s: %s", ticket, exc,
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
                # F-13 will address the JSON corruption below; F-06 does not.
                partial_tp_taken=list(state["partial_tps_taken"]),
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
                "Failed to persist management state for %s: %s", ticket, exc,
            )
            return False

    # ---------------------------------------------------------
    # Position initialization (unchanged)
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
        partial_tps_taken: List = []
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
                ) or []
                current_sl = saved_state.get("current_sl", current_sl)
                current_tp = saved_state.get("current_tp", current_tp)
                break_even_activated = saved_state.get(
                    "break_even_activated", break_even_activated,
                )
                trailing_active = saved_state.get(
                    "trailing_active", trailing_active,
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
    # Market updates and R calculations (unchanged)
    # ---------------------------------------------------------

    def update_position(
        self, ticket: int, current_price: float,
        profit: float, swap: float = 0,
    ):
        pos = self._managed_positions.get(ticket)
        if not pos:
            return
        r_multiple = self._calculate_r(
            pos["direction"], pos["entry_price"], current_price,
            pos["initial_risk_distance"],
        )
        pos["highest_r"] = max(pos["highest_r"], r_multiple)
        pos["lowest_r"] = min(pos["lowest_r"], r_multiple)
        self._persist_state(ticket)

    def _current_r(self, pos: Dict, current_price: float) -> float:
        return self._calculate_r(
            pos["direction"], pos["entry_price"], current_price,
            pos["initial_risk_distance"],
        )

    # ---------------------------------------------------------
    # Break-even (unchanged)
    # ---------------------------------------------------------

    def check_break_even(
        self, ticket: int, current_price: float,
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
                ticket, "BREAK_EVEN_ACTIVATED", pos["initial_sl"], new_sl,
                {"r_multiple": r_multiple},
            )
        return action

    # ---------------------------------------------------------
    # Trailing stop (unchanged)
    # ---------------------------------------------------------

    def check_trailing_stop(
        self, ticket: int, current_price: float,
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
                "Trailing stop activated for %s at %.2fR", ticket, r_multiple,
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
    # Partial take-profit (unchanged)
    # ---------------------------------------------------------

    def check_partial_tp(
        self, ticket: int, current_price: float,
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
                    ticket, tp_level["percent"],
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
                requested_volume, pos["remaining_volume"],
            )
            if requested_volume <= 0:
                continue
            action = self._partial_close(
                ticket, requested_volume, f"partial_tp_r{target_r}",
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
                    ticket, "PARTIAL_TP", None, requested_volume,
                    {
                        "target_r": target_r,
                        "actual_r": r_multiple,
                        "requested_percent": percent,
                    },
                )
                actions.append(action)
        return actions

    def _refresh_remaining_volume(self, ticket: int) -> bool:
        """Sync the in-memory remaining_volume with the broker.

        Distinguishes three outcomes that ``_get_mt5_position`` collapses
        into ``None``:

          * Query raised / returned None → **leave volume unchanged**,
            log loudly, retry next cycle. Silently zeroing was the bug
            (H-06 §6.7): it desynced partial-TP and ClosedTrade logic
            from the broker after a transient network hiccup.
          * Query succeeded, ticket absent → position is closed → set
            volume to 0.0.
          * Query succeeded, ticket present → set volume from broker.

        Returns True when the volume is now known to be correct.
        """
        pos = self._managed_positions.get(ticket)
        if not pos:
            return False

        try:
            positions = mt5_connector.get_positions()
        except Exception:
            logger.exception(
                "Position query raised while refreshing volume for %s; "
                "leaving stale volume in place.", ticket,
            )
            return False

        if positions is None:
            logger.error(
                "Position query returned None while refreshing volume "
                "for %s; leaving stale volume in place.", ticket,
            )
            return False

        for position in positions:
            if int(position["ticket"]) == int(ticket):
                pos["remaining_volume"] = float(position["volume"])
                return True

        # Query succeeded and the position is genuinely absent.
        pos["remaining_volume"] = 0.0
        return True

    # ---------------------------------------------------------
    # Maximum holding time (unchanged)
    # ---------------------------------------------------------

    def check_max_holding_time(
        self, ticket: int, open_time: datetime,
    ) -> Optional[ManagementAction]:
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
            ticket, elapsed,
        )
        return ManagementAction(
            "time_close", ticket, None, None, True,
            f"Maximum holding time ({max_hours}h) exceeded "
            f"(elapsed {elapsed:.2f}h)",
        )

    # ---------------------------------------------------------
    # Stop-loss modification (unchanged)
    # ---------------------------------------------------------

    def _modify_sl(
        self, ticket: int, new_sl: float, reason: str,
    ) -> ManagementAction:
        pos = self._managed_positions.get(ticket)
        if not pos:
            return ManagementAction(
                "sl_modified", ticket, None, new_sl, False,
                "Position not tracked",
            )
        old_sl = pos["current_sl"]
        direction = pos["direction"]
        live_position = self._get_mt5_position(ticket)
        if live_position is None:
            return ManagementAction(
                "sl_modified", ticket, old_sl, new_sl, False,
                "Position no longer exists in MT5",
            )
        market_price = float(live_position["price_current"])
        if old_sl > 0:
            if direction == "BUY" and new_sl <= old_sl:
                return ManagementAction(
                    "sl_modified", ticket, old_sl, new_sl, False,
                    "New BUY stop would loosen the existing stop",
                )
            if direction == "SELL" and new_sl >= old_sl:
                return ManagementAction(
                    "sl_modified", ticket, old_sl, new_sl, False,
                    "New SELL stop would loosen the existing stop",
                )
        if direction == "BUY" and new_sl >= market_price:
            return ManagementAction(
                "sl_modified", ticket, old_sl, new_sl, False,
                "BUY stop must be below the current market price",
            )
        if direction == "SELL" and new_sl <= market_price:
            return ManagementAction(
                "sl_modified", ticket, old_sl, new_sl, False,
                "SELL stop must be above the current market price",
            )
        result = mt5_connector.modify_position(ticket, sl=new_sl)
        if not result.success:
            error = result.error_message or "Unknown MT5 modification error"
            logger.error(
                "Failed to modify SL for %s: %s", ticket, error,
            )
            return ManagementAction(
                "sl_modified", ticket, old_sl, new_sl, False, error,
            )
        pos["current_sl"] = float(new_sl)
        self._persist_state(ticket)
        self._log_event(
            ticket, "SL_MODIFIED", old_sl, new_sl, {"reason": reason},
        )
        logger.info(
            "SL modified for %s: %.2f -> %.2f (%s)",
            ticket, old_sl, new_sl, reason,
        )
        return ManagementAction(
            "sl_modified", ticket, old_sl, new_sl, True, reason,
        )

    # ---------------------------------------------------------
    # Partial close (unchanged)
    # ---------------------------------------------------------

    def _partial_close(
        self, ticket: int, volume: float, reason: str,
    ) -> ManagementAction:
        pos = self._managed_positions.get(ticket)
        if not pos:
            return ManagementAction(
                "partial_tp", ticket, volume, None, False,
                "Position not tracked",
            )
        if volume <= 0 or volume >= pos["remaining_volume"]:
            return ManagementAction(
                "partial_tp", ticket, volume, None, False,
                "Invalid partial-close volume",
            )
        result = mt5_connector.close_position(ticket, volume)
        if not result.success:
            error = result.error_message or "Unknown MT5 close error"
            logger.error(
                "Partial close failed for %s: %s", ticket, error,
            )
            return ManagementAction(
                "partial_tp", ticket, volume, None, False, error,
            )
        self._refresh_remaining_volume(ticket)
        logger.info(
            "Partial close accepted for %s: requested %.3f lots; "
            "remaining %.3f",
            ticket, volume, pos["remaining_volume"],
        )
        return ManagementAction(
            "partial_tp", ticket, volume, pos["remaining_volume"], True,
            reason,
        )

    # ---------------------------------------------------------
    # Event logging (unchanged)
    # ---------------------------------------------------------

    def _log_event(
        self, ticket: int, event_type: str,
        old_value: Optional[float], new_value: Optional[float],
        details: Dict,
    ):
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
                "Failed to log management event for %s: %s", ticket, exc,
            )

    # ---------------------------------------------------------
    # State access (unchanged)
    # ---------------------------------------------------------

    def get_position_state(self, ticket: int) -> Optional[Dict]:
        return self._managed_positions.get(ticket)

    def remove_position(self, ticket: int):
        if ticket in self._managed_positions:
            del self._managed_positions[ticket]
            logger.info(
                "Removed position %s from memory tracking", ticket,
            )

    # =========================================================
    # F-05 / F-06: broker reconciliation and closed-trade recovery
    # =========================================================

    def sync_with_mt5(self) -> bool:
        """Backwards-compatible alias for reconcile_with_broker."""
        return self.reconcile_with_broker()

    def reconcile_with_broker(self) -> bool:
        """Full bidirectional reconciliation of DB position state with MT5.

        Returns True only when:
          - every mandatory query succeeded, AND
          - no bot-owned position remains unresolved.

        H-04: a database or broker query failure is a *distinct*
        failure mode from "some ticket failed to reconcile". The two
        are tracked separately, and either one forces a False return.
        Before this fix, a failure of
        ``db_manager.get_open_positions`` was substituted with ``[]``
        and then masked by an empty ``unresolved`` list — reconciliation
        reported success while the database was never read.
        """
        unresolved: List[int] = []
        errors: List[str] = []
        # H-04: independent of `unresolved`. Set to False on any
        # mandatory query failure so that an empty unresolved list
        # cannot mask the failure.
        mandatory_queries_ok = True

        try:
            mt5_positions = mt5_connector.get_positions()
        except Exception as exc:
            logger.exception("MT5 position query failed: %s", exc)
            self._persist_unresolved([], f"query failed: {exc}")
            return False

        if mt5_positions is None:
            logger.error(
                "MT5 returned no position data; reconciliation skipped"
            )
            self._persist_unresolved([], "position query returned None")
            return False

        live_tickets = {int(p["ticket"]) for p in mt5_positions}

        # 1. Live positions: resolve or mark unresolved.
        for position in mt5_positions:
            ticket = int(position["ticket"])
            try:
                ok = self._reconcile_live_position(position)
            except Exception as exc:
                logger.exception(
                    "Failed to reconcile live position %s", ticket,
                )
                errors.append(f"ticket {ticket}: exception: {exc}")
                unresolved.append(ticket)
                continue
            if not ok:
                errors.append(f"ticket {ticket}: could not reconstruct")
                unresolved.append(ticket)

        # 2. DB positions no longer in MT5: reconstruct closures.
        try:
            open_db_positions = db_manager.get_open_positions(
                environment=self.environment
            )
        except Exception as exc:
            logger.exception("Could not list OpenPosition rows.")
            open_db_positions = []
            errors.append(f"DB open-position list failed: {exc}")
            mandatory_queries_ok = False   # H-04

        for record in open_db_positions:
            ticket = int(record.ticket)
            if ticket in live_tickets:
                continue
            try:
                ok = self.reconstruct_closed_trade(
                    ticket, exit_reason="RECOVERED",
                )
            except Exception as exc:
                logger.exception(
                    "Failed to reconstruct closure for ticket %s", ticket,
                )
                errors.append(f"closure {ticket}: exception: {exc}")
                unresolved.append(ticket)
                continue
            if not ok:
                errors.append(f"closure {ticket}: reconstruct returned False")
                unresolved.append(ticket)

        # 3. Drop stale in-memory tracking.
        for ticket in list(self._managed_positions.keys()):
            if ticket not in live_tickets:
                self.remove_position(ticket)

        self._persist_unresolved(unresolved, "; ".join(errors))

        logger.info(
            "Broker reconciliation complete. MT5 positions: %s | "
            "Managed positions: %s | DB open rows: %s | "
            "unresolved: %s | queries_ok: %s",
            len(mt5_positions),
            len(self._managed_positions),
            len(open_db_positions),
            len(unresolved),
            mandatory_queries_ok,
        )

        # H-04: mandatory-query failure overrides the empty-unresolved
        # signal. A reconciliation that could not read the database is
        # not a successful reconciliation, even if no tickets were
        # outstanding at that moment.
        return mandatory_queries_ok and len(unresolved) == 0
   
    def _persist_unresolved(
        self, tickets: List[int], errors: str,
    ) -> None:
        key = f"unresolved_recon_{self.environment}"
        payload = {
            "tickets": sorted({int(t) for t in tickets}),
            "errors": errors or "",
        }
        try:
            with db_manager.get_session() as session:
                setting = session.query(AppSettings).filter_by(key=key).first()
                if setting is None:
                    session.add(AppSettings(
                        key=key, value=json.dumps(payload),
                    ))
                else:
                    setting.value = json.dumps(payload)
        except Exception:
            logger.exception(
                "Could not persist unresolved-reconciliation marker."
            )

    def has_unresolved_reconciliation(self) -> bool:
        """H-05: True when unresolved tickets exist OR the state is
        unverifiable.

        Unknown state is treated as unresolved — callers that cannot
        verify the marker must assume the worst. This is the method
        used by the LIVE opening boundary.
        """
        tickets = self.get_unresolved_tickets_strict()
        if tickets is None:
            return True
        return len(tickets) > 0

    def get_unresolved_tickets_strict(self) -> Optional[List[int]]:
        """H-05: return unresolved tickets, or None when the state is
        unverifiable.

        Return values:
            []              — verification succeeded, no unresolved tickets
            [t1, t2, ...]   — these tickets remain unresolved
            None            — state cannot be verified (fail closed):
                              DB read error, marker missing, empty value,
                              corrupt JSON, missing 'tickets' key, or
                              non-integer ticket values.

        Callers MUST NOT treat None as safe. Safety-critical consumers
        (EntryGate, MT5Connector opening boundary,
        has_unresolved_reconciliation) fail closed on None.
        """
        key = f"unresolved_recon_{self.environment}"

        try:
            setting = db_manager.query_one(AppSettings, key=key)
        except Exception:
            logger.exception(
                "Could not read unresolved-reconciliation marker "
                "(environment=%s); treating as unknown.",
                self.environment,
            )
            return None

        if setting is None:
            logger.warning(
                "Unresolved-reconciliation marker missing for "
                "environment=%s; treating as unknown (fail closed).",
                self.environment,
            )
            return None

        if not setting.value:
            logger.warning(
                "Unresolved-reconciliation marker is empty for "
                "environment=%s; treating as unknown (fail closed).",
                self.environment,
            )
            return None

        try:
            payload = json.loads(setting.value)
        except (json.JSONDecodeError, TypeError):
            logger.exception(
                "Corrupt unresolved-reconciliation marker for "
                "environment=%s; treating as unknown (fail closed).",
                self.environment,
            )
            return None

        if not isinstance(payload, dict) or "tickets" not in payload:
            logger.error(
                "Unresolved-reconciliation marker is malformed "
                "(missing 'tickets' key) for environment=%s; "
                "treating as unknown (fail closed).",
                self.environment,
            )
            return None

        try:
            return [int(t) for t in payload["tickets"]]
        except (TypeError, ValueError):
            logger.exception(
                "Unresolved-reconciliation marker has non-integer "
                "tickets for environment=%s; treating as unknown "
                "(fail closed).",
                self.environment,
            )
            return None

    def get_unresolved_tickets(self) -> List[int]:
        """H-05: backwards-compatible accessor.

        Returns [] when the state is unverifiable. Safety-critical
        paths MUST use get_unresolved_tickets_strict() so that
        'unknown' is distinguishable from 'no unresolved tickets'.
        """
        tickets = self.get_unresolved_tickets_strict()
        return tickets if tickets is not None else []

    def _reconcile_live_position(self, position: Dict) -> bool:
        """Ensure ExecutedOrder + OpenPosition + tracking exist for one
        live MT5 position, reconstructing from deal history if needed.

        Returns True when the position is fully tracked, False when it
        remains unresolved (no IN deal, or DB insert failed). H-05
        requires that unresolved positions are not reported as success.
        """
        ticket = int(position["ticket"])

        executed = db_manager.query_one(
            ExecutedOrder, ticket=ticket, environment=self.environment,
        )
        open_row = self._get_database_position(ticket)

        if executed is not None and open_row is not None:
            if ticket in self._managed_positions:
                self._refresh_remaining_volume(ticket)
            else:
                self.initialize_position_tracking(
                    ticket=ticket,
                    direction=position["type"],
                    entry_price=float(position["price_open"]),
                    sl=float(position.get("sl") or 0),
                    tp=float(position.get("tp") or 0),
                    volume=float(position["volume"]),
                    strategy=open_row.strategy or "UNKNOWN",
                )
            return True

        logger.warning(
            "F-05 RECOVERY | ticket=%s has MT5 position but %s missing",
            ticket,
            "both DB rows" if executed is None and open_row is None
            else ("ExecutedOrder" if executed is None else "OpenPosition"),
        )

        entry_deal = self._find_first_entry_deal(ticket)
        if entry_deal is None:
            logger.critical(
                "Cannot reconstruct ticket %s: no IN deal found. "
                "Position will remain untracked until resolved manually.",
                ticket,
            )
            return False

        entry_price = float(entry_deal["price"])
        entry_time = entry_deal["time"]
        entry_volume = float(entry_deal["volume"])
        direction = self._direction(entry_deal["type"])
        if direction not in ("BUY", "SELL"):
            direction = self._direction(position.get("type", ""))
        symbol = position.get("symbol") or entry_deal.get("symbol")
        magic = int(position.get("magic") or entry_deal.get("magic") or 0)

        strategy = self._lookup_strategy_for_ticket(ticket) or "RECOVERED"

        if executed is None:
            try:
                db_manager.add(ExecutedOrder(
                    order_request_id=None,
                    ticket=ticket,
                    magic_number=magic,
                    symbol=symbol,
                    order_type=direction,
                    volume=entry_volume,
                    price_open=entry_price,
                    price_current=float(position.get("price_current") or entry_price),
                    sl=float(position.get("sl") or 0),
                    tp=float(position.get("tp") or 0),
                    swap=float(position.get("swap") or 0),
                    profit=float(position.get("profit") or 0),
                    commission=0.0,
                    comment=str(entry_deal.get("comment") or ""),
                    time_open=entry_time,
                    environment=self.environment,
                ))
            except Exception:
                logger.exception(
                    "Failed to create recovered ExecutedOrder for %s", ticket,
                )

        if open_row is None:
            current_sl = float(position.get("sl") or 0)
            current_tp = float(position.get("tp") or 0)
            initial_risk = abs(entry_price - current_sl) if current_sl else 0.0
            try:
                db_manager.add(OpenPosition(
                    ticket=ticket,
                    magic_number=magic,
                    symbol=symbol,
                    direction=direction,
                    volume=entry_volume,
                    price_open=entry_price,
                    price_current=float(position.get("price_current") or entry_price),
                    sl=current_sl,
                    tp=current_tp,
                    swap=float(position.get("swap") or 0),
                    profit=float(position.get("profit") or 0),
                    commission=0.0,
                    strategy=strategy,
                    time_open=entry_time,
                    environment=self.environment,
                    initial_sl=current_sl,
                    initial_tp=current_tp,
                    initial_risk_distance=initial_risk,
                    remaining_volume=float(position["volume"]),
                    highest_r=0.0,
                    lowest_r=0.0,
                    break_even_activated=False,
                    trailing_active=False,
                    partial_tp_taken=[],
                    management_state={},
                ))
            except Exception:
                logger.exception(
                    "Failed to create recovered OpenPosition for %s", ticket,
                )
                return False

        self.initialize_position_tracking(
            ticket=ticket,
            direction=direction,
            entry_price=entry_price,
            sl=float(position.get("sl") or 0),
            tp=float(position.get("tp") or 0),
            volume=float(position["volume"]),
            strategy=strategy,
        )
        logger.info(
            "F-05 RECOVERY completed | ticket=%s entry=%.5f vol=%.3f "
            "strategy=%s",
            ticket, entry_price, entry_volume, strategy,
        )
        return True

    def _lookup_strategy_for_ticket(self, ticket: int) -> Optional[str]:
        req = db_manager.query_one(
            OrderRequest, mt5_ticket=ticket, environment=self.environment,
        )
        if req is not None and req.comment:
            return req.comment.replace("SMC_", "").strip() or None
        return None

    # ---------------------------------------------------------
    # Deal-history access helpers
    # ---------------------------------------------------------

    def _deals_for_ticket(self, ticket: int) -> Optional[List[Dict]]:
        """Query MT5 deal history for one position.

        H-04: returns None when the underlying MT5 history query fails,
        so callers can distinguish "query failed" from "no deals
        exist". The previous implementation returned [] for both and
        caused a query outage to be diagnosed as "no deals" (masking
        the broker failure). Returning None preserves diagnostic
        accuracy; the caller's decision (ticket is unresolved) is the
        same in both cases.
        """
        try:
            deals = mt5_connector.get_history_deals(
                from_date=datetime.now(timezone.utc) - timedelta(days=90),
                to_date=datetime.now(timezone.utc) + timedelta(minutes=1),
            )
        except Exception:
            logger.exception(
                "Could not fetch deal history while reconciling %s",
                ticket,
            )
            return None

        if deals is None:
            logger.error(
                "MT5 deal history returned None while reconciling %s: %s",
                ticket, mt5_connector.get_last_error(),
            )
            return None

        return [
            d for d in deals
            if int(d.get("position_id") or 0) == int(ticket)
        ]

    def _find_first_entry_deal(self, ticket: int) -> Optional[Dict]:
        deals = self._deals_for_ticket(ticket)
        if deals is None:
            logger.error(
                "Deal-history query failed while looking up first "
                "entry deal for ticket %s", ticket,
            )
            return None
        for deal in deals:
            if deal.get("entry") == "IN":
                return deal
        return None

    # =========================================================
    # F-06: closed-trade reconstruction
    # =========================================================

    def reconstruct_closed_trade(
        self, ticket: int, exit_reason: str = "RECOVERED",
    ) -> bool:
        """Create or complete a ClosedTrade row for one ticket.

        Idempotent: if a ClosedTrade already exists, this returns True
        and does nothing.
        """
        existing = db_manager.query_one(
            ClosedTrade, ticket=ticket, environment=self.environment,
        )
        if existing is not None:
            return True

        deals = self._deals_for_ticket(ticket)
        if deals is None:
            logger.error(
                "Cannot reconstruct ClosedTrade for ticket %s: MT5 "
                "deal-history query failed.", ticket,
            )
            return False
        if not deals:
            logger.warning(
                "Cannot reconstruct ClosedTrade for ticket %s: no deals "
                "returned by history query.", ticket,
            )
            return False

        entry_deals = [d for d in deals if d.get("entry") == "IN"]
        exit_deals = [
            d for d in deals
            if d.get("entry") in ("OUT", "OUT_BY", "INOUT")
        ]
        if not entry_deals or not exit_deals:
            logger.warning(
                "Cannot reconstruct ClosedTrade for ticket %s: deals "
                "incomplete (entries=%d, exits=%d).",
                ticket, len(entry_deals), len(exit_deals),
            )
            return False

        total_profit = sum(float(d.get("profit") or 0) for d in deals)
        total_swap = sum(float(d.get("swap") or 0) for d in deals)
        total_commission = sum(float(d.get("commission") or 0) for d in deals)
        net_profit = total_profit + total_swap + total_commission

        # Entry price: first IN deal.
        first_entry = entry_deals[0]
        entry_time = first_entry["time"]
        entry_price = float(first_entry["price"])
        entry_volume = float(first_entry["volume"])
        direction = self._direction(first_entry["type"])
        if direction not in ("BUY", "SELL"):
            direction = "BUY"
        symbol = first_entry.get("symbol") or ""
        magic = int(first_entry.get("magic") or 0)

        # Close price: volume-weighted average of OUT deals.
        total_exit_volume = sum(
            float(d["volume"]) for d in exit_deals if d["volume"]
        )
        if total_exit_volume > 0:
            close_price = sum(
                float(d["price"]) * float(d["volume"]) for d in exit_deals
            ) / total_exit_volume
        else:
            close_price = float(exit_deals[-1]["price"])

        close_time = exit_deals[-1]["time"]
        duration = max(
            0.0,
            (close_time - entry_time).total_seconds() / 3600.0,
        )

        executed = db_manager.query_one(
            ExecutedOrder, ticket=ticket, environment=self.environment,
        )
        original_sl = float(executed.sl or 0) if executed else 0.0
        original_tp = float(executed.tp or 0) if executed else 0.0
        initial_risk = abs(entry_price - original_sl) if original_sl else 0.0

        if initial_risk > 0:
            if direction == "BUY":
                realized_r = (close_price - entry_price) / initial_risk
            else:
                realized_r = (entry_price - close_price) / initial_risk
        else:
            realized_r = 0.0
            logger.warning(
                "ClosedTrade ticket %s: initial_risk unavailable; "
                "realized_r set to 0.",
                ticket,
            )

        strategy = (
            (executed.comment.replace("SMC_", "").strip() or "UNKNOWN")
            if executed and executed.comment
            else (self._lookup_strategy_for_ticket(ticket) or "RECOVERED")
        )

        try:
            db_manager.add(ClosedTrade(
                ticket=ticket,
                magic_number=magic,
                symbol=symbol,
                direction=direction,
                volume=entry_volume,
                price_open=entry_price,
                price_close=close_price,
                sl=original_sl,
                tp=original_tp,
                profit=total_profit,
                swap=total_swap,
                commission=total_commission,
                net_profit=net_profit,
                risk_percent=0.0,
                realized_r=realized_r,
                strategy=strategy,
                entry_reason=None,
                exit_reason=exit_reason,
                time_open=entry_time,
                time_close=close_time,
                duration_hours=duration,
                environment=self.environment,
            ))
        except Exception:
            logger.exception(
                "Failed to insert reconstructed ClosedTrade for %s", ticket,
            )
            return False

        # Update ExecutedOrder with actual realised values.
        if executed is not None:
            try:
                db_manager.update(
                    ExecutedOrder,
                    executed.id,
                    profit=total_profit,
                    swap=total_swap,
                    commission=total_commission,
                    time_close=close_time,
                    price_current=close_price,
                    reason=exit_reason,
                )
            except Exception:
                logger.exception(
                    "Could not update ExecutedOrder %s after reconstruction.",
                    ticket,
                )
        # L-02: the OpenPosition -> ClosedTrade transition is the single
        # authoritative closure step. Deleting the OpenPosition row and
        # dropping in-memory tracking are performed exactly once here;
        # duplicate blocks were removed to prevent a future edit from
        # updating only one path.
        try:
            with db_manager.get_session() as session:
                session.query(OpenPosition).filter(
                    OpenPosition.ticket == ticket,
                    OpenPosition.environment == self.environment,
                ).delete()
        except Exception:
            logger.exception(
                "Could not delete OpenPosition row for %s", ticket,
            )

        self.remove_position(ticket)

        logger.info(
            "F-06 RECOVERY | ClosedTrade reconstructed | ticket=%s "
            "profit=%.2f swap=%.2f comm=%.2f net=%.2f R=%.2f reason=%s",
            ticket, total_profit, total_swap, total_commission,
            net_profit, realized_r, exit_reason,
        )
        return True


# Global trade manager instance
trade_manager = TradeManager()