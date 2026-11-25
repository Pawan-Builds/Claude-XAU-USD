"""Risk management and broker-aware position sizing."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

from src.mt5_connector import mt5_connector, MT5Connector
from src.config_loader import config
from src.database_manager import db_manager, RiskCalculation, AppSettings
from src.logger import get_logger

logger = get_logger(__name__)

@dataclass(frozen=True)
class KillSwitchResult:
    """Result of a kill-switch persistence attempt.

    ``ok``     — True only if write AND read-back both agree with
                 the requested state.
    ``active`` — the state confirmed by read-back. On failure, this
                 is either the previous state or the read-back value
                 if it disagreed with the request.
    ``error``  — human-readable failure reason when ``ok`` is False.
    """
    ok: bool
    active: bool
    error: Optional[str] = None



@dataclass
class PositionSizeResult:
    lot_size: float
    risk_amount: float
    risk_pips: float
    sl_distance_pips: float
    tick_value: float
    tick_size: float
    contract_size: float
    equity: float
    risk_percent: float
    adjusted: bool
    reason: str = ""


class RiskManager:
    """Calculates position sizes and enforces account risk limits."""

    @property
    def ENVIRONMENT(self) -> str:
        """Runtime-derived environment for every risk query."""
        return config.trading.mode.lower()

    def __init__(self):
        self.risk_config = config.risk
        self.trading_config = config.trading
        self._last_equity = 0.0

    # ------------------------------------------------------------------
    # Kill Switch
    # ------------------------------------------------------------------

    def set_kill_switch(
        self, active: bool, reason: str = ""
    ) -> KillSwitchResult:
        """Persistently enable or disable the kill switch, with
        explicit success/failure result AND read-back verification.

        Returns:
            KillSwitchResult(ok=True, active=<confirmed>)
                only when the write succeeded AND read-back matches
                the requested state.
            KillSwitchResult(ok=False, active=<current>, error=...)
                on write failure, read-back failure, or mismatch.

        Never swallows DB errors silently — the caller (the
        dashboard/API) must see ok=False and report the failure.
        """
        try:
            with db_manager.get_session() as session:
                setting = (
                    session.query(AppSettings)
                    .filter_by(key="kill_switch")
                    .first()
                )
                if not setting:
                    setting = AppSettings(
                        key="kill_switch", value="false"
                    )
                    session.add(setting)
                setting.value = "true" if active else "false"
        except Exception as exc:
            logger.exception("Failed to set kill switch.")
            # Report the state that is actually readable, if any.
            try:
                current = self.get_kill_switch_state_strict()
            except Exception:
                current = None
            # Fail-closed if unreadable.
            return KillSwitchResult(
                ok=False,
                active=(
                    current
                    if current is not None
                    else (True if active else False)
                ),
                error=f"DB write failed: {exc}",
            )

        # Read-back verification.
        verified = self.get_kill_switch_state_strict()
        if verified is None:
            logger.critical(
                "Kill-switch read-back unavailable after write."
            )
            return KillSwitchResult(
                ok=False,
                active=not active,  # conservative: do not claim new state
                error="Read-back unavailable after write.",
            )
        if verified != active:
            logger.critical(
                "Kill-switch read-back mismatch: requested=%s "
                "actual=%s", active, verified,
            )
            return KillSwitchResult(
                ok=False,
                active=verified,
                error=(
                    f"Read-back mismatch: requested {active}, "
                    f"got {verified}."
                ),
            )

        if reason:
            logger.critical(
                "KILL SWITCH state changed to %s. Reason: %s",
                active, reason,
            )
        else:
            logger.info("KILL SWITCH state changed to %s.", active)

        return KillSwitchResult(ok=True, active=verified)

    def is_kill_switch_active(self) -> bool:
        """Check if the persistent kill switch is active."""
        try:
            setting = db_manager.query_one(AppSettings, key="kill_switch")
            return setting is not None and setting.value == "true"
        except Exception:
            logger.exception("Failed to read kill switch state; failing closed.")
            return True  # Fail safe: if DB is broken, assume active

    def get_kill_switch_state_strict(self) -> Optional[bool]:
        """Return True / False if readable, None if unreadable.

        This is the strict form: it distinguishes "state is
        definitively inactive" from "state could not be verified".
        Used by the LIVE opening boundary, which must fail closed
        when the state is unknown.

        Malformed values (None, '', non-"false"-like garbage) are
        treated as *active* (fail closed) — never as inactive.
        """
        try:
            setting = db_manager.query_one(
                AppSettings, key="kill_switch"
            )
            if setting is None:
                # No row → new install → inactive by design.
                return False
            value = str(setting.value).strip().lower()
            if value == "false":
                return False
            # "true", "1", "", None, garbage → treat as active.
            return True
        except Exception:
            logger.exception(
                "Strict kill-switch read failed; returning None."
            )
            return None
        
    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _failed_result(
        self,
        reason: str,
        equity: float = 0.0,
        risk_percent: float = 0.0,
        tick_value: float = 0.0,
        tick_size: float = 0.0,
        contract_size: float = 0.0,
        risk_pips: float = 0.0
    ) -> PositionSizeResult:
        return PositionSizeResult(
            lot_size=0.0,
            risk_amount=0.0,
            risk_pips=risk_pips,
            sl_distance_pips=risk_pips,
            tick_value=tick_value,
            tick_size=tick_size,
            contract_size=contract_size,
            equity=equity,
            risk_percent=risk_percent,
            adjusted=False,
            reason=reason
        )

    def calculate_position_size(
        self,
        entry_price: float,
        sl_price: float,
        direction: str,
        risk_percent: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Calculate volume from actual broker tick economics.
        """

        try:
            entry_price = float(entry_price)
            sl_price = float(sl_price)
        except (TypeError, ValueError):
            return self._failed_result("Entry or stop-loss price is invalid.")

        direction = str(direction).upper()

        if direction not in ("BUY", "SELL"):
            return self._failed_result(
                f"Unsupported trade direction: {direction}"
            )

        if entry_price <= 0 or sl_price <= 0:
            return self._failed_result(
                "Entry price and stop-loss price must be positive."
            )

        if direction == "BUY":
            sl_distance = entry_price - sl_price
        else:
            sl_distance = sl_price - entry_price

        if sl_distance <= 0:
            return self._failed_result(
                "Stop-loss is on the wrong side of the entry price."
            )

        if not mt5_connector.is_connected():
            return self._failed_result("MT5 is not connected.")

        account = mt5_connector.get_account_info()
        if account is None:
            return self._failed_result(
                "Could not retrieve MT5 account information."
            )

        equity = float(account.equity)
        if equity <= 0:
            return self._failed_result(
                "Account equity is zero or invalid.",
                equity=equity
            )

        self._last_equity = equity

        configured_risk = (
            self.risk_config.default_risk_percent
            if risk_percent is None
            else risk_percent
        )

        try:
            requested_risk = float(configured_risk)
            maximum_risk = float(self.risk_config.max_risk_percent)
        except (TypeError, ValueError, AttributeError):
            return self._failed_result(
                "Risk percentage configuration is invalid.",
                equity=equity
            )

        if requested_risk <= 0 or maximum_risk <= 0:
            return self._failed_result(
                "Risk percentage must be greater than zero.",
                equity=equity
            )

        risk_pct = min(requested_risk, maximum_risk)
        if risk_pct != requested_risk:
            logger.warning(
                "Requested risk %.4f%% capped to configured maximum %.4f%%.",
                requested_risk,
                risk_pct
            )

        symbol_info = mt5_connector.get_symbol_info()
        if symbol_info is None:
            return self._failed_result(
                "Could not retrieve broker symbol information.",
                equity=equity,
                risk_percent=risk_pct
            )

        point = float(symbol_info.point or 0)
        tick_size = float(symbol_info.tick_size or 0)
        tick_value = float(symbol_info.tick_value or 0)
        contract_size = float(symbol_info.contract_size or 0)

        if point <= 0:
            return self._failed_result(
                "Broker reported an invalid symbol point size.",
                equity=equity,
                risk_percent=risk_pct
            )

        if tick_size <= 0 or tick_value <= 0:
            return self._failed_result(
                "Broker tick size or tick value is missing/invalid. "
                "Cannot calculate monetary risk safely.",
                equity=equity,
                risk_percent=risk_pct,
                tick_value=tick_value,
                tick_size=tick_size,
                contract_size=contract_size
            )

        # FIX: use the broker-aware pip derivation from mt5_connector
        # so gold and 2/4-digit FX are handled the same way the
        # executor and signal validator handle them.
        # C-03 follow-up: call the class static method, not the mocked
        # instance. This guarantees a real float even when the connector
        # instance is patched in tests.
        pip_size = MT5Connector.pip_size_for(symbol_info)

        if not isinstance(pip_size, (int, float)) or pip_size <= 0:
            return self._failed_result(
                "Could not determine pip size from broker metadata.",
                equity=equity,
                risk_percent=risk_pct,
                tick_value=tick_value,
                tick_size=tick_size,
                contract_size=contract_size,
            )

        sl_distance_pips = sl_distance / pip_size
        risk_budget = equity * risk_pct / 100.0

        ticks_to_stop = sl_distance / tick_size
        risk_per_lot = ticks_to_stop * tick_value

        if risk_per_lot <= 0:
            return self._failed_result(
                "Calculated monetary risk per lot is invalid.",
                equity=equity,
                risk_percent=risk_pct,
                tick_value=tick_value,
                tick_size=tick_size,
                contract_size=contract_size,
                risk_pips=sl_distance_pips
            )

        raw_lots = risk_budget / risk_per_lot

        adjusted_lots, adjusted = self._adjust_lot_size(
            raw_lots,
            symbol_info
        )

        if adjusted_lots <= 0:
            minimum = float(symbol_info.volume_min)
            minimum_risk = risk_per_lot * minimum
            minimum_risk_pct = minimum_risk / equity * 100.0

            return self._failed_result(
                (
                    f"Calculated volume is below broker minimum. "
                    f"Minimum lot {minimum:.4f} would risk "
                    f"{minimum_risk_pct:.4f}% of equity, exceeding "
                    f"the configured {risk_pct:.4f}% risk."
                ),
                equity=equity,
                risk_percent=risk_pct,
                tick_value=tick_value,
                tick_size=tick_size,
                contract_size=contract_size,
                risk_pips=sl_distance_pips
            )

        actual_risk = risk_per_lot * adjusted_lots
        actual_risk_pct = actual_risk / equity * 100.0

        if actual_risk > risk_budget + 1e-8:
            return self._failed_result(
                "Normalized volume exceeds the configured monetary risk budget.",
                equity=equity,
                risk_percent=risk_pct,
                tick_value=tick_value,
                tick_size=tick_size,
                contract_size=contract_size,
                risk_pips=sl_distance_pips
            )

        # ------------------------------------------------------------------
        # MARGIN CHECK (Milestone 5)
        # Defensive: free_margin and margin_initial may be missing or
        # non-numeric depending on broker; fail safely if unavailable.
        # ------------------------------------------------------------------
        try:
            free_margin = float(account.free_margin)
        except (TypeError, ValueError, AttributeError):
            return self._failed_result(
                "Could not read free margin from account information.",
                equity=equity,
                risk_percent=risk_pct,
                tick_value=tick_value,
                tick_size=tick_size,
                contract_size=contract_size,
                risk_pips=sl_distance_pips
            )

        required_margin = 0.0
        try:
            margin_initial = float(symbol_info.margin_initial or 0)
        except (TypeError, ValueError, AttributeError):
            margin_initial = 0.0

        if margin_initial > 0:
            required_margin = margin_initial * adjusted_lots

        if required_margin <= 0:
            try:
                leverage = float(account.leverage or 0)
            except (TypeError, ValueError, AttributeError):
                leverage = 0

            if leverage <= 0:
                return self._failed_result(
                    "Cannot determine required margin: broker provided no "
                    "margin_initial and no valid leverage.",
                    equity=equity,
                    risk_percent=risk_pct,
                    tick_value=tick_value,
                    tick_size=tick_size,
                    contract_size=contract_size,
                    risk_pips=sl_distance_pips
                )

            required_margin = (
                adjusted_lots * contract_size * entry_price
            ) / leverage

        if required_margin > free_margin:
            return self._failed_result(
                f"Insufficient free margin. Required: "
                f"${required_margin:.2f}, Free: ${free_margin:.2f}",
                equity=equity,
                risk_percent=risk_pct,
                tick_value=tick_value,
                tick_size=tick_size,
                contract_size=contract_size,
                risk_pips=sl_distance_pips
            )

        # Include existing positions when checking portfolio-level risk.
        positions = mt5_connector.get_positions()
        if positions is None:
            return self._failed_result(
                "Could not retrieve open positions to calculate total account risk.",
                equity=equity,
                risk_percent=risk_pct,
                tick_value=tick_value,
                tick_size=tick_size,
                contract_size=contract_size,
                risk_pips=sl_distance_pips
            )

        existing_risk = self._calculate_open_positions_risk(
            positions,
            symbol_info
        )

        # H-03: pending-order exposure is part of total portfolio risk.
        try:
            pending_orders = mt5_connector.get_orders()
        except Exception:
            logger.exception(
                "Could not retrieve pending orders for position sizing."
            )
            return self._failed_result(
                "Could not retrieve pending orders for exposure calculation.",
                equity=equity,
                risk_percent=risk_pct,
                tick_value=tick_value,
                tick_size=tick_size,
                contract_size=contract_size,
                risk_pips=sl_distance_pips
            )

        if pending_orders is None:
            return self._failed_result(
                "Pending-order query returned None for exposure calculation.",
                equity=equity,
                risk_percent=risk_pct,
                tick_value=tick_value,
                tick_size=tick_size,
                contract_size=contract_size,
                risk_pips=sl_distance_pips
            )

        pending_risk = self._calculate_pending_orders_risk(
            pending_orders, symbol_info
        )
        if pending_risk is None:
            return self._failed_result(
                "A pending order has invalid or missing stop-loss risk data.",
                equity=equity,
                risk_percent=risk_pct,
                tick_value=tick_value,
                tick_size=tick_size,
                contract_size=contract_size,
                risk_pips=sl_distance_pips
            )

        if existing_risk is None:
            return self._failed_result(
                "An open position has invalid or missing stop-loss risk data.",
                equity=equity,
                risk_percent=risk_pct,
                tick_value=tick_value,
                tick_size=tick_size,
                contract_size=contract_size,
                risk_pips=sl_distance_pips
            )

        try:
            max_total_risk_pct = float(
                self.risk_config.max_total_open_risk_percent
            )
        except (TypeError, ValueError, AttributeError):
            return self._failed_result(
                "Maximum total open-risk configuration is invalid.",
                equity=equity,
                risk_percent=risk_pct
            )

        maximum_total_risk = equity * max_total_risk_pct / 100.0

        projected_total_risk = existing_risk + pending_risk + actual_risk
        if projected_total_risk > maximum_total_risk:
            projected_pct = projected_total_risk / equity * 100.0
            return self._failed_result(
                (
                    f"Trade would exceed maximum total open risk. "
                    f"Projected risk: {projected_pct:.4f}%; "
                    f"limit: {max_total_risk_pct:.4f}% "
                    f"(open={existing_risk:.2f}, pending={pending_risk:.2f}, "
                    f"new={actual_risk:.2f})."
                ),
                equity=equity,
                risk_percent=risk_pct,
                tick_value=tick_value,
                tick_size=tick_size,
                contract_size=contract_size,
                risk_pips=sl_distance_pips
            )

        self._log_risk_calculation(
            equity=equity,
            risk_percent=actual_risk_pct,
            sl_distance_pips=sl_distance_pips,
            tick_value=tick_value,
            tick_size=tick_size,
            contract_size=contract_size,
            calculated_lot=raw_lots,
            adjusted_lot=adjusted_lots,
            monetary_risk=actual_risk
        )

        logger.info(
            "Position size calculated: lots=%s risk=%.2f (%.4f%%) "
            "SL distance=%.2f pips existing risk=%.2f",
            adjusted_lots,
            actual_risk,
            actual_risk_pct,
            sl_distance_pips,
            existing_risk
        )

        return PositionSizeResult(
            lot_size=adjusted_lots,
            risk_amount=actual_risk,
            risk_pips=sl_distance_pips,
            sl_distance_pips=sl_distance_pips,
            tick_value=tick_value,
            tick_size=tick_size,
            contract_size=contract_size,
            equity=equity,
            risk_percent=actual_risk_pct,
            adjusted=adjusted,
            reason=(
                "Adjusted down to broker volume step"
                if adjusted
                else "OK"
            )
        )

    # ------------------------------------------------------------------
    # Broker volume normalization
    # ------------------------------------------------------------------

    def _adjust_lot_size(self, raw_lots: float, symbol_info) -> tuple:
        """Round volume down to the broker's permitted volume step."""

        try:
            raw = Decimal(str(raw_lots))
            minimum = Decimal(str(symbol_info.volume_min))
            maximum = Decimal(str(symbol_info.volume_max))
            step = Decimal(str(symbol_info.volume_step))
        except Exception:
            logger.exception("Invalid broker volume metadata.")
            return 0.0, False

        if raw <= 0 or minimum <= 0 or maximum < minimum or step <= 0:
            return 0.0, False

        min_precision = -minimum.as_tuple().exponent
        step_floor = Decimal("1").scaleb(-min_precision)
        if step < step_floor:
            logger.info(
                "Clamping broker step %s to volume_min precision %s",
                step, step_floor
            )
            step = step_floor

        if raw < minimum:
            return 0.0, True

        capped = min(raw, maximum)

        number_of_steps = (
            (capped - minimum) / step
        ).to_integral_value(rounding=ROUND_DOWN)

        normalized = minimum + number_of_steps * step
        normalized = min(normalized, maximum)
        normalized = normalized.quantize(step_floor, rounding=ROUND_DOWN)

        if normalized < minimum:
            return 0.0, True

        adjusted = normalized != raw

        return float(normalized), adjusted

    # ------------------------------------------------------------------
    # Existing open-position risk
    # ------------------------------------------------------------------

    def _calculate_open_positions_risk(
        self,
        positions: List[Dict[str, Any]],
        symbol_info
    ) -> Optional[float]:
        """Return total monetary risk to current stop losses."""

        total_risk = 0.0
        tick_size = float(symbol_info.tick_size or 0)
        tick_value = float(symbol_info.tick_value or 0)

        if tick_size <= 0 or tick_value <= 0:
            return None

        for position in positions:
            try:
                direction = str(
                    position.get("type", position.get("direction", ""))
                ).upper()

                entry = float(position["price_open"])
                stop = float(position.get("sl") or 0)
                volume = float(position["volume"])

                if volume <= 0 or entry <= 0:
                    return None

                if stop <= 0:
                    logger.error(
                        "Open position %s has no stop loss.",
                        position.get("ticket")
                    )
                    return None

                if direction == "BUY":
                    distance = entry - stop
                elif direction == "SELL":
                    distance = stop - entry
                else:
                    return None

                if distance <= 0:
                    logger.error(
                        "Open position %s has an invalid stop-loss side.",
                        position.get("ticket")
                    )
                    return None

                position_risk = (
                    (distance / tick_size) * tick_value * volume
                )
                total_risk += position_risk

            except (KeyError, TypeError, ValueError):
                logger.exception(
                    "Could not calculate risk for an open position."
                )
                return None

        return total_risk


    def _calculate_pending_orders_risk(
        self,
        orders: List[Dict[str, Any]],
        symbol_info,
    ) -> Optional[float]:
        """H-03: total monetary risk if all pending orders were filled.

        Pending orders without a stop-loss cannot have their risk
        quantified; return None so the caller fails closed. Duplicate
        tickets are deduplicated because MT5 sometimes returns the same
        ticket twice in a single batch.
        """
        tick_size = float(symbol_info.tick_size or 0)
        tick_value = float(symbol_info.tick_value or 0)

        if tick_size <= 0 or tick_value <= 0:
            return None

        total_risk = 0.0
        seen: set = set()

        for order in orders:
            try:
                ticket = int(order.get("ticket") or 0)
            except (TypeError, ValueError):
                continue
            if ticket <= 0 or ticket in seen:
                continue
            seen.add(ticket)

            try:
                price_open = float(order.get("price_open") or 0)
                stop = float(order.get("sl") or 0)
                volume = float(order.get("volume") or 0)
            except (TypeError, ValueError):
                logger.error(
                    "Pending order %s has non-numeric fields.", ticket,
                )
                return None

            if volume <= 0 or price_open <= 0:
                return None

            if stop <= 0:
                logger.error(
                    "Pending order %s has no stop-loss; risk cannot be "
                    "quantified.", ticket,
                )
                return None

            type_str = str(order.get("type", "")).upper()
            is_buy = type_str.startswith("BUY")
            is_sell = type_str.startswith("SELL")
            if not is_buy and not is_sell:
                logger.error(
                    "Pending order %s has unknown type: %s",
                    ticket, type_str,
                )
                return None

            if is_buy:
                distance = price_open - stop
            else:
                distance = stop - price_open

            if distance <= 0:
                logger.error(
                    "Pending order %s has invalid SL side "
                    "(distance=%s).", ticket, distance,
                )
                return None

            total_risk += (distance / tick_size) * tick_value * volume

        return total_risk

    def _calculate_account_open_positions_risk(
        self,
        positions: List[Dict[str, Any]],
    ) -> Optional[float]:
        """H-01: total monetary risk across ALL account positions.

        Each position's risk is computed with *that symbol's*
        tick_size / tick_value. Fails closed (returns None) if any
        symbol spec cannot be retrieved, any position lacks a valid
        stop-loss, or any position record is malformed. Duplicate
        tickets are deduplicated.
        """
        total_risk = 0.0
        seen_tickets: set = set()
        specs_cache: Dict[str, Dict[str, Any]] = {}

        for position in positions:
            try:
                ticket = int(position.get("ticket") or 0)
            except (TypeError, ValueError):
                logger.error("Account position has invalid ticket.")
                return None
            if ticket <= 0:
                logger.error("Account position has missing ticket.")
                return None
            if ticket in seen_tickets:
                continue
            seen_tickets.add(ticket)

            symbol = position.get("symbol")
            if not symbol:
                logger.error(
                    "Account position ticket=%s has no symbol.",
                    ticket,
                )
                return None

            specs = specs_cache.get(symbol)
            if specs is None:
                specs = mt5_connector.get_symbol_specs(symbol)
                if specs is None:
                    logger.error(
                        "Could not retrieve symbol specs for %s "
                        "while calculating account risk.", symbol,
                    )
                    return None
                specs_cache[symbol] = specs

            tick_size = float(specs["tick_size"])
            tick_value = float(specs["tick_value"])

            try:
                direction = str(
                    position.get(
                        "type", position.get("direction", "")
                    )
                ).upper()
                entry = float(position["price_open"])
                stop = float(position.get("sl") or 0)
                volume = float(position["volume"])
            except (KeyError, TypeError, ValueError):
                logger.exception(
                    "Could not read risk fields for position "
                    "ticket=%s.", ticket,
                )
                return None

            if volume <= 0 or entry <= 0:
                logger.error(
                    "Position ticket=%s has invalid volume/entry.",
                    ticket,
                )
                return None
            if stop <= 0:
                logger.error(
                    "Position ticket=%s has no stop loss; account "
                    "risk cannot be quantified.", ticket,
                )
                return None

            if direction == "BUY":
                distance = entry - stop
            elif direction == "SELL":
                distance = stop - entry
            else:
                logger.error(
                    "Position ticket=%s has unknown direction %r.",
                    ticket, direction,
                )
                return None

            if distance <= 0:
                logger.error(
                    "Position ticket=%s has invalid stop-loss side.",
                    ticket,
                )
                return None

            total_risk += (
                (distance / tick_size) * tick_value * volume
            )

        return total_risk

    def _calculate_account_pending_orders_risk(
        self,
        orders: List[Dict[str, Any]],
    ) -> Optional[float]:
        """H-01: total monetary risk across ALL account pending orders.

        Same per-symbol spec lookup and fail-closed behaviour as
        _calculate_account_open_positions_risk.
        """
        total_risk = 0.0
        seen: set = set()
        specs_cache: Dict[str, Dict[str, Any]] = {}

        for order in orders:
            try:
                ticket = int(order.get("ticket") or 0)
            except (TypeError, ValueError):
                logger.error("Account pending order has invalid ticket.")
                return None
            if ticket <= 0:
                logger.error("Account pending order has missing ticket.")
                return None
            if ticket in seen:
                continue
            seen.add(ticket)

            symbol = order.get("symbol")
            if not symbol:
                logger.error(
                    "Account pending order ticket=%s has no symbol.",
                    ticket,
                )
                return None

            specs = specs_cache.get(symbol)
            if specs is None:
                specs = mt5_connector.get_symbol_specs(symbol)
                if specs is None:
                    logger.error(
                        "Could not retrieve symbol specs for %s "
                        "while calculating account pending-order "
                        "risk.", symbol,
                    )
                    return None
                specs_cache[symbol] = specs

            tick_size = float(specs["tick_size"])
            tick_value = float(specs["tick_value"])

            try:
                price_open = float(order.get("price_open") or 0)
                stop = float(order.get("sl") or 0)
                volume = float(order.get("volume") or 0)
            except (TypeError, ValueError):
                logger.error(
                    "Account pending order ticket=%s has "
                    "non-numeric fields.", ticket,
                )
                return None

            if volume <= 0 or price_open <= 0:
                logger.error(
                    "Account pending order ticket=%s has invalid "
                    "price/volume.", ticket,
                )
                return None
            if stop <= 0:
                logger.error(
                    "Account pending order ticket=%s has no "
                    "stop-loss; risk cannot be quantified.", ticket,
                )
                return None

            type_str = str(order.get("type", "")).upper()
            is_buy = type_str.startswith("BUY")
            is_sell = type_str.startswith("SELL")
            if not is_buy and not is_sell:
                logger.error(
                    "Account pending order ticket=%s has unknown "
                    "type: %s", ticket, type_str,
                )
                return None

            if is_buy:
                distance = price_open - stop
            else:
                distance = stop - price_open

            if distance <= 0:
                logger.error(
                    "Account pending order ticket=%s has invalid "
                    "SL side.", ticket,
                )
                return None

            total_risk += (
                (distance / tick_size) * tick_value * volume
            )

        return total_risk

    # ------------------------------------------------------------------
    # Risk calculation persistence
    # ------------------------------------------------------------------

    def _log_risk_calculation(
        self,
        equity: float,
        risk_percent: float,
        sl_distance_pips: float,
        tick_value: float,
        tick_size: float,
        contract_size: float,
        calculated_lot: float,
        adjusted_lot: float,
        monetary_risk: float
    ) -> None:
        """Persist a completed risk calculation."""

        try:
            calculation = RiskCalculation(
                equity=equity,
                risk_percent=risk_percent,
                sl_distance_pips=sl_distance_pips,
                tick_value=tick_value,
                tick_size=tick_size,
                contract_size=contract_size,
                calculated_lot=calculated_lot,
                adjusted_lot=adjusted_lot,
                monetary_risk=monetary_risk,
                environment=self.ENVIRONMENT
            )
            db_manager.add(calculation)
        except Exception:
            logger.exception("Failed to save risk calculation.")

    # ------------------------------------------------------------------
    # Account risk limits
    # ------------------------------------------------------------------

    def check_risk_limits(self) -> Dict[str, Any]:
        """Check all configured account-level risk limits.

        H-01: position count and total-risk checks use ACCOUNT-WIDE
        positions (manual, other-EA, other-symbol) and ACCOUNT-WIDE
        pending orders. The bot-scoped get_positions() /
        get_orders() are intentionally not used here.
        """
        if not mt5_connector.is_connected():
            return {
                "ok": False,
                "reason": "MT5 is not connected.",
                "checks": {},
            }

        account = mt5_connector.get_account_info()
        if account is None or float(account.equity) <= 0:
            return {
                "ok": False,
                "reason": "Account information or equity is invalid.",
                "checks": {},
            }

        try:
            all_positions = mt5_connector.get_all_positions()
        except Exception:
            logger.exception(
                "Account-wide position query raised in "
                "check_risk_limits."
            )
            all_positions = None

        if all_positions is None:
            return {
                "ok": False,
                "reason": (
                    "Could not retrieve account-wide open positions."
                ),
                "checks": {},
            }

        checks = {}

        checks["kill_switch"] = not self.is_kill_switch_active()
        checks["daily_loss"] = self._check_daily_loss(
            float(account.equity)
        )
        checks["max_drawdown"] = self._check_max_drawdown(
            float(account.equity)
        )
        checks["consecutive_losses"] = self._check_consecutive_losses()
        checks["max_positions"] = (
            len(all_positions)
            < int(self.trading_config.max_positions)
        )
        checks["total_risk"] = self._check_total_risk(
            float(account.equity), all_positions
        )
        checks["emergency"] = self._check_emergency_drawdown(
            float(account.equity)
        )

        all_ok = all(checks.values())

        return {
            "ok": all_ok,
            "checks": checks,
            "equity": float(account.equity),
            "balance": float(account.balance),
        }  
        

    # ------------------------------------------------------------------
    # H-02: durable broker-day daily-loss baseline
    # ------------------------------------------------------------------

    def _get_broker_day_start_utc(self) -> Optional[datetime]:
        """UTC datetime for the start of the current broker day.

        Uses ``config.risk.broker_day_rollover_hour_utc`` if defined
        (0 <= h <= 23, default 0 = UTC midnight). Returns None on any
        configuration error so the caller fails closed.

        Note: XM and most MT5 brokers roll their daily at a server-side
        hour that is often 21:00–23:00 UTC. Operators should set
        ``broker_day_rollover_hour_utc`` accordingly. Defaulting to 0
        is the least-surprising choice when the broker's convention is
        unknown; it is never less safe than the previous behaviour.
        """
        try:
            rollover_hour = int(
                getattr(
                    self.risk_config,
                    "broker_day_rollover_hour_utc",
                    0,
                )
            )
        except (TypeError, ValueError):
            logger.error(
                "broker_day_rollover_hour_utc is not an integer; "
                "failing closed."
            )
            return None

        if not (0 <= rollover_hour <= 23):
            logger.error(
                "broker_day_rollover_hour_utc=%s is outside [0, 23]; "
                "failing closed.",
                rollover_hour,
            )
            return None

        now = datetime.now(timezone.utc)
        day_start = now.replace(
            hour=rollover_hour, minute=0, second=0, microsecond=0
        )
        if now < day_start:
            day_start -= timedelta(days=1)
        return day_start

    def _reconstruct_opening_balance_from_history(
        self,
        day_start: datetime,
    ) -> Optional[float]:
        """H-02: reconstruct the balance at ``day_start`` from broker
        deal history.

        opening_balance = current_balance
                        − Σ(profit + swap + commission) over deals
                          in [day_start, now]

        Returns None on any failure (fail closed). Deals from any
        symbol, magic, or origin (manual, other EAs, bot) are included
        because the daily-loss limit is account-wide.
        """
        account = mt5_connector.get_account_info()
        if account is None:
            logger.error(
                "Cannot reconstruct daily baseline: account info "
                "unavailable."
            )
            return None

        try:
            current_balance = float(account.balance)
        except (TypeError, ValueError, AttributeError):
            logger.error(
                "Cannot reconstruct daily baseline: account balance "
                "invalid."
            )
            return None

        if current_balance <= 0:
            logger.error(
                "Cannot reconstruct daily baseline: current balance "
                "is not positive."
            )
            return None

        try:
            from src.mt5_connector import mt5 as _mt5
            deals = _mt5.history_deals_get(
                day_start, datetime.now(timezone.utc)
            )
        except Exception:
            logger.exception(
                "Broker deal history query raised during baseline "
                "reconstruction."
            )
            return None

        if deals is None:
            logger.error(
                "Broker deal history unavailable during baseline "
                "reconstruction."
            )
            return None

        day_net_flow = 0.0
        for deal in deals:
            try:
                day_net_flow += (
                    float(deal.profit or 0)
                    + float(deal.swap or 0)
                    + float(deal.commission or 0)
                )
            except (TypeError, ValueError, AttributeError):
                logger.exception(
                    "History deal has non-numeric fields; cannot "
                    "reconstruct daily baseline. ticket=%s",
                    getattr(deal, "ticket", "?"),
                )
                return None

        opening_balance = current_balance - day_net_flow

        if opening_balance <= 0:
            logger.error(
                "Reconstructed opening balance is invalid: %.2f",
                opening_balance,
            )
            return None

        return opening_balance

    def _get_or_establish_daily_baseline(
        self,
        day_start: datetime,
    ) -> Optional[float]:
        """H-02: return the day's opening balance, durably establishing
        it once per broker day.

        Order of resolution:
          1. If ``AppSettings`` carries a baseline whose stored date
             matches ``day_start.date()``, return it. Restart does NOT
             recompute — that is the entire point of persistence.
          2. Otherwise reconstruct from broker deal history (covers
             manual trades, other EAs, bot-offline losses, bot-start-
             after-losses).
          3. Persist the reconstructed value for today's date.

        Returns None on any failure (fail closed).
        """
        env = self.ENVIRONMENT
        date_key = day_start.date().isoformat()
        date_setting_key = f"daily_baseline_date_{env}"
        balance_setting_key = f"daily_baseline_balance_{env}"

        # Step 1: try persisted baseline for today.
        try:
            with db_manager.get_session() as session:
                date_row = (
                    session.query(AppSettings)
                    .filter_by(key=date_setting_key)
                    .first()
                )
                balance_row = (
                    session.query(AppSettings)
                    .filter_by(key=balance_setting_key)
                    .first()
                )
        except Exception:
            logger.exception(
                "Could not read persisted daily baseline; will "
                "attempt reconstruction from broker history."
            )
            date_row = None
            balance_row = None

        if (
            date_row is not None
            and balance_row is not None
            and str(date_row.value) == date_key
        ):
            try:
                persisted = float(balance_row.value)
            except (TypeError, ValueError):
                persisted = None
            if persisted is not None and persisted > 0:
                return persisted
            logger.warning(
                "Persisted daily baseline is malformed; reconstructing "
                "from broker history."
            )

        # Step 2: reconstruct from broker history.
        opening_balance = self._reconstruct_opening_balance_from_history(
            day_start
        )
        if opening_balance is None:
            return None

        # Step 3: persist for durable reuse across restarts.
        try:
            with db_manager.get_session() as session:
                for key, val in (
                    (date_setting_key, date_key),
                    (balance_setting_key, str(opening_balance)),
                ):
                    row = (
                        session.query(AppSettings)
                        .filter_by(key=key)
                        .first()
                    )
                    if row is None:
                        session.add(AppSettings(key=key, value=val))
                    else:
                        row.value = val
            logger.info(
                "Daily baseline established and persisted | "
                "date=%s balance=%.2f",
                date_key,
                opening_balance,
            )
        except Exception:
            logger.exception(
                "Could not persist daily baseline. It will be "
                "reconstructed from broker history on the next call; "
                "reconstruction is deterministic, so this is not a "
                "safety regression."
            )

        return opening_balance

    def _check_daily_loss(self, equity: float) -> bool:
        """H-02: account-wide daily-loss check using a durable,
        broker-day-correct opening-balance baseline.

        Definition:
            daily_opening_balance  = reconstructed from broker history
            daily_loss_pct         = max(0, opening − equity) / opening * 100
            pass                   ⇔ daily_loss_pct < limit

        Fails closed on: invalid equity, unreadable day boundary,
        unreadable baseline, non-positive baseline, invalid limit.
        """
        if equity <= 0:
            return False

        day_start = self._get_broker_day_start_utc()
        if day_start is None:
            return False

        opening_balance = self._get_or_establish_daily_baseline(day_start)
        if opening_balance is None or opening_balance <= 0:
            return False

        equity_loss = max(0.0, opening_balance - equity)
        total_loss_pct = equity_loss / opening_balance * 100.0

        try:
            limit = float(self.risk_config.daily_loss_limit_percent)
        except (TypeError, ValueError, AttributeError):
            logger.error(
                "daily_loss_limit_percent config is invalid; failing "
                "closed."
            )
            return False

        if limit <= 0:
            logger.error(
                "daily_loss_limit_percent is not positive; failing "
                "closed."
            )
            return False

        return total_loss_pct < limit

    def _get_peak_equity(self) -> Optional[float]:
        """Durable high-water-mark equity.

        H-04: reads a persisted peak from AppSettings, seeds it from
        account snapshots on first use, and updates it to a new peak
        whenever the current equity exceeds the stored value. The peak
        is a one-way ratchet — equity declines never lower it.
        """
        key = f"peak_equity_{self.ENVIRONMENT}"

        persisted: Optional[float] = None
        try:
            setting = db_manager.query_one(AppSettings, key=key)
            if setting is not None and setting.value:
                try:
                    candidate = float(setting.value)
                except (TypeError, ValueError):
                    candidate = None
                if candidate is not None and candidate > 0:
                    persisted = candidate
        except Exception:
            logger.exception("Could not read persisted peak equity.")

        account = mt5_connector.get_account_info()
        current = float(account.equity) if account is not None else 0.0

        if persisted is None or persisted <= 0:
            from src.database_manager import AccountSnapshot
            try:
                with db_manager.get_session() as session:
                    snapshots = session.query(AccountSnapshot).filter(
                        AccountSnapshot.environment == self.ENVIRONMENT
                    ).order_by(
                        AccountSnapshot.timestamp.desc()
                    ).limit(1000).all()
            except Exception:
                logger.exception(
                    "Could not seed peak equity from snapshots."
                )
                snapshots = []

            if snapshots:
                seeded = max(float(s.equity) for s in snapshots)
                if seeded > 0:
                    persisted = seeded
            elif current > 0:
                persisted = current

        peak = max(persisted or 0.0, current)
        if peak <= 0:
            return None

        if persisted is None or peak > persisted:
            try:
                with db_manager.get_session() as session:
                    setting = (
                        session.query(AppSettings)
                        .filter_by(key=key).first()
                    )
                    if setting is None:
                        session.add(AppSettings(
                            key=key, value=str(peak),
                        ))
                    else:
                        setting.value = str(peak)
            except Exception:
                logger.exception(
                    "Could not persist peak equity update."
                )

        return peak

    def _check_max_drawdown(self, equity: float) -> bool:
        if equity <= 0:
            return False

        peak_equity = self._get_peak_equity()
        if peak_equity is None:
            return False

        drawdown_pct = max(
            0.0,
            (peak_equity - equity) / peak_equity * 100.0
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

            consecutive_losses = 0

            for trade in trades:
                if float(trade.net_profit or 0) < 0:
                    consecutive_losses += 1
                else:
                    break

            return consecutive_losses < int(
                self.risk_config.max_consecutive_losses
            )

        except Exception:
            logger.exception("Consecutive-loss check failed.")
            return False

    def _check_total_risk(
        self,
        equity: float,
        positions: List[Dict[str, Any]],
    ) -> bool:
        """H-01: total account risk across all symbols.

        `positions` is expected to be account-wide (from
        mt5_connector.get_all_positions()). Pending orders are
        fetched account-wide here. Any query or spec failure
        returns False (fail closed).
        """
        if equity <= 0:
            return False

        open_risk = self._calculate_account_open_positions_risk(
            positions
        )
        if open_risk is None:
            return False

        try:
            pending_orders = mt5_connector.get_all_orders()
        except Exception:
            logger.exception(
                "Account-wide order query raised in _check_total_risk."
            )
            return False

        if pending_orders is None:
            logger.error(
                "Account-wide order query returned None; "
                "failing closed."
            )
            return False

        pending_risk = self._calculate_account_pending_orders_risk(
            pending_orders
        )
        if pending_risk is None:
            return False

        total_risk = open_risk + pending_risk

        maximum_risk = (
            float(self.risk_config.max_total_open_risk_percent)
            / 100.0
            * equity
        )

        return total_risk <= maximum_risk

    def _check_emergency_drawdown(self, equity: float) -> bool:
        if equity <= 0:
            return False

        peak_equity = self._get_peak_equity()
        if peak_equity is None:
            return False

        drawdown_pct = max(
            0.0,
            (peak_equity - equity) / peak_equity * 100.0
        )

        return drawdown_pct < float(
            self.risk_config.emergency_drawdown_percent
        )

    # ------------------------------------------------------------------
    # Compounding
    # ------------------------------------------------------------------

    def should_compound(self) -> bool:
        return bool(self.risk_config.compound_enabled)

    def get_compound_factor(self) -> float:
        """
        Position sizing already uses current equity.
        Therefore no additional multiplier is applied here.
        """
        return 1.0


# Global risk manager instance
risk_manager = RiskManager()