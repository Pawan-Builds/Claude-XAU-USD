"""Main trading agent - DEMO and LIVE orchestration with account safeguards."""

import asyncio
import os
import signal
import time
import psutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

import src.logger as logger_module

from src.config_loader import config
from src.logger import setup_logging, get_logger
from src.database_manager import db_manager, AccountSnapshot, TradePlan
from src.mt5_connector import mt5_connector, mt5
from src.market_data import market_data
from src.smc_strategy import smc_engine, SignalDirection, StrategyType, TradeSignal
from src.risk_manager import risk_manager
from src.trade_manager import trade_manager
from src.order_executor import order_executor
from src.session_filter import session_filter
from src.news_filter import news_filter
from src.telegram_bot import telegram_bot
from src.signal_validator import signal_validator
from src.entry_gate import EntryGate
from src.trading_controller import trading_controller

logger = get_logger(__name__)


class TradingAgent:
    """Trading orchestrator with explicit DEMO/LIVE safeguards."""

    LIVE_CONFIRMATION = "I_ACCEPT_LIVE_TRADING_RISK"

    def __init__(self):
        self._running = False
        self._paused = True
        self._shutdown_requested = False
        self._shutdown_started = False

        self._current_mode = "DEMO"

        self._analysis_interval = 5
        self._last_analysis_time = 0.0

        self._last_waiting_log = 0.0
        self._waiting_log_interval = 60.0

        self._error_count = 0
        self._max_errors = max(
            1, int(config.safety.max_errors_before_pause)
        )

        self._health_check_interval = max(
            5, int(config.safety.health_check_interval)
        )
        self._last_health_check = 0.0

        self._last_daily_report_date = None

        self._telegram_ready = False
        self._telegram_logging_started = False
        self._market_data_started = False
        self._mt5_started = False
        
        # MILESTONE 11: Single-Instance Lock
        self._lock_file = Path("bot.lock")
        self._lock_acquired = False
        self._entry_gate = EntryGate(
            is_paused_getter=lambda: self._paused,
            is_shutdown_getter=lambda: self._shutdown_requested,
        )

    # =========================================================
    # MODE AND ACCOUNT VERIFICATION
    # =========================================================

    def _verify_selected_mode(self) -> str:
        mode = str(config.trading.mode).strip().upper()

        if mode not in ("DEMO", "LIVE"):
            raise RuntimeError(
                f"Invalid trading mode '{mode}'. Expected DEMO or LIVE."
            )

        config.validate_mode_configuration(mode)

        if mode == "LIVE":
            if config.safety.demo_only:
                raise RuntimeError(
                    "LIVE trading is disabled by safety.demo_only."
                )

            enabled = os.getenv(
                "LIVE_TRADING_ENABLED", ""
            ).strip().lower()

            if enabled not in ("true", "1", "yes"):
                raise RuntimeError(
                    "LIVE trading refused. "
                    "LIVE_TRADING_ENABLED must be true."
                )

            allowed_account = os.getenv(
                "LIVE_ALLOWED_ACCOUNT", ""
            ).strip()

            if not allowed_account.isdigit():
                raise RuntimeError(
                    "LIVE_ALLOWED_ACCOUNT must contain the approved "
                    "numeric MT5 account login."
                )

            confirmation = os.getenv(
                "LIVE_TRADING_CONFIRMATION", ""
            ).strip()

            if confirmation != self.LIVE_CONFIRMATION:
                raise RuntimeError(
                    "LIVE_TRADING_CONFIRMATION is missing or incorrect."
                )

        return mode

    def _verify_connected_account(self, account):
        if account is None:
            raise RuntimeError(
                "Could not retrieve MT5 account information."
            )

        actual_mode = str(account.trade_mode).strip().upper()

        if self._current_mode == "DEMO":
            if actual_mode != "DEMO" or not account.is_demo:
                raise RuntimeError(
                    "Account mismatch: DEMO mode selected, "
                    f"but MT5 reports {actual_mode}."
                )

        elif self._current_mode == "LIVE":
            if actual_mode != "REAL" or account.is_demo:
                raise RuntimeError(
                    "Account mismatch: LIVE mode selected, "
                    f"but MT5 reports {actual_mode}."
                )

            allowed_login = os.getenv(
                "LIVE_ALLOWED_ACCOUNT", ""
            ).strip()

            if not allowed_login.isdigit():
                raise RuntimeError(
                    "LIVE_ALLOWED_ACCOUNT is invalid."
                )

            if int(account.login) != int(allowed_login):
                raise RuntimeError(
                    "Connected LIVE account does not match the approved "
                    f"account. Expected {allowed_login}, "
                    f"connected to {account.login}."
                )

        else:
            raise RuntimeError(
                f"Unsupported trading mode: {self._current_mode}"
            )

        logger.debug(
            "MT5 account verified: mode=%s login=%s server=%s",
            actual_mode,
            account.login,
            account.server
        )

    def _verify_current_connection(self):
        if not mt5_connector.is_connected():
            raise RuntimeError("MT5 is disconnected.")

        if not mt5_connector.update_account_info():
            raise RuntimeError(
                "Could not refresh MT5 account information."
            )

        account = mt5_connector.get_account_info()
        self._verify_connected_account(account)

        return account
    
    # =========================================================
    # ACCOUNT SNAPSHOTS
    # =========================================================

    def _get_positions_or_raise(self):
        if not mt5_connector.is_connected():
            raise RuntimeError(
                "Cannot query positions: MT5 is disconnected."
            )

        positions = mt5_connector.get_positions()

        if positions is None:
            raise RuntimeError(
                "MT5 position query failed."
            )

        return positions

    @staticmethod
    def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
        """Normalize a datetime to timezone-aware UTC.

        SQLAlchemy's SQLite DateTime type stores values as a fixed
        naive-looking string and returns them naive on read, regardless of
        whether the write was tz-aware. Every value we persist is UTC, so
        this helper restores the explicit UTC tzinfo and makes comparisons
        against ``datetime.now(timezone.utc)`` safe. Safe to call on
        already-aware datetimes.
        """
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
        

    def _save_account_snapshot(self):
        account = mt5_connector.get_account_info()

        if account is None:
            raise RuntimeError(
                "Cannot save account snapshot: account unavailable."
            )

        positions = self._get_positions_or_raise()


        floating_pl = sum(
            float(position.get("profit", 0.0))
            for position in positions
        )

        snapshot = AccountSnapshot(
            timestamp=datetime.now(timezone.utc),
            balance=float(account.balance),
            equity=float(account.equity),
            margin=float(account.margin),
            free_margin=float(account.free_margin),
            margin_level=float(account.margin_level),
            floating_pl=floating_pl,
            environment=self._current_mode.lower()
        )

        db_manager.add(snapshot)


    # =========================================================
    # STARTUP
    # =========================================================

    async def start(self):
        # --- MILESTONE 11: SINGLE-INSTANCE LOCK ---
        if self._lock_file.exists():
            try:
                pid = int(self._lock_file.read_text().strip())
                # Check if the process is actually still running
                if psutil.pid_exists(pid):
                    logger.critical(f"ABORTING: Another bot instance is already running (PID: {pid}).")
                    return False
            except Exception:
                pass  # If the file is corrupt or psutil fails, overwrite it
                
        self._lock_file.write_text(str(os.getpid()))
        self._lock_acquired = True
        # ------------------------------------------

        logger.info("=" * 65)
        logger.info("Starting XAU/USD SMC Trading Agent")
        logger.info("=" * 65)

        try:
            self._current_mode = self._verify_selected_mode()

            logger.info(
                "Selected trading mode: %s",
                self._current_mode
            )

            logger.info("Initializing database...")
            db_manager.create_tables()

            try:
                self._telegram_ready = bool(
                    await telegram_bot.initialize()
                )
            except Exception:
                self._telegram_ready = False
                logger.exception(
                    "Telegram initialization failed. "
                    "Continuing without Telegram."
                )

            if self._telegram_ready:
                handler = logger_module.telegram_handler

                if handler:
                    handler.start()
                    self._telegram_logging_started = True

            mt5_config = config.mt5

            logger.info("Connecting to MT5...")
            logger.info("MT5 path: %s", mt5_config.path)
            logger.info("MT5 server: %s", mt5_config.server)
            logger.info(
                "MT5 login ending: %s",
                str(mt5_config.login)[-4:]
            )

            if not mt5_connector.initialize():
                raise RuntimeError(
                    "MT5 initialization failed: "
                    f"{mt5_connector.get_last_error()}"
                )

            self._mt5_started = True

            account = self._verify_current_connection()

            logger.info(
                "MT5 connected | Mode: %s | Login: %s | Server: %s",
                self._current_mode,
                account.login,
                account.server
            )

            if self._telegram_ready:
                await telegram_bot.notify_mt5_connected()

            self._save_account_snapshot()

            logger.info("Initializing market data...")

            if not market_data.initialize_cache():
                raise RuntimeError(
                    "Market data cache initialization failed."
                )

            market_data.start()
            self._market_data_started = True

            logger.info("Market data engine started.")

            try:
                order_executor.reconcile_pending_orders()
            except Exception:
                logger.exception(
                    "Startup pending-order reconciliation failed."
                )

            self._recover_state()

            self._last_health_check = time.time()
            self._last_daily_report_date = datetime.now(
                timezone.utc
            ).date()

            startup_gate = self._entry_gate.probe()
            if startup_gate.ok:
                self._paused = False
                logger.info(
                    "Startup checks passed. New entries enabled."
                )
            else:
                self._paused = True
                logger.warning(
                    "Startup completed, but new entries remain paused. "
                    "Reason: %s. The health monitor will retry checks.",
                    startup_gate.reason,
                )
            trading_controller.register_agent(self, self._entry_gate)
            logger.info(
                "TradingController registered with live TradingAgent."
            )
            self._running = True

            logger.info(
                "Trading agent started successfully in %s mode.",
                self._current_mode
            )

            if self._telegram_ready:
                await telegram_bot.notify_startup()

            await self._main_loop()

            return True

        except Exception as exc:
            logger.exception(
                "Trading agent startup/runtime failure: %s",
                exc
            )

            self._pause_trading()

            if self._telegram_ready:
                try:
                    await telegram_bot.notify_error(str(exc))
                except Exception:
                    logger.exception(
                        "Could not send Telegram error notification."
                    )

            return False

        finally:
            await self.shutdown()

    # =========================================================
    # POSITION RECOVERY
    # =========================================================

    def _recover_state(self):
        logger.info(
            "Recovering positions from MT5 and database..."
        )

        positions = self._get_positions_or_raise()

        if not trade_manager.sync_with_mt5():
            raise RuntimeError(
                "Trade manager failed to synchronize MT5 positions."
            )

        logger.info(
            "Position recovery completed. MT5 positions: %d",
            len(positions)
        )

    # =========================================================
    # MAIN LOOP
    # =========================================================

    async def _main_loop(self):
        while self._running and not self._shutdown_requested:
            loop_start = time.time()

            try:
                if (
                    time.time() - self._last_health_check
                    >= self._health_check_interval
                ):
                    await self._health_check()
                    self._last_health_check = time.time()

                if not mt5_connector.is_connected():
                    self._pause_trading()

                    logger.warning(
                        "MT5 disconnected. Trading and position "
                        "management are suspended."
                    )

                    await asyncio.sleep(5)
                    continue

                market_data.update_live_data()

                try:
                    order_executor.reconcile_pending_orders()
                except Exception:
                    logger.exception(
                        "Pending-order reconciliation failed."
                    )

                today = datetime.now(timezone.utc).date()

                if today != self._last_daily_report_date:
                    if self._telegram_ready:
                        await telegram_bot.notify_daily_report()

                    self._last_daily_report_date = today

                if not self._paused and self._should_analyze():
                    await self._trading_cycle()

                if not self._paused:
                    await self._process_trade_plans()

                await self._manage_positions()

                # Phase 3A: honour config.sessions.close_before_weekend_hours.
                self._handle_weekend_close()

                self._error_count = max(
                    0, self._error_count - 1
                )

            except asyncio.CancelledError:
                raise

            except Exception:
                self._error_count += 1

                logger.exception(
                    "Main-loop error (%d/%d).",
                    self._error_count,
                    self._max_errors
                )

                if self._error_count >= self._max_errors:
                    self._pause_trading()

                    if self._telegram_ready:
                        try:
                            await telegram_bot.notify_error(
                                "Maximum error count reached. "
                                "New entries have been paused."
                            )
                        except Exception:
                            logger.exception(
                                "Telegram error notification failed."
                            )

            elapsed = time.time() - loop_start

            await asyncio.sleep(
                max(0.1, self._analysis_interval - elapsed)
            )

    def _should_analyze(self) -> bool:
        return (
            time.time() - self._last_analysis_time
            >= self._analysis_interval
        )

    # =========================================================
    # SIGNAL GENERATION AND TRADE PLAN CREATION
    # =========================================================

    async def _trading_cycle(self):
        """Phase 1 & 2: Market Analysis and Trade Plan Creation."""
        self._last_analysis_time = time.time()

        gate_result = self._entry_gate.check()
        if not gate_result.ok:
            logger.debug(
                "Skipping signal generation; entry gate blocked: %s",
                gate_result.reason,
            )
            return

        try:
            signals = smc_engine.analyze_all_strategies()
        except Exception:
            logger.exception("SMC strategy analysis failed.")
            raise

        if not signals:
            now = time.time()

            if (
                now - self._last_waiting_log
                >= self._waiting_log_interval
            ):
                logger.info(
                    "STRATEGY ENGINE | Waiting for SMC signal | "
                    "Symbol: %s | Mode: %s",
                    config.trading.symbol,
                    self._current_mode
                )

                self._last_waiting_log = now

            return

        logger.info(
            "SIGNAL GENERATION | %d signal(s) detected.",
            len(signals)
        )

        for trading_signal in signals:
            try:
                with db_manager.get_session() as session:
                    existing = session.query(TradePlan).filter(
                        TradePlan.environment == self._current_mode.lower(),
                        TradePlan.symbol == trading_signal.symbol,
                        TradePlan.direction == trading_signal.direction.value,
                        TradePlan.strategy == trading_signal.strategy.value,
                        TradePlan.entry_price == trading_signal.entry_price,
                        TradePlan.status == "WAITING"
                    ).first()

                    if existing:
                        continue

                    plan = TradePlan(
                        timestamp=datetime.now(timezone.utc),
                        symbol=trading_signal.symbol,
                        direction=trading_signal.direction.value,
                        strategy=trading_signal.strategy.value,
                        entry_price=trading_signal.entry_price,
                        sl_price=trading_signal.sl_price,
                        tp_price=trading_signal.tp_price,
                        risk_reward=trading_signal.risk_reward,
                        status="WAITING",
                        expiry_time=datetime.now(timezone.utc) + timedelta(hours=config.trading.pending_expiry_hours),
                        reason=str([c.details for c in trading_signal.confirmations]),
                        environment=self._current_mode.lower()
                    )
                    session.add(plan)
                    logger.info(
                        "TRADE PLAN CREATED | %s %s @ %.5f | Strategy: %s",
                        plan.direction,
                        plan.symbol,
                        plan.entry_price,
                        plan.strategy
                    )
            except Exception:
                logger.exception("Failed to create TradePlan")

    # =========================================================
    # TRADE PLAN PROCESSING
    # =========================================================

    async def _process_trade_plans(self):
        """Phase 3, 4, 5 & 6: Wait for price, Reconfirm setup, Risk Validation, Execution."""
        try:
            now = datetime.now(timezone.utc)
            current_tick = market_data.get_current_tick()
            if not current_tick:
                return
            
            bid = current_tick["bid"]
            ask = current_tick["ask"]

            with db_manager.get_session() as session:
                plans = session.query(TradePlan).filter(
                    TradePlan.environment == self._current_mode.lower(),
                    TradePlan.status == "WAITING"
                ).all()

                for plan in plans:
                    if now >= self._as_utc(plan.expiry_time):
                        plan.status = "EXPIRED"
                        plan.invalidated_reason = "Time expired"
                        logger.info(
                            "TRADE PLAN EXPIRED | %s %s", 
                            plan.direction, 
                            plan.entry_price
                        )
                        continue

                    entry_reached = False
                    if plan.direction == "BUY":
                        if ask <= plan.entry_price:
                            entry_reached = True
                    elif plan.direction == "SELL":
                        if bid >= plan.entry_price:
                            entry_reached = True

                    if entry_reached:
                        logger.info(
                            "TRADE PLAN READY | Price reached entry for %s %s @ %.5f",
                            plan.direction,
                            plan.symbol,
                            plan.entry_price
                        )

                        # H-01: reconfirm the SMC setup on fresh closed-
                        # candle data. Prevents a plan whose premise has
                        # expired from becoming a live order.
                        try:
                            reconfirmed, reconfirm_reason = (
                                smc_engine.reconfirm_plan(plan)
                            )
                        except Exception:
                            logger.exception(
                                "Plan reconfirmation raised; failing closed."
                            )
                            reconfirmed = False
                            reconfirm_reason = "reconfirmation error"

                        if not reconfirmed:
                            plan.status = "INVALIDATED"
                            plan.invalidated_reason = (
                                f"Reconfirmation: {reconfirm_reason}"
                            )
                            logger.warning(
                                "TRADE PLAN INVALIDATED AT RECONFIRM | "
                                "%s %s @ %.5f | %s",
                                plan.direction,
                                plan.symbol,
                                plan.entry_price,
                                reconfirm_reason,
                            )
                            continue
                        
                        # F-01 FIX: Independent fail-closed permission check
                        # F-01: fail-closed gate. Pause is honoured here.
                        gate_result = self._entry_gate.check()
                        if not gate_result.ok:
                            logger.warning(
                                "TRADE PLAN BLOCKED AT ENTRY | %s | %s %s",
                                gate_result.reason,
                                plan.direction,
                                plan.symbol,
                            )
                            plan.status = "INVALIDATED"
                            plan.invalidated_reason = (
                                f"Entry gate: {gate_result.reason}"
                            )
                            continue

                        signal = TradeSignal(
                            timestamp=now,
                            symbol=plan.symbol,
                            direction=SignalDirection(plan.direction),
                            entry_price=plan.entry_price,
                            sl_price=plan.sl_price,
                            tp_price=plan.tp_price,
                            lot_size=0.0,
                            risk_percent=config.risk.default_risk_percent,
                            risk_reward=plan.risk_reward,
                            strategy=StrategyType(plan.strategy),
                            confirmations=[],
                            timeframe_analysis={},
                            entry_tf="m5",
                            setup_tf="m15",
                            structure_tf="h1",
                            bias_tf="h4"
                        )

                        validation = signal_validator.validate_signal(signal, market_entry=True)
                        if not validation.valid:
                            plan.status = "INVALIDATED"
                            plan.invalidated_reason = validation.reason
                            logger.warning(
                                "TRADE PLAN INVALIDATED AT ENTRY | %s",
                                validation.reason
                            )
                            continue

                        result = order_executor.execute_signal(
                            signal,
                            market_entry=True,
                            gate=self._entry_gate,
                        )

                        if result.success:
                            plan.status = "EXECUTED"
                            logger.info(
                                "TRADE PLAN EXECUTED | %s %s @ %.5f",
                                plan.direction,
                                plan.symbol,
                                plan.entry_price
                            )
                            if self._telegram_ready:
                                await telegram_bot.notify_new_trade(signal, result)
                        elif getattr(result, "uncertain", False):
                            # Submission outcome is not definitively known.
                            # Leave the plan WAITING; the fingerprint
                            # duplicate check blocks an immediate retry,
                            # and the executor's reconciler will resolve
                            # the uncertain request to filled /
                            # rejected_notfound on a later cycle.
                            logger.warning(
                                "TRADE PLAN UNCERTAIN | %s %s @ %.5f | %s",
                                plan.direction,
                                plan.symbol,
                                plan.entry_price,
                                result.error,
                            )
                            if self._telegram_ready:
                                await telegram_bot.notify_error(
                                    "Order outcome uncertain for "
                                    f"{plan.direction} {plan.symbol}: "
                                    f"{result.error}"
                                )
                        else:
                            plan.status = "INVALIDATED"
                            plan.invalidated_reason = result.error
                            logger.warning(
                                "TRADE PLAN EXECUTION FAILED | %s",
                                result.error
                            )
                            if self._telegram_ready:
                                await telegram_bot.notify_order_rejected(
                                    signal,
                                    result.error or "Unknown error"
                                )

        except Exception:
            logger.exception("Failed to process trade plans.")

    # =========================================================
    # POSITION MANAGEMENT
    # =========================================================

    def _get_position_direction(self, position) -> Optional[str]:
        position_type = position.get("type")

        if isinstance(position_type, str):
            normalized = position_type.strip().upper()

            if normalized in ("BUY", "SELL"):
                return normalized

        if position_type == mt5.POSITION_TYPE_BUY:
            return "BUY"

        if position_type == mt5.POSITION_TYPE_SELL:
            return "SELL"

        return None

    def _get_management_price(self, position) -> Optional[float]:
        symbol = position.get("symbol")
        direction = self._get_position_direction(position)

        if not symbol or direction is None:
            return None

        tick = mt5.symbol_info_tick(symbol)

        if tick is None:
            return None

        price = float(
            tick.bid if direction == "BUY" else tick.ask
        )

        return price if price > 0 else None

    async def _manage_positions(self):
        try:
            risk_status = risk_manager.check_risk_limits()
            checks = risk_status.get("checks", {})
            
            if checks.get("emergency") is False:
                logger.critical(
                    "EMERGENCY DRAWDOWN BREACHED. Closing all positions and activating kill switch."
                )
                order_executor.close_all_positions(reason="EMERGENCY_DRAWDOWN")
                risk_manager.set_kill_switch(True, "Emergency drawdown threshold breached")
                self._pause_trading()
                
                if self._telegram_ready:
                    await telegram_bot.notify_error(
                        "EMERGENCY: Maximum drawdown breached. All positions closed. Kill switch activated."
                    )
                return
        except Exception:
            logger.exception("Failed to check emergency drawdown status.")

        positions = self._get_positions_or_raise()

        for position in positions:
            position_magic = position.get("magic")

            if position_magic is None:
                logger.warning(
                    "Skipping position %s: magic number unavailable.",
                    position.get("ticket", "unknown")
                )
                continue

            if int(position_magic) != int(config.trading.magic_number):
                continue

            ticket = int(position["ticket"])
            current_price = self._get_management_price(position)

            if current_price is None:
                logger.warning(
                    "Skipping position %s: closing quote unavailable.",
                    ticket
                )
                continue

            trade_manager.update_position(
                ticket,
                current_price,
                float(position.get("profit", 0.0)),
                float(position.get("swap", 0.0))
            )

            be_action = trade_manager.check_break_even(
                ticket,
                current_price
            )

            if be_action and be_action.success:
                logger.info(
                    "BREAK EVEN | Position %s | New SL: %s",
                    ticket,
                    be_action.new_value
                )

                if self._telegram_ready:
                    await telegram_bot.notify_break_even(
                        ticket,
                        be_action.new_value
                    )

            trail_action = trade_manager.check_trailing_stop(
                ticket,
                current_price
            )

            if trail_action and trail_action.success:
                state = trade_manager.get_position_state(ticket)
                highest_r = state["highest_r"] if state else 0.0

                logger.info(
                    "TRAILING STOP | Position %s | New SL: %s",
                    ticket,
                    trail_action.new_value
                )

                if self._telegram_ready:
                    await telegram_bot.notify_trailing_stop(
                        ticket,
                        trail_action.new_value,
                        highest_r
                    )

            partial_actions = trade_manager.check_partial_tp(
                ticket,
                current_price
            )

            for action in partial_actions or []:
                if action.success:
                    state = trade_manager.get_position_state(ticket)
                    highest_r = state["highest_r"] if state else 0.0

                    logger.info(
                        "PARTIAL TAKE PROFIT | Position %s",
                        ticket
                    )

                    if self._telegram_ready:
                        await telegram_bot.notify_partial_tp(
                            ticket,
                            action.old_value,
                            highest_r,
                            0
                        )

            time_action = trade_manager.check_max_holding_time(
                ticket,
                position["time_open"]
            )

            if time_action and time_action.success:
                logger.warning(
                    "Maximum holding time reached for position %s. "
                    "Requesting closure.",
                    ticket
                )

                close_result = order_executor.close_position(ticket)

                if not close_result or not close_result.success:
                    logger.error(
                        "Could not close position %s after maximum holding time: %s",
                        ticket,
                        getattr(close_result, "error_message", "unknown error"),
                    )

                    self._pause_trading()


    # =========================================================
    # WEEKEND CLOSE
    # =========================================================

    _WEEKEND_MARKER_KEY = "last_weekend_close_date"

    def _handle_weekend_close(self):
        """Close bot-owned positions once per weekend when configured.

        Behaviour:
        - Fires only when ``config.sessions.weekend_protection`` is True
          AND ``config.sessions.close_before_weekend_hours > 0`` AND
          ``market_data.is_weekend_protection_active()`` is True.
        - Persists a per-weekend marker (Friday's ISO date) in AppSettings
          so it fires exactly once per calendar weekend, even if the
          agent restarts mid-session.
        - A failure to record the marker is logged but does NOT skip the
          close — the operator's safety beats the idempotency guard.
        """
        try:
            sessions_cfg = config.sessions
        except Exception:
            logger.exception("Could not read session config.")
            return

        if not bool(getattr(sessions_cfg, "weekend_protection", False)):
            return

        try:
            hours_before = int(
                getattr(sessions_cfg, "close_before_weekend_hours", 0)
            )
        except (TypeError, ValueError):
            hours_before = 0

        if hours_before <= 0:
            return

        try:
            if not market_data.is_weekend_protection_active():
                return
        except Exception:
            logger.exception("Weekend protection check raised.")
            return

        # Key on Friday of the current calendar week, so re-running
        # during the weekend does not double-fire.
        now = datetime.now(timezone.utc)
        if now.weekday() == 4:                     # Friday
            week_key = now.date().isoformat()
        elif now.weekday() >= 5:                    # Sat / Sun
            week_key = (now - timedelta(
                days=now.weekday() - 4
            )).date().isoformat()
        else:
            return  # Mon–Thu: nothing to do

        from src.database_manager import AppSettings
        try:
            setting = db_manager.query_one(
                AppSettings, key=self._WEEKEND_MARKER_KEY,
            )
        except Exception:
            logger.exception("Could not read weekend-close marker.")
            setting = None

        if setting is not None and setting.value == week_key:
            return  # Already closed for this weekend.

        logger.warning(
            "Weekend protection active and close_before_weekend_hours=%d; "
            "closing bot-owned positions (weekend=%s).",
            hours_before, week_key,
        )

        result = trading_controller.close_all_for_weekend()

        if not result.get("query_ok", False):
            logger.error(
                "Weekend close could not verify position query; "
                "will retry next cycle. Details: %s", result,
            )
            return  # Do NOT mark as done — retry next loop.

        if result.get("failed", 0) > 0:
            logger.error(
                "Weekend close partially failed: closed=%d failed=%d. "
                "Will retry next cycle.", result["closed"], result["failed"],
            )
            return  # Do NOT mark — retry next loop.

        # Successfully closed. Persist the marker.
        try:
            with db_manager.get_session() as session:
                row = (
                    session.query(AppSettings)
                    .filter_by(key=self._WEEKEND_MARKER_KEY).first()
                )
                if row is None:
                    session.add(AppSettings(
                        key=self._WEEKEND_MARKER_KEY, value=week_key,
                    ))
                else:
                    row.value = week_key
        except Exception:
            logger.exception(
                "Could not persist weekend-close marker; close already "
                "executed. Next tick will re-attempt close but should "
                "find nothing to close (idempotent at broker level)."
            )

        if self._telegram_ready:
            try:
                asyncio.create_task(
                    telegram_bot.notify(
                        "Weekend Close",
                        f"Closed {result['closed']} bot-owned position(s) "
                        f"ahead of the weekend.",
                        "🕒",
                    )
                )
            except Exception:
                logger.exception("Weekend-close Telegram notify failed.")

    # =========================================================
    # HEALTH CHECKS AND RECONNECTION
    # =========================================================

    async def _health_check(self):
        if not mt5_connector.is_connected():
            logger.warning(
                "MT5 connection lost. Attempting reconnection..."
            )

            self._pause_trading()

            if self._telegram_ready:
                await telegram_bot.notify_connection_lost()

            if not mt5_connector.reconnect():
                logger.error(
                    "MT5 reconnection failed."
                )
                return

            logger.info("MT5 reconnected.")

            self._verify_current_connection()

            if self._telegram_ready:
                await telegram_bot.notify_reconnected()

            if not market_data.initialize_cache():
                raise RuntimeError(
                    "Market data cache failed after reconnection."
                )

            self._recover_state()

        self._verify_current_connection()

        mt5_connector.update_account_info()

        self._save_account_snapshot()

        # F-05 / H-05 / H-06: periodic reconciliation. Fail closed when
        # any bot-owned position remains unresolved.
        if not self._shutdown_requested:
            try:
                reconciled = trade_manager.reconcile_with_broker()

                if not reconciled:
                    self._pause_trading()
                    unresolved = trade_manager.get_unresolved_tickets()
                    logger.error(
                        "Broker reconciliation incomplete; new entries "
                        "paused. Unresolved tickets: %s", unresolved,
                    )
            except Exception:
                logger.exception(
                    "Periodic broker reconciliation failed."
                )
                self._pause_trading()

        # Phase 3D: surface stuck pending-cancel attempts. Not a pause
        # trigger — stuck cancels are a housekeeping issue, not a safety
        # issue; positions are unaffected. Logged WARNING so it appears
        # in the operator's Telegram feed and dashboard.
        try:
            stuck = order_executor.get_stuck_cancellations()
            if stuck:
                logger.warning(
                    "Stuck pending cancellations require operator "
                    "attention: %s", stuck,
                )
        except Exception:
            logger.exception(
                "Could not check stuck pending cancellations."
            )

        if self._paused and not self._shutdown_requested:
            probe = self._entry_gate.probe()
            if probe.ok:
                self._paused = False
                self._error_count = 0
                logger.info("Health checks passed. New entries resumed.")
            else:
                logger.debug(
                    "Health probe: still blocked - %s", probe.reason
                )

    # =========================================================
    # PAUSE AND RESUME
    # =========================================================

    def _pause_trading(self):
        if not self._paused:
            logger.warning("New trading entries paused.")

        self._paused = True

    def _resume_trading(self):
        if self._shutdown_requested:
            logger.warning(
                "Cannot resume: shutdown is in progress."
            )
            return

        probe = self._entry_gate.probe()
        if probe.ok:
            self._paused = False
            self._error_count = 0
            logger.info("Trading entries resumed through Telegram.")
        else:
            self._paused = True
            logger.warning(
                "Resume rejected. Reason: %s", probe.reason
            )

    def _request_shutdown(self):
        logger.info("Shutdown requested.")
        self._shutdown_requested = True

    # =========================================================
    # SHUTDOWN
    # =========================================================

    async def shutdown(self):
        if self._shutdown_started:
            return

        self._shutdown_started = True
        self._running = False
        self._paused = True
        try:
            trading_controller.unregister_agent()
        except Exception:
            logger.exception("TradingController.unregister_agent failed.")
        logger.info("Shutting down trading agent...")

        if self._mt5_started and mt5_connector.is_connected():
            try:
                self._save_account_snapshot()
            except Exception:
                logger.exception("Final account snapshot could not be saved.")

        if self._telegram_ready:
            try:
                await telegram_bot.notify_shutdown()
            except Exception:
                logger.exception("Shutdown notification failed.")

        if self._market_data_started:
            try:
                market_data.stop()
            except Exception:
                logger.exception("Market data shutdown failed.")
            self._market_data_started = False

        # F-07: configurable pending-order shutdown policy.
        policy = str(
            getattr(config.trading, "pending_orders_on_shutdown", "keep")
        ).lower()
        if policy == "cancel" and self._mt5_started:
            try:
                cancelled = order_executor.cancel_all_pending(
                    reason="SHUTDOWN"
                )
                if not getattr(cancelled, "query_ok", True):
                    logger.error(
                        "Shutdown cancel: pending-order query failed; "
                        "could not confirm cancellation."
                    )
                else:
                    failed = getattr(cancelled, "failed", 0)
                    total = len(cancelled)
                    if failed:
                        logger.error(
                            "Shutdown cancel: %d/%d pending order(s) "
                            "failed to cancel. Check stuck list.",
                            failed, total,
                        )
                    else:
                        logger.info(
                            "Shutdown policy: cancelled %d bot-owned "
                            "pending order(s).", total,
                        )
            except Exception:
                logger.exception(
                    "Failed to cancel bot-owned pending orders at shutdown."
                )

        if self._mt5_started:
            try:
                mt5_connector.shutdown()
            except Exception:
                logger.exception("MT5 shutdown failed.")
            self._mt5_started = False

        if self._telegram_logging_started:
            handler = logger_module.telegram_handler
            if handler:
                try:
                    await handler.shutdown_async()
                except Exception:
                    logger.exception("Telegram logging handler shutdown failed.")
            self._telegram_logging_started = False

        try:
            await telegram_bot.shutdown()
        except Exception:
            logger.exception("Telegram shutdown failed.")

        self._telegram_ready = False
        
        # MILESTONE 11: Release the lock
        if getattr(self, '_lock_acquired', False) and self._lock_file.exists():
            try:
                self._lock_file.unlink()
            except Exception:
                pass
                
        logger.info("Trading agent shutdown complete.")


# =============================================================
# APPLICATION ENTRY POINT
# =============================================================

async def main():
    setup_logging()
    agent = TradingAgent()

    def signal_handler(signum, frame):
        logger.info("Received operating-system signal: %s", signum)
        agent._request_shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())