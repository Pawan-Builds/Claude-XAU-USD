
"""Main trading agent - DEMO and LIVE orchestration with account safeguards."""

import asyncio
import os
import signal
import time
from datetime import datetime, timezone
from typing import Optional

import src.logger as logger_module

from src.config_loader import config
from src.logger import setup_logging, get_logger
from src.database_manager import db_manager, AccountSnapshot
from src.mt5_connector import mt5_connector, mt5
from src.market_data import market_data
from src.smc_strategy import smc_engine
from src.risk_manager import risk_manager
from src.trade_manager import trade_manager
from src.order_executor import order_executor
from src.session_filter import session_filter
from src.news_filter import news_filter
from src.telegram_bot import telegram_bot


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

        account = mt5_connector.get_account_info()
        self._verify_connected_account(account)

        return account

    # =========================================================
    # ENTRY PERMISSION CHECKS
    # =========================================================

    def _check_news_permission(self) -> bool:
        try:
            result = news_filter.should_block_trading()
        except Exception:
            logger.exception("News filter execution failed.")
            return False

        if not isinstance(result, dict):
            logger.error(
                "News filter returned an invalid result."
            )
            return False

        if result.get("block", True):
            logger.warning(
                "Entry blocked by news filter: %s",
                result.get("reason", "News filter blocked trading.")
            )
            return False

        return True

    def _check_risk_permission(self) -> bool:
        try:
            result = risk_manager.check_risk_limits()
        except Exception:
            logger.exception("Risk manager execution failed.")
            return False

        if not isinstance(result, dict):
            logger.error(
                "Risk manager returned an invalid result."
            )
            return False

        if not result.get("ok", False):
            logger.warning(
                "Risk checks failed: %s",
                result.get("checks")
            )
            return False

        return True

    def _check_entry_permission(
        self,
        allow_paused: bool = False
    ) -> bool:
        """Check every restriction before permitting new entries."""

        if self._shutdown_requested:
            return False

        if self._paused and not allow_paused:
            return False

        try:
            self._verify_current_connection()
        except Exception:
            logger.exception(
                "Entry blocked: MT5 account verification failed."
            )
            self._pause_trading()
            return False

        try:
            if not session_filter.is_trading_allowed():
                return False
        except Exception:
            logger.exception("Session filter failed.")
            return False

        try:
            if market_data.is_weekend_protection_active():
                return False
        except Exception:
            logger.exception("Weekend protection check failed.")
            return False

        if not self._check_news_permission():
            return False

        if not self._check_risk_permission():
            return False

        logger.debug("All entry-permission checks passed.")
        return True

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

            telegram_bot.set_shutdown_callback(
                self._request_shutdown
            )
            telegram_bot.set_pause_callback(
                self._pause_trading
            )
            telegram_bot.set_resume_callback(
                self._resume_trading
            )

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

            # Start the managed Telegram logging worker only after
            # Telegram initialization has configured its bot instance.
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

            self._recover_state()

            self._last_health_check = time.time()

            self._last_daily_report_date = datetime.now(
                timezone.utc
            ).date()

            if self._check_entry_permission(allow_paused=True):
                self._paused = False
                logger.info(
                    "Startup checks passed. New entries enabled."
                )
            else:
                self._paused = True
                logger.warning(
                    "Startup completed, but new entries remain paused. "
                    "The health monitor will retry permission checks."
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

        result = trade_manager.sync_with_mt5()

        if result is False:
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

                today = datetime.now(timezone.utc).date()

                if today != self._last_daily_report_date:
                    if self._telegram_ready:
                        await telegram_bot.notify_daily_report()

                    self._last_daily_report_date = today

                if not self._paused and self._should_analyze():
                    await self._trading_cycle()

                # Continue managing existing bot positions even when
                # new entries are paused, provided MT5 is connected.
                await self._manage_positions()

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
    # SIGNAL GENERATION AND EXECUTION
    # =========================================================

    async def _trading_cycle(self):
        self._last_analysis_time = time.time()

        if not self._check_entry_permission():
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
            if not self._check_entry_permission():
                logger.warning(
                    "SIGNAL EXECUTION | Stopped because entry permission failed."
                )
                return

            logger.info(
                "SIGNAL EXECUTION | Submitting signal to order executor."
            )

            result = order_executor.execute_signal(
                trading_signal
            )

            if result.success:
                if result.error:
                    logger.critical(
                        "ORDER RECONCILIATION ERROR | %s",
                        result.error
                    )

                    self._pause_trading()

                    if self._telegram_ready:
                        await telegram_bot.notify_error(
                            result.error
                        )

                    return

                logger.info(
                    "ORDER EXECUTION | Successful | Signal ID: %s",
                    result.signal_id
                )

                if self._telegram_ready:
                    await telegram_bot.notify_new_trade(
                        trading_signal,
                        result
                    )

                # One successful entry per analysis cycle.
                break

            logger.warning(
                "ORDER REJECTED | %s",
                result.error or "Unknown execution error"
            )

            if self._telegram_ready:
                await telegram_bot.notify_order_rejected(
                    trading_signal,
                    result.error or "Unknown execution error"
                )

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
        positions = self._get_positions_or_raise()

        for position in positions:
            # Only manage positions positively identified as belonging
            # to this bot. Never manage unknown/manual positions.
            position_magic = position.get("magic")

            if position_magic is None:
                logger.warning(
                    "Skipping position %s: magic number unavailable.",
                    position.get("ticket", "unknown")
                )
                continue

            if int(position_magic) != int(
                config.trading.magic_number
            ):
                continue

            ticket = int(position["ticket"])

            current_price = self._get_management_price(
                position
            )

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
                state = trade_manager.get_position_state(
                    ticket
                )

                highest_r = (
                    state["highest_r"] if state else 0.0
                )

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
                    state = trade_manager.get_position_state(
                        ticket
                    )

                    highest_r = (
                        state["highest_r"] if state else 0.0
                    )

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

                close_result = order_executor.close_position(
                    ticket
                )

                if not close_result:
                    logger.error(
                        "Could not close position %s after "
                        "maximum holding time.",
                        ticket
                    )

                    self._pause_trading()

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

        if self._paused and not self._shutdown_requested:
            if self._check_entry_permission(
                allow_paused=True
            ):
                self._paused = False
                self._error_count = 0

                logger.info(
                    "Health checks passed. New entries resumed."
                )

    # =========================================================
    # PAUSE AND RESUME
    # =========================================================

    def _pause_trading(self):
        if not self._paused:
            logger.warning(
                "New trading entries paused."
            )

        self._paused = True

    def _resume_trading(self):
        if self._shutdown_requested:
            logger.warning(
                "Cannot resume: shutdown is in progress."
            )
            return

        if self._check_entry_permission(
            allow_paused=True
        ):
            self._paused = False
            self._error_count = 0

            logger.info(
                "Trading entries resumed through Telegram."
            )
        else:
            self._paused = True

            logger.warning(
                "Resume rejected. One or more entry checks failed."
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

        logger.info("Shutting down trading agent...")

        # Save the final account state before disconnecting MT5.
        if (
            self._mt5_started
            and mt5_connector.is_connected()
        ):
            try:
                self._save_account_snapshot()
            except Exception:
                logger.exception(
                    "Final account snapshot could not be saved."
                )

        # Send the shutdown notification while Telegram is available.
        if self._telegram_ready:
            try:
                await telegram_bot.notify_shutdown()
            except Exception:
                logger.exception(
                    "Shutdown notification failed."
                )

        # Stop market data before disconnecting MT5.
        if self._market_data_started:
            try:
                market_data.stop()
            except Exception:
                logger.exception(
                    "Market data shutdown failed."
                )

            self._market_data_started = False

        # Disconnect MT5.
        if self._mt5_started:
            try:
                mt5_connector.shutdown()
            except Exception:
                logger.exception(
                    "MT5 shutdown failed."
                )

            self._mt5_started = False

        # CRITICAL: Drain Telegram logging messages before the
        # Telegram HTTP client is closed.
        if self._telegram_logging_started:
            handler = logger_module.telegram_handler

            if handler:
                try:
                    await handler.shutdown_async()
                except Exception:
                    logger.exception(
                        "Telegram logging handler shutdown failed."
                    )

            self._telegram_logging_started = False

        # Close the Telegram bot only after the logging worker stops.
        try:
            await telegram_bot.shutdown()
        except Exception:
            logger.exception(
                "Telegram shutdown failed."
            )

        self._telegram_ready = False

        logger.info(
            "Trading agent shutdown complete."
        )


# =============================================================
# APPLICATION ENTRY POINT
# =============================================================

async def main():
    setup_logging()

    agent = TradingAgent()

    def signal_handler(signum, frame):
        logger.info(
            "Received operating-system signal: %s",
            signum
        )
        agent._request_shutdown()

    signal.signal(
        signal.SIGINT,
        signal_handler
    )

    if hasattr(signal, "SIGTERM"):
        signal.signal(
            signal.SIGTERM,
            signal_handler
        )

    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
