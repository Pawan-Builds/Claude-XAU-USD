"""Telegram bot for notifications and remote control."""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

from telegram import Update, Bot
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ContextTypes, MessageHandler, filters
)
from telegram.error import TelegramError

from src.config_loader import config
from src.mt5_connector import mt5_connector
from src.database_manager import db_manager, AccountSnapshot, ClosedTrade, OpenPosition
from src.risk_manager import risk_manager
from src import logger as app_logger
from src.logger import get_logger

logger = get_logger(__name__)


class BotState(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    EMERGENCY = "emergency"


class TelegramBot:
    """Telegram bot for notifications and control."""

    def __init__(self):
        self.telegram_config = config.telegram
        self.bot_token = self.telegram_config.bot_token
        self.chat_id = self.telegram_config.chat_id
        self.parse_mode = self.telegram_config.parse_mode

        self._application: Optional[Application] = None
        self._bot: Optional[Bot] = None
        self._state = BotState.STOPPED
        self._authorized_users = {self.chat_id}
        self._command_handlers: Dict[str, Callable] = {}

    async def initialize(self):
        """Initialize the Telegram bot."""
        if not self.bot_token:
            logger.warning("Telegram bot token not configured")
            return False

        try:
            self._application = Application.builder().token(self.bot_token).build()
            self._bot = self._application.bot

            # Connect Telegram bot to the logging handler
            if app_logger.telegram_handler:
                app_logger.telegram_handler.set_bot(self._bot)
            else:
                logger.warning("Telegram logging handler not available")

            # Register handlers
            self._register_handlers()

            # Start the bot
            await self._application.initialize()
            await self._application.start()
            await self._application.updater.start_polling()

            self._state = BotState.RUNNING
            logger.info("Telegram bot started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            return False

    def _register_handlers(self):
        """Register read-only commands and the single emergency trigger."""
        self._application.add_handler(CommandHandler("start", self._cmd_start))
        self._application.add_handler(CommandHandler("status", self._cmd_status))
        self._application.add_handler(CommandHandler("balance", self._cmd_balance))
        self._application.add_handler(CommandHandler("positions", self._cmd_positions))
        self._application.add_handler(CommandHandler("history", self._cmd_history))
        self._application.add_handler(CommandHandler("performance", self._cmd_performance))
        self._application.add_handler(CommandHandler("kill", self._cmd_kill))
        self._application.add_handler(CommandHandler("help", self._cmd_help))

    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized."""
        return user_id in self._authorized_users

    async def _send_message(self, text: str, parse_mode: str = "HTML", reply_markup=None):
        """Send message to authorized chat."""
        if self._bot is None:
            logger.warning(f"Telegram bot not initialized, cannot send message: {text[:100]}")
            return
        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup
            )
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")

    # --- Command Handlers ---

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized")
            return

        from src.trading_controller import trading_controller
        status = trading_controller.status()
        emoji = {
            "running": "🟢", "paused": "🟡",
            "stopped": "🔴", "emergency": "🚨",
        }.get(status.state, "❔")

        await update.message.reply_text(
            f"{emoji} <b>XAU/USD SMC Trading Bot</b>\n\n"
            f"Agent state: <b>{status.state.upper()}</b>\n"
            f"Mode: {status.mode}\n"
            f"Kill switch: {'ACTIVE' if status.kill_switch else 'inactive'}\n\n"
            f"Use /help for available commands.",
            parse_mode="HTML",
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized")
            return

        # Get MT5 connection status
        mt5_connected = mt5_connector.is_connected()
        account = mt5_connector.get_account_info()

        text = f"📊 <b>System Status</b>\n\n"
        from src.trading_controller import trading_controller
        controller_status = trading_controller.status()
        text = f"📊 <b>System Status</b>\n\n"
        text += f"Agent State: <b>{controller_status.state.upper()}</b>\n"
        text += f"MT5 Connected: {'✅' if mt5_connected else '❌'}\n"
        text += f"Kill Switch: {'🚨 ACTIVE' if controller_status.kill_switch else '✅ inactive'}\n"
        text += f"MT5 Connected: {'✅' if mt5_connected else '❌'}\n"

        if account:
            text += f"Account: <code>{account.login}</code>\n"
            text += f"Server: {account.server}\n"
            text += f"Type: {'🎮 DEMO' if account.is_demo else '💰 REAL'}\n"
            text += f"Balance: <b>${account.balance:,.2f}</b>\n"
            text += f"Equity: <b>${account.equity:,.2f}</b>\n"
            text += f"Free Margin: ${account.free_margin:,.2f}\n"
            text += f"Margin Level: {account.margin_level:.1f}%\n"
        else:
            text += "Account: Not connected\n"

        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized")
            return

        account = mt5_connector.get_account_info()
        if not account:
            await update.message.reply_text("❌ Account not connected")
            return

        text = f"💰 <b>Account Balance</b>\n\n"
        text += f"Login: <code>{account.login}</code>\n"
        text += f"Currency: {account.currency}\n"
        text += f"Leverage: 1:{account.leverage}\n\n"
        text += f"Balance: <b>${account.balance:,.2f}</b>\n"
        text += f"Equity: <b>${account.equity:,.2f}</b>\n"
        text += f"Margin: ${account.margin:,.2f}\n"
        text += f"Free Margin: ${account.free_margin:,.2f}\n"
        text += f"Margin Level: {account.margin_level:.2f}%\n\n"

        # Daily P/L
        from datetime import datetime, timezone
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        env = config.trading.mode.lower()

        with db_manager.get_session() as session:
            trades = session.query(ClosedTrade).filter(
                ClosedTrade.environment == env,
                ClosedTrade.time_close >= today_start
            ).all()

        daily_pl = sum(t.net_profit for t in trades)
        text += f"Daily P/L: <b>${daily_pl:+,.2f}</b>\n"

        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized")
            return

        positions = mt5_connector.get_positions()

        if not positions:
            await update.message.reply_text("📭 No open positions")
            return

        text = f"📈 <b>Open Positions ({len(positions)})</b>\n\n"

        for pos in positions:
            direction_emoji = "🟢" if pos["type"] == "BUY" else "🔴"
            pnl_emoji = "🟢" if pos["profit"] >= 0 else "🔴"

            text += f"{direction_emoji} <b>{pos['type']}</b> {pos['symbol']}\n"
            text += f"  Ticket: <code>{pos['ticket']}</code>\n"
            text += f"  Volume: {pos['volume']} lots\n"
            text += f"  Entry: {pos['price_open']:.5f}\n"
            text += f"  Current: {pos['price_current']:.5f}\n"
            text += f"  SL: {pos['sl']:.5f}\n"
            text += f"  TP: {pos['tp']:.5f}\n"
            text += f"  {pnl_emoji} P/L: ${pos['profit']:+,.2f}\n"
            text += f"  Swap: ${pos['swap']:+,.2f}\n"
            text += f"  Opened: {pos['time_open'].strftime('%Y-%m-%d %H:%M')}\n\n"

        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized")
            return

        env = config.trading.mode.lower()

        with db_manager.get_session() as session:
            trades = session.query(ClosedTrade).filter(
                ClosedTrade.environment == env
            ).order_by(ClosedTrade.time_close.desc()).limit(10).all()

        if not trades:
            await update.message.reply_text("📭 No trade history")
            return

        text = f"📜 <b>Recent Trades (Last 10)</b>\n\n"

        for trade in trades:
            direction_emoji = "🟢" if trade.direction == "BUY" else "🔴"
            pnl_emoji = "🟢" if trade.net_profit >= 0 else "🔴"

            text += f"{direction_emoji} <b>{trade.direction}</b> {trade.symbol}\n"
            text += f"  Ticket: <code>{trade.ticket}</code>\n"
            text += f"  Entry: {trade.price_open:.5f} → Exit: {trade.price_close:.5f}\n"
            text += f"  Volume: {trade.volume} lots\n"
            text += f"  {pnl_emoji} Net P/L: ${trade.net_profit:+,.2f}\n"
            text += f"  R-Multiple: {trade.realized_r:+.2f}R\n"
            text += f"  Strategy: {trade.strategy}\n"
            text += f"  Exit: {trade.exit_reason}\n"
            text += f"  Closed: {trade.time_close.strftime('%Y-%m-%d %H:%M')}\n\n"

        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized")
            return

        env = config.trading.mode.lower()

        with db_manager.get_session() as session:
            trades = session.query(ClosedTrade).filter(
                ClosedTrade.environment == env
            ).all()

        if not trades:
            await update.message.reply_text("📭 No trades for performance calculation")
            return

        total = len(trades)
        winners = [t for t in trades if t.net_profit > 0]
        losers = [t for t in trades if t.net_profit < 0]

        win_rate = len(winners) / total * 100 if total > 0 else 0
        gross_profit = sum(t.net_profit for t in winners)
        gross_loss = abs(sum(t.net_profit for t in losers))
        net_profit = sum(t.net_profit for t in trades)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = gross_profit / len(winners) if winners else 0
        avg_loss = gross_loss / len(losers) if losers else 0
        expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)

        text = f"📊 <b>Performance Statistics</b>\n\n"
        text += f"Total Trades: {total}\n"
        text += f"Winning: {len(winners)} | Losing: {len(losers)}\n"
        text += f"Win Rate: <b>{win_rate:.1f}%</b>\n"
        text += f"Profit Factor: <b>{profit_factor:.2f}</b>\n"
        text += f"Net Profit: <b>${net_profit:+,.2f}</b>\n"
        text += f"Gross Profit: ${gross_profit:,.2f}\n"
        text += f"Gross Loss: ${gross_loss:,.2f}\n"
        text += f"Avg Win: ${avg_win:,.2f}\n"
        text += f"Avg Loss: ${avg_loss:,.2f}\n"
        text += f"Expectancy: ${expectancy:,.2f}\n"

        await update.message.reply_text(text, parse_mode="HTML")
        
    async def _cmd_kill(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE,
    ):
        """Emergency: activate the durable kill switch.

        This is the ONLY state-changing Telegram command. It writes to
        the database, requires no live agent, blocks all new entries,
        and never closes positions.
        """
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized")
            return

        from src.risk_manager import risk_manager
        try:
            risk_manager.set_kill_switch(
                True, reason="Telegram /kill",
            )
        except Exception as exc:
            logger.exception("Telegram /kill failed.")
            await update.message.reply_text(
                f"❌ Kill switch write failed: {exc}\n\n"
                "No action taken. Use the dashboard or host terminal.",
            )
            return

        try:
            active = risk_manager.is_kill_switch_active()
        except Exception:
            active = None

        if active is True:
            await update.message.reply_text(
                "🚨 <b>KILL SWITCH ACTIVE</b>\n\n"
                "• New entries blocked\n"
                "• Open positions remain open\n"
                "• Clear from dashboard or DB",
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text(
                "⚠️ Kill-switch write reported success, but read-back "
                "was inconclusive. Verify in dashboard.",
            )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized")
            return

        text = (
            "📖 <b>Available Commands</b>\n\n"
            "<b>Read-only</b>\n"
            "/start - Bot status overview\n"
            "/status - Detailed system status\n"
            "/balance - Account balance & equity\n"
            "/positions - Open positions\n"
            "/history - Recent trade history\n"
            "/performance - Performance statistics\n"
            "/help - This help message\n\n"
            "<b>Emergency</b>\n"
            "/kill - Activate the durable kill switch. Blocks new "
            "entries. Does <b>not</b> close open positions. Clear via "
            "dashboard or DB.\n\n"
            "<i>All other control (pause, resume, close, shutdown) is "
            "performed on the agent host or via the local dashboard.</i>"
        )
        await update.message.reply_text(text, parse_mode="HTML")


    async def notify(self, title: str, message: str, emoji: str = "ℹ️"):
        """Send a notification."""
        text = f"{emoji} <b>{title}</b>\n\n{message}"
        await self._send_message(text)

    async def notify_startup(self):
        await self.notify("Bot Started", "XAU/USD SMC Trading Bot has started successfully.", "🚀")

    async def notify_mt5_connected(self):
        await self.notify("MT5 Connected", "Successfully connected to XM MetaTrader 5.", "🔗")

    async def notify_demo_verified(self):
        await self.notify("Demo Account Verified", "Account verified as DEMO. Trading enabled.", "✅")

    async def notify_shutdown(self):
        await self.notify("Bot Shutdown", "XAU/USD SMC Trading Bot has been shut down.", "🛑")

    async def notify_new_trade(self, signal, result):
        direction_emoji = "🟢" if signal.direction.value == "BUY" else "🔴"
        text = (
            f"{direction_emoji} <b>New {signal.direction.value} Trade</b>\n\n"
            f"Symbol: {signal.symbol}\n"
            f"Strategy: {signal.strategy.value}\n"
            f"Entry: {signal.entry_price:.5f}\n"
            f"SL: {signal.sl_price:.5f}\n"
            f"TP: {signal.tp_price:.5f}\n"
            f"Volume: {signal.lot_size} lots\n"
            f"Risk: {signal.risk_percent}% (${result.position_size.risk_amount:.2f})\n"
            f"R:R: 1:{signal.risk_reward:.2f}\n\n"
            f"Ticket: <code>{result.order_result.ticket}</code>"
        )
        await self._send_message(text)

    async def notify_sl_hit(self, ticket: int, direction: str, price: float, pnl: float):
        emoji = "🔴"
        text = (
            f"{emoji} <b>Stop Loss Hit</b>\n\n"
            f"Ticket: <code>{ticket}</code>\n"
            f"Direction: {direction}\n"
            f"Exit Price: {price:.5f}\n"
            f"P/L: ${pnl:+,.2f}"
        )
        await self._send_message(text)

    async def notify_tp_hit(self, ticket: int, direction: str, price: float, pnl: float):
        text = (
            f"🟢 <b>Take Profit Hit</b>\n\n"
            f"Ticket: <code>{ticket}</code>\n"
            f"Direction: {direction}\n"
            f"Exit Price: {price:.5f}\n"
            f"P/L: ${pnl:+,.2f}"
        )
        await self._send_message(text)

    async def notify_break_even(self, ticket: int, new_sl: float):
        text = (
            f"🔵 <b>Break-Even Activated</b>\n\n"
            f"Ticket: <code>{ticket}</code>\n"
            f"New SL: {new_sl:.5f}"
        )
        await self._send_message(text)

    async def notify_trailing_stop(self, ticket: int, new_sl: float, r_multiple: float):
        text = (
            f"🔵 <b>Trailing Stop Updated</b>\n\n"
            f"Ticket: <code>{ticket}</code>\n"
            f"New SL: {new_sl:.5f}\n"
            f"Current R: {r_multiple:.2f}"
        )
        await self._send_message(text)

    async def notify_partial_tp(self, ticket: int, volume: float, r_multiple: float, pnl: float):
        text = (
            f"🟢 <b>Partial Take Profit</b>\n\n"
            f"Ticket: <code>{ticket}</code>\n"
            f"Volume Closed: {volume} lots\n"
            f"R-Multiple: {r_multiple:.2f}\n"
            f"P/L: ${pnl:+,.2f}"
        )
        await self._send_message(text)

    async def notify_order_rejected(self, signal, reason: str):
        text = (
            f"❌ <b>Order Rejected</b>\n\n"
            f"Direction: {signal.direction.value}\n"
            f"Reason: {reason}"
        )
        await self._send_message(text)

    async def notify_connection_lost(self):
        await self.notify("MT5 Connection Lost", "Lost connection to MetaTrader 5. Attempting reconnection...", "⚠️")

    async def notify_reconnected(self):
        await self.notify("MT5 Reconnected", "Successfully reconnected to MetaTrader 5.", "✅")

    async def notify_daily_report(self):
        """Send daily P/L report."""
        account = mt5_connector.get_account_info()
        if not account:
            return

        from datetime import datetime, timezone
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        env = config.trading.mode.lower()

        with db_manager.get_session() as session:
            trades = session.query(ClosedTrade).filter(
                ClosedTrade.environment == env,
                ClosedTrade.time_close >= today_start
            ).all()

        if not trades:
            return

        total_pl = sum(t.net_profit for t in trades)
        winners = [t for t in trades if t.net_profit > 0]
        losers = [t for t in trades if t.net_profit < 0]

        text = (
            f"📊 <b>Daily Report</b>\n\n"
            f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
            f"Trades: {len(trades)} (W: {len(winners)} | L: {len(losers)})\n"
            f"Net P/L: <b>${total_pl:+,.2f}</b>\n"
            f"Balance: ${account.balance:,.2f}\n"
            f"Equity: ${account.equity:,.2f}"
        )
        await self._send_message(text)

    async def notify_risk_limit(self, limit_type: str, details: str):
        text = (
            f"⚠️ <b>Risk Limit Reached</b>\n\n"
            f"Type: {limit_type}\n"
            f"Details: {details}\n"
            f"New trades paused."
        )
        await self._send_message(text)

    async def notify_emergency_shutdown(self, reason: str):
        text = (
            f"🚨 <b>EMERGENCY SHUTDOWN</b>\n\n"
            f"Reason: {reason}\n"
            f"All trading stopped."
        )
        await self._send_message(text)

    async def notify_error(self, error: str):
        text = f"❌ <b>Error</b>\n\n{error}"
        await self._send_message(text)

    def get_state(self) -> BotState:
        return self._state

    def is_running(self) -> bool:
        return self._state == BotState.RUNNING

    def is_paused(self) -> bool:
        return self._state == BotState.PAUSED

    async def shutdown(self):
        """Shutdown the bot."""
        if self._application:
            await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()
        self._state = BotState.STOPPED
        logger.info("Telegram bot shut down")


# Global telegram bot instance
telegram_bot = TelegramBot()