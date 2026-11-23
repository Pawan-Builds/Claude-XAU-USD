
"""Logging configuration with managed asynchronous Telegram delivery."""

import asyncio
import logging
import logging.config
from pathlib import Path
from typing import Optional

import yaml


class TelegramHandler(logging.Handler):
    """Queue warning/error logs for managed Telegram delivery."""

    def __init__(self, bot_token: str = "", chat_id: int = 0):
        super().__init__()
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._bot = None
        self._queue = None
        self._worker = None
        self._loop = None
        self._closing = False

    def set_bot(self, bot):
        self._bot = bot

    def set_credentials(self, bot_token: str, chat_id: int):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def start(self):
        """Start the Telegram delivery worker inside the running event loop."""
        if not self._bot or not self.bot_token or not self.chat_id:
            return

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._closing = False

        if self._worker is None or self._worker.done():
            self._worker = self._loop.create_task(
                self._send_worker(),
                name="telegram-log-worker"
            )

    def emit(self, record):
        """Queue warning and error messages without creating unmanaged tasks."""
        if record.levelno < logging.WARNING:
            return

        if self._closing or not self._queue or not self._loop:
            return

        try:
            message = self.format(record)
            self._loop.call_soon_threadsafe(
                self._enqueue,
                f"<b>{record.levelname}</b> | {record.name}\n{message}"
            )
        except Exception:
            self.handleError(record)

    def _enqueue(self, message):
        if self._queue is not None and not self._closing:
            try:
                self._queue.put_nowait(message)
            except Exception:
                pass

    async def _send_worker(self):
        while True:
            message = await self._queue.get()

            try:
                if message is None:
                    return

                if not self._bot or self._closing:
                    continue

                try:
                    await self._bot.send_message(
                        chat_id=self.chat_id,
                        text=message,
                        parse_mode="HTML"
                    )
                except Exception:
                    # Avoid recursively logging Telegram delivery failures.
                    pass

            finally:
                self._queue.task_done()

    async def shutdown_async(self):
        """Drain queued messages and stop the worker before closing Telegram."""
        if self._closing:
            return

        self._closing = True

        if self._queue is not None and self._worker is not None:
            try:
                await asyncio.wait_for(
                    self._queue.join(),
                    timeout=10
                )
            except asyncio.TimeoutError:
                pass

            if not self._worker.done():
                self._queue.put_nowait(None)

                try:
                    await asyncio.wait_for(self._worker, timeout=5)
                except asyncio.TimeoutError:
                    self._worker.cancel()
                    await asyncio.gather(
                        self._worker,
                        return_exceptions=True
                    )

        self._worker = None
        self._queue = None
        self._loop = None

    def close(self):
        """Synchronous logging-handler close."""
        super().close()


telegram_handler: Optional[TelegramHandler] = None


def setup_logging(config_path: str = "config/logging.yaml"):
    global telegram_handler

    Path("logs").mkdir(exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as file:
        log_config = yaml.safe_load(file)

    logging.config.dictConfig(log_config)

    for name in (
        "httpx",
        "httpcore",
        "telegram",
        "telegram.ext",
        "urllib3",
        "asyncio",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    root_logger = logging.getLogger()
    telegram_handler = None

    for handler in root_logger.handlers:
        if isinstance(handler, TelegramHandler):
            telegram_handler = handler
            break

    try:
        from src.config_loader import config

        if telegram_handler:
            telegram_handler.set_credentials(
                config.telegram.bot_token,
                config.telegram.chat_id
            )

    except Exception:
        logging.getLogger(__name__).exception(
            "Could not configure Telegram logging credentials"
        )

    return telegram_handler


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
