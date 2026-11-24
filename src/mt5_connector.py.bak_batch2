"""MT5 Connector for XM MetaTrader 5 integration."""

import os
import time
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import MetaTrader5 as mt5

from src.config_loader import config
from src.logger import get_logger

logger = get_logger(__name__)


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"


@dataclass
class AccountInfo:
    login: int
    server: str
    name: str
    company: str
    currency: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    leverage: int
    trade_mode: str
    trade_allowed: bool
    trade_expert: bool
    margin_mode: str
    is_demo: bool


@dataclass
class SymbolInfo:
    name: str
    bid: float
    ask: float
    spread: float
    point: float
    digits: int
    trade_mode: int
    volume_min: float
    volume_max: float
    volume_step: float
    contract_size: float
    tick_value: float
    tick_size: float
    swap_long: float
    swap_short: float
    margin_initial: float
    margin_maintenance: float
    session_deals: int
    session_buy_orders: int
    session_sell_orders: int
    volume_limit: float


@dataclass
class OrderResult:
    success: bool
    ticket: Optional[int] = None
    order_id: Optional[int] = None
    volume: Optional[float] = None
    price: Optional[float] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    retcode: Optional[int] = None
    comment: Optional[str] = None
    deal_id: Optional[int] = None


class MT5Connector:
    """Manages MT5 connection and trading operations."""

    def __init__(self):
        self.mt5_config = config.mt5
        self.trading_config = config.trading

        self.state = ConnectionState.DISCONNECTED
        self._account_info: Optional[AccountInfo] = None
        self._symbol_info: Optional[SymbolInfo] = None
        self._symbol_name: Optional[str] = None
        self._last_error: Optional[str] = None
        self._reconnect_attempts = 0
        self._initialized = False

    # ---------------------------------------------------------
    # Connection
    # ---------------------------------------------------------

    def initialize(self) -> bool:
        """Initialize MT5, ensure the configured account is active, verify."""

        logger.info("Initializing MT5 connection...")

        terminal_path = self.mt5_config.path

        if not terminal_path or not os.path.exists(terminal_path):
            return self._connection_failure(
                f"MT5 terminal not found: {terminal_path!r}"
            )

        if self._initialized:
            try:
                mt5.shutdown()
            except Exception:
                logger.exception("Prior MT5 shutdown raised.")
            self._initialized = False

    # Step 1: attach to the terminal WITHOUT credentials. This is the
    # pre-existing successful path when the user has already logged
    # into the terminal manually.
        if not mt5.initialize(
            path=terminal_path,
            timeout=self.mt5_config.timeout,
        ):
            return self._connection_failure(
                f"MT5 initialization failed: {mt5.last_error()}"
            )

        self._initialized = True
        self.state = ConnectionState.CONNECTING

        expected_login = int(self.mt5_config.login)
        expected_server = str(self.mt5_config.server)

        account = mt5.account_info()

        attached_ok = (
            account is not None
            and int(account.login) == expected_login
            and str(account.server) == expected_server
        )

        if not attached_ok:
        # Step 2: the terminal is on a different account. Attempt an
        # explicit login using the credentials for the SELECTED mode.
        # This is a fallback — not the primary path — so a stale
        # terminal cannot silently route orders to the wrong account.
            current_login = account.login if account else None
            current_server = account.server if account else None
            logger.warning(
                "Terminal is on account %s/%s but configuration expects "
                "%s/%s. Attempting explicit login...",
                current_login,
                current_server,
                expected_login,
                expected_server,
            )

            if not mt5.login(
                login=expected_login,
                password=self.mt5_config.password,
                server=expected_server,
                timeout=self.mt5_config.timeout,
            ):
                error = mt5.last_error()
                return self._connection_failure(
                    "MT5 explicit login failed. Check that "
                    f"{expected_login}/{expected_server} is a valid account "
                    "and that DEMO/LIVE credentials in .env are correct "
                    f"and free of inline comments. Last error: {error}"
                )

            account = mt5.account_info()

        if account is None:
            return self._connection_failure(
                f"Could not read MT5 account: {mt5.last_error()}"
            )

        if not self._verify_account_identity(account):
            return False

        if not self.verify_connection():
            error = self._last_error or "MT5 verification failed"
            return self._connection_failure(error)

        self.state = ConnectionState.AUTHENTICATED
        self._reconnect_attempts = 0
        self._last_error = None

        logger.info(
            "MT5 authenticated | account=%s | server=%s | symbol=%s",
            account.login,
            account.server,
            self._symbol_name,
        )

        return True

    def _verify_account_identity(self, account) -> bool:
        """FIX #4 / #5: enforce DEMO<->demo and LIVE<->real at connect."""
        expected_login = int(self.mt5_config.login)
        expected_server = str(self.mt5_config.server)

        if int(account.login) != expected_login:
            return self._connection_failure(
                f"Connected login {account.login} does not match "
                f"configured login {expected_login}"
            )

        if str(account.server) != expected_server:
            return self._connection_failure(
                f"Connected server {account.server!r} does not match "
                f"configured server {expected_server!r}"
            )

        mode = config.trading.mode.upper()
        is_demo = account.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO

        if mode == "DEMO" and not is_demo:
            return self._connection_failure(
                "Trading mode DEMO but connected account is REAL."
            )

        if mode == "LIVE" and is_demo:
            return self._connection_failure(
                "Trading mode LIVE but connected account is DEMO."
            )

        return True

    def _connection_failure(self, message: str) -> bool:
        self._last_error = message
        self.state = ConnectionState.ERROR

        logger.error(message)

        try:
            mt5.shutdown()
        except Exception:
            logger.exception("MT5 shutdown during failure handling raised.")

        self._initialized = False
        self._account_info = None
        self._symbol_info = None
        self._symbol_name = None

        return False

    def login(self) -> bool:
        """Explicitly authenticate using configured credentials.

        Kept for backward compatibility. initialize() already performs
        login, so this is only needed when a caller wants to force a
        re-authentication on an already-initialized session.
        """

        if not self._initialized:
            logger.error("Cannot login: MT5 is not initialized")
            return False

        authorized = mt5.login(
            login=int(self.mt5_config.login),
            password=self.mt5_config.password,
            server=self.mt5_config.server,
            timeout=self.mt5_config.timeout,
        )

        if not authorized:
            self._last_error = f"MT5 login failed: {mt5.last_error()}"
            logger.error(self._last_error)
            return False

        self.state = ConnectionState.CONNECTED
        return self.verify_connection()

    def verify_connection(self) -> bool:
        """Verify account, permissions, and trading symbol."""

        if not self._initialized:
            self._last_error = "MT5 is not initialized"
            return False

        if mt5.terminal_info() is None:
            self._last_error = f"Terminal unavailable: {mt5.last_error()}"
            return False

        if not self.update_account_info():
            self._last_error = "Could not retrieve account information"
            return False

        account = self._account_info

        logger.info(
            "Account=%s | Server=%s | Currency=%s | "
            "Balance=%.2f | Equity=%.2f | Demo=%s | Trading=%s",
            account.login,
            account.server,
            account.currency,
            account.balance,
            account.equity,
            account.is_demo,
            account.trade_allowed,
        )

        # FIX #4: DEMO<->demo and LIVE<->real in every mode.
        if not self._verify_account_identity(mt5.account_info()):
            return False

        if config.safety.demo_only and not account.is_demo:
            self._last_error = (
                "Real account detected. Demo-only safety is enabled."
            )
            logger.critical(self._last_error)
            return False

        # FIX #6: trade permissions are always required, not only in DEMO.
        if not account.trade_allowed:
            self._last_error = "Trading is disabled on this account"
            logger.critical(self._last_error)
            return False

        if not account.trade_expert:
            self._last_error = "Expert Advisor trading is disabled"
            logger.critical(self._last_error)
            return False

        if not self.identify_symbol():
            return False

        if not self.verify_symbol():
            return False

        self.state = ConnectionState.AUTHENTICATED
        return True

    # ---------------------------------------------------------
    # Account information
    # ---------------------------------------------------------

    def _trade_mode_to_str(self, mode: int) -> str:
        modes = {
            mt5.ACCOUNT_TRADE_MODE_DEMO: "DEMO",
            mt5.ACCOUNT_TRADE_MODE_CONTEST: "CONTEST",
            mt5.ACCOUNT_TRADE_MODE_REAL: "REAL",
        }
        return modes.get(mode, "UNKNOWN")

    def _margin_mode_to_str(self, mode: int) -> str:
        modes = {
            mt5.ACCOUNT_MARGIN_MODE_RETAIL_NETTING: "RETAIL_NETTING",
            mt5.ACCOUNT_MARGIN_MODE_EXCHANGE: "EXCHANGE",
            mt5.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING: "RETAIL_HEDGING",
        }
        return modes.get(mode, "UNKNOWN")

    def update_account_info(self) -> bool:
        """Refresh account data directly from MT5."""

        if not self._initialized:
            return False

        account = mt5.account_info()

        if account is None:
            self._last_error = (
                f"Account information unavailable: {mt5.last_error()}"
            )
            return False

        self._account_info = AccountInfo(
            login=account.login,
            server=account.server,
            name=account.name,
            company=account.company,
            currency=account.currency,
            balance=account.balance,
            equity=account.equity,
            margin=account.margin,
            free_margin=account.margin_free,
            margin_level=account.margin_level,
            leverage=account.leverage,
            trade_mode=self._trade_mode_to_str(account.trade_mode),
            trade_allowed=account.trade_allowed,
            trade_expert=account.trade_expert,
            margin_mode=self._margin_mode_to_str(account.margin_mode),
            is_demo=account.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO,
        )

        return True

    def get_account_info(self) -> Optional[AccountInfo]:
        return self._account_info

    # ---------------------------------------------------------
    # Symbol discovery and specifications
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_symbol_name(name: str) -> str:
        return (
            name.upper()
            .replace("/", "")
            .replace("_", "")
            .replace(".", "")
        )

    def identify_symbol(self) -> bool:
        """Identify a gold symbol without selecting unrelated instruments."""

        logger.info("Identifying XAU/USD symbol...")

        symbols = mt5.symbols_get()

        # FIX: distinguish MT5 error from "no symbols"
        if symbols is None:
            self._last_error = (
                f"Could not retrieve symbols: {mt5.last_error()}"
            )
            logger.error(self._last_error)
            return False

        if len(symbols) == 0:
            self._last_error = "Broker returned an empty symbol list"
            logger.error(self._last_error)
            return False

        by_name = {symbol.name.upper(): symbol for symbol in symbols}

        preferred = [
            "GOLD",
            "XAUUSD",
            "XAUUSD.",
            "XAUUSDm",
            "XAUUSDpro",
            "GOLD.",
            "GOLDm",
            "GOLDpro",
            "XAU/USD",
            "XAU_USD",
        ]

        candidates = []

        for name in preferred:
            symbol = by_name.get(name.upper())
            if symbol is not None and symbol not in candidates:
                candidates.append(symbol)

        for symbol in symbols:
            normalized = self._normalize_symbol_name(symbol.name)

            if normalized in ("XAUUSD", "GOLD"):
                if symbol not in candidates:
                    candidates.append(symbol)

        for symbol in candidates:
            if symbol.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
                continue

            if not mt5.symbol_select(symbol.name, True):
                continue

            info = mt5.symbol_info(symbol.name)

            if info is None:
                continue

            if info.trade_tick_size <= 0 or info.point <= 0:
                continue

            self._symbol_name = symbol.name

            logger.info("Selected gold symbol: %s", self._symbol_name)
            return True

        self._last_error = "No valid tradable XAUUSD/GOLD symbol found"
        logger.error(self._last_error)
        return False

    def verify_symbol(self) -> bool:
        """Load and verify broker symbol specifications."""

        if not self._symbol_name:
            self._last_error = "No symbol selected"
            return False

        if not mt5.symbol_select(self._symbol_name, True):
            self._last_error = (
                f"Could not select symbol {self._symbol_name}: "
                f"{mt5.last_error()}"
            )
            return False

        info = mt5.symbol_info(self._symbol_name)

        if info is None:
            self._last_error = (
                f"Symbol information unavailable: {self._symbol_name}"
            )
            return False

        if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            self._last_error = f"Trading disabled for {self._symbol_name}"
            return False

        if (
            info.volume_min <= 0
            or info.volume_step <= 0
            or info.volume_max < info.volume_min
            or info.trade_tick_size <= 0
        ):
            self._last_error = "Invalid broker symbol specifications"
            return False

        tick = mt5.symbol_info_tick(self._symbol_name)

        if tick is None or tick.bid <= 0 or tick.ask <= 0:
            self._last_error = "No valid live quote for selected symbol"
            return False

        self._symbol_info = SymbolInfo(
            name=info.name,
            bid=tick.bid,
            ask=tick.ask,
            spread=tick.ask - tick.bid,
            point=info.point,
            digits=info.digits,
            trade_mode=info.trade_mode,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            contract_size=info.trade_contract_size,
            tick_value=info.trade_tick_value,
            tick_size=info.trade_tick_size,
            swap_long=info.swap_long,
            swap_short=info.swap_short,
            margin_initial=info.margin_initial,
            margin_maintenance=info.margin_maintenance,
            session_deals=info.session_deals,
            session_buy_orders=info.session_buy_orders,
            session_sell_orders=info.session_sell_orders,
            volume_limit=info.volume_limit,
        )

        logger.info(
            "Symbol=%s | Digits=%s | Point=%s | Tick size=%s | "
            "Tick value=%s | Min lot=%s | Max lot=%s | Step=%s | "
            "Pip size=%s",
            info.name,
            info.digits,
            info.point,
            info.trade_tick_size,
            info.trade_tick_value,
            info.volume_min,
            info.volume_max,
            info.volume_step,
            self.pip_size_for(self._symbol_info),
        )

        return True

    @staticmethod
    def pip_size_for(symbol_info: SymbolInfo) -> float:
        """FIX #10/#18: central broker-aware pip-size derivation.

        Gold is conventionally quoted like JPY: pip = 10 * point.
        Standard 3/5-digit FX also uses 10 * point.
        2/4-digit FX uses 1 point.
        Return 0.0 if the broker spec is invalid.
        """

        try:
            point = float(symbol_info.point)
            digits = int(symbol_info.digits)
            name = str(symbol_info.name).upper()
        except (TypeError, ValueError, AttributeError):
            return 0.0

        if point <= 0:
            return 0.0

        if "XAU" in name or "GOLD" in name:
            return point * 10

        if digits in (3, 5):
            return point * 10

        return point

    def get_symbol_info(self) -> Optional[SymbolInfo]:
        if not self._symbol_info or not self._symbol_name:
            return None

        tick = mt5.symbol_info_tick(self._symbol_name)

        if tick is not None:
            self._symbol_info.bid = tick.bid
            self._symbol_info.ask = tick.ask
            self._symbol_info.spread = tick.ask - tick.bid

        return self._symbol_info

    def get_symbol_name(self) -> Optional[str]:
        return self._symbol_name

    # ---------------------------------------------------------
    # Market data
    # ---------------------------------------------------------

    def get_tick(self) -> Optional[Dict[str, Any]]:
        if not self._symbol_name:
            return None

        tick = mt5.symbol_info_tick(self._symbol_name)

        if tick is None:
            logger.warning("Could not retrieve latest tick: %s", mt5.last_error())
            return None

        return {
            "symbol": self._symbol_name,
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "volume": tick.volume,
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
            "time_msc": tick.time_msc,
            "flags": tick.flags,
            "spread": tick.ask - tick.bid,
        }

    @staticmethod
    def _timeframe_map():
        return {
            1: mt5.TIMEFRAME_M1,
            5: mt5.TIMEFRAME_M5,
            15: mt5.TIMEFRAME_M15,
            30: mt5.TIMEFRAME_M30,
            60: mt5.TIMEFRAME_H1,
            240: mt5.TIMEFRAME_H4,
            1440: mt5.TIMEFRAME_D1,
        }

    @staticmethod
    def _rates_to_dict(rates) -> List[Dict]:
        return [
            {
                "time": datetime.fromtimestamp(rate[0], tz=timezone.utc),
                "open": float(rate[1]),
                "high": float(rate[2]),
                "low": float(rate[3]),
                "close": float(rate[4]),
                "tick_volume": int(rate[5]),
                "spread": int(rate[6]),
                "real_volume": int(rate[7]),
            }
            for rate in rates
        ]

    def get_rates(
        self,
        timeframe: int,
        count: int,
        from_date: Optional[datetime] = None,
    ) -> Optional[List[Dict]]:
        if not self._symbol_name:
            return None

        if count <= 0:
            logger.error("Candle count must be positive")
            return None

        mt5_tf = self._timeframe_map().get(timeframe)

        if mt5_tf is None:
            logger.error("Unsupported timeframe: %s", timeframe)
            return None

        if from_date is not None:
            rates = mt5.copy_rates_from(
                self._symbol_name, mt5_tf, from_date, count
            )
        else:
            rates = mt5.copy_rates_from_pos(
                self._symbol_name, mt5_tf, 0, count
            )

        if rates is None:
            logger.error("Failed to get rates: %s", mt5.last_error())
            return None

        return self._rates_to_dict(rates)

    def get_rates_range(
        self,
        timeframe: int,
        from_date: datetime,
        to_date: datetime,
    ) -> Optional[List[Dict]]:
        if not self._symbol_name:
            return None

        if from_date >= to_date:
            logger.error("Invalid historical date range")
            return None

        mt5_tf = self._timeframe_map().get(timeframe)

        if mt5_tf is None:
            logger.error("Unsupported timeframe: %s", timeframe)
            return None

        rates = mt5.copy_rates_range(
            self._symbol_name,
            mt5_tf,
            from_date,
            to_date,
        )

        if rates is None:
            logger.error("Failed to get rates range: %s", mt5.last_error())
            return None

        return self._rates_to_dict(rates)

    # ---------------------------------------------------------
    # Position and order queries
    # ---------------------------------------------------------

    def get_positions(self) -> Optional[List[Dict]]:
        """Return bot-owned positions, or None if the query failed.

        FIX #1: returning None (not []) on MT5 error so callers can
        distinguish "no positions" from "could not query".
        """

        if not self._symbol_name:
            return []

        positions = mt5.positions_get(symbol=self._symbol_name)

        if positions is None:
            logger.error(
                "Could not retrieve positions: %s", mt5.last_error()
            )
            return None

        result = []

        for pos in positions:
            if pos.magic != self.trading_config.magic_number:
                continue

            result.append({
                "ticket": pos.ticket,
                "magic": pos.magic,
                "magic_number": pos.magic,
                "symbol": pos.symbol,
                "type": (
                    "BUY"
                    if pos.type == mt5.POSITION_TYPE_BUY
                    else "SELL"
                ),
                "volume": pos.volume,
                "price_open": pos.price_open,
                "price_current": pos.price_current,
                "sl": pos.sl,
                "tp": pos.tp,
                "swap": pos.swap,
                "profit": pos.profit,
                "commission": getattr(pos, "commission", 0.0),
                "comment": pos.comment,
                "time_open": datetime.fromtimestamp(
                    pos.time, tz=timezone.utc
                ),
                "time_update": (
                    datetime.fromtimestamp(pos.time_update, tz=timezone.utc)
                    if getattr(pos, "time_update", 0)
                    else None
                ),
            })

        return result

    def get_orders(self) -> Optional[List[Dict]]:
        """Return bot-owned pending orders, or None if the query failed."""

        if not self._symbol_name:
            return []

        orders = mt5.orders_get(symbol=self._symbol_name)

        if orders is None:
            logger.error(
                "Could not retrieve pending orders: %s", mt5.last_error()
            )
            return None

        result = []

        for order in orders:
            if order.magic != self.trading_config.magic_number:
                continue

            result.append({
                "ticket": order.ticket,
                "magic": order.magic,
                "symbol": order.symbol,
                "type": self._order_type_to_str(order.type),
                "volume": order.volume_initial,
                "price_open": order.price_open,
                "sl": order.sl,
                "tp": order.tp,
                "comment": order.comment,
                "time_setup": datetime.fromtimestamp(
                    order.time_setup, tz=timezone.utc
                ),
                "time_expiration": (
                    datetime.fromtimestamp(
                        order.time_expiration, tz=timezone.utc
                    )
                    if order.time_expiration
                    else None
                ),
            })

        return result

    def get_history_deals(
        self,
        from_date: datetime,
        to_date: Optional[datetime] = None,
    ) -> Optional[List[Dict]]:
        """Return bot-owned historical deals, or None if the query failed.

        FIX #1: returns None on MT5 error rather than [].
        """

        if to_date is None:
            to_date = datetime.now(timezone.utc)

        deals = mt5.history_deals_get(from_date, to_date)

        if deals is None:
            logger.error(
                "Could not retrieve deal history: %s", mt5.last_error()
            )
            return None

        result = []

        for deal in deals:
            if deal.magic != self.trading_config.magic_number:
                continue

            result.append({
                "ticket": deal.ticket,
                "order": deal.order,
                "position_id": getattr(deal, "position_id", None),
                "magic": deal.magic,
                "symbol": deal.symbol,
                "type": self._deal_type_to_str(deal.type),
                "entry": self._deal_entry_to_str(
                    getattr(deal, "entry", None)
                ),                "volume": deal.volume,
                "price": deal.price,
                "commission": deal.commission,
                "swap": deal.swap,
                "profit": deal.profit,
                "comment": deal.comment,
                "time": datetime.fromtimestamp(
                    deal.time, tz=timezone.utc
                ),
            })

        return result

    def _order_type_to_str(self, type_id: int) -> str:
        types = {
            mt5.ORDER_TYPE_BUY: "BUY",
            mt5.ORDER_TYPE_SELL: "SELL",
            mt5.ORDER_TYPE_BUY_LIMIT: "BUY_LIMIT",
            mt5.ORDER_TYPE_SELL_LIMIT: "SELL_LIMIT",
            mt5.ORDER_TYPE_BUY_STOP: "BUY_STOP",
            mt5.ORDER_TYPE_SELL_STOP: "SELL_STOP",
            mt5.ORDER_TYPE_BUY_STOP_LIMIT: "BUY_STOP_LIMIT",
            mt5.ORDER_TYPE_SELL_STOP_LIMIT: "SELL_STOP_LIMIT",
        }
        return types.get(type_id, "UNKNOWN")

    def _deal_type_to_str(self, type_id: int) -> str:
        types = {
            mt5.DEAL_TYPE_BUY: "BUY",
            mt5.DEAL_TYPE_SELL: "SELL",
            mt5.DEAL_TYPE_BALANCE: "BALANCE",
            mt5.DEAL_TYPE_CREDIT: "CREDIT",
        }
        return types.get(type_id, "UNKNOWN")

    @staticmethod
    def _deal_entry_to_str(type_id) -> str:
        mapping = {
            mt5.DEAL_ENTRY_IN: "IN",
            mt5.DEAL_ENTRY_OUT: "OUT",
            mt5.DEAL_ENTRY_INOUT: "INOUT",
            mt5.DEAL_ENTRY_OUT_BY: "OUT_BY",
        }
        return mapping.get(type_id, "UNKNOWN")

    # ---------------------------------------------------------
    # Trading safety and execution helpers
    # ---------------------------------------------------------

    def _trading_precheck(self) -> Optional[str]:
        """FIX #2/#3: refresh account and re-verify login/server and
        mode-to-account-type before every trade."""

        if not self._initialized:
            return "MT5 is not initialized"

        if mt5.terminal_info() is None:
            return "MT5 terminal is not running"

        if not self.update_account_info():
            return "Could not refresh MT5 account information"

        account = self._account_info
        expected_login = int(self.mt5_config.login)
        expected_server = str(self.mt5_config.server)

        if int(account.login) != expected_login:
            return (
                f"Account mismatch: connected login {account.login} "
                f"does not match configured login {expected_login}"
            )

        if str(account.server) != expected_server:
            return (
                f"Server mismatch: connected server {account.server!r} "
                f"does not match configured server {expected_server!r}"
            )

        mode = config.trading.mode.upper()
        if mode == "DEMO" and not account.is_demo:
            return (
                f"Trading mode DEMO but connected account is "
                f"{account.trade_mode}"
            )

        if mode == "LIVE" and account.is_demo:
            return (
                f"Trading mode LIVE but connected account is "
                f"{account.trade_mode}"
            )

        if config.safety.demo_only and not account.is_demo:
            return "Demo-only safety: real account trading is blocked"

        if not account.trade_allowed:
            return "Trading is disabled on the account"

        if not account.trade_expert:
            return "Expert Advisor trading is disabled"

        return None

    def _get_raw_symbol_info(self, symbol: Optional[str] = None):
        symbol = symbol or self._symbol_name

        if not symbol:
            return None

        return mt5.symbol_info(symbol)

    def _normalize_volume(self, volume: float, symbol: str) -> Optional[float]:
        """Round volume down to the broker's volume step."""

        info = self._get_raw_symbol_info(symbol)

        if info is None:
            return None

        try:
            requested = Decimal(str(volume))
            minimum = Decimal(str(info.volume_min))
            maximum = Decimal(str(info.volume_max))
            step = Decimal(str(info.volume_step))

            if requested <= 0 or step <= 0:
                return None

            if requested < minimum or requested > maximum:
                return None

            steps = ((requested - minimum) / step).to_integral_value(
                rounding=ROUND_DOWN
            )

            normalized = minimum + steps * step

            if normalized < minimum or normalized > maximum:
                return None

            precision = max(0, -step.as_tuple().exponent)
            normalized = normalized.quantize(
                Decimal("1").scaleb(-precision)
            )

            return float(normalized)

        except Exception as exc:
            logger.error("Volume normalization failed: %s", exc)
            return None

    def _get_filling_mode(self, symbol: str) -> int:
        """Choose a supported filling mode from broker symbol flags."""

        info = self._get_raw_symbol_info(symbol)

        if info is None:
            return mt5.ORDER_FILLING_IOC

        filling_flags = info.filling_mode

        if filling_flags & mt5.SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK

        if filling_flags & mt5.SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC

        return mt5.ORDER_FILLING_IOC

    def _validate_owned_position(self, ticket: int):
        """Return (position, error).

        FIX #7: distinguish MT5 query failure from "position not found"
        so that close_position / modify_position are not silently
        skipped during an outage.
        """

        try:
            positions = mt5.positions_get(ticket=int(ticket))
        except Exception as exc:
            logger.exception("Position query raised for %s: %s", ticket, exc)
            return None, f"MT5 position query failed: {exc}"

        if positions is None:
            error = mt5.last_error()
            logger.error(
                "MT5 position query failed for %s: %s", ticket, error
            )
            return None, f"MT5 position query failed: {error}"

        if len(positions) == 0:
            return None, "Position not found"

        position = positions[0]

        if position.magic != self.trading_config.magic_number:
            return None, "Position does not belong to this bot"

        if self._symbol_name and position.symbol != self._symbol_name:
            return None, (
                "Position symbol does not match configured symbol"
            )

        return position, None

    def _send_order_request(self, request: Dict) -> OrderResult:
        """Send an MT5 request with safety checks and consistent results."""

        safety_error = self._trading_precheck()

        if safety_error:
            logger.error("Trading request blocked: %s", safety_error)
            return OrderResult(success=False, error_message=safety_error)

        payload = dict(request)

        action = payload.get("action")
        symbol = payload.get("symbol") or self._symbol_name

        if not symbol:
            return OrderResult(
                success=False,
                error_message="No trading symbol specified",
            )

        if self._symbol_name and symbol != self._symbol_name:
            return OrderResult(
                success=False,
                error_message="Request symbol differs from configured symbol",
            )

        payload["symbol"] = symbol
        payload["magic"] = self.trading_config.magic_number

        if not payload.get("comment"):
            payload["comment"] = "SMC_BOT"

        if action == mt5.TRADE_ACTION_DEAL:
            tick = mt5.symbol_info_tick(symbol)

            if tick is None or tick.bid <= 0 or tick.ask <= 0:
                return OrderResult(
                    success=False,
                    error_message="No valid market quote available",
                )

            order_type = payload.get("type")

            if order_type == mt5.ORDER_TYPE_BUY:
                payload["price"] = tick.ask
            elif order_type == mt5.ORDER_TYPE_SELL:
                payload["price"] = tick.bid
            else:
                return OrderResult(
                    success=False,
                    error_message="Invalid market order direction",
                )

            payload.setdefault("deviation", 20)
            payload["type_filling"] = self._get_filling_mode(symbol)

        elif action == mt5.TRADE_ACTION_PENDING:
            order_type = payload.get("type")
            if order_type not in (
                mt5.ORDER_TYPE_BUY_LIMIT,
                mt5.ORDER_TYPE_SELL_LIMIT,
                mt5.ORDER_TYPE_BUY_STOP,
                mt5.ORDER_TYPE_SELL_STOP,
                mt5.ORDER_TYPE_BUY_STOP_LIMIT,
                mt5.ORDER_TYPE_SELL_STOP_LIMIT,
            ):
                return OrderResult(
                    success=False,
                    error_message="Invalid pending order type",
                )

            if "price" not in payload or payload["price"] is None:
                return OrderResult(
                    success=False,
                    error_message="Pending order requires a price",
                )

            payload.setdefault("type_filling", mt5.ORDER_FILLING_RETURN)
            payload.setdefault("type_time", mt5.ORDER_TIME_GTC)

        result = mt5.order_send(payload)

        if result is None:
            error = mt5.last_error()

            logger.error("MT5 order_send returned None: %s", error)

            return OrderResult(
                success=False,
                error_code=error[0] if error else None,
                error_message=str(error),
            )

        accepted_codes = {
            mt5.TRADE_RETCODE_DONE,
            mt5.TRADE_RETCODE_DONE_PARTIAL,
        }

        if action == mt5.TRADE_ACTION_PENDING:
            accepted_codes.add(mt5.TRADE_RETCODE_PLACED)

        success = result.retcode in accepted_codes

        if not success:
            logger.error(
                "MT5 request rejected | retcode=%s | comment=%s",
                result.retcode,
                result.comment,
            )

            return OrderResult(
                success=False,
                ticket=getattr(result, "order", None),
                order_id=getattr(result, "order", None),
                volume=getattr(result, "volume", None),
                price=getattr(result, "price", None),
                error_code=result.retcode,
                error_message=result.comment,
                retcode=result.retcode,
                comment=result.comment,
                deal_id=getattr(result, "deal", None),
            )

        logger.info(
            "MT5 request accepted | order=%s | deal=%s | "
            "volume=%s | price=%s | retcode=%s",
            getattr(result, "order", None),
            getattr(result, "deal", None),
            getattr(result, "volume", None),
            getattr(result, "price", None),
            result.retcode,
        )

        return OrderResult(
            success=True,
            ticket=getattr(result, "order", None),
            order_id=getattr(result, "order", None),
            volume=getattr(result, "volume", None),
            price=getattr(result, "price", None),
            retcode=result.retcode,
            comment=result.comment,
            deal_id=getattr(result, "deal", None),
        )

    # ---------------------------------------------------------
    # Trading operations
    # ---------------------------------------------------------

    def send_order(self, request: Dict) -> OrderResult:
        """Send a market, pending, or supported trade-management request."""

        if not isinstance(request, dict):
            return OrderResult(
                success=False,
                error_message="Order request must be a dictionary",
            )

        return self._send_order_request(request)

    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> OrderResult:
        """Modify SL/TP after verifying ownership and broker constraints."""

        if sl is None and tp is None:
            return OrderResult(
                success=False,
                error_message="No SL or TP modification supplied",
            )

        position, error = self._validate_owned_position(ticket)

        if error:
            return OrderResult(success=False, error_message=error)

        info = self._get_raw_symbol_info(position.symbol)
        tick = mt5.symbol_info_tick(position.symbol)

        if info is None or tick is None:
            return OrderResult(
                success=False,
                error_message=(
                    "Could not retrieve symbol specifications or tick"
                ),
            )

        digits = info.digits
        point = info.point
        stops_distance = info.trade_stops_level * point
        freeze_distance = info.trade_freeze_level * point

        is_buy = position.type == mt5.POSITION_TYPE_BUY

        new_sl = position.sl if sl is None else round(float(sl), digits)
        new_tp = position.tp if tp is None else round(float(tp), digits)

        if is_buy:
            if new_sl > 0:
                if new_sl >= tick.bid:
                    return OrderResult(
                        success=False,
                        error_message="BUY stop-loss must be below Bid",
                    )

                if tick.bid - new_sl < max(
                    stops_distance, freeze_distance
                ):
                    return OrderResult(
                        success=False,
                        error_message=(
                            "BUY stop-loss is inside broker "
                            "stop/freeze distance"
                        ),
                    )

            if new_tp > 0:
                if new_tp <= tick.bid:
                    return OrderResult(
                        success=False,
                        error_message="BUY take-profit must be above Bid",
                    )

                if new_tp - tick.bid < stops_distance:
                    return OrderResult(
                        success=False,
                        error_message=(
                            "BUY take-profit is inside broker "
                            "stop distance"
                        ),
                    )

        else:
            if new_sl > 0:
                if new_sl <= tick.ask:
                    return OrderResult(
                        success=False,
                        error_message="SELL stop-loss must be above Ask",
                    )

                if new_sl - tick.ask < max(
                    stops_distance, freeze_distance
                ):
                    return OrderResult(
                        success=False,
                        error_message=(
                            "SELL stop-loss is inside broker "
                            "stop/freeze distance"
                        ),
                    )

            if new_tp > 0:
                if new_tp >= tick.ask:
                    return OrderResult(
                        success=False,
                        error_message=(
                            "SELL take-profit must be below Ask"
                        ),
                    )

                if tick.ask - new_tp < stops_distance:
                    return OrderResult(
                        success=False,
                        error_message=(
                            "SELL take-profit is inside broker "
                            "stop distance"
                        ),
                    )

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "symbol": position.symbol,
            "magic": self.trading_config.magic_number,
            "sl": new_sl,
            "tp": new_tp,
        }

        return self._send_order_request(request)

    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
    ) -> OrderResult:
        """Close all or part of a bot-owned position."""

        position, error = self._validate_owned_position(ticket)

        if error:
            return OrderResult(success=False, error_message=error)

        info = self._get_raw_symbol_info(position.symbol)

        if info is None:
            return OrderResult(
                success=False,
                error_message=(
                    "Could not retrieve broker volume specifications"
                ),
            )

        current_volume = Decimal(str(position.volume))

        if volume is None:
            close_volume = current_volume
        else:
            normalized = self._normalize_volume(volume, position.symbol)

            if normalized is None:
                return OrderResult(
                    success=False,
                    error_message=(
                        "Requested close volume violates broker "
                        "volume limits"
                    ),
                )

            close_volume = Decimal(str(normalized))

        if close_volume <= 0 or close_volume > current_volume:
            return OrderResult(
                success=False,
                error_message=(
                    "Close volume exceeds current position volume"
                ),
            )

        remaining = current_volume - close_volume
        minimum = Decimal(str(info.volume_min))

        if remaining > 0 and remaining < minimum:
            return OrderResult(
                success=False,
                error_message=(
                    "Partial close would leave less than broker minimum "
                    "volume. Close the full position or choose a smaller "
                    "close volume."
                ),
            )

        if remaining > 0:
            step = Decimal(str(info.volume_step))
            steps = (remaining / step).to_integral_value(
                rounding=ROUND_DOWN
            )

            valid_remaining = steps * step

            if abs(valid_remaining - remaining) > Decimal("0.00000001"):
                return OrderResult(
                    success=False,
                    error_message=(
                        "Remaining position volume would violate "
                        "broker volume step"
                    ),
                )

        close_type = (
            mt5.ORDER_TYPE_SELL
            if position.type == mt5.POSITION_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": int(ticket),
            "symbol": position.symbol,
            "volume": float(close_volume),
            "type": close_type,
            "magic": self.trading_config.magic_number,
            "comment": "SMC_BOT_CLOSE",
        }

        return self._send_order_request(request)

    def delete_order(self, ticket: int) -> OrderResult:
        """Delete a pending order belonging to this bot."""

        try:
            orders = mt5.orders_get(ticket=int(ticket))
        except Exception as exc:
            logger.exception("Order query raised: %s", exc)
            return OrderResult(
                success=False,
                error_message=f"Order query failed: {exc}",
            )

        if orders is None:
            return OrderResult(
                success=False,
                error_message=(
                    f"Order query failed: {mt5.last_error()}"
                ),
            )

        if len(orders) == 0:
            return OrderResult(
                success=False,
                error_message="Pending order not found",
            )

        order = orders[0]

        if order.magic != self.trading_config.magic_number:
            return OrderResult(
                success=False,
                error_message="Pending order does not belong to this bot",
            )

        if self._symbol_name and order.symbol != self._symbol_name:
            return OrderResult(
                success=False,
                error_message=(
                    "Pending order symbol does not match configured symbol"
                ),
            )

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": int(ticket),
            "symbol": order.symbol,
            "magic": self.trading_config.magic_number,
            "comment": "SMC_BOT_CANCEL",
        }

        return self._send_order_request(request)

    # ---------------------------------------------------------
    # Connection health and shutdown
    # ---------------------------------------------------------

    def is_connected(self) -> bool:
        """FIX #3: verify terminal, account, AND account identity.

        If the terminal account switches mid-session, this returns False
        so the main loop pauses and the health check triggers a reconnect
        with the configured credentials.
        """

        if not self._initialized:
            return False

        terminal = mt5.terminal_info()
        account = mt5.account_info()

        if terminal is None or account is None:
            return False

        if int(account.login) != int(self.mt5_config.login):
            logger.error(
                "Connected account %s != configured login %s",
                account.login,
                self.mt5_config.login,
            )
            return False

        if str(account.server) != str(self.mt5_config.server):
            logger.error(
                "Connected server %r != configured server %r",
                account.server,
                self.mt5_config.server,
            )
            return False

        return True

    def reconnect(self) -> bool:
        logger.warning("Attempting MT5 reconnection...")

        self.shutdown()

        for attempt in range(self.mt5_config.retry_attempts):
            self._reconnect_attempts = attempt + 1

            logger.info(
                "Reconnection attempt %s/%s",
                attempt + 1,
                self.mt5_config.retry_attempts,
            )

            time.sleep(self.mt5_config.retry_delay)

            if self.initialize():
                logger.info("MT5 reconnection successful")
                return True

        logger.error("All MT5 reconnection attempts failed")
        return False

    def shutdown(self):
        if self._initialized:
            try:
                mt5.shutdown()
            except Exception:
                logger.exception("MT5 shutdown raised.")

        self._initialized = False
        self._account_info = None
        self._symbol_info = None
        self._symbol_name = None
        self.state = ConnectionState.DISCONNECTED

        logger.info("MT5 connection closed")

    def get_last_error(self) -> Optional[str]:
        return self._last_error


# Global MT5 connector instance
mt5_connector = MT5Connector()