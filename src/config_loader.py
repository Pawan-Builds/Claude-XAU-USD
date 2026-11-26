
"""Configuration loader with secure DEMO/LIVE account separation."""

import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
LIVE_CONFIRMATION_PHRASE = "I_ACCEPT_LIVE_TRADING_RISK"


# ---------------------------------------------------------------------
# Safe conversion helpers
# ---------------------------------------------------------------------

def _is_unresolved(value: Any) -> bool:
    if value is None:
        return True

    text = str(value).strip()
    return (
        not text
        or (text.startswith("${") and text.endswith("}"))
    )


def _safe_int(value: Any, default: int = 0) -> int:
    if _is_unresolved(value):
        return default

    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if _is_unresolved(value):
        return default

    try:
        result = float(str(value).strip())
        return result if result == result and abs(result) != float("inf") else default
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_str(value: Any, default: str = "") -> str:
    if _is_unresolved(value):
        return default
    return str(value).strip()


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value

    if _is_unresolved(value):
        return default

    text = str(value).strip().lower()

    if text in ("true", "1", "yes", "on"):
        return True

    if text in ("false", "0", "no", "off"):
        return False

    return default


def _resolve_env_value(value: Any) -> Any:
    """Resolve ${VARIABLE} references without unsafe YAML text replacement."""

    if not isinstance(value, str):
        return value

    match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value.strip())

    if not match:
        return value

    variable_name = match.group(1)
    return os.environ.get(variable_name, value)


def _resolve_tree(value: Any) -> Any:
    """Resolve environment placeholders recursively in parsed YAML."""

    if isinstance(value, dict):
        return {
            key: _resolve_tree(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_resolve_tree(item) for item in value]

    return _resolve_env_value(value)


def _load_env_section(section_prefix: str) -> Dict[str, str]:
    """Load environment variables belonging to one account prefix."""

    prefix = section_prefix.upper() + "_"

    return {
        key[len(prefix):].lower(): value
        for key, value in os.environ.items()
        if key.upper().startswith(prefix)
    }


# ---------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------

@dataclass
class MT5Config:
    login: int
    password: str
    server: str
    path: str
    timeout: int
    retry_attempts: int
    retry_delay: int


@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: int
    parse_mode: str


@dataclass
class TradingConfig:
    mode: str
    symbol: str
    risk_percent: float
    min_risk_reward: float
    magic_number: int
    max_positions: int
    max_daily_trades: int
    max_daily_loss_percent: float
    max_drawdown_percent: float
    max_consecutive_losses: int
    max_holding_hours: int
    entry_mode: str
    pending_expiry_hours: int
    max_pending_orders: int
    pending_orders_on_shutdown: str = "keep"



@dataclass
class TimeframeConfig:
    h4: int
    h1: int
    m15: int
    m5: int
    bias_tf: str
    structure_tf: str
    setup_tf: str
    entry_tf: str


@dataclass
class SMCConfig:
    strategies: list
    swing_lookback: int
    fvg_min_size_pips: float
    ob_lookback: int
    liquidity_lookback: int
    displacement_threshold: float
    equal_high_low_tolerance: float


@dataclass
class RiskConfig:
    default_risk_percent: float
    min_risk_reward: float
    max_risk_percent: float
    daily_loss_limit_percent: float
    max_account_drawdown_percent: float
    max_consecutive_losses: int
    max_open_positions: int
    max_total_open_risk_percent: float
    emergency_drawdown_percent: float
    compound_enabled: bool
    compound_method: str


@dataclass
class TradeManagementConfig:
    break_even_enabled: bool
    break_even_trigger_r: float
    break_even_buffer_pips: float
    trailing_stop_enabled: bool
    trailing_stop_start_r: float
    trailing_stop_step_pips: float
    trailing_stop_method: str
    partial_tp_enabled: bool
    partial_tp_levels: list
    max_slippage_pips: float
    max_spread_pips: float


@dataclass
class SessionConfig:
    london: dict
    new_york: dict
    asian: dict
    timezone: str
    weekend_protection: bool
    close_before_weekend_hours: int


@dataclass
class NewsConfig:
    enabled: bool
    high_impact_only: bool
    buffer_minutes_before: int
    buffer_minutes_after: int
    source: str
    calendar_file: str = "data/economic_calendar.json"
    max_calendar_age_hours: int = 24


@dataclass
class BacktestingConfig:
    default_start_date: str
    default_end_date: str
    default_balance: float
    default_risk_percent: float
    spread_pips: float
    slippage_pips: float
    commission_per_lot: float
    swap_enabled: bool
    data_source: str


@dataclass
class DashboardConfig:
    host: str
    port: int
    secret_key: str
    cors_origins: list
    websocket_ping_interval: int


@dataclass
class DatabaseConfig:
    url: str
    echo: bool
    pool_size: int
    max_overflow: int


@dataclass
class LoggingConfig:
    level: str
    file: str
    max_size_mb: int
    backup_count: int
    format: str
    date_format: str


@dataclass
class SafetyConfig:
    demo_only: bool
    verify_demo_on_startup: bool
    kill_switch_enabled: bool
    max_errors_before_pause: int
    health_check_interval: int
    trading_mode: str = "DEMO"


# ---------------------------------------------------------------------
# Main configuration class
# ---------------------------------------------------------------------

class Config:
    """Loads settings and selects one explicitly configured MT5 account."""

    VALID_MODES = {"DEMO", "LIVE"}
    VALID_ENTRY_MODES = {"planned"}

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = (
            Path(config_path).resolve()
            if config_path
            else DEFAULT_CONFIG_PATH
        )

        self._raw_config: Dict[str, Any] = {}
        self._env_config: Dict[str, Dict[str, str]] = {
            "demo": {},
            "live": {},
        }

        self._load_config()

    def _load_config(self):
        """Load .env, parse YAML and resolve placeholders."""

        env_path = PROJECT_ROOT / ".env"

        if env_path.exists():
            load_dotenv(env_path, override=False)

        self._env_config = {
            "demo": _load_env_section("DEMO"),
            "live": _load_env_section("LIVE"),
        }

        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with self.config_path.open("r", encoding="utf-8") as file:
            parsed = yaml.safe_load(file) or {}

        if not isinstance(parsed, dict):
            raise ValueError("The YAML configuration root must be a mapping.")

        self._raw_config = _resolve_tree(parsed)

        # Account credentials are loaded independently. They are never
        # copied from one account section into the other.
        self._load_account_credentials()

        mode = self.trading.mode

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid TRADING_MODE {mode!r}. "
                "Use DEMO or LIVE."
            )

    def _load_account_credentials(self):
        """Resolve DEMO and LIVE credentials independently."""

        for mode in ("demo", "live"):
            section = self._raw_config.setdefault(mode, {})
            env_values = self._env_config.get(mode, {})

            section["login"] = self._first_value(
                env_values.get("mt5_login"),
                section.get("mt5_login"),
                section.get("login"),
            )

            section["password"] = self._first_value(
                env_values.get("mt5_password"),
                section.get("mt5_password"),
                section.get("password"),
            )

            section["server"] = self._first_value(
                env_values.get("mt5_server"),
                section.get("mt5_server"),
                section.get("server"),
            )

            section["path"] = self._first_value(
                env_values.get("mt5_path"),
                section.get("mt5_path"),
                section.get("path"),
            )

            section["login"] = _safe_int(section["login"], 0)
            section["password"] = _safe_str(section["password"])
            section["server"] = _safe_str(section["server"])
            section["path"] = _safe_str(section["path"])

        live = self._raw_config["live"]

        live_env = self._env_config.get("live", {})

        enabled_value = self._first_value(
            live_env.get("trading_enabled"),
            live.get("enabled"),
            os.getenv("LIVE_TRADING_ENABLED"),
        )

        allowed_account_value = self._first_value(
            live_env.get("allowed_account"),
            live.get("allowed_account"),
            os.getenv("LIVE_ALLOWED_ACCOUNT"),
        )

        live["enabled"] = _safe_bool(enabled_value, False)
        live["allowed_account"] = _safe_int(
            allowed_account_value,
            0
        )

    @staticmethod
    def _first_value(*values):
        """Return the first nonempty, resolved value."""
        for value in values:
            if not _is_unresolved(value):
                return value
        return None

    # -----------------------------------------------------------------
    # Mode and account selection
    # -----------------------------------------------------------------

    @property
    def trading_mode(self) -> str:
        return self.trading.mode

    @property
    def is_demo(self) -> bool:
        return self.trading.mode == "DEMO"

    @property
    def is_live(self) -> bool:
        return self.trading.mode == "LIVE"

    @property
    def live_trading_enabled(self) -> bool:
        return _safe_bool(
            self._raw_config.get("live", {}).get("enabled"),
            False
        )

    @property
    def live_allowed_account(self) -> int:
        return _safe_int(
            self._raw_config.get("live", {}).get("allowed_account"),
            0
        )

    @property
    def live_trading_confirmation(self) -> str:
        """M-13: return the current LIVE confirmation string.

        Must be a property, not a class attribute. The prior form
        evaluated os.getenv(...) at module-import time — before
        load_dotenv() runs — so a value placed only in .env read as
        "" at the connector submission boundary even though the
        startup check (which calls os.getenv() fresh) saw the
        correct value. Reading fresh on every access also ensures
        the boundary check cannot be satisfied by a stale snapshot.
        """
        return os.getenv("LIVE_TRADING_CONFIRMATION", "").strip()

    def validate_mode_configuration(self, mode: Optional[str] = None):
        """Validate the selected account configuration before MT5 startup."""

        selected_mode = (
            str(mode or self.trading.mode).strip().upper()
        )

        if selected_mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported trading mode: {selected_mode!r}"
            )
        self.validate_entry_mode()

        if selected_mode == "LIVE":
            if not self.live_trading_enabled:
                raise RuntimeError(
                    "LIVE mode is disabled. Set "
                    "LIVE_TRADING_ENABLED=true only after completing "
                    "the required safety and testing checks."
                )

            if self.live_allowed_account <= 0:
                raise RuntimeError(
                    "LIVE_ALLOWED_ACCOUNT is missing or invalid."
                )

        credentials = self.get_account_info_for_mode(selected_mode)

        if credentials["login"] <= 0:
            raise RuntimeError(
                f"{selected_mode} MT5 login is missing or invalid."
            )

        if not credentials["password"]:
            raise RuntimeError(
                f"{selected_mode} MT5 password is missing."
            )

        if not credentials["server"]:
            raise RuntimeError(
                f"{selected_mode} MT5 server is missing."
            )

        if selected_mode == "LIVE":
            if credentials["login"] != self.live_allowed_account:
                raise RuntimeError(
                    "Configured LIVE login does not match "
                    "LIVE_ALLOWED_ACCOUNT."
                )

        return True

    def validate_entry_mode(self) -> bool:
        """M-07: fail closed on unsupported trading.entry_mode.

        Only "planned" is implemented: the bot generates a trade plan,
        waits for the planned entry, reconfirms the SMC setup, runs the
        entry gate, then submits a MARKET order (TRADE_ACTION_DEAL).

        "immediate" is reserved and NOT implemented. Any other value
        (including "limit" and "market", which previously appeared in
        the misleading entry_order_type key) raises RuntimeError before
        MT5 connects.

        Raises:
            RuntimeError: on any value outside VALID_ENTRY_MODES.
        """
        mode = self.trading.entry_mode
        if mode not in self.VALID_ENTRY_MODES:
            raise RuntimeError(
                f"Unsupported trading.entry_mode {mode!r}. "
                f"Only {sorted(self.VALID_ENTRY_MODES)} is implemented. "
                f"'immediate' is reserved and not implemented."
            )
        return True

    def get_account_info_for_mode(self, mode: str) -> Dict[str, Any]:
        """Return credentials for exactly the requested mode."""

        selected_mode = str(mode).strip().upper()

        if selected_mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported account mode: {selected_mode!r}"
            )

        section = self._raw_config.get(
            selected_mode.lower(),
            {}
        )

        return {
            "login": _safe_int(section.get("login"), 0),
            "password": _safe_str(section.get("password")),
            "server": _safe_str(section.get("server")),
            "path": _safe_str(section.get("path")),
        }

    @property
    def mt5(self) -> MT5Config:
        """Return MT5 credentials for the selected mode only."""

        mode = self.trading.mode

        self.validate_mode_configuration(mode)

        credentials = self.get_account_info_for_mode(mode)
        mt5_settings = self._raw_config.get("mt5", {})

        return MT5Config(
            login=credentials["login"],
            password=credentials["password"],
            server=credentials["server"],
            path=credentials["path"],
            timeout=_safe_int(mt5_settings.get("timeout"), 60000),
            retry_attempts=_safe_int(
                mt5_settings.get("retry_attempts"), 3
            ),
            retry_delay=_safe_int(
                mt5_settings.get("retry_delay"), 5
            ),
        )

    # -----------------------------------------------------------------
    # Individual configuration sections
    # -----------------------------------------------------------------

    @property
    def telegram(self) -> TelegramConfig:
        c = self._raw_config.get("telegram", {})

        return TelegramConfig(
            bot_token=_safe_str(c.get("bot_token")),
            chat_id=_safe_int(c.get("chat_id"), 0),
            parse_mode=_safe_str(c.get("parse_mode"), "HTML"),
        )

    @property
    def trading(self) -> TradingConfig:
        c = self._raw_config.get("trading", {})

        mode = _safe_str(
            c.get("mode"),
            _safe_str(os.getenv("TRADING_MODE"), "DEMO")
        ).upper()

        return TradingConfig(
            mode=mode,
            symbol=_safe_str(c.get("symbol"), "GOLD"),
            risk_percent=_safe_float(c.get("risk_percent"), 1.0),
            min_risk_reward=_safe_float(
                c.get("min_risk_reward"), 2.0
            ),
            magic_number=_safe_int(
                c.get("magic_number"), 123456789
            ),
            max_positions=_safe_int(c.get("max_positions"), 3),
            max_daily_trades=_safe_int(
                c.get("max_daily_trades"), 20
            ),
            max_daily_loss_percent=_safe_float(
                c.get("max_daily_loss_percent"), 5.0
            ),
            max_drawdown_percent=_safe_float(
                c.get("max_drawdown_percent"), 10.0
            ),
            max_consecutive_losses=_safe_int(
                c.get("max_consecutive_losses"), 5
            ),
            max_holding_hours=_safe_int(
                c.get("max_holding_hours"), 48
            ),
            entry_mode=_safe_str(
                c.get("entry_mode"), "planned"
            ).lower(),
            pending_expiry_hours=_safe_int(
                c.get("pending_expiry_hours"), 4
            ),
            max_pending_orders=_safe_int(
                c.get("max_pending_orders"), 5
            ),
            pending_orders_on_shutdown=_safe_str(
                c.get("pending_orders_on_shutdown"), "keep"
            ).lower(),
        )

    @property
    def timeframes(self) -> TimeframeConfig:
        c = self._raw_config.get("timeframes", {})

        return TimeframeConfig(
            h4=_safe_int(c.get("h4"), 240),
            h1=_safe_int(c.get("h1"), 60),
            m15=_safe_int(c.get("m15"), 15),
            m5=_safe_int(c.get("m5"), 5),
            bias_tf=_safe_str(c.get("bias_tf"), "h4"),
            structure_tf=_safe_str(c.get("structure_tf"), "h1"),
            setup_tf=_safe_str(c.get("setup_tf"), "m15"),
            entry_tf=_safe_str(c.get("entry_tf"), "m5"),
        )

    @property
    def smc(self) -> SMCConfig:
        c = self._raw_config.get("smc", {})

        return SMCConfig(
            strategies=c.get("strategies", []),
            swing_lookback=_safe_int(
                c.get("swing_lookback"), 50
            ),
            fvg_min_size_pips=_safe_float(
                c.get("fvg_min_size_pips"), 5.0
            ),
            ob_lookback=_safe_int(c.get("ob_lookback"), 20),
            liquidity_lookback=_safe_int(
                c.get("liquidity_lookback"), 100
            ),
            displacement_threshold=_safe_float(
                c.get("displacement_threshold"), 1.5
            ),
            equal_high_low_tolerance=_safe_float(
                c.get("equal_high_low_tolerance"), 2.0
            ),
        )

    @property
    def risk(self) -> RiskConfig:
        c = self._raw_config.get("risk", {})
        mode = self.trading.mode

        # Optional per-mode overrides take precedence over the common cap.
        if mode == "DEMO":
            default_cap = 2.0
            mode_cap = c.get("demo_max_risk_percent")
            mode_default = c.get("demo_default_risk_percent")
        else:
            default_cap = 1.0
            mode_cap = c.get("live_max_risk_percent")
            mode_default = c.get("live_default_risk_percent")

        common_cap = _safe_float(
            c.get("max_risk_percent"),
            default_cap
        )

        max_risk = _safe_float(mode_cap, common_cap)

        if max_risk <= 0:
            raise ValueError(
                f"{mode} max risk percent must be greater than zero."
            )

        common_default = _safe_float(
            c.get("default_risk_percent"),
            self.trading.risk_percent
        )

        default_risk = _safe_float(
            mode_default,
            common_default
        )

        # The configured default must never exceed the selected mode cap.
        default_risk = min(default_risk, max_risk)

        return RiskConfig(
            default_risk_percent=default_risk,
            min_risk_reward=_safe_float(
                c.get("min_risk_reward"),
                self.trading.min_risk_reward
            ),
            max_risk_percent=max_risk,
            daily_loss_limit_percent=_safe_float(
                c.get("daily_loss_limit_percent"),
                self.trading.max_daily_loss_percent
            ),
            max_account_drawdown_percent=_safe_float(
                c.get("max_account_drawdown_percent"),
                self.trading.max_drawdown_percent
            ),
            max_consecutive_losses=_safe_int(
                c.get("max_consecutive_losses"),
                self.trading.max_consecutive_losses
            ),
            max_open_positions=_safe_int(
                c.get("max_open_positions"),
                self.trading.max_positions
            ),
            max_total_open_risk_percent=_safe_float(
                c.get("max_total_open_risk_percent"),
                4.0 if mode == "LIVE" else 6.0
            ),
            emergency_drawdown_percent=_safe_float(
                c.get("emergency_drawdown_percent"),
                10.0 if mode == "LIVE" else 15.0
            ),
            compound_enabled=_safe_bool(
                c.get("compound_enabled"), True
            ),
            compound_method=_safe_str(
                c.get("compound_method"), "equity_based"
            ),
        )

    @property
    def trade_management(self) -> TradeManagementConfig:
        c = self._raw_config.get("trade_management", {})

        return TradeManagementConfig(
            break_even_enabled=_safe_bool(
                c.get("break_even_enabled"), True
            ),
            break_even_trigger_r=_safe_float(
                c.get("break_even_trigger_r"), 1.0
            ),
            break_even_buffer_pips=_safe_float(
                c.get("break_even_buffer_pips"), 5.0
            ),
            trailing_stop_enabled=_safe_bool(
                c.get("trailing_stop_enabled"), True
            ),
            trailing_stop_start_r=_safe_float(
                c.get("trailing_stop_start_r"), 1.5
            ),
            trailing_stop_step_pips=_safe_float(
                c.get("trailing_stop_step_pips"), 10.0
            ),
            trailing_stop_method=_safe_str(
                c.get("trailing_stop_method"), "structure"
            ),
            partial_tp_enabled=_safe_bool(
                c.get("partial_tp_enabled"), True
            ),
            partial_tp_levels=c.get("partial_tp_levels", []),
            max_slippage_pips=_safe_float(
                c.get("max_slippage_pips"), 3.0
            ),
            max_spread_pips=_safe_float(
                c.get("max_spread_pips"), 5.0
            ),
        )

    @property
    def sessions(self) -> SessionConfig:
        c = self._raw_config.get("sessions", {})

        return SessionConfig(
            london=c.get("london", {}),
            new_york=c.get("new_york", {}),
            asian=c.get("asian", {}),
            timezone=_safe_str(c.get("timezone"), "UTC"),
            weekend_protection=_safe_bool(
                c.get("weekend_protection"), True
            ),
            close_before_weekend_hours=_safe_int(
                c.get("close_before_weekend_hours"), 4
            ),
        )

    @property
    def news(self) -> NewsConfig:
        c = self._raw_config.get("news", {})

        return NewsConfig(
            enabled=_safe_bool(c.get("enabled"), True),
            high_impact_only=_safe_bool(
                c.get("high_impact_only"), True
            ),
            buffer_minutes_before=_safe_int(
                c.get("buffer_minutes_before"), 30
            ),
            buffer_minutes_after=_safe_int(
                c.get("buffer_minutes_after"), 30
            ),
            source=_safe_str(c.get("source"), "local_json"),
            calendar_file=_safe_str(
                c.get("calendar_file"),
                "data/economic_calendar.json"
            ),
            max_calendar_age_hours=_safe_int(
                c.get("max_calendar_age_hours"), 24
            ),
        )

    @property
    def backtesting(self) -> BacktestingConfig:
        c = self._raw_config.get("backtesting", {})

        return BacktestingConfig(
            default_start_date=_safe_str(
                c.get("default_start_date"), "2023-01-01"
            ),
            default_end_date=_safe_str(
                c.get("default_end_date"), "2024-12-31"
            ),
            default_balance=_safe_float(
                c.get("default_balance"), 10000.0
            ),
            default_risk_percent=_safe_float(
                c.get("default_risk_percent"), 1.0
            ),
            spread_pips=_safe_float(c.get("spread_pips"), 2.0),
            slippage_pips=_safe_float(c.get("slippage_pips"), 1.0),
            commission_per_lot=_safe_float(
                c.get("commission_per_lot"), 7.0
            ),
            swap_enabled=_safe_bool(c.get("swap_enabled"), True),
            data_source=_safe_str(c.get("data_source"), "mt5"),
        )

    @property
    def dashboard(self) -> DashboardConfig:
        c = self._raw_config.get("dashboard", {})

        return DashboardConfig(
            host=_safe_str(c.get("host"), "127.0.0.1"),
            port=_safe_int(c.get("port"), 8000),
            secret_key=_safe_str(c.get("secret_key")),
            cors_origins=c.get("cors_origins", []),
            websocket_ping_interval=_safe_int(
                c.get("websocket_ping_interval"), 30
            ),
        )

    @property
    def database(self) -> DatabaseConfig:
        c = self._raw_config.get("database", {})

        return DatabaseConfig(
            url=_safe_str(
                c.get("url"), "sqlite:///database/trading.db"
            ),
            echo=_safe_bool(c.get("echo"), False),
            pool_size=_safe_int(c.get("pool_size"), 5),
            max_overflow=_safe_int(c.get("max_overflow"), 10),
        )

    @property
    def logging(self) -> LoggingConfig:
        c = self._raw_config.get("logging", {})

        return LoggingConfig(
            level=_safe_str(c.get("level"), "INFO"),
            file=_safe_str(c.get("file"), "logs/trading.log"),
            max_size_mb=_safe_int(c.get("max_size_mb"), 50),
            backup_count=_safe_int(c.get("backup_count"), 5),
            format=_safe_str(
                c.get("format"),
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            ),
            date_format=_safe_str(
                c.get("date_format"), "%Y-%m-%d %H:%M:%S"
            ),
        )

    @property
    def safety(self) -> SafetyConfig:
        c = self._raw_config.get("safety", {})

        return SafetyConfig(
            demo_only=_safe_bool(c.get("demo_only"), True),
            verify_demo_on_startup=_safe_bool(
                c.get("verify_demo_on_startup"), True
            ),
            kill_switch_enabled=_safe_bool(
                c.get("kill_switch_enabled"), True
            ),
            max_errors_before_pause=_safe_int(
                c.get("max_errors_before_pause"), 10
            ),
            health_check_interval=_safe_int(
                c.get("health_check_interval"), 60
            ),
            trading_mode=self.trading.mode,
        )


# Global configuration instance
config = Config()
