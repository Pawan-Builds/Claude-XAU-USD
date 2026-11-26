"""Central fail-closed entry permission gate.

This is the single place every new-entry decision must pass through.
No caller may bypass pause, shutdown, account verification, kill switch,
or risk limits. If any check cannot be evaluated, it fails closed.

`check()` honours the pause flag; use before every order submission.
`probe()` ignores the pause flag; use only from startup, health-check,
and resume paths.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Optional
from src.trade_manager import trade_manager
from src.config_loader import config
from src.database_manager import db_manager, ClosedTrade, ExecutedOrder
from src.market_data import market_data
from src.mt5_connector import mt5_connector
from src.news_filter import news_filter
from src.risk_manager import risk_manager
from src.session_filter import session_filter
from src.logger import get_logger
from src.gate_token import issue as _issue_gate_token
logger = get_logger(__name__)


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reason: str
    checks: Dict[str, bool] = field(default_factory=dict)
    token: str = ""

    def __bool__(self) -> bool:
        return self.ok


class EntryGate:
    """Central, fail-closed entry permission gate."""

    def __init__(
        self,
        is_paused_getter: Optional[Callable[[], bool]] = None,
        is_shutdown_getter: Optional[Callable[[], bool]] = None,
    ):
        self._is_paused_getter: Optional[Callable[[], bool]] = is_paused_getter
        self._is_shutdown_getter: Optional[Callable[[], bool]] = is_shutdown_getter

    @property
    def _configured(self) -> bool:
        """Derived, not stored: cannot drift out of sync with the getters."""
        return (
            self._is_paused_getter is not None
            and self._is_shutdown_getter is not None
        )

    def set_paused_getter(self, getter: Callable[[], bool]) -> None:
        self._is_paused_getter = getter

    def set_shutdown_getter(self, getter: Callable[[], bool]) -> None:
        self._is_shutdown_getter = getter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> GateResult:
        """Full gate: honour pause. Use before every order submission."""
        return self._run(require_unpaused=True)

    def probe(self) -> GateResult:
        """Probe that ignores the pause flag (startup, health, resume).

        Every other check still applies; still fails closed.
        """
        return self._run(require_unpaused=False)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self, require_unpaused: bool) -> GateResult:
        if not self._configured:
            return GateResult(
                ok=False,
                reason=(
                    "EntryGate is not configured with pause/shutdown "
                    "getters. Refusing to authorize any entry."
                ),
                checks={"configured": False},
            )

        checks: Dict[str, bool] = {"configured": True}

        def fail(name: str, reason: str) -> GateResult:
            checks[name] = False
            logger.debug("Entry gate blocked | %s | %s", name, reason)
            return GateResult(
                ok=False, reason=reason, checks=dict(checks), token="",
            )

        def step(name: str) -> None:
            checks[name] = True

        # 1. Shutdown
        try:
            if self._is_shutdown_getter():
                return fail("shutdown", "Shutdown requested.")
        except Exception:
            logger.exception("Shutdown-state lookup failed.")
            return fail("shutdown", "Shutdown state unavailable.")
        step("shutdown")

        # 2. Paused (bypassable only by probe())
        if require_unpaused:
            try:
                if self._is_paused_getter():
                    return fail("paused", "Trading is paused.")
            except Exception:
                logger.exception("Paused-state lookup failed.")
                return fail("paused", "Paused state unavailable.")
        step("paused")

        # 3. MT5 connected + account refresh + environment match
        account = None
        try:
            if not mt5_connector.is_connected():
                return fail("mt5_connected", "MT5 is not connected.")
            if not mt5_connector.update_account_info():
                return fail("mt5_connected", "MT5 account refresh failed.")
            account = mt5_connector.get_account_info()
            if account is None:
                return fail(
                    "account_verified",
                    "MT5 account information unavailable.",
                )

            expected_mode = str(config.trading.mode).strip().upper()
            actual_mode = str(account.trade_mode).strip().upper()

            if expected_mode == "DEMO":
                if actual_mode != "DEMO" or not account.is_demo:
                    return fail(
                        "account_verified",
                        f"Mode DEMO but connected account is {actual_mode}.",
                    )
            elif expected_mode == "LIVE":
                if actual_mode != "REAL" or account.is_demo:
                    return fail(
                        "account_verified",
                        f"Mode LIVE but connected account is {actual_mode}.",
                    )
            else:
                return fail(
                    "account_verified",
                    f"Unsupported trading mode {expected_mode!r}.",
                )
        except Exception:
            logger.exception("Account verification failed in gate.")
            return fail("account_verified", "Account verification error.")
        step("mt5_connected")
        step("account_verified")

        # 4. Broker trading permissions
        try:
            if not bool(account.trade_allowed):
                return fail("trade_allowed", "Account trading is disabled.")
            if not bool(account.trade_expert):
                return fail("trade_expert", "Expert trading is disabled.")
        except Exception:
            logger.exception("Broker permission check failed.")
            return fail("trade_expert", "Broker permission check failed.")
        step("trade_allowed")
        step("trade_expert")

        # 5. Kill switch (direct read, not via check_risk_limits)
        try:
            if risk_manager.is_kill_switch_active():
                return fail("kill_switch", "Kill switch is active.")
        except Exception:
            logger.exception("Kill switch check failed.")
            return fail("kill_switch", "Kill switch state unavailable.")
        step("kill_switch")
        try:
            unresolved = trade_manager.get_unresolved_tickets_strict()
        except Exception:
            logger.exception(
                "Reconciliation state lookup raised in gate."
            )
            return fail(
                "reconciliation",
                "Reconciliation state unavailable.",
            )
        if unresolved is None:
            return fail(
                "reconciliation",
                "Reconciliation state unknown (fail-closed).",
            )
        if unresolved:
            return fail(
                "reconciliation",
                f"Unresolved broker reconciliation: "
                f"tickets {unresolved}.",
            )
        step("reconciliation")

        # 6. Trading session (timezone-aware, weekend-safe)
        try:
            if not session_filter.is_trading_allowed():
                return fail(
                    "session",
                    "Outside configured trading sessions.",
                )
        except Exception:
            logger.exception("Session filter raised.")
            return fail("session", "Session check failed.")
        step("session")

        # 7. Weekend protection
        try:
            if market_data.is_weekend_protection_active():
                return fail("weekend", "Weekend protection is active.")
        except Exception:
            logger.exception("Weekend protection check raised.")
            return fail("weekend", "Weekend check failed.")
        step("weekend")

        # 8a. Market-data reliability (M-10): fresh tick plus fresh
        # closed candles for every required timeframe. Fail-closed:
        # a stale cache blocks new entries rather than letting the
        # strategy act on stale data. Position management and
        # emergency paths do not call the gate, so this cannot block
        # them.
        try:
            data_ok, data_reason = market_data.is_market_data_reliable()
        except Exception:
            logger.exception("Market-data reliability check raised.")
            return fail(
                "market_data",
                "Market-data reliability check failed.",
            )
        if not data_ok:
            return fail(
                "market_data",
                f"Market data unreliable: {data_reason}",
            )
        step("market_data")

        # 8b. Spread (M-10): missing, zero, unavailable, stale, or
        # above-max spread must fail closed. get_spread_status() is
        # the M-05 disambiguator — is_high_spread() collapses all
        # three failure modes into "high", which hides the root cause
        # from the operator and permits a genuine-zero spread to
        # pass. New entries require a strictly-positive, verified,
        # within-max spread; anything else is refused.
        try:
            spread_pips, spread_status = market_data.get_spread_status()
        except Exception:
            logger.exception("Spread status lookup raised.")
            return fail(
                "spread",
                "Spread data unavailable (lookup error).",
            )

        if spread_status != "ok":
            if spread_status == "stale":
                return fail("spread", "Spread data is stale.")
            return fail("spread", "Spread data unavailable.")

        if spread_pips is None:
            return fail("spread", "Spread value missing.")

        if spread_pips <= 0.0:
            return fail(
                "spread",
                f"Spread is {spread_pips:.2f} pips (zero or "
                "negative; no verifiable market quote).",
            )

        max_spread = float(config.trade_management.max_spread_pips)
        if spread_pips > max_spread:
            return fail(
                "spread",
                f"Spread {spread_pips:.2f} pips exceeds maximum "
                f"{max_spread:.2f} pips.",
            )
        step("spread")

        # 9. News filter (fails closed when enabled and data unavailable)
        try:
            result = news_filter.should_block_trading()
            if not isinstance(result, dict):
                return fail(
                    "news", "News filter returned an invalid result."
                )
            if result.get("block", True):
                return fail(
                    "news",
                    str(result.get("reason", "News filter blocked trading.")),
                )
        except Exception:
            logger.exception("News filter raised.")
            return fail("news", "News filter failed closed.")
        step("news")

        # 10. Risk manager limits (daily loss, drawdown, consecutive losses,
        #     max positions, total open risk, emergency).
        try:
            result = risk_manager.check_risk_limits()
            if not isinstance(result, dict):
                return fail(
                    "risk", "Risk manager returned an invalid result."
                )
            if not result.get("ok", False):
                return fail(
                    "risk",
                    "Risk limits failed: "
                    f"{result.get('checks') or result.get('reason')}",
                )
        except Exception:
            logger.exception("Risk manager raised.")
            return fail("risk", "Risk check failed.")
        step("risk")

        # 11. Daily trade limit
        try:
            today_start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            env = str(config.trading.mode).lower()
            with db_manager.get_session() as session:
                closed_count = (
                    session.query(ClosedTrade)
                    .filter(
                        ClosedTrade.environment == env,
                        ClosedTrade.time_close >= today_start,
                    )
                    .count()
                )
                open_count = (
                    session.query(ExecutedOrder)
                    .filter(
                        ExecutedOrder.environment == env,
                        ExecutedOrder.time_close.is_(None),
                        ExecutedOrder.time_open >= today_start,
                    )
                    .count()
                )
            total_trades = int(closed_count) + int(open_count)
            limit = int(config.trading.max_daily_trades)
            if total_trades >= limit:
                return fail(
                    "daily_trades",
                    f"Daily trade limit reached ({total_trades}/{limit}).",
                )
        except Exception:
            logger.exception("Daily trade limit check failed.")
            return fail("daily_trades", "Daily trade limit check failed.")
        step("daily_trades")

        logger.debug("Entry gate passed all checks.")
        return GateResult(
            ok=True,
            reason="OK",
            checks=dict(checks),
            token=_issue_gate_token(),
        )




# Global gate instance. TradingAgent overrides the paused/shutdown getters
# at construction time so the gate reflects live agent state.
entry_gate = EntryGate()