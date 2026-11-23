
"""Timezone-aware trading session filter."""

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.config_loader import config
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SessionState:
    """Current trading-session state."""

    is_london: bool
    is_new_york: bool
    is_asian: bool
    is_overlap: bool
    active_session: str
    next_session_change: Optional[datetime]
    london_open: datetime
    london_close: datetime
    ny_open: datetime
    ny_close: datetime
    asian_open: datetime
    asian_close: datetime


class SessionFilter:
    """Evaluates configured trading sessions in their configured timezone."""

    SESSION_NAMES = ("london", "new_york", "asian")

    def __init__(self):
        self.sessions = config.sessions
        self.timezone_name = str(
            self._get(self.sessions, "timezone", "UTC")
        )

        try:
            self.tz = ZoneInfo(self.timezone_name)
        except (ZoneInfoNotFoundError, ValueError):
            logger.exception(
                "Invalid session timezone %r. Falling back to UTC.",
                self.timezone_name
            )
            self.timezone_name = "UTC"
            self.tz = timezone.utc

        self._schedule = {}

        for name in self.SESSION_NAMES:
            section = self._get(self.sessions, name, {})
            enabled = self._as_bool(
                self._get(section, "enabled", name != "asian")
            )
            open_time = self._parse_time(
                self._get(section, "open", "00:00")
            )
            close_time = self._parse_time(
                self._get(section, "close", "00:00")
            )

            self._schedule[name] = {
                "enabled": enabled,
                "open": open_time,
                "close": close_time,
            }

    @staticmethod
    def _get(source: Any, key: str, default: Any = None) -> Any:
        """Read a setting from either a dictionary or an object."""
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)

    @staticmethod
    def _parse_time(value: Any) -> time:
        """Parse a session time in HH:MM format."""
        if isinstance(value, time):
            return value.replace(tzinfo=None)

        if not isinstance(value, str):
            raise ValueError(f"Session time must be HH:MM, got {value!r}")

        try:
            return datetime.strptime(value.strip(), "%H:%M").time()
        except ValueError as exc:
            raise ValueError(
                f"Invalid session time {value!r}; expected HH:MM."
            ) from exc

    def _session_datetimes(self, name: str, date):
        """Return the session's actual local opening and closing datetimes."""
        item = self._schedule[name]

        opening = datetime.combine(
            date,
            item["open"],
            tzinfo=self.tz
        )

        closing = datetime.combine(
            date,
            item["close"],
            tzinfo=self.tz
        )

        # Equal open/close times are treated as a disabled-length session,
        # not as a 24-hour session.
        if closing <= opening:
            closing += timedelta(days=1)

        return opening, closing

    def _is_active(self, name: str, now: datetime) -> bool:
        """Check today's and yesterday's session windows."""
        item = self._schedule[name]

        if not item["enabled"]:
            return False

        for offset in (-1, 0):
            session_date = (now + timedelta(days=offset)).date()
            opening, closing = self._session_datetimes(
                name,
                session_date
            )

            if opening <= now < closing:
                return True

        return False

    def _next_session_change(self, now: datetime) -> Optional[datetime]:
        """Find the next enabled session opening or closing boundary."""
        boundaries = []

        for name in self.SESSION_NAMES:
            if not self._schedule[name]["enabled"]:
                continue

            for offset in range(-1, 9):
                date = (now + timedelta(days=offset)).date()
                opening, closing = self._session_datetimes(name, date)

                if opening > now:
                    boundaries.append(opening)

                if closing > now:
                    boundaries.append(closing)

        return min(boundaries) if boundaries else None

    def get_current_session(self) -> SessionState:
        """Return the current session state and next boundary."""
        now = datetime.now(self.tz)

        is_london = self._is_active("london", now)
        is_new_york = self._is_active("new_york", now)
        is_asian = self._is_active("asian", now)

        is_overlap = is_london and is_new_york

        if is_overlap:
            active = "london_new_york_overlap"
        elif is_london:
            active = "london"
        elif is_new_york:
            active = "new_york"
        elif is_asian:
            active = "asian"
        else:
            active = "closed"

        today = now.date()

        london_open, london_close = self._session_datetimes(
            "london", today
        )
        ny_open, ny_close = self._session_datetimes(
            "new_york", today
        )
        asian_open, asian_close = self._session_datetimes(
            "asian", today
        )

        return SessionState(
            is_london=is_london,
            is_new_york=is_new_york,
            is_asian=is_asian,
            is_overlap=is_overlap,
            active_session=active,
            next_session_change=self._next_session_change(now),
            london_open=london_open,
            london_close=london_close,
            ny_open=ny_open,
            ny_close=ny_close,
            asian_open=asian_open,
            asian_close=asian_close,
        )

    def is_trading_allowed(self) -> bool:
        """Return whether at least one enabled session is active."""
        state = self.get_current_session()
        return state.is_london or state.is_new_york or state.is_asian

    def is_high_liquidity_session(self) -> bool:
        """Return whether London and New York sessions overlap."""
        return self.get_current_session().is_overlap

    def get_session_info(self) -> Dict[str, Any]:
        """Return session information for the dashboard."""
        state = self.get_current_session()
        now_utc = datetime.now(timezone.utc)

        result = {
            "current_time_utc": now_utc.strftime("%H:%M:%S"),
            "current_time_local": datetime.now(self.tz).isoformat(),
            "timezone": self.timezone_name,
            "active_session": state.active_session,
            "is_london": state.is_london,
            "is_new_york": state.is_new_york,
            "is_asian": state.is_asian,
            "is_overlap": state.is_overlap,
            "trading_allowed": (
                state.is_london or state.is_new_york or state.is_asian
            ),
            "high_liquidity": state.is_overlap,
            "next_change": (
                state.next_session_change.isoformat()
                if state.next_session_change else None
            ),
        }

        for name in self.SESSION_NAMES:
            item = self._schedule[name]
            active = getattr(
                state,
                {
                    "london": "is_london",
                    "new_york": "is_new_york",
                    "asian": "is_asian",
                }[name]
            )

            result[name] = {
                "enabled": item["enabled"],
                "open": item["open"].strftime("%H:%M"),
                "close": item["close"].strftime("%H:%M"),
                "active": active,
            }

        return result


# Global session filter instance
session_filter = SessionFilter()
