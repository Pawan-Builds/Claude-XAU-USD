
"""Fail-closed economic news filter.

Reads economic calendar events from data/economic_calendar.json.
A real calendar provider must populate this file. Missing or stale
calendar data blocks new trades while the news filter is enabled.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config_loader import config
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NewsEvent:
    """Normalized economic calendar event."""

    timestamp: datetime
    currency: str
    event: str
    impact: str
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None


class NewsFilter:
    """Blocks trading around configured high-impact news events."""

    def __init__(self):
        self.news_config = config.news
        self._cached_events: List[NewsEvent] = []
        self._last_fetch: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=5)

        # The file can be overridden in settings.yaml with calendar_file.
        configured_path = getattr(
            self.news_config,
            "calendar_file",
            "data/economic_calendar.json"
        )

        self.calendar_path = Path(configured_path)

        if not self.calendar_path.is_absolute():
            self.calendar_path = Path.cwd() / self.calendar_path

        # Require a recently updated calendar file.
        self.max_file_age = timedelta(
            hours=int(
                getattr(self.news_config, "max_calendar_age_hours", 24)
            )
        )

    def is_news_filter_active(self) -> bool:
        return bool(getattr(self.news_config, "enabled", False))

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        """Parse ISO-8601 timestamps and normalize them to UTC."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Event timestamp is missing or invalid.")

        normalized = value.strip().replace("Z", "+00:00")

        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Invalid event timestamp: {value!r}"
            ) from exc

        # Naive timestamps are rejected because their timezone is unknown.
        if parsed.tzinfo is None:
            raise ValueError(
                f"Event timestamp must include a timezone: {value!r}"
            )

        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _optional_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        return str(value)

    def _load_calendar_file(self) -> List[NewsEvent]:
        """Load and validate the local calendar JSON file."""
        if not self.calendar_path.is_file():
            raise FileNotFoundError(
                f"Economic calendar file not found: {self.calendar_path}"
            )

        modified_at = datetime.fromtimestamp(
            self.calendar_path.stat().st_mtime,
            tz=timezone.utc
        )

        age = datetime.now(timezone.utc) - modified_at

        if age < timedelta(seconds=0) or age > self.max_file_age:
            raise ValueError(
                "Economic calendar file is stale or has an invalid "
                f"modification time. File age: {age}."
            )

        with self.calendar_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            raise ValueError(
                "Calendar JSON root must be an object."
            )

        events_data = payload.get("events")

        if not isinstance(events_data, list):
            raise ValueError(
                "Calendar JSON must contain an 'events' list."
            )

        if not events_data:
            raise ValueError(
                "Calendar contains no events; news protection "
                "cannot be verified."
            )

        events: List[NewsEvent] = []

        for index, item in enumerate(events_data):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Calendar event at index {index} is not an object."
                )

            timestamp = self._parse_timestamp(
                item.get("timestamp")
            )

            currency = str(
                item.get("currency", "")
            ).strip().upper()

            event_name = str(
                item.get("event", "")
            ).strip()

            impact = str(
                item.get("impact", "")
            ).strip().lower()

            if not currency or not event_name:
                raise ValueError(
                    f"Calendar event at index {index} "
                    "is missing currency or event name."
                )

            if impact not in ("high", "medium", "low"):
                raise ValueError(
                    f"Invalid impact at event index {index}: {impact!r}"
                )

            events.append(
                NewsEvent(
                    timestamp=timestamp,
                    currency=currency,
                    event=event_name,
                    impact=impact,
                    actual=self._optional_string(item.get("actual")),
                    forecast=self._optional_string(item.get("forecast")),
                    previous=self._optional_string(item.get("previous")),
                )
            )

        if bool(getattr(self.news_config, "high_impact_only", True)):
            events = [
                event for event in events
                if event.impact == "high"
            ]

        return events

    def fetch_economic_calendar(self) -> List[NewsEvent]:
        """Load cached calendar events or refresh from the JSON file.

        This method does not scrape websites or claim to connect to
        ForexFactory or Investing.com.
        """
        if not self.is_news_filter_active():
            return []

        now = datetime.now(timezone.utc)

        if (
            self._last_fetch is not None
            and now - self._last_fetch < self._cache_duration
        ):
            return list(self._cached_events)

        try:
            events = self._load_calendar_file()

            self._cached_events = events
            self._last_fetch = now

            logger.info(
                "Loaded %d economic calendar events from %s.",
                len(events),
                self.calendar_path
            )

            return list(events)

        except Exception:
            # Never turn missing or invalid calendar data into permission
            # to trade. Clear the cache so stale events are not trusted.
            self._cached_events = []
            self._last_fetch = None

            logger.exception(
                "Economic calendar unavailable. News protection "
                "cannot be verified."
            )

            raise

    def check_news_proximity(
        self,
        buffer_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Check whether relevant news falls inside the buffer window."""
        if not self.is_news_filter_active():
            return {
                "blocked": False,
                "reason": "News filter disabled.",
                "data_available": True,
                "events": [],
            }

        try:
            events = self.fetch_economic_calendar()
        except Exception as exc:
            return {
                "blocked": True,
                "reason": (
                    "News filter fail-closed: calendar data unavailable. "
                    f"{exc}"
                ),
                "data_available": False,
                "events": [],
            }

        now = datetime.now(timezone.utc)

        before_buffer = (
            int(buffer_minutes)
            if buffer_minutes is not None
            else int(self.news_config.buffer_minutes_before)
        )

        after_buffer = int(
            self.news_config.buffer_minutes_after
        )

        if before_buffer < 0 or after_buffer < 0:
            return {
                "blocked": True,
                "reason": "News buffer configuration is invalid.",
                "data_available": False,
                "events": [],
            }

        upcoming_events = []

        for event in events:
            # Gold is particularly sensitive to USD macroeconomic events.
            # Other currencies are not automatically treated as relevant.
            if event.currency != "USD":
                continue

            minutes_until = (
                event.timestamp - now
            ).total_seconds() / 60.0

            if -after_buffer <= minutes_until <= before_buffer:
                upcoming_events.append({
                    "event": event.event,
                    "currency": event.currency,
                    "impact": event.impact,
                    "time_utc": event.timestamp.isoformat(),
                    "minutes_until": round(minutes_until),
                    "forecast": event.forecast,
                    "previous": event.previous,
                    "actual": event.actual,
                })

        blocked = bool(upcoming_events)

        return {
            "blocked": blocked,
            "reason": (
                f"{len(upcoming_events)} relevant high-impact "
                "event(s) inside the news buffer."
                if blocked
                else "No relevant high-impact events inside the news buffer."
            ),
            "events": upcoming_events,
            "data_available": True,
            "buffer_before_minutes": before_buffer,
            "buffer_after_minutes": after_buffer,
        }

    def get_upcoming_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Return relevant upcoming events within the requested horizon."""
        if hours < 0:
            raise ValueError("hours must not be negative.")

        if not self.is_news_filter_active():
            return []

        events = self.fetch_economic_calendar()

        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours)

        upcoming = []

        for event in events:
            if (
                event.currency == "USD"
                and now <= event.timestamp <= cutoff
            ):
                upcoming.append({
                    "event": event.event,
                    "currency": event.currency,
                    "impact": event.impact,
                    "time_utc": event.timestamp.isoformat(),
                    "forecast": event.forecast,
                    "previous": event.previous,
                    "actual": event.actual,
                })

        return upcoming

    def should_block_trading(self) -> Dict[str, Any]:
        """Return an explicit trading decision."""
        result = self.check_news_proximity()

        return {
            "block": bool(result["blocked"]),
            "reason": result["reason"],
            "events": result.get("events", []),
            "data_available": bool(result.get("data_available", False)),
        }


# Global news filter instance
news_filter = NewsFilter()
