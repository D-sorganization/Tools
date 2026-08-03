"""Thread-safe application boundary for the professional alarm manager."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from alarm_lifecycle import (
    AlarmDefinition,
    AlarmManager,
    AlarmPerformanceReport,
    AlarmPriority,
    AlarmSnapshot,
)
from identity import Principal
from models import RoutingConfig

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017


def manager_from_routing(config: RoutingConfig) -> AlarmManager:
    """Adapt generic four-tier routing limits into supervisory lifecycle alarms."""
    if not isinstance(config, RoutingConfig):
        raise TypeError("config must be a RoutingConfig")
    definitions: list[AlarmDefinition] = []
    for tag, limits in config.interlocks.items():
        ordered = (
            limits.lolo_limit < limits.low_limit < limits.high_limit < limits.hihi_limit
        )
        if not ordered:
            raise ValueError(f"alarm limits for {tag!r} must be strictly ordered")
        span = limits.high_limit - limits.low_limit
        definitions.append(
            AlarmDefinition(
                tag=tag,
                low_limit=limits.low_limit,
                high_limit=limits.high_limit,
                priority=AlarmPriority.HIGH,
                deadband=span * 0.01,
                on_delay=timedelta(seconds=1),
                off_delay=timedelta(seconds=1),
                help_text=("Review signal quality and the generic process context."),
                suppression_rules=frozenset({"synthetic.maintenance"}),
            )
        )
    return AlarmManager(definitions)


class AlarmService:
    """Serialize poll and API access through a narrow manager interface."""

    def __init__(
        self,
        manager: AlarmManager,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(manager, AlarmManager):
            raise TypeError("manager must be an AlarmManager")
        self._manager = manager
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()

    def reconfigure(self, manager: AlarmManager) -> None:
        """Atomically replace definitions; protected activation is handled by F04."""
        if not isinstance(manager, AlarmManager):
            raise TypeError("manager must be an AlarmManager")
        with self._lock:
            self._manager = manager

    def _now(self) -> datetime:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError("clock must return an aware datetime")
        return now

    def observe(self, values: dict[str, float], now: datetime | None = None) -> None:
        if not isinstance(values, dict):
            raise TypeError("values must be a dict")
        stamp = now or self._now()
        with self._lock:
            for tag, value in values.items():
                try:
                    self._manager.evaluate(tag, value, stamp)
                except KeyError:
                    continue

    def active(self) -> list[AlarmSnapshot]:
        with self._lock:
            snapshots: list[AlarmSnapshot] = self._manager.active_snapshots(self._now())
            return snapshots

    def acknowledge(self, tag: str, principal: Principal) -> AlarmSnapshot:
        with self._lock:
            return self._manager.acknowledge(tag, principal, self._now())

    def shelve(
        self,
        tag: str,
        principal: Principal,
        reason: str,
        duration: timedelta,
    ) -> AlarmSnapshot:
        if not isinstance(duration, timedelta) or duration <= timedelta(0):
            raise ValueError("duration must be a positive timedelta")
        now = self._now()
        with self._lock:
            return self._manager.shelve(tag, principal, reason, now + duration, now)

    def unshelve(self, tag: str, principal: Principal) -> AlarmSnapshot:
        with self._lock:
            return self._manager.unshelve(tag, principal, self._now())

    def suppress(self, tag: str, rule: str, active: bool) -> AlarmSnapshot:
        with self._lock:
            return self._manager.set_suppression(tag, rule, active, self._now())

    def performance(self) -> AlarmPerformanceReport:
        with self._lock:
            return self._manager.performance_report()
