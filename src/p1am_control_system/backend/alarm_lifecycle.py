"""Deterministic alarm lifecycle, shelving, suppression, first-out, and metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from identity import Principal, Role

from shared.python.compatibility import StrEnum


class AlarmPriority(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlarmLifecycle(StrEnum):
    INACTIVE = "inactive"
    UNACKNOWLEDGED = "unacknowledged"
    ACKNOWLEDGED = "acknowledged"
    RETURNED_UNACKNOWLEDGED = "returned_unacknowledged"
    SHELVED = "shelved"
    SUPPRESSED = "suppressed"


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _aware(value: object, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


@dataclass(frozen=True)
class AlarmDefinition:
    tag: str
    low_limit: float
    high_limit: float
    priority: AlarmPriority
    deadband: float
    on_delay: timedelta
    off_delay: timedelta
    help_text: str
    suppression_rules: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tag", _required_text(self.tag, "tag"))
        object.__setattr__(
            self, "help_text", _required_text(self.help_text, "help_text")
        )
        if not isinstance(self.priority, AlarmPriority):
            raise TypeError("priority must be an AlarmPriority")
        for name in ("low_limit", "high_limit", "deadband"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.low_limit >= self.high_limit:
            raise ValueError("low_limit must be below high_limit")
        if self.deadband < 0 or self.deadband * 2 >= self.high_limit - self.low_limit:
            raise ValueError("deadband must be nonnegative and smaller than alarm span")
        for name in ("on_delay", "off_delay"):
            value = getattr(self, name)
            if not isinstance(value, timedelta) or value < timedelta(0):
                raise ValueError(f"{name} must be a nonnegative timedelta")
        rules = frozenset(
            _required_text(rule, "suppression rule") for rule in self.suppression_rules
        )
        object.__setattr__(self, "suppression_rules", rules)


@dataclass(frozen=True)
class AlarmSnapshot:
    tag: str
    priority: AlarmPriority
    lifecycle: AlarmLifecycle
    condition: str
    acknowledged_by: str | None
    shelved_by: str | None
    shelf_reason: str | None
    shelf_until: datetime | None
    suppression_rule: str | None
    first_out_sequence: int | None
    active_since: datetime | None
    help_text: str


@dataclass(frozen=True)
class AlarmPerformanceReport:
    activations: int
    acknowledged_activations: int
    mean_acknowledgement_seconds: float | None


@dataclass
class _Runtime:
    condition: str = "normal"
    pending_condition: str | None = None
    pending_since: datetime | None = None
    lifecycle: AlarmLifecycle = AlarmLifecycle.INACTIVE
    acknowledged_by: str | None = None
    active_since: datetime | None = None
    first_out_sequence: int | None = None
    shelved_by: str | None = None
    shelf_reason: str | None = None
    shelf_until: datetime | None = None
    suppression_rule: str | None = None


class AlarmManager:
    """Own all lifecycle state for a validated set of alarm definitions."""

    def __init__(self, definitions: list[AlarmDefinition]) -> None:
        if not isinstance(definitions, list) or not definitions:
            raise ValueError("definitions must be a non-empty list")
        if not all(isinstance(item, AlarmDefinition) for item in definitions):
            raise TypeError("definitions must contain AlarmDefinition values")
        self._definitions = {item.tag: item for item in definitions}
        if len(self._definitions) != len(definitions):
            raise ValueError("alarm tags must be unique")
        self._runtime = {tag: _Runtime() for tag in self._definitions}
        self._first_out_counter = 0
        self._activations = 0
        self._acknowledgement_times: list[float] = []

    def _definition(self, tag: str) -> AlarmDefinition:
        try:
            return self._definitions[tag]
        except KeyError as exc:
            raise KeyError(f"unknown alarm tag {tag!r}") from exc

    @staticmethod
    def _candidate(definition: AlarmDefinition, runtime: _Runtime, value: float) -> str:
        if (
            runtime.condition == "high"
            and value >= definition.high_limit - definition.deadband
        ):
            return "high"
        if (
            runtime.condition == "low"
            and value <= definition.low_limit + definition.deadband
        ):
            return "low"
        if value >= definition.high_limit:
            return "high"
        if value <= definition.low_limit:
            return "low"
        return "normal"

    def _commit_condition(
        self, runtime: _Runtime, condition: str, now: datetime
    ) -> None:
        prior = runtime.condition
        runtime.condition = condition
        runtime.pending_condition = None
        runtime.pending_since = None
        if prior == "normal" and condition != "normal":
            self._first_out_counter += 1
            self._activations += 1
            runtime.first_out_sequence = self._first_out_counter
            runtime.active_since = now
            runtime.acknowledged_by = None
            runtime.lifecycle = AlarmLifecycle.UNACKNOWLEDGED
        elif prior != "normal" and condition == "normal":
            runtime.lifecycle = (
                AlarmLifecycle.INACTIVE
                if runtime.acknowledged_by is not None
                else AlarmLifecycle.RETURNED_UNACKNOWLEDGED
            )
        elif condition != "normal" and runtime.lifecycle is AlarmLifecycle.INACTIVE:
            runtime.lifecycle = AlarmLifecycle.UNACKNOWLEDGED

    @staticmethod
    def _expire_shelf(runtime: _Runtime, now: datetime) -> None:
        if runtime.shelf_until is not None and now >= runtime.shelf_until:
            runtime.shelved_by = None
            runtime.shelf_reason = None
            runtime.shelf_until = None

    def _snapshot(self, tag: str, now: datetime) -> AlarmSnapshot:
        definition = self._definition(tag)
        runtime = self._runtime[tag]
        self._expire_shelf(runtime, now)
        lifecycle = runtime.lifecycle
        if runtime.suppression_rule is not None:
            lifecycle = AlarmLifecycle.SUPPRESSED
        elif runtime.shelf_until is not None:
            lifecycle = AlarmLifecycle.SHELVED
        return AlarmSnapshot(
            tag=tag,
            priority=definition.priority,
            lifecycle=lifecycle,
            condition=runtime.condition,
            acknowledged_by=runtime.acknowledged_by,
            shelved_by=runtime.shelved_by,
            shelf_reason=runtime.shelf_reason,
            shelf_until=runtime.shelf_until,
            suppression_rule=runtime.suppression_rule,
            first_out_sequence=runtime.first_out_sequence,
            active_since=runtime.active_since,
            help_text=definition.help_text,
        )

    def evaluate(self, tag: str, value: float, now: datetime) -> AlarmSnapshot:
        definition = self._definition(tag)
        runtime = self._runtime[tag]
        stamp = _aware(now, "now")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("alarm value must be finite")
        candidate = self._candidate(definition, runtime, numeric)
        if candidate == runtime.condition:
            runtime.pending_condition = None
            runtime.pending_since = None
            return self._snapshot(tag, stamp)
        if runtime.pending_condition != candidate:
            runtime.pending_condition = candidate
            runtime.pending_since = stamp
        assert runtime.pending_since is not None
        delay = definition.off_delay if candidate == "normal" else definition.on_delay
        if stamp - runtime.pending_since >= delay:
            self._commit_condition(runtime, candidate, stamp)
        return self._snapshot(tag, stamp)

    def acknowledge(
        self, tag: str, principal: Principal, now: datetime
    ) -> AlarmSnapshot:
        if not isinstance(principal, Principal) or not principal.allows(Role.OPERATOR):
            raise PermissionError("alarm acknowledgement requires operator role")
        runtime = self._runtime[self._definition(tag).tag]
        stamp = _aware(now, "now")
        if runtime.lifecycle not in {
            AlarmLifecycle.UNACKNOWLEDGED,
            AlarmLifecycle.RETURNED_UNACKNOWLEDGED,
        }:
            raise ValueError("alarm is not awaiting acknowledgement")
        runtime.acknowledged_by = principal.subject
        if runtime.active_since is not None:
            self._acknowledgement_times.append(
                (stamp - runtime.active_since).total_seconds()
            )
        runtime.lifecycle = (
            AlarmLifecycle.INACTIVE
            if runtime.condition == "normal"
            else AlarmLifecycle.ACKNOWLEDGED
        )
        return self._snapshot(tag, stamp)

    def shelve(
        self,
        tag: str,
        principal: Principal,
        reason: str,
        until: datetime,
        now: datetime,
    ) -> AlarmSnapshot:
        if not isinstance(principal, Principal) or not principal.allows(Role.OPERATOR):
            raise PermissionError("alarm shelving requires operator role")
        stamp = _aware(now, "now")
        expiry = _aware(until, "until")
        if expiry <= stamp or expiry - stamp > timedelta(days=1):
            raise ValueError("shelf expiry must be within the next day")
        runtime = self._runtime[self._definition(tag).tag]
        runtime.shelved_by = principal.subject
        runtime.shelf_reason = _required_text(reason, "reason")
        runtime.shelf_until = expiry
        return self._snapshot(tag, stamp)

    def set_suppression(
        self,
        tag: str,
        rule: str,
        active: bool,
        now: datetime,
    ) -> AlarmSnapshot:
        definition = self._definition(tag)
        rule_id = _required_text(rule, "suppression rule")
        if rule_id not in definition.suppression_rules:
            raise ValueError("suppression rule is not designed for this alarm")
        if not isinstance(active, bool):
            raise TypeError("active must be a bool")
        runtime = self._runtime[tag]
        runtime.suppression_rule = rule_id if active else None
        return self._snapshot(tag, _aware(now, "now"))

    def unshelve(
        self,
        tag: str,
        principal: Principal,
        now: datetime,
    ) -> AlarmSnapshot:
        if not isinstance(principal, Principal) or not principal.allows(Role.OPERATOR):
            raise PermissionError("alarm unshelving requires operator role")
        runtime = self._runtime[self._definition(tag).tag]
        runtime.shelved_by = None
        runtime.shelf_reason = None
        runtime.shelf_until = None
        return self._snapshot(tag, _aware(now, "now"))

    def snapshot(self, tag: str, now: datetime) -> AlarmSnapshot:
        return self._snapshot(tag, _aware(now, "now"))

    def active_snapshots(self, now: datetime) -> list[AlarmSnapshot]:
        stamp = _aware(now, "now")
        snapshots = [self._snapshot(tag, stamp) for tag in self._definitions]
        active = [
            item for item in snapshots if item.lifecycle is not AlarmLifecycle.INACTIVE
        ]
        return sorted(active, key=lambda item: item.first_out_sequence or math.inf)

    def performance_report(self) -> AlarmPerformanceReport:
        count = len(self._acknowledgement_times)
        mean = sum(self._acknowledgement_times) / count if count else None
        return AlarmPerformanceReport(self._activations, count, mean)
