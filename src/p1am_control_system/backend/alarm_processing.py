"""Alarm-event processing for the poll loop.

Feeds a scan's tag values through the (Rust) alarm engine, folds the resulting
state transitions into the live ``active_alarms`` map, and returns the EventLog
rows to persist. Pulling this out of the poll loop makes it a pure, unit-testable
function (no DB, no network) and removes a tangled nested block from main.py.

LOD: imports only the ORM model. The alarm engine is duck-typed (anything with
``update_tag(name, value) -> list[dict]``), so tests pass a tiny fake.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from models import EventLog

try:
    from datetime import UTC
except ImportError:  # Python 3.10 — repo supports 3.10+
    UTC = timezone.utc  # noqa: UP017

# Alarm states that map to each severity (single source of truth — DRY).
_SEVERITY_BY_STATE: dict[str, int] = {
    "Low": 1,
    "High": 1,
    "LoLo": 2,
    "HiHi": 2,
}


def severity_for_state(state: str) -> int:
    """Map an alarm state name to a severity (0 normal, 1 Lo/Hi, 2 LoLo/HiHi)."""
    return _SEVERITY_BY_STATE.get(state, 0)


def process_alarm_events(
    alarm_engine: Any,
    tags: dict[str, float],
    active_alarms: dict[str, dict[str, Any]],
    *,
    now: datetime | None = None,
) -> list[EventLog]:
    """Update ``active_alarms`` in place from this scan and return EventLog rows.

    Args:
        alarm_engine: Object exposing ``update_tag(name, value) -> list[dict]``;
            each event dict carries a ``current_state`` (e.g. ``State.LoLo``).
        tags: Mapping of tag name -> value for this scan.
        active_alarms: Live alarm map, keyed by tag; mutated in place. A tag
            returning to Normal is dropped once acknowledged, otherwise marked
            Normal so the operator still sees it cleared.
        now: Timestamp for new-alarm records; defaults to now (UTC).

    Returns:
        EventLog rows for every transition this scan (caller persists them).

    Raises:
        TypeError: If ``tags`` or ``active_alarms`` is not a dict, or
            ``alarm_engine`` has no ``update_tag``.
    """
    if not isinstance(tags, dict):
        raise TypeError(f"tags must be a dict, got {type(tags).__name__}")
    if not isinstance(active_alarms, dict):
        raise TypeError(
            f"active_alarms must be a dict, got {type(active_alarms).__name__}"
        )
    if not callable(getattr(alarm_engine, "update_tag", None)):
        raise TypeError("alarm_engine must expose a callable update_tag(name, value)")

    stamp = (now if now is not None else datetime.now(UTC)).isoformat()
    events: list[EventLog] = []

    for tag_name, value in tags.items():
        for ev in alarm_engine.update_tag(tag_name, value):
            state = str(ev["current_state"]).split(".")[-1]
            sev = severity_for_state(state)

            if state == "Normal":
                existing = active_alarms.get(tag_name)
                if existing is not None:
                    if existing["acknowledged"]:
                        del active_alarms[tag_name]
                    else:
                        existing["state"] = "Normal"
            else:
                active_alarms[tag_name] = {
                    "tag_id": tag_name,
                    "tag_name": tag_name,
                    "state": state,
                    "severity": sev,
                    "acknowledged": False,
                    "timestamp": stamp,
                }

            events.append(
                EventLog(
                    event_type="ALARM",
                    description=(
                        f"Tag {tag_name} crossed limit. State: {state} Value: {value}"
                    ),
                    severity=sev,
                )
            )

    return events
