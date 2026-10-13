"""Per-tag alarm priority (SCADA F03, issue #5526).

The alarm engines (Rust ``tools_core.scada`` and the Python fallback) own the
BASE severity tier of a state: 0 normal, 1 Lo/Hi, 2 LoLo/HiHi/BadQuality. An
operator may re-rank a tag by giving its interlock an explicit ``alarm_priority``;
the override is applied here, ABOVE the engine, so both engines behave
identically and the Rust crate needs no change.

Contract:

* ``None`` / absent override => the default state->severity map, unchanged.
* A configured override must be ``WARNING`` (1) or ``TRIP`` (2), given as the
  int, the (case-insensitive) name, or the enum. ``0`` is rejected: severity 0
  means "Normal" everywhere downstream, so an *active* alarm must never carry it.
* ``Normal`` is always severity 0; an override cannot manufacture an alarm.
* ``BadQuality`` is a safety floor: always trip severity, whatever the override.
* FAIL CLOSED: an override that cannot be read at resolve time (it passed
  validation at the boundary, so this means corruption) yields trip severity
  (``FAIL_CLOSED_SEVERITY``), the most conservative tier, never a quieter one.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Any

logger = logging.getLogger("dcs_backend.alarm_priority")

__all__ = [
    "BAD_QUALITY_STATE",
    "DEFAULT_SEVERITY_BY_STATE",
    "FAIL_CLOSED_SEVERITY",
    "AlarmPriority",
    "parse_alarm_priority",
    "resolve_severity",
]

#: The state both engines emit for a NaN/Inf reading. Never "Normal".
BAD_QUALITY_STATE = "BadQuality"


class AlarmPriority(IntEnum):
    """Explicit alarm tier an operator may assign to a tag."""

    WARNING = 1
    TRIP = 2


#: Severity returned when a stored override cannot be interpreted: most
#: conservative documented tier.
FAIL_CLOSED_SEVERITY: int = int(AlarmPriority.TRIP)

#: Default severity per alarm state (single source of truth - DRY). Mirrors the
#: Rust ``AlarmState::severity`` and ``scada_fallback._SEVERITY``.
DEFAULT_SEVERITY_BY_STATE: dict[str, int] = {
    "Low": 1,
    "High": 1,
    "LoLo": 2,
    "HiHi": 2,
    # Non-finite reading (sensor/register fault). Ranks with the trip tier: an
    # alarmed tag whose value cannot be read cannot be shown to be safe (#3973).
    BAD_QUALITY_STATE: 2,
}


def parse_alarm_priority(raw: Any) -> AlarmPriority | None:
    """Validate a configured priority at the config boundary.

    Args:
        raw: ``None`` (no override), an ``AlarmPriority``, ``1``/``2``, or
            ``"warning"``/``"trip"`` in any case.

    Returns:
        The validated priority, or ``None`` when no override was given.

    Raises:
        ValueError: If ``raw`` is anything else (including ``0``, ``bool``,
            floats and unknown names), naming the accepted values.
    """
    if raw is None:
        return None
    if isinstance(raw, AlarmPriority):
        return raw
    parsed: AlarmPriority | None = None
    if isinstance(raw, str):
        parsed = AlarmPriority.__members__.get(raw.strip().upper())
    elif isinstance(raw, int) and not isinstance(raw, bool):
        parsed = next((p for p in AlarmPriority if p.value == raw), None)
    if parsed is None:
        raise ValueError(
            f"alarm_priority must be one of 1/'warning' or 2/'trip' "
            f"(0 is reserved for Normal), got {raw!r}"
        )
    return parsed


def resolve_severity(state: str, priority: AlarmPriority | None = None) -> int:
    """Severity of ``state`` for a tag with an optional ``priority`` override."""
    base = DEFAULT_SEVERITY_BY_STATE.get(state, 0)
    if priority is None or base == 0 or state == BAD_QUALITY_STATE:
        return base
    try:
        return int(AlarmPriority(priority))
    except ValueError:
        logger.error(
            "Unreadable alarm priority %r; failing closed to severity %d",
            priority,
            FAIL_CLOSED_SEVERITY,
        )
        return FAIL_CLOSED_SEVERITY
