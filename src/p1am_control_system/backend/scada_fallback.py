"""Pure-Python fallback for the ``tools_core.scada`` Rust extension.

The P1AM/DCS backend prefers the Rust-accelerated ``tools_core.scada`` module
for its SCADA alarm engine and signal-smoothing helpers. That module is a
PyO3 extension shipped as a compiled wheel; when the wheel is not installed
(e.g. a fresh checkout, a developer environment without the Rust toolchain, or
a slim deployment image) ``from tools_core import scada`` raises
``ModuleNotFoundError`` at import time and the entire backend fails to load.

This module provides drop-in, behaviour-compatible Python implementations of
the three symbols the backend binds at import time:

* :class:`AlarmEngine`
* :func:`exponential_smoothing`
* :func:`moving_average`

The numeric algorithms intentionally mirror the Rust implementations in
``rust_core/tools-core/src/scada.rs`` so results are identical regardless of
which backend is active:

* ``moving_average`` is a *centered* moving average matching NumPy ``"same"``
  convolution semantics (same windowing the Rust kernel uses).
* ``exponential_smoothing`` is a first-order recursive filter seeded with the
  first sample.

The fallback ``AlarmEngine`` reproduces the Rust engine's public contract:
LoLo/Low/Normal/High/HiHi/BadQuality state classification, at-most-32-tags and
monotonic threshold validation (``ValueError``), acknowledgment tracking, and
the ``update_tag`` / ``acknowledge_alarm`` / ``get_active_alarms`` /
``get_alarm_state`` methods used by the backend.

A non-finite value (NaN/Inf) is classified ``BadQuality`` -- an *active* alarm
state -- never ``Normal``. All four band comparisons are False for NaN, so the
naive classifier resolved a live HiHi to Normal on a burned-out register read
(issue #3973). ``tests/test_scada_fallback.py`` pins this against the Rust
engine.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum
from typing import Any

__all__ = ["AlarmEngine", "AlarmState", "exponential_smoothing", "moving_average"]

_MAX_TAGS = 32


class AlarmState(Enum):
    """Alarm severity classification matching the Rust ``AlarmState`` enum."""

    NORMAL = "Normal"
    LOW = "Low"
    LOLO = "LoLo"
    HIGH = "High"
    HIHI = "HiHi"
    #: Non-finite reading: a sensor/register fault, held as an active alarm.
    BAD_QUALITY = "BadQuality"


#: Severity per state, mirrored by ``alarm_processing.severity_for_state`` and
#: the Rust ``get_active_alarms``. BadQuality ranks with the trip tier: a fault
#: on an alarmed tag cannot be shown to be safe.
_SEVERITY: dict[AlarmState, int] = {
    AlarmState.NORMAL: 0,
    AlarmState.LOW: 1,
    AlarmState.HIGH: 1,
    AlarmState.LOLO: 2,
    AlarmState.HIHI: 2,
    AlarmState.BAD_QUALITY: 2,
}


def moving_average(values: Sequence[float], window_size: int) -> list[float]:
    """Centered moving-average smoothing (matches NumPy ``"same"``).

    Mirrors ``moving_average`` in ``rust_core/tools-core/src/scada.rs``.

    Args:
        values: Input samples.
        window_size: Smoothing window; must be >= 1.

    Returns:
        A list the same length as ``values``.

    Raises:
        ValueError: If ``window_size`` < 1.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    data = [float(v) for v in values]
    n = len(data)
    if n == 0:
        return []

    prefix = [0.0]
    for value in data:
        prefix.append(prefix[-1] + value)

    full_len = n + window_size - 1
    start = (window_size - 1) // 2
    out: list[float] = []
    for k in range(start, min(start + n, full_len)):
        input_start = max(k - (window_size - 1), 0)
        input_end = min(k, n - 1) + 1
        out.append((prefix[input_end] - prefix[input_start]) / float(window_size))
    return out


def exponential_smoothing(values: Sequence[float], alpha: float) -> list[float]:
    """First-order recursive exponential smoothing.

    Mirrors ``exponential_smoothing`` in
    ``rust_core/tools-core/src/scada.rs``.

    Args:
        values: Input samples.
        alpha: Smoothing factor in ``(0, 1]``.

    Returns:
        A list the same length as ``values``.

    Raises:
        ValueError: If ``alpha`` is not in ``(0, 1]``.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")

    data = [float(v) for v in values]
    if not data:
        return []

    out = [data[0]]
    for value in data[1:]:
        out.append(alpha * value + (1.0 - alpha) * out[-1])
    return out


class AlarmEngine:
    """Pure-Python SCADA alarm engine matching the Rust ``AlarmEngine``.

    Tracks active state, severity, and acknowledgments for up to 32 tags.

    Args:
        limits: Mapping of ``tag_id`` -> ``{"lolo", "low", "high", "hihi"}``.

    Raises:
        ValueError: If more than 32 tags are supplied, if a tag is missing a
            required limit key, or if the limits are not monotonic
            (``lolo <= low <= high <= hihi``).
    """

    def __init__(self, limits: dict[str, dict[str, float]]) -> None:
        if len(limits) > _MAX_TAGS:
            raise ValueError("AlarmEngine supports at most 32 tags")

        self.tag_limits: dict[str, dict[str, float]] = {}
        self._tag_values: dict[str, float] = {}
        self._tag_states: dict[str, AlarmState] = {}
        self._tag_acknowledged: dict[str, bool] = {}
        self._tag_acknowledged_by: dict[str, str | None] = {}

        for tag_id, limit_map in limits.items():
            try:
                lolo = float(limit_map["lolo"])
                low = float(limit_map["low"])
                high = float(limit_map["high"])
                hihi = float(limit_map["hihi"])
            except KeyError as exc:  # pragma: no cover - defensive
                missing = exc.args[0]
                raise ValueError(f"Missing '{missing}' limit for tag {tag_id}") from exc

            if lolo > low or low > high or high > hihi:
                raise ValueError(
                    f"Limits for tag '{tag_id}' must satisfy "
                    f"lolo <= low <= high <= hihi (got lolo={lolo}, low={low}, "
                    f"high={high}, hihi={hihi})"
                )

            self.tag_limits[tag_id] = {
                "low": low,
                "lolo": lolo,
                "high": high,
                "hihi": hihi,
            }
            self._tag_values[tag_id] = 0.0
            self._tag_states[tag_id] = AlarmState.NORMAL
            self._tag_acknowledged[tag_id] = False
            self._tag_acknowledged_by[tag_id] = None

    def update_tag(self, tag_id: str, value: float) -> list[dict[str, Any]]:
        """Update a tag value and re-evaluate its alarm state.

        Returns a list of state-change events (empty if the state did not
        change). Acknowledgment is reset whenever the state changes.

        Raises:
            KeyError: If ``tag_id`` is not registered.
        """
        if tag_id not in self.tag_limits:
            raise KeyError(f"Tag '{tag_id}' not registered")

        limits = self.tag_limits[tag_id]
        old_state = self._tag_states.get(tag_id, AlarmState.NORMAL)
        if not math.isfinite(value):
            new_state = AlarmState.BAD_QUALITY
        elif value <= limits["lolo"]:
            new_state = AlarmState.LOLO
        elif value <= limits["low"]:
            new_state = AlarmState.LOW
        elif value >= limits["hihi"]:
            new_state = AlarmState.HIHI
        elif value >= limits["high"]:
            new_state = AlarmState.HIGH
        else:
            new_state = AlarmState.NORMAL

        self._tag_values[tag_id] = value

        events: list[dict[str, Any]] = []
        if new_state != old_state:
            self._tag_states[tag_id] = new_state
            self._tag_acknowledged[tag_id] = False
            self._tag_acknowledged_by[tag_id] = None
            events.append(
                {
                    "tag_id": tag_id,
                    "previous_state": old_state,
                    "current_state": new_state,
                    "value": value,
                }
            )
        return events

    def acknowledge_alarm(self, tag_id: str, user: str) -> bool:
        """Acknowledge an active alarm. Returns ``True`` if acknowledged.

        Raises:
            KeyError: If ``tag_id`` is not registered.
        """
        if tag_id not in self.tag_limits:
            raise KeyError(f"Tag '{tag_id}' not registered")

        if self._tag_states.get(tag_id, AlarmState.NORMAL) == AlarmState.NORMAL:
            return False
        self._tag_acknowledged[tag_id] = True
        self._tag_acknowledged_by[tag_id] = user
        return True

    def get_active_alarms(self) -> list[dict[str, Any]]:
        """Return the list of currently active (non-Normal) alarms."""
        active: list[dict[str, Any]] = []
        for tag_id, state in self._tag_states.items():
            if state == AlarmState.NORMAL:
                continue
            active.append(
                {
                    "tag_id": tag_id,
                    "state": state,
                    "severity": _SEVERITY[state],
                    "acknowledged": self._tag_acknowledged.get(tag_id, False),
                    "acknowledged_by": self._tag_acknowledged_by.get(tag_id),
                    "value": self._tag_values.get(tag_id, 0.0),
                }
            )
        return active

    def get_alarm_state(self, tag_id: str) -> dict[str, Any]:
        """Return the current state and details of a single tag.

        Raises:
            KeyError: If ``tag_id`` is not registered.
        """
        if tag_id not in self.tag_limits:
            raise KeyError(f"Tag '{tag_id}' not registered")
        return {
            "tag_id": tag_id,
            "state": self._tag_states.get(tag_id, AlarmState.NORMAL),
            "acknowledged": self._tag_acknowledged.get(tag_id, False),
            "acknowledged_by": self._tag_acknowledged_by.get(tag_id),
            "value": self._tag_values.get(tag_id, 0.0),
        }
