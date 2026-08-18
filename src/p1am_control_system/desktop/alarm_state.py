"""Alarm state machine for the P1AM desktop HMI.

Deliberately Qt-free so the annunciation rules can be exercised without a
display, and so ``main_window`` stays a thin coordinator.

Standard alarm management keeps two independent facts about every alarm:

``active``
    The process condition is present *right now*. This drives the annunciator's
    colour/severity — an operator must be able to see that a tag is still in
    alarm even after acknowledging it.

``unacknowledged``
    Nobody has confirmed that they have seen the alarm. This drives flashing
    versus steady.

Consequences that the previous implementation got backwards (issue #4012):

* Acknowledging a still-active alarm makes it *steady*, not invisible.
* A value returning to its normal band drops the alarm from **both** sets, so a
  long-cleared alarm stops flashing the ACK button.

Trip points come from the deployed four-tier interlock configuration
(``lolo_limit``/``low_limit``/``high_limit``/``hihi_limit``) so the HMI's
severity matches the firmware's (issue #4019).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

__all__ = [
    "ALARM_TYPES",
    "AlarmEventDebouncer",
    "AlarmStateMachine",
    "AlarmTransition",
    "AnnunciatorState",
    "InterlockLimitError",
    "SEVERE_ALARM_TYPES",
    "WARNING_ALARM_TYPES",
    "classify_value",
    "interlock_for_index",
    "validate_interlocks",
]

logger = logging.getLogger("p1am_control.desktop.alarm_state")

#: Alarm types in descending severity order.
ALARM_TYPES: tuple[str, ...] = ("LL", "HH", "L", "H")

#: Types that annunciate red (the firmware trips on these).
SEVERE_ALARM_TYPES = frozenset({"HH", "LL"})

#: Types that annunciate amber.
WARNING_ALARM_TYPES = frozenset({"H", "L"})

#: Canonical broker tag-name template used as the interlock mapping key.
TAG_KEY_TEMPLATE = "TAG_{index}"

_LIMIT_FIELDS = ("lolo_limit", "low_limit", "high_limit", "hihi_limit")

AlarmKey = tuple[int, str]


class InterlockLimitError(ValueError):
    """Raised when an interlock's four trip points are not monotonic.

    Subclasses :class:`ValueError` so existing ``except ValueError`` handlers
    keep working.
    """


class AnnunciatorState(NamedTuple):
    """What the header should show.

    ``has_*`` describe the *condition* (colour); ``unacked_*`` describe whether
    anyone has seen it (flash versus steady).
    """

    has_hl: bool
    has_hhll: bool
    unacked_hl: bool
    unacked_hhll: bool


@dataclass(frozen=True)
class AlarmTransition:
    """A single alarm edge produced by :meth:`AlarmStateMachine.evaluate`."""

    kind: str  # "raised" | "cleared"
    tag_id: int
    alarm_type: str
    message: str

    @property
    def key(self) -> AlarmKey:
        """The ``(tag_id, alarm_type)`` identity of this alarm."""
        return (self.tag_id, self.alarm_type)


def _require_number(value: Any, name: str) -> float:
    """Return ``value`` as a finite float.

    Non-finite input is rejected, not passed through. ``classify_value`` decides
    the alarm band with four comparisons, and *every* comparison against NaN is
    False — so a NaN reading would be classified as "inside the normal band" and
    ``AlarmState.evaluate`` would emit ``cleared`` transitions and drop the tag
    from both the active and unacknowledged sets. A garbled or unscaled register
    would therefore silence a live High-High on the heater. ``json.loads``
    accepts bare ``NaN``, so this is reachable straight off the wire.

    Raising instead means ``_evaluate_alarms`` logs the tag and skips it, and
    because ``evaluate`` is per-tag and clears nothing before this guard runs,
    any alarm already latched for that tag stays latched — the fail-safe
    direction.

    Raises:
        TypeError: If ``value`` is not a real number (``bool`` is rejected).
        ValueError: If ``value`` is NaN or infinite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(
            f"{name} must be finite, got {numeric!r}; a non-finite reading is a "
            "sensor fault, not a measurement, and must not be compared against "
            "trip limits"
        )
    return numeric


def _limits(interlock: Any) -> tuple[float, float, float, float]:
    """Return ``(lolo, low, high, hihi)`` for ``interlock``.

    Raises:
        TypeError: If a field is missing or is not a real number.
    """
    values = []
    for field in _LIMIT_FIELDS:
        if not hasattr(interlock, field):
            raise TypeError(f"interlock is missing required field {field!r}")
        values.append(_require_number(getattr(interlock, field), field))
    return (values[0], values[1], values[2], values[3])


def validate_interlocks(interlocks: Any) -> None:
    """Validate every configured interlock's trip-point ordering.

    Preconditions for each entry: ``lolo_limit <= low_limit <= high_limit <=
    hihi_limit``. A configuration that violates this is a deployment fault —
    the HMI cannot classify severity consistently with the firmware — so it is
    rejected rather than silently used.

    Args:
        interlocks: Mapping of tag key to interlock config, or a sequence of
            interlock configs.

    Raises:
        TypeError: If ``interlocks`` is not a mapping/sequence, or an entry's
            limits are not real numbers.
        InterlockLimitError: If any entry's limits are not monotonic. The
            message names every offending key.
    """
    if isinstance(interlocks, Mapping):
        items: list[tuple[Any, Any]] = list(interlocks.items())
    elif isinstance(interlocks, Sequence) and not isinstance(interlocks, (str, bytes)):
        items = list(enumerate(interlocks))
    else:
        raise TypeError(
            f"interlocks must be a mapping or sequence, got {type(interlocks).__name__}"
        )

    offenders: list[str] = []
    for key, interlock in items:
        lolo, low, high, hihi = _limits(interlock)
        if not (lolo <= low <= high <= hihi):
            offenders.append(
                f"{key}: lolo={lolo:g} low={low:g} high={high:g} hihi={hihi:g}"
            )

    if offenders:
        raise InterlockLimitError(
            "interlock limits must satisfy lolo <= low <= high <= hihi; "
            "offending tags: " + "; ".join(offenders)
        )


def classify_value(value: Any, interlock: Any) -> str | None:
    """Classify ``value`` against the four configured trip points.

    Args:
        value: The engineering-unit process value.
        interlock: Config exposing ``lolo_limit``/``low_limit``/``high_limit``/
            ``hihi_limit``.

    Returns:
        ``"LL"``, ``"HH"``, ``"L"``, ``"H"`` or ``None`` when the value is
        inside the normal band.

    Raises:
        TypeError: If ``value`` or any limit is not a real number.
    """
    reading = _require_number(value, "value")
    lolo, low, high, hihi = _limits(interlock)

    if reading <= lolo:
        return "LL"
    if reading >= hihi:
        return "HH"
    if reading <= low:
        return "L"
    if reading >= high:
        return "H"
    return None


def interlock_for_index(interlocks: Any, tag_id: int) -> Any | None:
    """Resolve the interlock configured for broker index ``tag_id``.

    The backend serves ``dict[str, InterlockConfig]`` keyed ``"TAG_<n>"``; a
    plain sequence is also accepted for simpler callers/tests.

    Returns:
        The interlock config, or ``None`` when the tag is not configured.

    Raises:
        TypeError: If ``tag_id`` is not an int.
    """
    if isinstance(tag_id, bool) or not isinstance(tag_id, int):
        raise TypeError(f"tag_id must be an int, got {type(tag_id).__name__}")

    if isinstance(interlocks, Mapping):
        for key in (TAG_KEY_TEMPLATE.format(index=tag_id), str(tag_id), tag_id):
            if key in interlocks:
                return interlocks[key]
        return None
    if isinstance(interlocks, Sequence) and not isinstance(interlocks, (str, bytes)):
        if 0 <= tag_id < len(interlocks):
            return interlocks[tag_id]
        return None
    return None


_MESSAGE_TEMPLATES = {
    "LL": "Tag {tag} Low-Low limit violation ({value:.2f} <= {limit:.2f})",
    "L": "Tag {tag} Low limit violation ({value:.2f} <= {limit:.2f})",
    "H": "Tag {tag} High limit violation ({value:.2f} >= {limit:.2f})",
    "HH": "Tag {tag} High-High limit violation ({value:.2f} >= {limit:.2f})",
}

_LIMIT_INDEX = {"LL": 0, "L": 1, "H": 2, "HH": 3}


class AlarmStateMachine:
    """Tracks active and unacknowledged alarms for the HMI annunciator.

    Attributes:
        active_alarms: ``(tag_id, alarm_type)`` whose condition is present now.
        unacknowledged_alarms: Subset that no operator has acknowledged.

    Invariant: ``unacknowledged_alarms <= active_alarms``.
    """

    def __init__(self) -> None:
        self.active_alarms: set[AlarmKey] = set()
        self.unacknowledged_alarms: set[AlarmKey] = set()

    def evaluate(
        self, tag_id: int, value: Any, interlock: Any
    ) -> list[AlarmTransition]:
        """Fold one scan value for ``tag_id`` into the alarm state.

        Exactly one alarm type can be active per tag; promoting H to HH clears
        the H edge so no stale severity lingers.

        Args:
            tag_id: Broker tag index.
            value: The scanned engineering-unit value.
            interlock: Config exposing the four trip points.

        Returns:
            The transitions this scan produced (``"raised"``/``"cleared"``), in
            clear-before-raise order so a promotion reads naturally in the log.

        Raises:
            TypeError: If ``tag_id`` is not an int, or the value/limits are not
                real numbers.
        """
        if isinstance(tag_id, bool) or not isinstance(tag_id, int):
            raise TypeError(f"tag_id must be an int, got {type(tag_id).__name__}")

        reading = _require_number(value, "value")
        limits = _limits(interlock)
        current = classify_value(reading, interlock)

        transitions: list[AlarmTransition] = []

        for alarm_type in ALARM_TYPES:
            if alarm_type == current:
                continue
            key = (tag_id, alarm_type)
            if key in self.active_alarms:
                self.active_alarms.discard(key)
                self.unacknowledged_alarms.discard(key)
                transitions.append(
                    AlarmTransition(
                        kind="cleared",
                        tag_id=tag_id,
                        alarm_type=alarm_type,
                        message=f"Tag {tag_id} alarm {alarm_type} cleared.",
                    )
                )

        if current is not None:
            key = (tag_id, current)
            if key not in self.active_alarms:
                self.active_alarms.add(key)
                self.unacknowledged_alarms.add(key)
                transitions.append(
                    AlarmTransition(
                        kind="raised",
                        tag_id=tag_id,
                        alarm_type=current,
                        message=_MESSAGE_TEMPLATES[current].format(
                            tag=tag_id,
                            value=reading,
                            limit=limits[_LIMIT_INDEX[current]],
                        ),
                    )
                )

        return transitions

    def acknowledge(self, keys: Any) -> list[AlarmKey]:
        """Acknowledge exactly the alarms in ``keys``.

        Only the alarms that were on screen when the operator pressed ACK may be
        acknowledged; a blanket clear would silently acknowledge alarms that
        arrived between the render and the click (issue #4012).

        Args:
            keys: An iterable of the ``(tag_id, alarm_type)`` pairs the header
                was showing.

        Returns:
            The keys that were actually acknowledged, sorted for stable logging.

        Raises:
            TypeError: If ``keys`` is not an iterable of pairs.
        """
        if not isinstance(keys, Iterable) or isinstance(keys, (str, bytes)):
            raise TypeError(
                "keys must be an iterable of (tag_id, alarm_type) pairs, "
                f"got {type(keys).__name__}"
            )
        requested: set[AlarmKey] = set(keys)

        acknowledged = sorted(self.unacknowledged_alarms & requested)
        self.unacknowledged_alarms -= requested
        return acknowledged

    def annunciator_state(self) -> AnnunciatorState:
        """Return the colour/flash flags for the header ACK button."""
        return AnnunciatorState(
            has_hl=any(a[1] in WARNING_ALARM_TYPES for a in self.active_alarms),
            has_hhll=any(a[1] in SEVERE_ALARM_TYPES for a in self.active_alarms),
            unacked_hl=any(
                a[1] in WARNING_ALARM_TYPES for a in self.unacknowledged_alarms
            ),
            unacked_hhll=any(
                a[1] in SEVERE_ALARM_TYPES for a in self.unacknowledged_alarms
            ),
        )


class AlarmEventDebouncer:
    """Coalesce rapidly repeating alarm events into one counted entry.

    A thermocouple dithering on its trip point produces one raise and one clear
    per 100 ms scan. Logging each of those is both a readability problem and —
    because every event was an fsync-backed SQLite commit on the Qt GUI thread —
    the reason the HMI froze exactly while an alarm was active (issue #4022).

    The first occurrence of a key is released immediately. Further occurrences
    inside ``window_s`` are counted and released as a single summary once the
    window expires.
    """

    def __init__(self, window_s: float = 5.0) -> None:
        """Create a debouncer.

        Args:
            window_s: Coalescing window in seconds. Must be > 0.

        Raises:
            TypeError: If ``window_s`` is not a real number.
            ValueError: If ``window_s`` is not strictly positive.
        """
        window = _require_number(window_s, "window_s")
        if window <= 0.0:
            raise ValueError(f"window_s must be > 0, got {window}")
        self.window_s = window
        # key -> [last_emit_time, suppressed_count, last_level, last_message]
        self._pending: dict[Any, list[Any]] = {}

    def submit(
        self, key: Any, level: str, message: str, now: float | None = None
    ) -> list[tuple[str, str]]:
        """Offer an event for logging.

        Args:
            key: Identity used for coalescing, e.g. ``(tag_id, type, level)``.
            level: HMI log level (``"ALARM"``, ``"CLEAR"``, ...).
            message: Human-readable event text.
            now: Monotonic timestamp; defaults to :func:`time.monotonic`.

        Returns:
            The ``(level, message)`` pairs that should be written now — empty
            when the event was coalesced into a pending summary.
        """
        if not isinstance(level, str) or not isinstance(message, str):
            raise TypeError("level and message must be strings")
        stamp = time.monotonic() if now is None else _require_number(now, "now")

        released = self._release_expired(stamp, exclude=key)

        entry = self._pending.get(key)
        if entry is None or stamp - entry[0] >= self.window_s:
            if entry is not None and entry[1]:
                released.append(self._summary(entry))
            self._pending[key] = [stamp, 0, level, message]
            released.append((level, message))
            return released

        entry[1] += 1
        entry[2] = level
        entry[3] = message
        return released

    def flush(self, now: float | None = None) -> list[tuple[str, str]]:
        """Release summaries for every window that has expired.

        Returns:
            The ``(level, message)`` pairs to write now.
        """
        stamp = time.monotonic() if now is None else _require_number(now, "now")
        return self._release_expired(stamp, exclude=None)

    def _release_expired(self, stamp: float, exclude: Any) -> list[tuple[str, str]]:
        released: list[tuple[str, str]] = []
        for key in list(self._pending):
            if key == exclude:
                continue
            entry = self._pending[key]
            if stamp - entry[0] < self.window_s:
                continue
            if entry[1]:
                released.append(self._summary(entry))
            del self._pending[key]
        return released

    @staticmethod
    def _summary(entry: list[Any]) -> tuple[str, str]:
        count = entry[1]
        return (entry[2], f"{entry[3]} (repeated {count} more times)")
