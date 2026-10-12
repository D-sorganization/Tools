"""Advisory asset-health detectors: flatline and command/feedback mismatch.

SCADA F10 (issue #5531). Two pure functions inspect a window of timestamped
samples and return :class:`HealthFinding` records that an operator may read as
maintenance hints (a stuck sensor, an actuator that is not following its
command).

Safety property -- ADVISORY ONLY:
    - Findings are tagged ``finding_type == "advisory"``. The tag is fixed (not
      an ``__init__`` argument) and the record is frozen, so a finding can
      neither be created as, nor mutated into, an alarm or trip.
    - A finding has no priority, severity or trip field. It can never raise or
      clear a trip, change a setpoint or mode, or write to the PLC.
    - The module is pure: it imports only the stdlib and the pure
      ``signal_stats`` helper, and receives read-only sample data from the
      caller (Law of Demeter). It has no path to PLC clients, command APIs,
      alarm or trip state; a test asserts this by inspecting the imports.
    - Nothing here sits in the E-stop or de-energizing command path.

Design by Contract:
    - ``min_duration``, ``tolerance`` and ``window`` are REQUIRED, with no
      defaults, so the thresholds are always explicit at the call site. Each
      must be a real number (not bool), finite and strictly positive.
    - Timestamps (seconds) must be finite and strictly increasing.
    - A missing (``None``) or non-finite (NaN/inf) value is bad quality. It
      ends any run in progress and is never counted as flat or as a mismatch.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar

from signal_stats import _is_real_number

from shared.python.compatibility import StrEnum

ADVISORY = "advisory"

_State = TypeVar("_State")


class FindingKind(StrEnum, Enum):
    """What an advisory finding observed."""

    FLATLINE = "flatline"
    COMMAND_FEEDBACK_MISMATCH = "command_feedback_mismatch"


@dataclass(frozen=True)
class TimedSample:
    """One reading: ``time_s`` seconds and a value; ``None`` means missing."""

    time_s: float
    value: float | None

    def __post_init__(self) -> None:
        if not _is_real_number(self.time_s) or not math.isfinite(self.time_s):
            raise TypeError(f"time_s must be a finite real number, got {self.time_s!r}")
        if self.value is not None and not _is_real_number(self.value):
            raise TypeError(f"value must be None or int|float, got {self.value!r}")

    @property
    def is_good(self) -> bool:
        """True when the value is present and finite."""
        return self.value is not None and math.isfinite(self.value)


@dataclass(frozen=True)
class HealthFinding:
    """An advisory observation over ``[start_s, end_s]``. Never a trip."""

    kind: FindingKind
    start_s: float
    end_s: float
    message: str
    finding_type: str = field(default=ADVISORY, init=False)

    @property
    def duration_s(self) -> float:
        """Span covered by the finding in seconds."""
        return self.end_s - self.start_s


def _require_positive(name: str, value: object) -> float:
    """Return ``value`` if it is a finite real number > 0, else raise."""
    if not _is_real_number(value):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return number


def _require_increasing(name: str, samples: Sequence[TimedSample]) -> None:
    """Raise unless every element is a TimedSample with increasing time."""
    previous = -math.inf
    for index, sample in enumerate(samples):
        if not isinstance(sample, TimedSample):
            raise TypeError(f"{name}[{index}] must be a TimedSample")
        if sample.time_s <= previous:
            raise ValueError(f"{name} timestamps must be strictly increasing")
        previous = sample.time_s


def _runs(
    times: Sequence[float],
    keep: Sequence[bool],
    start: Callable[[int], _State],
    extend: Callable[[_State, int], _State | None],
    minimum_span: float,
) -> list[tuple[float, float]]:
    """Time spans of maximal runs of kept samples lasting ``minimum_span``.

    ``start(i)`` makes the run state for a run beginning at sample ``i`` and
    ``extend(state, i)`` returns the updated state, or ``None`` when sample
    ``i`` does not belong to the run. A sample that is not kept ends the run;
    a kept sample that does not extend it starts a new run at itself. State is
    updated incrementally so the scan is O(n).
    """
    spans: list[tuple[float, float]] = []
    first = last = -1
    state: _State | None = None

    def close() -> None:
        if state is not None and times[last] - times[first] >= minimum_span:
            spans.append((times[first], times[last]))

    for i, ok in enumerate(keep):
        if not ok:
            close()
            state = None
            continue
        grown = extend(state, i) if state is not None else None
        if grown is None:
            close()
            first, state = i, start(i)
        else:
            state = grown
        last = i
    close()
    return spans


def detect_flatline(
    samples: Sequence[TimedSample], *, min_duration: float, tolerance: float
) -> list[HealthFinding]:
    """Flag spans where a signal stays within ``tolerance`` for ``min_duration``.

    Preconditions: ``samples`` is a time-ordered sequence of ``TimedSample``;
    ``min_duration`` (seconds) and ``tolerance`` (signal units, max - min
    across the run) are finite and > 0 with no defaults.
    Postconditions: every finding is ``advisory``; runs contain only good
    samples; ``end_s - start_s >= min_duration``; input is not mutated.
    """
    min_duration = _require_positive("min_duration", min_duration)
    tolerance = _require_positive("tolerance", tolerance)
    _require_increasing("samples", samples)

    times = [s.time_s for s in samples]
    values = [s.value if s.value is not None else 0.0 for s in samples]
    good = [s.is_good for s in samples]

    def start(i: int) -> tuple[float, float]:
        return (values[i], values[i])

    def extend(state: tuple[float, float], i: int) -> tuple[float, float] | None:
        low, high = min(state[0], values[i]), max(state[1], values[i])
        return (low, high) if high - low <= tolerance else None

    return [
        HealthFinding(
            FindingKind.FLATLINE,
            first_s,
            last_s,
            f"signal flat within {tolerance:g} for {last_s - first_s:g}s "
            f"(threshold {min_duration:g}s)",
        )
        for first_s, last_s in _runs(times, good, start, extend, min_duration)
    ]


def detect_command_feedback_mismatch(
    cmd: Sequence[TimedSample],
    fb: Sequence[TimedSample],
    *,
    tolerance: float,
    window: float,
) -> list[HealthFinding]:
    """Flag spans where |command - feedback| exceeds ``tolerance`` for ``window``.

    Preconditions: ``cmd`` and ``fb`` are time-ordered, equal length, with
    identical timestamps; ``tolerance`` (signal units) and ``window`` (dwell,
    seconds) are finite and > 0 with no defaults.
    Postconditions: every finding is ``advisory``; a sample where either side
    is bad quality ends the run and is never evidence of a mismatch; a gap
    shorter than ``window`` is not reported; input is not mutated.
    """
    tolerance = _require_positive("tolerance", tolerance)
    window = _require_positive("window", window)
    _require_increasing("cmd", cmd)
    _require_increasing("fb", fb)
    if len(cmd) != len(fb) or any(
        c.time_s != f.time_s for c, f in zip(cmd, fb, strict=True)
    ):
        raise ValueError("cmd and fb must have identical timestamps")

    times = [c.time_s for c in cmd]
    good = [c.is_good and f.is_good for c, f in zip(cmd, fb, strict=True)]
    gap = [
        abs(c.value - f.value) > tolerance if ok else False  # type: ignore[operator]
        for c, f, ok in zip(cmd, fb, good, strict=True)
    ]

    def start(_i: int) -> bool:
        return True

    def extend(_state: bool, _i: int) -> bool | None:
        return True

    return [
        HealthFinding(
            FindingKind.COMMAND_FEEDBACK_MISMATCH,
            first_s,
            last_s,
            f"feedback off command by more than {tolerance:g} for "
            f"{last_s - first_s:g}s (dwell {window:g}s)",
        )
        for first_s, last_s in _runs(times, gap, start, extend, window)
    ]
