"""Clock-domain and frame-timestamp evidence contracts."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import require_finite, require_nonnegative_integer, require_text
from .enums import ClockKind


@dataclass(frozen=True, slots=True)
class ClockDomain:
    """One timestamp domain with declared resolution and monotonic behavior."""

    clock_id: str
    kind: ClockKind
    tick_period_seconds: float
    monotonic: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "clock_id", require_text(self.clock_id, "clock_id"))
        if not isinstance(self.kind, ClockKind):
            raise TypeError("kind must be a ClockKind")
        period = require_finite(self.tick_period_seconds, "tick_period_seconds")
        if period <= 0.0:
            raise ValueError("tick_period_seconds must be positive")
        if not isinstance(self.monotonic, bool):
            raise TypeError("monotonic must be a boolean")
        object.__setattr__(self, "tick_period_seconds", period)


@dataclass(frozen=True, slots=True)
class FrameStamp:
    """Frame identity and timing evidence without an inferred sync claim."""

    source_id: str
    stream_id: str
    sequence_number: int
    clock_id: str
    capture_timestamp_ns: int
    host_monotonic_ns: int
    timing_uncertainty_ns: int
    exposure_start_ns: int | None = None
    exposure_end_ns: int | None = None

    def __post_init__(self) -> None:
        for name in ("source_id", "stream_id", "clock_id"):
            object.__setattr__(self, name, require_text(getattr(self, name), name))
        for name in (
            "sequence_number",
            "capture_timestamp_ns",
            "host_monotonic_ns",
            "timing_uncertainty_ns",
        ):
            object.__setattr__(
                self, name, require_nonnegative_integer(getattr(self, name), name)
            )
        if (self.exposure_start_ns is None) != (self.exposure_end_ns is None):
            raise ValueError("exposure start and end must be provided together")
        if self.exposure_start_ns is not None and self.exposure_end_ns is not None:
            start = require_nonnegative_integer(
                self.exposure_start_ns, "exposure_start_ns"
            )
            end = require_nonnegative_integer(self.exposure_end_ns, "exposure_end_ns")
            if start > end:
                raise ValueError("exposure start must not exceed exposure end")

    @property
    def exposure_duration_ns(self) -> int | None:
        """Return exposure duration when the source supplied an interval."""
        if self.exposure_start_ns is None or self.exposure_end_ns is None:
            return None
        return self.exposure_end_ns - self.exposure_start_ns


__all__: list[str] = []
