"""Canonical value, timing, quality, diagnostic, source, and sequence model."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType

try:
    from datetime import UTC
except ImportError:  # Python 3.10 support
    UTC = timezone.utc  # noqa: UP017


class SignalQuality(StrEnum):
    """Small, transport-stable signal quality vocabulary."""

    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    STALE = "stale"
    SIMULATED = "simulated"


_ALARM_ELIGIBLE = frozenset(
    {SignalQuality.GOOD, SignalQuality.UNCERTAIN, SignalQuality.SIMULATED}
)


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _aware(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


@dataclass(frozen=True)
class SignalSample:
    """One fully attributed signal sample at a specific scan sequence."""

    value: float
    source_timestamp: datetime
    server_timestamp: datetime
    quality: SignalQuality
    diagnostic_reason: str | None
    sequence: int
    source: str

    def __post_init__(self) -> None:
        try:
            value = float(self.value)
        except (TypeError, ValueError) as exc:
            raise TypeError("value must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        object.__setattr__(self, "value", value)
        object.__setattr__(
            self,
            "source_timestamp",
            _aware(self.source_timestamp, "source_timestamp"),
        )
        object.__setattr__(
            self,
            "server_timestamp",
            _aware(self.server_timestamp, "server_timestamp"),
        )
        if self.source_timestamp > self.server_timestamp:
            raise ValueError("source_timestamp cannot be after server_timestamp")
        if not isinstance(self.quality, SignalQuality):
            raise TypeError("quality must be a SignalQuality")
        if not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("sequence must be a positive integer")
        object.__setattr__(self, "source", _required_text(self.source, "source"))
        if self.quality is SignalQuality.GOOD:
            if self.diagnostic_reason is not None:
                raise ValueError("good quality cannot have a diagnostic_reason")
        else:
            if self.diagnostic_reason is None:
                raise ValueError("degraded quality requires a diagnostic_reason")
            object.__setattr__(
                self,
                "diagnostic_reason",
                _required_text(self.diagnostic_reason, "diagnostic_reason"),
            )

    def age_seconds(self, now: datetime) -> float:
        """Return source age at an aware reference time."""
        return (_aware(now, "now") - self.source_timestamp).total_seconds()

    def to_payload(self) -> dict[str, object]:
        """Return the canonical JSON-safe wire representation."""
        return {
            "value": self.value,
            "source_timestamp": self.source_timestamp.isoformat(),
            "server_timestamp": self.server_timestamp.isoformat(),
            "quality": self.quality.value,
            "diagnostic_reason": self.diagnostic_reason,
            "sequence": self.sequence,
            "source": self.source,
        }


@dataclass(frozen=True)
class SignalFrame:
    """Immutable scan of samples sharing one server time and sequence."""

    samples: Mapping[str, SignalSample]
    server_timestamp: datetime
    sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.samples, Mapping):
            raise TypeError("samples must be a mapping")
        if not self.samples:
            raise ValueError("samples must contain at least one signal")
        timestamp = _aware(self.server_timestamp, "server_timestamp")
        normalized: dict[str, SignalSample] = {}
        for name, sample in self.samples.items():
            tag_name = _required_text(name, "signal name")
            if not isinstance(sample, SignalSample):
                raise TypeError("samples must contain SignalSample values")
            if sample.server_timestamp != timestamp or sample.sequence != self.sequence:
                raise ValueError("all samples must share frame time and sequence")
            normalized[tag_name] = sample
        object.__setattr__(self, "samples", MappingProxyType(normalized))

    @property
    def values(self) -> dict[str, float]:
        return {name: sample.value for name, sample in self.samples.items()}

    @property
    def alarm_eligible(self) -> bool:
        return all(
            sample.quality in _ALARM_ELIGIBLE for sample in self.samples.values()
        )

    def to_payload(self) -> dict[str, object]:
        return {name: sample.to_payload() for name, sample in self.samples.items()}


class SignalFrameFactory:
    """Sequence and source-time owner for raw driver scan adaptation."""

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sequence = 0
        self._last_source_times: dict[str, datetime] = {}

    def _next(
        self,
        values: dict[str, float],
        quality: SignalQuality,
        source: str,
        reason: str | None,
        retain_source_time: bool,
    ) -> SignalFrame:
        if not isinstance(values, dict):
            raise TypeError("values must be a dict")
        if not values:
            raise ValueError("values must contain at least one signal")
        now = _aware(self._clock(), "clock result")
        self._sequence += 1
        samples: dict[str, SignalSample] = {}
        for name, value in values.items():
            source_time = (
                self._last_source_times.get(name, now) if retain_source_time else now
            )
            sample = SignalSample(
                value=value,
                source_timestamp=source_time,
                server_timestamp=now,
                quality=quality,
                diagnostic_reason=reason,
                sequence=self._sequence,
                source=source,
            )
            samples[name] = sample
            if not retain_source_time:
                self._last_source_times[name] = now
        return SignalFrame(samples, now, self._sequence)

    def good(self, values: dict[str, float], source: str = "driver") -> SignalFrame:
        return self._next(values, SignalQuality.GOOD, source, None, False)

    def stale(
        self,
        values: dict[str, float],
        source: str = "driver",
        reason: str = "read_failed",
    ) -> SignalFrame:
        return self._next(values, SignalQuality.STALE, source, reason, True)

    def simulated(
        self,
        values: dict[str, float],
        source: str = "simulator",
    ) -> SignalFrame:
        return self._next(
            values,
            SignalQuality.SIMULATED,
            source,
            "synthetic_source",
            False,
        )
