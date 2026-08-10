"""Result-only value types for the flight-to-ground v1 contract."""

from __future__ import annotations

from dataclasses import dataclass

from .contract_types import (
    GroundTerminationReason,
    GroundWarningSeverity,
    _finite,
    _integer,
    _nonnegative,
    _text,
    _WireRecord,
)


@dataclass(frozen=True)
class GroundSummary(_WireRecord):
    """Distinct displacement metrics and post-first-contact bounce count."""

    carry_distance_m: float
    bounce_air_distance_m: float
    skid_distance_m: float
    roll_distance_m: float
    surface_path_distance_m: float
    total_distance_m: float
    final_downrange_m: float
    final_offline_m: float
    bounce_count: int

    def __post_init__(self) -> None:
        for name in (
            "carry_distance_m",
            "bounce_air_distance_m",
            "skid_distance_m",
            "roll_distance_m",
            "surface_path_distance_m",
            "total_distance_m",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(
            self,
            "final_downrange_m",
            _finite(self.final_downrange_m, "final_downrange_m"),
        )
        object.__setattr__(
            self, "final_offline_m", _finite(self.final_offline_m, "final_offline_m")
        )
        object.__setattr__(
            self,
            "bounce_count",
            _integer(self.bounce_count, "bounce_count"),
        )


@dataclass(frozen=True)
class GroundTermination(_WireRecord):
    """Typed terminal state with a fail-closed completion matrix."""

    reason: GroundTerminationReason
    time_s: float
    completed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", GroundTerminationReason(self.reason))
        object.__setattr__(
            self, "time_s", _nonnegative(self.time_s, "termination time_s")
        )
        if not isinstance(self.completed, bool):
            raise ValueError("completed must be a boolean")
        expected = self.reason in {
            GroundTerminationReason.REST,
            GroundTerminationReason.LEFT_SURFACE,
        }
        if self.completed is not expected:
            raise ValueError("completed does not match termination reason")


@dataclass(frozen=True)
class GroundWarning(_WireRecord):
    """Typed, non-fatal model qualification."""

    code: str
    severity: GroundWarningSeverity
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _text(self.code, "warning code"))
        object.__setattr__(self, "severity", GroundWarningSeverity(self.severity))
        object.__setattr__(self, "message", _text(self.message, "warning message"))


__all__ = ["GroundSummary", "GroundTermination", "GroundWarning"]
