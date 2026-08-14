"""Versioned robust capability-optimization request contracts and facade."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from shared.python.compatibility import StrEnum

from .capability_profile import (
    CapabilityParameter,
    ClubCapability,
    PlayerCapabilityProfile,
)

REQUEST_SCHEMA_VERSION = "capability-optimization-request/v1"


class CapabilityObjective(StrEnum):
    """Supported robust shot-selection objectives."""

    MAXIMIZE_CARRY = "maximize_carry"
    MINIMIZE_EXPECTED_MISS = "minimize_expected_miss"
    MAXIMIZE_TARGET_HOLD = "maximize_target_hold"
    MINIMIZE_VARIABILITY = "minimize_variability"
    MINIMIZE_DOWNSIDE = "minimize_downside"
    DISTANCE_CONTROL_PARETO = "distance_control_pareto"


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _text(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be nonempty")
    return normalized


def _exact(payload: dict[str, Any], fields: set[str], name: str) -> None:
    if set(payload) != fields:
        raise ValueError(f"{name} fields do not match v1 schema")


@dataclass(frozen=True)
class TargetDefinition:
    """Serializable landing target compatible with the solver target geometry."""

    kind: str
    distance_m: float
    lateral_m: float
    radius_m: float
    band_half_length_m: float
    half_width_m: float

    def __post_init__(self) -> None:
        if self.kind not in {"green", "fairway"}:
            raise ValueError("target kind must be green or fairway")
        for name in (
            "distance_m",
            "lateral_m",
            "radius_m",
            "band_half_length_m",
            "half_width_m",
        ):
            object.__setattr__(
                self, name, _finite(getattr(self, name), f"target {name}")
            )
        if (
            self.distance_m <= 0
            or self.radius_m <= 0
            or self.band_half_length_m <= 0
            or self.half_width_m <= 0
        ):
            raise ValueError("target sizes and distance must be positive")

    def to_dict(self) -> dict[str, object]:
        """Return the strict target representation."""
        return {
            "band_half_length_m": self.band_half_length_m,
            "distance_m": self.distance_m,
            "half_width_m": self.half_width_m,
            "kind": self.kind,
            "lateral_m": self.lateral_m,
            "radius_m": self.radius_m,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TargetDefinition:
        """Parse a strict target definition."""
        _exact(
            payload,
            {
                "band_half_length_m",
                "distance_m",
                "half_width_m",
                "kind",
                "lateral_m",
                "radius_m",
            },
            "target definition",
        )
        return cls(
            str(payload["kind"]),
            float(payload["distance_m"]),
            float(payload["lateral_m"]),
            float(payload["radius_m"]),
            float(payload["band_half_length_m"]),
            float(payload["half_width_m"]),
        )


@dataclass(frozen=True)
class OptimizationRequest:
    """Bounded deterministic ensemble optimization request."""

    problem_id: str
    objective: CapabilityObjective
    club_ids: tuple[str, ...]
    target: TargetDefinition
    candidate_budget: int
    ensemble_size: int
    alternatives_count: int
    seed: int
    cvar_alpha: float
    minimum_success_fraction: float
    schema_version: str = REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", _text(self.problem_id, "problem_id"))
        object.__setattr__(self, "objective", CapabilityObjective(self.objective))
        clubs = tuple(_text(item, "request club_id") for item in self.club_ids)
        if not clubs or len(set(clubs)) != len(clubs):
            raise ValueError("request club IDs must be nonempty and unique")
        for name in ("candidate_budget", "ensemble_size", "alternatives_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.alternatives_count > self.candidate_budget:
            raise ValueError("alternatives_count must not exceed candidate_budget")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        for name in ("cvar_alpha", "minimum_success_fraction"):
            value = _finite(getattr(self, name), name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must lie within (0, 1]")
            object.__setattr__(self, name, value)
        if self.schema_version != REQUEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        object.__setattr__(self, "club_ids", clubs)

    def to_dict(self) -> dict[str, object]:
        """Return the strict v1 wire representation."""
        return {
            "alternatives_count": self.alternatives_count,
            "candidate_budget": self.candidate_budget,
            "club_ids": list(self.club_ids),
            "cvar_alpha": self.cvar_alpha,
            "ensemble_size": self.ensemble_size,
            "minimum_success_fraction": self.minimum_success_fraction,
            "objective": self.objective.value,
            "problem_id": self.problem_id,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "target": self.target.to_dict(),
        }

    def to_json(self) -> str:
        """Serialize deterministically."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OptimizationRequest:
        """Parse a strict v1 request."""
        _exact(
            payload,
            {
                "alternatives_count",
                "candidate_budget",
                "club_ids",
                "cvar_alpha",
                "ensemble_size",
                "minimum_success_fraction",
                "objective",
                "problem_id",
                "schema_version",
                "seed",
                "target",
            },
            "optimization request",
        )
        return cls(
            str(payload["problem_id"]),
            CapabilityObjective(payload["objective"]),
            tuple(str(item) for item in payload["club_ids"]),
            TargetDefinition.from_dict(payload["target"]),
            payload["candidate_budget"],
            payload["ensemble_size"],
            payload["alternatives_count"],
            payload["seed"],
            float(payload["cvar_alpha"]),
            float(payload["minimum_success_fraction"]),
            str(payload["schema_version"]),
        )


from .capability_result import OptimizationAlternative, OptimizationResult  # noqa: E402

__all__ = [
    "CapabilityObjective",
    "CapabilityParameter",
    "ClubCapability",
    "OptimizationAlternative",
    "OptimizationRequest",
    "OptimizationResult",
    "PlayerCapabilityProfile",
    "TargetDefinition",
]
