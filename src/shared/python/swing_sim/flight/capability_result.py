"""Versioned robust capability-optimization result contracts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

RESULT_SCHEMA_VERSION = "capability-optimization-result/v1"


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _exact(payload: dict[str, Any], fields: set[str], name: str) -> None:
    if set(payload) != fields:
        raise ValueError(f"{name} fields do not match v1 schema")


def _nonnegative_integer(value: int, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _boolean(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True)
class OptimizationAlternative:
    """One robustly evaluated club and delivery alternative."""

    rank: int
    club_id: str
    parameters: tuple[tuple[str, float], ...]
    score: float
    mean_carry_m: float
    expected_miss_m: float
    dispersion_rms_m: float
    target_hold_probability: float
    cvar_miss_m: float
    downside_carry_m: float
    sample_count: int
    successful_count: int
    no_impact_count: int
    failed_count: int
    failure_fraction: float
    confidence: float
    limiting_constraints: tuple[str, ...]
    extrapolated: bool
    pareto_efficient: bool

    def __post_init__(self) -> None:
        _nonnegative_integer(self.rank, "rank", positive=True)
        _nonnegative_integer(self.sample_count, "sample_count", positive=True)
        for name in ("successful_count", "no_impact_count", "failed_count"):
            _nonnegative_integer(getattr(self, name), name)
        if (
            self.successful_count + self.no_impact_count + self.failed_count
            != self.sample_count
        ):
            raise ValueError("alternative diagnostic counts must sum to sample_count")
        _boolean(self.extrapolated, "extrapolated")
        _boolean(self.pareto_efficient, "pareto_efficient")
        for name in (
            "score",
            "mean_carry_m",
            "expected_miss_m",
            "dispersion_rms_m",
            "target_hold_probability",
            "cvar_miss_m",
            "downside_carry_m",
            "failure_fraction",
            "confidence",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if any(
            not 0 <= getattr(self, name) <= 1
            for name in ("target_hold_probability", "failure_fraction", "confidence")
        ):
            raise ValueError("probabilities and confidence must lie within [0, 1]")
        parameters = tuple(
            (key.strip(), _finite(value, key)) for key, value in self.parameters
        )
        if any(not key for key, _value in parameters) or len(
            {key for key, _value in parameters}
        ) != len(parameters):
            raise ValueError("alternative parameter IDs must be nonempty and unique")
        object.__setattr__(self, "parameters", parameters)
        constraints = tuple(self.limiting_constraints)
        if any(not item.strip() for item in constraints) or len(
            set(constraints)
        ) != len(constraints):
            raise ValueError("limiting constraints must be nonempty and unique")
        object.__setattr__(self, "limiting_constraints", constraints)

    def to_dict(self) -> dict[str, object]:
        """Return the strict v1 wire representation."""
        return {
            "club_id": self.club_id,
            "confidence": self.confidence,
            "cvar_miss_m": self.cvar_miss_m,
            "dispersion_rms_m": self.dispersion_rms_m,
            "downside_carry_m": self.downside_carry_m,
            "expected_miss_m": self.expected_miss_m,
            "extrapolated": self.extrapolated,
            "failed_count": self.failed_count,
            "failure_fraction": self.failure_fraction,
            "limiting_constraints": list(self.limiting_constraints),
            "mean_carry_m": self.mean_carry_m,
            "no_impact_count": self.no_impact_count,
            "parameters": [
                {"parameter_id": key, "value": value} for key, value in self.parameters
            ],
            "pareto_efficient": self.pareto_efficient,
            "rank": self.rank,
            "sample_count": self.sample_count,
            "score": self.score,
            "successful_count": self.successful_count,
            "target_hold_probability": self.target_hold_probability,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OptimizationAlternative:
        """Parse a strict v1 alternative."""
        _exact(
            payload,
            {
                "club_id",
                "confidence",
                "cvar_miss_m",
                "dispersion_rms_m",
                "downside_carry_m",
                "expected_miss_m",
                "extrapolated",
                "failed_count",
                "failure_fraction",
                "limiting_constraints",
                "mean_carry_m",
                "no_impact_count",
                "parameters",
                "pareto_efficient",
                "rank",
                "sample_count",
                "score",
                "successful_count",
                "target_hold_probability",
            },
            "optimization alternative",
        )
        parameters = payload["parameters"]
        if any(set(item) != {"parameter_id", "value"} for item in parameters):
            raise ValueError("alternative parameter fields do not match v1 schema")
        return cls(
            payload["rank"],
            str(payload["club_id"]),
            tuple(
                (str(item["parameter_id"]), float(item["value"])) for item in parameters
            ),
            float(payload["score"]),
            float(payload["mean_carry_m"]),
            float(payload["expected_miss_m"]),
            float(payload["dispersion_rms_m"]),
            float(payload["target_hold_probability"]),
            float(payload["cvar_miss_m"]),
            float(payload["downside_carry_m"]),
            payload["sample_count"],
            payload["successful_count"],
            payload["no_impact_count"],
            payload["failed_count"],
            float(payload["failure_fraction"]),
            float(payload["confidence"]),
            tuple(str(item) for item in payload["limiting_constraints"]),
            payload["extrapolated"],
            payload["pareto_efficient"],
        )


@dataclass(frozen=True)
class OptimizationResult:
    """Deterministic robust optimization result and aggregate diagnostics."""

    problem_id: str
    status: str
    alternatives: tuple[OptimizationAlternative, ...]
    evaluations_attempted: int
    evaluations_completed: int
    no_impact_count: int
    failed_count: int
    provenance: tuple[tuple[str, str], ...]
    schema_version: str = RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESULT_SCHEMA_VERSION or self.status not in {
            "solved",
            "nonconverged",
        }:
            raise ValueError("unsupported result schema or status")
        alternatives = tuple(self.alternatives)
        if [item.rank for item in alternatives] != list(
            range(1, len(alternatives) + 1)
        ):
            raise ValueError("alternative ranks must be contiguous")
        for name in (
            "evaluations_attempted",
            "evaluations_completed",
            "no_impact_count",
            "failed_count",
        ):
            _nonnegative_integer(getattr(self, name), name)
        if (
            self.evaluations_completed + self.no_impact_count + self.failed_count
            != self.evaluations_attempted
        ):
            raise ValueError(
                "result diagnostic counts must sum to evaluations_attempted"
            )
        object.__setattr__(self, "alternatives", alternatives)
        provenance = tuple(self.provenance)
        if any(not key.strip() or not value.strip() for key, value in provenance):
            raise ValueError("result provenance entries must be nonempty")
        if len({key for key, _value in provenance}) != len(provenance):
            raise ValueError("result provenance keys must be unique")
        object.__setattr__(self, "provenance", provenance)

    def to_dict(self) -> dict[str, object]:
        """Return the strict v1 wire representation."""
        return {
            "alternatives": [item.to_dict() for item in self.alternatives],
            "evaluations_attempted": self.evaluations_attempted,
            "evaluations_completed": self.evaluations_completed,
            "failed_count": self.failed_count,
            "no_impact_count": self.no_impact_count,
            "problem_id": self.problem_id,
            "provenance": [
                {"key": key, "value": value} for key, value in self.provenance
            ],
            "schema_version": self.schema_version,
            "status": self.status,
        }

    def to_json(self) -> str:
        """Serialize deterministically."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OptimizationResult:
        """Parse a strict v1 result."""
        _exact(
            payload,
            {
                "alternatives",
                "evaluations_attempted",
                "evaluations_completed",
                "failed_count",
                "no_impact_count",
                "problem_id",
                "provenance",
                "schema_version",
                "status",
            },
            "optimization result",
        )
        provenance = payload["provenance"]
        if any(set(item) != {"key", "value"} for item in provenance):
            raise ValueError("result provenance fields do not match v1 schema")
        return cls(
            str(payload["problem_id"]),
            str(payload["status"]),
            tuple(
                OptimizationAlternative.from_dict(item)
                for item in payload["alternatives"]
            ),
            payload["evaluations_attempted"],
            payload["evaluations_completed"],
            payload["no_impact_count"],
            payload["failed_count"],
            tuple((str(item["key"]), str(item["value"])) for item in provenance),
            str(payload["schema_version"]),
        )


__all__ = ["OptimizationAlternative", "OptimizationResult", "RESULT_SCHEMA_VERSION"]
