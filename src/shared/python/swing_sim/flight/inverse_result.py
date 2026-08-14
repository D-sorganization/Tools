"""Immutable result records for desired-flight inverse solving."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .inverse_contract import (
    RESULT_SCHEMA_VERSION,
    ObjectiveMode,
    SolverStatus,
    _exact,
    _wire_number,
)
from .result_contract import FlightMetricId


@dataclass(frozen=True)
class ParameterValue:
    """One decision-variable value in a returned candidate."""

    parameter_id: str
    unit: str
    value: float

    def __post_init__(self) -> None:
        if not self.parameter_id.strip() or not self.unit.strip():
            raise ValueError("parameter value ID and unit must be nonempty")
        if isinstance(self.value, bool) or not math.isfinite(self.value):
            raise ValueError("parameter value must be finite")

    def to_dict(self) -> dict[str, object]:
        """Return the wire representation."""
        return {
            "parameter_id": self.parameter_id,
            "unit": self.unit,
            "value": _wire_number(self.value),
        }


@dataclass(frozen=True)
class ObjectiveResidual:
    """Traceable residual and constraint violation for one objective."""

    metric_id: FlightMetricId
    unit: str
    mode: ObjectiveMode
    actual_value: float
    target_value: float | None
    normalized_residual: float
    constraint_violation: float
    provenance: str

    def __post_init__(self) -> None:
        values = (
            self.actual_value,
            self.normalized_residual,
            self.constraint_violation,
        )
        if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise ValueError("objective residual values must be finite")
        if self.target_value is not None and not math.isfinite(self.target_value):
            raise ValueError("objective target_value must be finite")
        if self.constraint_violation < 0.0:
            raise ValueError("constraint_violation must be nonnegative")
        if not self.unit.strip() or not self.provenance.strip():
            raise ValueError("objective residual unit and provenance must be nonempty")

    def to_dict(self) -> dict[str, object]:
        """Return the wire representation."""
        return {
            "actual_value": _wire_number(self.actual_value),
            "constraint_violation": _wire_number(self.constraint_violation),
            "metric_id": self.metric_id.value,
            "mode": self.mode.value,
            "normalized_residual": _wire_number(self.normalized_residual),
            "provenance": self.provenance,
            "target_value": (
                None if self.target_value is None else _wire_number(self.target_value)
            ),
            "unit": self.unit,
        }


@dataclass(frozen=True)
class SolutionCandidate:
    """One ranked, reproducible bounded-search candidate."""

    rank: int
    evaluation_index: int
    feasible: bool
    score: float
    parameters: tuple[ParameterValue, ...]
    residuals: tuple[ObjectiveResidual, ...]

    def __post_init__(self) -> None:
        if self.rank < 1 or self.evaluation_index < 0:
            raise ValueError("candidate rank and evaluation_index are out of range")
        if isinstance(self.feasible, bool) is False or not math.isfinite(self.score):
            raise ValueError("candidate feasible and score values are invalid")
        if not self.parameters or not self.residuals:
            raise ValueError("candidate parameters and residuals must be nonempty")
        if len({item.parameter_id for item in self.parameters}) != len(self.parameters):
            raise ValueError("candidate parameter IDs must be unique")
        if len({item.metric_id for item in self.residuals}) != len(self.residuals):
            raise ValueError("candidate residual metric IDs must be unique")

    def parameter(self, parameter_id: str) -> ParameterValue:
        """Return one parameter by stable ID."""
        return next(
            item for item in self.parameters if item.parameter_id == parameter_id
        )

    def residual(self, metric_id: FlightMetricId) -> ObjectiveResidual:
        """Return one objective residual by stable metric ID."""
        return next(item for item in self.residuals if item.metric_id is metric_id)

    def to_dict(self) -> dict[str, object]:
        """Return the wire representation."""
        return {
            "evaluation_index": self.evaluation_index,
            "feasible": self.feasible,
            "parameters": [item.to_dict() for item in self.parameters],
            "rank": self.rank,
            "residuals": [item.to_dict() for item in self.residuals],
            "score": _wire_number(self.score),
        }


ManifestFields = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class InverseFlightResult:
    """Terminal inverse-solver result with complete diagnostic counts."""

    problem_id: str
    status: SolverStatus
    termination_reason: str
    evaluations_attempted: int
    evaluations_completed: int
    no_impact_count: int
    failed_count: int
    candidates: tuple[SolutionCandidate, ...]
    provenance: ManifestFields
    schema_version: str = RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESULT_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if not self.problem_id.strip() or not self.termination_reason.strip():
            raise ValueError(
                "result problem_id and termination_reason must be nonempty"
            )
        counts = (
            self.evaluations_attempted,
            self.evaluations_completed,
            self.no_impact_count,
            self.failed_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise ValueError("result diagnostic counts must be nonnegative integers")
        if (
            self.evaluations_completed + self.no_impact_count + self.failed_count
            != self.evaluations_attempted
        ):
            raise ValueError(
                "result diagnostic counts must sum to evaluations_attempted"
            )
        if tuple(item.rank for item in self.candidates) != tuple(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("candidate ranks must be contiguous")
        normalized_provenance = tuple(sorted(self.provenance))
        if any(
            not key.strip() or not value.strip() for key, value in normalized_provenance
        ):
            raise ValueError("result provenance keys and values must be nonempty")
        if len({key for key, _ in normalized_provenance}) != len(normalized_provenance):
            raise ValueError("result provenance keys must be unique")
        object.__setattr__(self, "provenance", normalized_provenance)

    def to_dict(self) -> dict[str, object]:
        """Return the strict v1 wire representation."""
        return {
            "candidates": [item.to_dict() for item in self.candidates],
            "evaluations_attempted": self.evaluations_attempted,
            "evaluations_completed": self.evaluations_completed,
            "failed_count": self.failed_count,
            "no_impact_count": self.no_impact_count,
            "problem_id": self.problem_id,
            "provenance": {key: value for key, value in self.provenance},
            "schema_version": self.schema_version,
            "status": self.status.value,
            "termination_reason": self.termination_reason,
        }

    def to_json(self) -> str:
        """Serialize deterministically with sorted keys."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InverseFlightResult:
        """Parse solver-owned output after strict field validation."""
        _exact(
            payload,
            {
                "candidates",
                "evaluations_attempted",
                "evaluations_completed",
                "failed_count",
                "no_impact_count",
                "problem_id",
                "provenance",
                "schema_version",
                "status",
                "termination_reason",
            },
            "inverse result",
        )
        if payload["schema_version"] != RESULT_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {payload['schema_version']}")
        if not isinstance(payload["candidates"], list) or not isinstance(
            payload["provenance"], dict
        ):
            raise ValueError("result candidates and provenance have invalid types")
        candidates = tuple(_candidate_from_dict(item) for item in payload["candidates"])
        provenance = tuple(
            sorted(
                (str(key), str(value)) for key, value in payload["provenance"].items()
            )
        )
        return cls(
            str(payload["problem_id"]),
            SolverStatus(payload["status"]),
            str(payload["termination_reason"]),
            payload["evaluations_attempted"],
            payload["evaluations_completed"],
            payload["no_impact_count"],
            payload["failed_count"],
            candidates,
            provenance,
            str(payload["schema_version"]),
        )


def _candidate_from_dict(payload: dict[str, Any]) -> SolutionCandidate:
    """Parse one solver-owned candidate wire record."""
    _exact(
        payload,
        {"evaluation_index", "feasible", "parameters", "rank", "residuals", "score"},
        "solution candidate",
    )
    if not isinstance(payload["parameters"], list) or not isinstance(
        payload["residuals"], list
    ):
        raise ValueError("candidate parameters and residuals must be lists")
    parameters = tuple(_parameter_from_dict(item) for item in payload["parameters"])
    residuals = tuple(_residual_from_dict(item) for item in payload["residuals"])
    return SolutionCandidate(
        payload["rank"],
        payload["evaluation_index"],
        payload["feasible"],
        float(payload["score"]),
        parameters,
        residuals,
    )


def _parameter_from_dict(payload: dict[str, Any]) -> ParameterValue:
    """Parse one strict parameter-value record."""
    _exact(payload, {"parameter_id", "unit", "value"}, "parameter value")
    return ParameterValue(
        str(payload["parameter_id"]), str(payload["unit"]), float(payload["value"])
    )


def _residual_from_dict(payload: dict[str, Any]) -> ObjectiveResidual:
    """Parse one strict objective-residual record."""
    _exact(
        payload,
        {
            "actual_value",
            "constraint_violation",
            "metric_id",
            "mode",
            "normalized_residual",
            "provenance",
            "target_value",
            "unit",
        },
        "objective residual",
    )
    return ObjectiveResidual(
        FlightMetricId(payload["metric_id"]),
        str(payload["unit"]),
        ObjectiveMode(payload["mode"]),
        float(payload["actual_value"]),
        None if payload["target_value"] is None else float(payload["target_value"]),
        float(payload["normalized_residual"]),
        float(payload["constraint_violation"]),
        str(payload["provenance"]),
    )


__all__ = [
    "InverseFlightResult",
    "ObjectiveResidual",
    "ParameterValue",
    "SolutionCandidate",
]
