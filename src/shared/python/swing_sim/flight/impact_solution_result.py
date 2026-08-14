"""Immutable result records for impact delivery solution families."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .impact_solution_contract import (
    RESULT_SCHEMA_VERSION,
    ForwardStatus,
    ModelManifest,
    _wire_number,
)
from .inverse_contract import (
    ObjectiveResidual,
    ParameterValue,
    SolverStatus,
)
from .result_contract import FlightMetricId


@dataclass(frozen=True)
class MetricValue:
    """One frame-independent scalar with units, event and provenance."""

    metric_id: FlightMetricId
    unit: str
    value: float
    reference_event: str
    provenance: str

    def __post_init__(self) -> None:
        if (
            not self.unit.strip()
            or not self.reference_event.strip()
            or not self.provenance.strip()
        ):
            raise ValueError(
                "metric unit, reference_event and provenance must be nonempty"
            )
        if isinstance(self.value, bool) or not math.isfinite(self.value):
            raise ValueError("metric value must be finite")

    def to_dict(self) -> dict[str, object]:
        """Return the strict wire record."""
        return {
            "metric_id": self.metric_id.value,
            "provenance": self.provenance,
            "reference_event": self.reference_event,
            "unit": self.unit,
            "value": _wire_number(self.value),
        }


@dataclass(frozen=True)
class ParameterInterval:
    """Observed feasible interval for one family delivery parameter."""

    parameter_id: str
    unit: str
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        if not self.parameter_id.strip() or not self.unit.strip():
            raise ValueError("interval parameter_id and unit must be nonempty")
        if (
            not math.isfinite(self.lower_bound)
            or not math.isfinite(self.upper_bound)
            or self.lower_bound > self.upper_bound
        ):
            raise ValueError("interval bounds must be finite and ordered")

    def to_dict(self) -> dict[str, object]:
        """Return the strict wire record."""
        return {
            "lower_bound": _wire_number(self.lower_bound),
            "parameter_id": self.parameter_id,
            "unit": self.unit,
            "upper_bound": _wire_number(self.upper_bound),
        }


@dataclass(frozen=True)
class ParameterCorrelation:
    """Pearson correlation between two delivery parameters in one family."""

    left_parameter_id: str
    right_parameter_id: str
    coefficient: float
    sample_count: int

    def __post_init__(self) -> None:
        if (
            not self.left_parameter_id.strip()
            or not self.right_parameter_id.strip()
            or self.left_parameter_id >= self.right_parameter_id
        ):
            raise ValueError("correlation parameter IDs must be nonempty and ordered")
        if not math.isfinite(self.coefficient) or abs(self.coefficient) > 1.0:
            raise ValueError("correlation coefficient must be within [-1, 1]")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 2
        ):
            raise ValueError("correlation sample_count must be at least two")

    def to_dict(self) -> dict[str, object]:
        """Return the strict wire record."""
        return {
            "coefficient": _wire_number(self.coefficient),
            "left_parameter_id": self.left_parameter_id,
            "right_parameter_id": self.right_parameter_id,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class MetricSensitivity:
    """Local finite-difference derivative at a family representative."""

    parameter_id: str
    parameter_unit: str
    metric_id: FlightMetricId
    metric_unit: str
    derivative: float
    method: str

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.parameter_id,
                self.parameter_unit,
                self.metric_unit,
                self.method,
            )
        ):
            raise ValueError("sensitivity text fields must be nonempty")
        if not math.isfinite(self.derivative):
            raise ValueError("sensitivity derivative must be finite")

    def to_dict(self) -> dict[str, object]:
        """Return the strict wire record."""
        return {
            "derivative": _wire_number(self.derivative),
            "method": self.method,
            "metric_id": self.metric_id.value,
            "metric_unit": self.metric_unit,
            "parameter_id": self.parameter_id,
            "parameter_unit": self.parameter_unit,
        }


@dataclass(frozen=True)
class FamilyMember:
    """One ranked inverse candidate with cached launch and residual outputs."""

    evaluation_index: int
    feasible: bool
    score: float
    parameters: tuple[ParameterValue, ...]
    launch_values: tuple[MetricValue, ...]
    launch_residuals: tuple[ObjectiveResidual, ...]
    flight_residuals: tuple[ObjectiveResidual, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.evaluation_index, bool)
            or not isinstance(self.evaluation_index, int)
            or self.evaluation_index < 0
            or not isinstance(self.feasible, bool)
            or not math.isfinite(self.score)
        ):
            raise ValueError("family member index, feasible and score are invalid")
        if (
            not self.parameters
            or not self.launch_values
            or not (self.launch_residuals or self.flight_residuals)
        ):
            raise ValueError(
                "family member requires parameters, launch values and residuals"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the strict wire record."""
        return {
            "evaluation_index": self.evaluation_index,
            "feasible": self.feasible,
            "flight_residuals": [item.to_dict() for item in self.flight_residuals],
            "launch_residuals": [item.to_dict() for item in self.launch_residuals],
            "launch_values": [item.to_dict() for item in self.launch_values],
            "parameters": [item.to_dict() for item in self.parameters],
            "score": _wire_number(self.score),
        }


@dataclass(frozen=True)
class SolutionFamily:
    """A ranked connected region of delivery solutions."""

    family_id: str
    rank: int
    representative_evaluation_index: int
    members: tuple[FamilyMember, ...]
    intervals: tuple[ParameterInterval, ...]
    correlations: tuple[ParameterCorrelation, ...]
    sensitivities: tuple[MetricSensitivity, ...]
    launch_residuals: tuple[ObjectiveResidual, ...]
    flight_residuals: tuple[ObjectiveResidual, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or isinstance(self.representative_evaluation_index, bool)
            or not isinstance(self.representative_evaluation_index, int)
            or not self.family_id.strip()
            or self.rank < 1
            or self.representative_evaluation_index < 0
        ):
            raise ValueError("family identity fields are invalid")
        if not self.members or not self.intervals:
            raise ValueError("family requires members and intervals")
        if self.members[0].evaluation_index != self.representative_evaluation_index:
            raise ValueError("first family member must be the representative")

    def to_dict(self) -> dict[str, object]:
        """Return the strict wire record."""
        return {
            "correlations": [item.to_dict() for item in self.correlations],
            "family_id": self.family_id,
            "flight_residuals": [item.to_dict() for item in self.flight_residuals],
            "intervals": [item.to_dict() for item in self.intervals],
            "launch_residuals": [item.to_dict() for item in self.launch_residuals],
            "members": [item.to_dict() for item in self.members],
            "rank": self.rank,
            "representative_evaluation_index": self.representative_evaluation_index,
            "sensitivities": [item.to_dict() for item in self.sensitivities],
        }


@dataclass(frozen=True)
class RejectedCandidate:
    """One sampled delivery not retained in a returned family."""

    evaluation_index: int
    status: ForwardStatus
    reason: str
    parameters: tuple[ParameterValue, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ForwardStatus(self.status))
        if (
            isinstance(self.evaluation_index, bool)
            or not isinstance(self.evaluation_index, int)
            or self.evaluation_index < 0
            or not self.reason.strip()
            or not self.parameters
        ):
            raise ValueError("rejected candidate fields are invalid")

    def to_dict(self) -> dict[str, object]:
        """Return the strict wire record."""
        return {
            "evaluation_index": self.evaluation_index,
            "parameters": [item.to_dict() for item in self.parameters],
            "reason": self.reason,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class ImpactSolutionResult:
    """Terminal family result with complete model and rejection diagnostics."""

    problem_id: str
    status: SolverStatus
    termination_reason: str
    evaluations_attempted: int
    families: tuple[SolutionFamily, ...]
    rejected_candidates: tuple[RejectedCandidate, ...]
    model_manifest: ModelManifest
    provenance: tuple[tuple[str, str], ...]
    schema_version: str = RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SolverStatus(self.status))
        if (
            self.schema_version != RESULT_SCHEMA_VERSION
            or not self.problem_id.strip()
            or not self.termination_reason.strip()
        ):
            raise ValueError("invalid impact solution result identity")
        if (
            isinstance(self.evaluations_attempted, bool)
            or not isinstance(self.evaluations_attempted, int)
            or self.evaluations_attempted < 0
        ):
            raise ValueError("evaluations_attempted must be a nonnegative integer")
        if tuple(item.rank for item in self.families) != tuple(
            range(1, len(self.families) + 1)
        ):
            raise ValueError("family ranks must be contiguous")
        retained = sum(len(item.members) for item in self.families)
        if retained + len(self.rejected_candidates) != self.evaluations_attempted:
            raise ValueError(
                "retained and rejected evaluation counts must match attempted"
            )
        evaluation_indices = [
            member.evaluation_index
            for family in self.families
            for member in family.members
        ] + [item.evaluation_index for item in self.rejected_candidates]
        if sorted(evaluation_indices) != list(range(self.evaluations_attempted)):
            raise ValueError("each attempted evaluation must appear exactly once")
        provenance = tuple(sorted(self.provenance))
        if not provenance or len({key for key, _ in provenance}) != len(provenance):
            raise ValueError("result provenance must be nonempty and unique")
        object.__setattr__(self, "provenance", provenance)

    def to_dict(self) -> dict[str, object]:
        """Return the strict v1 wire representation."""
        return {
            "evaluations_attempted": self.evaluations_attempted,
            "families": [item.to_dict() for item in self.families],
            "model_manifest": self.model_manifest.to_dict(),
            "problem_id": self.problem_id,
            "provenance": dict(self.provenance),
            "rejected_candidates": [
                item.to_dict() for item in self.rejected_candidates
            ],
            "schema_version": self.schema_version,
            "status": self.status.value,
            "termination_reason": self.termination_reason,
        }

    def to_json(self) -> str:
        """Serialize deterministically with sorted keys."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ImpactSolutionResult:
        """Parse a strict v1 result."""
        from .impact_solution_parsing import parse_impact_solution_result

        return parse_impact_solution_result(payload)


__all__ = [
    "FamilyMember",
    "ImpactSolutionResult",
    "MetricSensitivity",
    "MetricValue",
    "ParameterCorrelation",
    "ParameterInterval",
    "RejectedCandidate",
    "SolutionFamily",
]
