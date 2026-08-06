"""Strict contracts for deterministic desired-flight inverse solving."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .result_contract import FlightMetricId, flight_metric_catalog

REQUEST_SCHEMA_VERSION = "inverse-flight-request/v1"
RESULT_SCHEMA_VERSION = "inverse-flight-result/v1"
_EXPORT_DECIMAL_PLACES = 11


class ObjectiveMode(StrEnum):
    """Supported objective directions."""

    TARGET = "target"
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class EvaluationStatus(StrEnum):
    """Outcome reported by the injected forward-model evaluator."""

    COMPLETE = "complete"
    NO_IMPACT = "no_impact"
    FAILED = "failed"
    NONCONVERGED = "nonconverged"


class SolverStatus(StrEnum):
    """Typed terminal state of one inverse solve."""

    SOLVED = "solved"
    INFEASIBLE = "infeasible"
    NO_IMPACT = "no_impact"
    NONCONVERGED = "nonconverged"


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _text(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be nonempty")
    return normalized


def _wire_number(value: float) -> float | int:
    rounded = round(value, _EXPORT_DECIMAL_PLACES)
    return int(rounded) if rounded.is_integer() else rounded


def _wire_optional(value: float | None) -> float | int | None:
    """Return an optional finite number in canonical wire precision."""
    return None if value is None else _wire_number(value)


def _optional_float(value: Any) -> float | None:
    """Parse a nullable numeric wire value."""
    return None if value is None else float(value)


def _exact(payload: dict[str, Any], fields: set[str], name: str) -> None:
    if set(payload) != fields:
        raise ValueError(f"{name} fields do not match v1 schema")


@dataclass(frozen=True)
class DecisionVariable:
    """One bounded, independently sampled forward-model parameter."""

    parameter_id: str
    unit: str
    lower_bound: float
    upper_bound: float
    initial_value: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parameter_id", _text(self.parameter_id, "parameter_id")
        )
        object.__setattr__(self, "unit", _text(self.unit, "variable unit"))
        lower = _finite(self.lower_bound, "variable lower_bound")
        upper = _finite(self.upper_bound, "variable upper_bound")
        initial = _finite(self.initial_value, "variable initial_value")
        if lower > upper:
            raise ValueError("variable lower_bound must not exceed upper_bound")
        if not lower <= initial <= upper:
            raise ValueError("variable initial_value must lie within bounds")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "initial_value", initial)

    def to_dict(self) -> dict[str, object]:
        """Return the strict v1 wire representation."""
        return {
            "initial_value": _wire_number(self.initial_value),
            "lower_bound": _wire_number(self.lower_bound),
            "parameter_id": self.parameter_id,
            "unit": self.unit,
            "upper_bound": _wire_number(self.upper_bound),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DecisionVariable:
        """Parse a strict v1 variable."""
        _exact(
            payload,
            {"initial_value", "lower_bound", "parameter_id", "unit", "upper_bound"},
            "decision variable",
        )
        return cls(
            str(payload["parameter_id"]),
            str(payload["unit"]),
            float(payload["lower_bound"]),
            float(payload["upper_bound"]),
            float(payload["initial_value"]),
        )


@dataclass(frozen=True)
class FlightObjective:
    """One canonical scalar flight metric objective and feasibility contract."""

    metric_id: FlightMetricId
    unit: str
    mode: ObjectiveMode
    target_value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    tolerance: float = 1.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        try:
            metric_id = FlightMetricId(self.metric_id)
            mode = ObjectiveMode(self.mode)
        except ValueError as exc:
            raise ValueError("objective metric_id and mode must be supported") from exc
        object.__setattr__(self, "metric_id", metric_id)
        object.__setattr__(self, "mode", mode)
        definition = flight_metric_catalog().definition(self.metric_id)
        if not definition.solver_objective:
            raise ValueError(f"{self.metric_id.value} is not solver-eligible")
        if self.unit != definition.unit:
            raise ValueError(
                f"{self.metric_id.value} canonical unit is {definition.unit}, "
                f"not {self.unit}"
            )
        target = self.target_value
        if self.mode is ObjectiveMode.TARGET and target is None:
            raise ValueError("target objectives require target_value")
        if self.mode is not ObjectiveMode.TARGET and target is not None:
            raise ValueError(
                "maximize/minimize objectives must not define target_value"
            )
        for name in ("target_value", "lower_bound", "upper_bound"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("objective lower_bound must not exceed upper_bound")
        tolerance = _finite(self.tolerance, "objective tolerance")
        weight = _finite(self.weight, "objective weight")
        if tolerance <= 0.0 or weight <= 0.0:
            raise ValueError("objective tolerance and weight must be positive")
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "weight", weight)

    def to_dict(self) -> dict[str, object]:
        """Return the strict v1 wire representation."""
        return {
            "lower_bound": _wire_optional(self.lower_bound),
            "metric_id": self.metric_id.value,
            "mode": self.mode.value,
            "target_value": _wire_optional(self.target_value),
            "tolerance": _wire_number(self.tolerance),
            "unit": self.unit,
            "upper_bound": _wire_optional(self.upper_bound),
            "weight": _wire_number(self.weight),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FlightObjective:
        """Parse a strict v1 objective."""
        _exact(
            payload,
            {
                "lower_bound",
                "metric_id",
                "mode",
                "target_value",
                "tolerance",
                "unit",
                "upper_bound",
                "weight",
            },
            "flight objective",
        )
        return cls(
            FlightMetricId(payload["metric_id"]),
            str(payload["unit"]),
            ObjectiveMode(payload["mode"]),
            _optional_float(payload["target_value"]),
            _optional_float(payload["lower_bound"]),
            _optional_float(payload["upper_bound"]),
            float(payload["tolerance"]),
            float(payload["weight"]),
        )


@dataclass(frozen=True)
class InverseFlightRequest:
    """Validated bounded inverse-flight problem definition."""

    problem_id: str
    variables: tuple[DecisionVariable, ...]
    objectives: tuple[FlightObjective, ...]
    max_evaluations: int
    candidate_count: int
    schema_version: str = REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", _text(self.problem_id, "problem_id"))
        if self.schema_version != REQUEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        variables = tuple(self.variables)
        objectives = tuple(self.objectives)
        if not variables or not objectives:
            raise ValueError("inverse solve requires variables and objectives")
        if len({item.parameter_id for item in variables}) != len(variables):
            raise ValueError("decision variable IDs must be unique")
        if len({item.metric_id for item in objectives}) != len(objectives):
            raise ValueError("objective metric IDs must be unique")
        if (
            isinstance(self.max_evaluations, bool)
            or not isinstance(self.max_evaluations, int)
            or self.max_evaluations < 1
        ):
            raise ValueError("max_evaluations must be a positive integer")
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or not 1 <= self.candidate_count <= self.max_evaluations
        ):
            raise ValueError("candidate_count must be between one and max_evaluations")
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "objectives", objectives)

    def to_dict(self) -> dict[str, object]:
        """Return the strict v1 wire representation."""
        return {
            "candidate_count": self.candidate_count,
            "max_evaluations": self.max_evaluations,
            "objectives": [item.to_dict() for item in self.objectives],
            "problem_id": self.problem_id,
            "schema_version": self.schema_version,
            "variables": [item.to_dict() for item in self.variables],
        }

    def to_json(self) -> str:
        """Serialize deterministically with sorted keys."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InverseFlightRequest:
        """Parse a strict v1 request."""
        _exact(
            payload,
            {
                "candidate_count",
                "max_evaluations",
                "objectives",
                "problem_id",
                "schema_version",
                "variables",
            },
            "inverse request",
        )
        if not isinstance(payload["variables"], list) or not isinstance(
            payload["objectives"], list
        ):
            raise ValueError("variables and objectives must be lists")
        return cls(
            str(payload["problem_id"]),
            tuple(DecisionVariable.from_dict(item) for item in payload["variables"]),
            tuple(FlightObjective.from_dict(item) for item in payload["objectives"]),
            payload["max_evaluations"],
            payload["candidate_count"],
            str(payload["schema_version"]),
        )


@dataclass(frozen=True)
class EvaluatedMetric:
    """One canonical scalar metric returned by a forward evaluator."""

    metric_id: FlightMetricId
    value: float
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite(self.value, "evaluated metric value"))
        object.__setattr__(
            self, "provenance", _text(self.provenance, "metric provenance")
        )


@dataclass(frozen=True)
class SolverEvaluation:
    """Forward-model outcome consumed by the inverse solver."""

    status: EvaluationStatus
    metrics: tuple[EvaluatedMetric, ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        metrics = tuple(self.metrics)
        if len({item.metric_id for item in metrics}) != len(metrics):
            raise ValueError("evaluated metric IDs must be unique")
        if self.status is EvaluationStatus.COMPLETE:
            if not metrics or self.reason is not None:
                raise ValueError("complete evaluations require metrics and no reason")
        elif metrics or self.reason is None or not self.reason.strip():
            raise ValueError("incomplete evaluations require a reason and no metrics")
        object.__setattr__(self, "metrics", metrics)


from .inverse_result import (  # noqa: E402 - facade export after base types exist
    InverseFlightResult,
    ObjectiveResidual,
    ParameterValue,
    SolutionCandidate,
)

__all__ = [
    "DecisionVariable",
    "EvaluatedMetric",
    "EvaluationStatus",
    "FlightObjective",
    "InverseFlightRequest",
    "InverseFlightResult",
    "ObjectiveMode",
    "ObjectiveResidual",
    "ParameterValue",
    "SolutionCandidate",
    "SolverEvaluation",
    "SolverStatus",
]
