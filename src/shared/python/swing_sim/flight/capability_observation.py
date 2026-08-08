"""Immutable streaming observations for capability-optimizer sample attempts."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from .inverse_contract import EvaluationStatus
from .result_contract import FlightMetricId

CAPABILITY_SAMPLE_OBSERVATION_SCHEMA_VERSION = "capability-sample-observation/v1"


class CapabilitySampleStatus(StrEnum):
    """Normalized terminal state of one attempted capability sample."""

    COMPLETE = "complete"
    NO_IMPACT = "no_impact"
    FAILED = "failed"


_SOURCELESS_FAILURE_CODES = {"evaluator_exception", "invalid_evaluator_result"}


@dataclass(frozen=True)
class CapabilitySampleParameter:
    """One declared capability parameter at nominal and perturbed values."""

    parameter_id: str
    unit: str
    nominal_value: float
    perturbed_value: float

    def __post_init__(self) -> None:
        if not self.parameter_id.strip() or not self.unit.strip():
            raise ValueError("observation parameter identity must be nonempty")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("observation parameter values must be finite")

    @property
    def values(self) -> tuple[float, float]:
        """Return numeric values for compact invariant validation."""
        return self.nominal_value, self.perturbed_value

    def to_wire(self) -> dict[str, object]:
        """Return the exact v1 wire representation."""
        return {
            "parameter_id": self.parameter_id,
            "unit": self.unit,
            "nominal_value": self.nominal_value,
            "perturbed_value": self.perturbed_value,
        }


@dataclass(frozen=True)
class CapabilitySampleMetric:
    """One evaluator metric preserved in evaluator declaration order."""

    metric_id: FlightMetricId
    value: float
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_id", FlightMetricId(self.metric_id))
        if not math.isfinite(self.value):
            raise ValueError("observation metric value must be finite")
        if not self.provenance.strip():
            raise ValueError("observation metric provenance must be nonempty")

    def to_wire(self) -> dict[str, object]:
        """Return the exact v1 wire representation."""
        return {
            "metric_id": self.metric_id.value,
            "value": self.value,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class CapabilitySampleObservation:
    """One immutable, non-retained capability sample observation."""

    problem_id: str
    attempt_ordinal: int
    attempted_count: int
    total_count: int
    candidate_ordinal: int
    club_candidate_ordinal: int
    sample_ordinal: int
    club_id: str
    parameters: tuple[CapabilitySampleParameter, ...]
    source_status: EvaluationStatus | None
    effective_status: CapabilitySampleStatus
    reason_code: str | None
    source_reason: str | None
    metrics: tuple[CapabilitySampleMetric, ...]
    schema_version: str = CAPABILITY_SAMPLE_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CAPABILITY_SAMPLE_OBSERVATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if not self.problem_id.strip() or not self.club_id.strip():
            raise ValueError("observation problem_id and club_id must be nonempty")
        ordinals = self.identity_ordinals
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in ordinals
        ):
            raise ValueError("observation ordinals must be nonnegative integers")
        if self.attempted_count != self.attempt_ordinal + 1:
            raise ValueError("attempted_count must equal attempt_ordinal + 1")
        if self.total_count < self.attempted_count:
            raise ValueError("total_count must include the attempted sample")
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if not self.parameters or not all(
            isinstance(item, CapabilitySampleParameter) for item in self.parameters
        ):
            raise ValueError("observation parameters must be nonempty")
        if not all(isinstance(item, CapabilitySampleMetric) for item in self.metrics):
            raise ValueError("observation metrics must use the v1 metric contract")
        if len({item.parameter_id for item in self.parameters}) != len(self.parameters):
            raise ValueError("observation parameter IDs must be unique")
        if len({item.metric_id for item in self.metrics}) != len(self.metrics):
            raise ValueError("observation metric IDs must be unique")
        object.__setattr__(
            self, "effective_status", CapabilitySampleStatus(self.effective_status)
        )
        if self.source_status is not None:
            object.__setattr__(
                self, "source_status", EvaluationStatus(self.source_status)
            )
        _validate_status_contract(self)

    @property
    def identity_ordinals(self) -> tuple[int, ...]:
        """Return ordering integers for compact invariant validation."""
        return (
            self.attempt_ordinal,
            self.attempted_count,
            self.total_count,
            self.candidate_ordinal,
            self.club_candidate_ordinal,
            self.sample_ordinal,
        )

    def to_wire(self) -> dict[str, object]:
        """Return the exact ordered v1 wire representation."""
        return {
            "schema_version": self.schema_version,
            "problem_id": self.problem_id,
            "attempt_ordinal": self.attempt_ordinal,
            "attempted_count": self.attempted_count,
            "total_count": self.total_count,
            "candidate_ordinal": self.candidate_ordinal,
            "club_candidate_ordinal": self.club_candidate_ordinal,
            "sample_ordinal": self.sample_ordinal,
            "club_id": self.club_id,
            "parameters": [item.to_wire() for item in self.parameters],
            "source_status": (
                None if self.source_status is None else self.source_status.value
            ),
            "effective_status": self.effective_status.value,
            "reason_code": self.reason_code,
            "source_reason": self.source_reason,
            "metrics": [item.to_wire() for item in self.metrics],
        }


def _validate_status_contract(observation: CapabilitySampleObservation) -> None:
    source = observation.source_status
    effective = observation.effective_status
    code = observation.reason_code
    reason = observation.source_reason
    if any(value is not None and not value.strip() for value in (code, reason)):
        raise ValueError("observation reasons must be nonempty when present")
    if effective is CapabilitySampleStatus.COMPLETE:
        valid = source is EvaluationStatus.COMPLETE and code is None and reason is None
    elif effective is CapabilitySampleStatus.NO_IMPACT:
        valid = (
            source is EvaluationStatus.NO_IMPACT and code is not None and code == reason
        )
    elif source is None:
        valid = code in _SOURCELESS_FAILURE_CODES and reason is None
    elif source is EvaluationStatus.COMPLETE:
        valid = code == "missing_required_landing_metrics" and reason is None
    else:
        valid = source in {EvaluationStatus.FAILED, EvaluationStatus.NONCONVERGED}
        valid = valid and code is not None and code == reason
    if not valid:
        raise ValueError("observation status and reason fields are inconsistent")


ObservationSink: TypeAlias = Callable[[CapabilitySampleObservation], None]
CancellationCheck: TypeAlias = Callable[[], bool]


@dataclass(frozen=True)
class CapabilityOptimizationHooks:
    """Optional synchronous observation and cooperative-cancellation hooks."""

    observation_sink: ObservationSink | None = None
    should_cancel: CancellationCheck | None = None

    def __post_init__(self) -> None:
        for name in ("observation_sink", "should_cancel"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable")


class CapabilityOptimizationCancelled(RuntimeError):
    """Typed cooperative cancellation with deterministic progress counts."""

    def __init__(self, attempted_count: int, total_count: int) -> None:
        if (
            isinstance(attempted_count, bool)
            or not isinstance(attempted_count, int)
            or attempted_count < 0
        ):
            raise ValueError("attempted_count must be a nonnegative integer")
        if (
            isinstance(total_count, bool)
            or not isinstance(total_count, int)
            or total_count < attempted_count
        ):
            raise ValueError("total_count must be an integer at least attempted_count")
        self.attempted_count = attempted_count
        self.total_count = total_count
        super().__init__(
            f"capability optimization cancelled after {attempted_count} "
            f"of {total_count} attempts"
        )


__all__ = [
    "CAPABILITY_SAMPLE_OBSERVATION_SCHEMA_VERSION",
    "CancellationCheck",
    "CapabilityOptimizationCancelled",
    "CapabilityOptimizationHooks",
    "CapabilitySampleMetric",
    "CapabilitySampleObservation",
    "CapabilitySampleParameter",
    "CapabilitySampleStatus",
    "ObservationSink",
]
