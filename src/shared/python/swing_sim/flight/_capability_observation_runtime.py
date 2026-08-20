"""Private runtime normalization for streamed capability observations."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .capability_contract import OptimizationRequest
from .capability_observation import (
    CapabilityOptimizationHooks,
    CapabilitySampleMetric,
    CapabilitySampleObservation,
    CapabilitySampleParameter,
    CapabilitySampleStatus,
)
from .capability_profile import ClubCapability, PlayerCapabilityProfile
from .inverse_contract import EvaluatedMetric, EvaluationStatus, SolverEvaluation
from .result_contract import FlightMetricId


@dataclass(frozen=True)
class _SampleAttempt:
    problem_id: str
    attempt_ordinal: int
    total_count: int
    candidate_ordinal: int
    club_candidate_ordinal: int
    sample_ordinal: int
    club: ClubCapability
    nominal: Mapping[str, float]
    perturbed: Mapping[str, float]


@dataclass(frozen=True)
class _OptimizationContext:
    profile: PlayerCapabilityProfile
    request: OptimizationRequest
    evaluator: Callable[[str, dict[str, float]], SolverEvaluation]
    hooks: CapabilityOptimizationHooks

    @property
    def total_count(self) -> int:
        return int(self.request.candidate_budget * self.request.ensemble_size)


@dataclass(frozen=True)
class _CandidateIdentity:
    candidate_ordinal: int
    club_candidate_ordinal: int


@dataclass(frozen=True)
class _NormalizedEvaluation:
    evaluation: SolverEvaluation | None
    source_status: EvaluationStatus | None
    effective_status: CapabilitySampleStatus
    reason_code: str | None
    source_reason: str | None


def _has_required_landing(evaluation: SolverEvaluation) -> bool:
    values = {item.metric_id: item.value for item in evaluation.metrics}
    return all(
        metric_id in values and math.isfinite(values[metric_id])
        for metric_id in (FlightMetricId.CARRY_DISTANCE, FlightMetricId.CARRY_OFFLINE)
    )


def _validated_evaluation(value: SolverEvaluation) -> SolverEvaluation | None:
    """Rebuild an evaluator result through every canonical runtime contract."""
    try:
        status: object = value.status
        if not isinstance(status, EvaluationStatus):
            return None
        metrics = tuple(
            EvaluatedMetric(FlightMetricId(item.metric_id), item.value, item.provenance)
            for item in value.metrics
        )
        return SolverEvaluation(status, metrics, value.reason)
    except Exception:
        return None


def _normalize(raw: object, *, evaluator_failed: bool) -> _NormalizedEvaluation:
    if evaluator_failed:
        return _NormalizedEvaluation(
            None, None, CapabilitySampleStatus.FAILED, "evaluator_exception", None
        )
    if not isinstance(raw, SolverEvaluation):
        return _invalid_result()
    evaluation = _validated_evaluation(raw)
    if evaluation is None:
        return _invalid_result()
    if evaluation.status is EvaluationStatus.COMPLETE:
        return _normalize_complete(evaluation)
    status = (
        CapabilitySampleStatus.NO_IMPACT
        if evaluation.status is EvaluationStatus.NO_IMPACT
        else CapabilitySampleStatus.FAILED
    )
    return _NormalizedEvaluation(
        evaluation,
        evaluation.status,
        status,
        evaluation.reason,
        evaluation.reason,
    )


def _invalid_result() -> _NormalizedEvaluation:
    return _NormalizedEvaluation(
        None,
        None,
        CapabilitySampleStatus.FAILED,
        "invalid_evaluator_result",
        None,
    )


def _normalize_complete(evaluation: SolverEvaluation) -> _NormalizedEvaluation:
    if _has_required_landing(evaluation):
        return _NormalizedEvaluation(
            evaluation,
            evaluation.status,
            CapabilitySampleStatus.COMPLETE,
            None,
            None,
        )
    return _NormalizedEvaluation(
        evaluation,
        evaluation.status,
        CapabilitySampleStatus.FAILED,
        "missing_required_landing_metrics",
        None,
    )


def _build_observation(
    attempt: _SampleAttempt, normalized: _NormalizedEvaluation
) -> CapabilitySampleObservation:
    evaluation = normalized.evaluation
    metrics = () if evaluation is None else evaluation.metrics
    return CapabilitySampleObservation(
        attempt.problem_id,
        attempt.attempt_ordinal,
        attempt.attempt_ordinal + 1,
        attempt.total_count,
        attempt.candidate_ordinal,
        attempt.club_candidate_ordinal,
        attempt.sample_ordinal,
        attempt.club.club_id,
        tuple(
            CapabilitySampleParameter(
                item.parameter_id,
                item.unit,
                attempt.nominal[item.parameter_id],
                attempt.perturbed[item.parameter_id],
            )
            for item in attempt.club.parameters
        ),
        normalized.source_status,
        normalized.effective_status,
        normalized.reason_code,
        normalized.source_reason,
        tuple(
            CapabilitySampleMetric(item.metric_id, item.value, item.provenance)
            for item in metrics
        ),
    )


def _evaluate_sample(
    attempt: _SampleAttempt,
    evaluator: Callable[[str, dict[str, float]], SolverEvaluation],
    hooks: CapabilityOptimizationHooks,
) -> _NormalizedEvaluation:
    """Evaluate, normalize, and synchronously emit one attempted sample."""
    try:
        raw: object = evaluator(attempt.club.club_id, dict(attempt.perturbed))
    except Exception:
        normalized = _normalize(None, evaluator_failed=True)
    else:
        normalized = _normalize(raw, evaluator_failed=False)
    observation = _build_observation(attempt, normalized)
    if hooks.observation_sink is not None:
        hooks.observation_sink(observation)
    return normalized


__all__: list[str] = []
