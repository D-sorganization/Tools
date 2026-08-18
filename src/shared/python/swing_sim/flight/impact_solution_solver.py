"""Rank and summarize deterministic impact-delivery solution families."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Protocol

from .impact_solution_contract import (
    ForwardEvaluation,
    ForwardStatus,
    ImpactSolutionRequest,
    ModelManifest,
)
from .impact_solution_result import (
    FamilyMember,
    ImpactSolutionResult,
    MetricSensitivity,
    MetricValue,
    ParameterCorrelation,
    ParameterInterval,
    RejectedCandidate,
    SolutionFamily,
)
from .inverse_contract import (
    ParameterValue,
    SolutionCandidate,
    SolverEvaluation,
    SolverStatus,
)
from .inverse_solver import solve_inverse_flight
from .result_contract import FlightMetricId, flight_metric_catalog

_LAUNCH_METRICS = {
    FlightMetricId.BALL_SPEED,
    FlightMetricId.VERTICAL_LAUNCH_ANGLE,
    FlightMetricId.LAUNCH_DIRECTION,
    FlightMetricId.TOTAL_SPIN,
    FlightMetricId.SPIN_AXIS_TILT,
}


class ImpactForwardEvaluator(Protocol):
    """Minimal rich evaluator injected into the family solver."""

    @property
    def model_manifest(self) -> ModelManifest:
        """Return requested model identities and availability."""
        ...

    def evaluate(self, supplied: Mapping[str, float]) -> ForwardEvaluation:
        """Evaluate one exact decision-variable mapping."""
        ...


def _parameter_mapping(parameters: tuple[ParameterValue, ...]) -> dict[str, float]:
    return {item.parameter_id: item.value for item in parameters}


def _metric_value(evaluation: ForwardEvaluation, metric_id: FlightMetricId) -> float:
    for metric in evaluation.launch_metrics + evaluation.flight_metrics:
        if metric.metric_id is metric_id:
            return metric.value
    raise KeyError(metric_id.value)


def _member(
    candidate: SolutionCandidate, evaluation: ForwardEvaluation
) -> FamilyMember:
    catalog = flight_metric_catalog()
    launch_values = tuple(
        MetricValue(
            item.metric_id,
            catalog.definition(item.metric_id).unit,
            item.value,
            catalog.definition(item.metric_id).reference_event,
            item.provenance,
        )
        for item in evaluation.launch_metrics
    )
    launch_residuals = tuple(
        item for item in candidate.residuals if item.metric_id in _LAUNCH_METRICS
    )
    flight_residuals = tuple(
        item for item in candidate.residuals if item.metric_id not in _LAUNCH_METRICS
    )
    return FamilyMember(
        candidate.evaluation_index,
        candidate.feasible,
        candidate.score,
        candidate.parameters,
        launch_values,
        launch_residuals,
        flight_residuals,
    )


def _distance(
    request: ImpactSolutionRequest, left: FamilyMember, right: FamilyMember
) -> float:
    left_values = _parameter_mapping(left.parameters)
    right_values = _parameter_mapping(right.parameters)
    squared = 0.0
    for variable in request.inverse_request.variables:
        span = variable.upper_bound - variable.lower_bound
        delta = (
            0.0
            if span == 0.0
            else (
                left_values[variable.parameter_id] - right_values[variable.parameter_id]
            )
            / span
        )
        squared += delta * delta
    return math.sqrt(squared)


def _cluster(
    request: ImpactSolutionRequest, members: tuple[FamilyMember, ...]
) -> tuple[tuple[FamilyMember, ...], ...]:
    clusters: list[list[FamilyMember]] = []
    for member in members:
        target = next(
            (
                cluster
                for cluster in clusters
                if _distance(request, member, cluster[0]) <= request.family_radius
            ),
            None,
        )
        if target is not None:
            target.append(member)
        elif len(clusters) < request.family_count:
            clusters.append([member])
    return tuple(tuple(cluster) for cluster in clusters)


def _intervals(members: tuple[FamilyMember, ...]) -> tuple[ParameterInterval, ...]:
    intervals = []
    for parameter in members[0].parameters:
        values = [
            _parameter_mapping(member.parameters)[parameter.parameter_id]
            for member in members
        ]
        intervals.append(
            ParameterInterval(
                parameter.parameter_id, parameter.unit, min(values), max(values)
            )
        )
    return tuple(intervals)


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator == 0.0:
        return 0.0
    return max(
        -1.0,
        min(
            1.0,
            sum(a * b for a, b in zip(left_delta, right_delta, strict=True))
            / denominator,
        ),
    )


def _correlations(
    members: tuple[FamilyMember, ...],
) -> tuple[ParameterCorrelation, ...]:
    if len(members) < 2:
        return ()
    parameter_ids = sorted(item.parameter_id for item in members[0].parameters)
    mappings = [_parameter_mapping(member.parameters) for member in members]
    correlations = []
    for left_index, left_id in enumerate(parameter_ids):
        for right_id in parameter_ids[left_index + 1 :]:
            correlations.append(
                ParameterCorrelation(
                    left_id,
                    right_id,
                    _pearson(
                        [item[left_id] for item in mappings],
                        [item[right_id] for item in mappings],
                    ),
                    len(members),
                )
            )
    return tuple(correlations)


def _sensitivity(
    request: ImpactSolutionRequest,
    evaluator: ImpactForwardEvaluator,
    representative: FamilyMember,
    base_evaluation: ForwardEvaluation,
) -> tuple[MetricSensitivity, ...]:
    base = _parameter_mapping(representative.parameters)
    sensitivities = []
    for variable in request.inverse_request.variables:
        span = variable.upper_bound - variable.lower_bound
        step = request.sensitivity_fraction * span
        if step == 0.0:
            for objective in request.inverse_request.objectives:
                sensitivities.append(
                    MetricSensitivity(
                        variable.parameter_id,
                        variable.unit,
                        objective.metric_id,
                        objective.unit,
                        0.0,
                        "fixed_parameter",
                    )
                )
            continue
        lower_value = max(variable.lower_bound, base[variable.parameter_id] - step)
        upper_value = min(variable.upper_bound, base[variable.parameter_id] + step)
        lower_parameters = dict(base)
        upper_parameters = dict(base)
        lower_parameters[variable.parameter_id] = lower_value
        upper_parameters[variable.parameter_id] = upper_value
        lower = evaluator.evaluate(lower_parameters)
        upper = evaluator.evaluate(upper_parameters)
        for objective in request.inverse_request.objectives:
            derivative, method = _derivative(
                objective.metric_id,
                base[variable.parameter_id],
                _metric_value(base_evaluation, objective.metric_id),
                lower_value,
                lower,
                upper_value,
                upper,
            )
            if derivative is not None:
                sensitivities.append(
                    MetricSensitivity(
                        variable.parameter_id,
                        variable.unit,
                        objective.metric_id,
                        objective.unit,
                        derivative,
                        method,
                    )
                )
    return tuple(sensitivities)


def _derivative(
    metric_id: FlightMetricId,
    base_parameter: float,
    base_metric: float,
    lower_parameter: float,
    lower: ForwardEvaluation,
    upper_parameter: float,
    upper: ForwardEvaluation,
) -> tuple[float | None, str]:
    if (
        lower.status is ForwardStatus.COMPLETE
        and upper.status is ForwardStatus.COMPLETE
        and upper_parameter > lower_parameter
    ):
        return (
            (_metric_value(upper, metric_id) - _metric_value(lower, metric_id))
            / (upper_parameter - lower_parameter),
            "central_bounded_difference",
        )
    if upper.status is ForwardStatus.COMPLETE and upper_parameter > base_parameter:
        return (
            (_metric_value(upper, metric_id) - base_metric)
            / (upper_parameter - base_parameter),
            "forward_bounded_difference",
        )
    if lower.status is ForwardStatus.COMPLETE and base_parameter > lower_parameter:
        return (
            (base_metric - _metric_value(lower, metric_id))
            / (base_parameter - lower_parameter),
            "backward_bounded_difference",
        )
    return None, "unavailable"


def _family(
    rank: int,
    members: tuple[FamilyMember, ...],
    request: ImpactSolutionRequest,
    evaluator: ImpactForwardEvaluator,
    evaluations: list[ForwardEvaluation],
) -> SolutionFamily:
    representative = members[0]
    sensitivities = _sensitivity(
        request, evaluator, representative, evaluations[representative.evaluation_index]
    )
    return SolutionFamily(
        f"family-{rank:02d}",
        rank,
        representative.evaluation_index,
        members,
        _intervals(members),
        _correlations(members),
        sensitivities,
        representative.launch_residuals,
        representative.flight_residuals,
    )


def solve_impact_solution_families(
    request: ImpactSolutionRequest, evaluator: ImpactForwardEvaluator
) -> ImpactSolutionResult:
    """Solve desired flight and preserve distinct delivery families and misses."""
    evaluations: list[ForwardEvaluation] = []
    sampled_parameters: list[tuple[ParameterValue, ...]] = []

    def evaluate(parameters: dict[str, float]) -> SolverEvaluation:
        evaluation = evaluator.evaluate(parameters)
        evaluations.append(evaluation)
        sampled_parameters.append(
            tuple(
                ParameterValue(
                    item.parameter_id, item.unit, parameters[item.parameter_id]
                )
                for item in request.inverse_request.variables
            )
        )
        return evaluation.to_solver_evaluation()

    inverse = solve_inverse_flight(request.inverse_request, evaluate)
    feasible_members = tuple(
        _member(candidate, evaluations[candidate.evaluation_index])
        for candidate in inverse.candidates
        if candidate.feasible
    )
    clusters = _cluster(request, feasible_members)
    families = tuple(
        _family(index + 1, members, request, evaluator, evaluations)
        for index, members in enumerate(clusters)
    )
    retained = {
        member.evaluation_index for family in families for member in family.members
    }
    candidate_by_index = {item.evaluation_index: item for item in inverse.candidates}
    rejected = []
    for index, (evaluation, parameters) in enumerate(
        zip(evaluations, sampled_parameters, strict=True)
    ):
        if index in retained:
            continue
        candidate = candidate_by_index.get(index)
        if evaluation.status is not ForwardStatus.COMPLETE:
            reason = evaluation.reason or "forward_evaluation_failed"
        elif candidate is not None and not candidate.feasible:
            reason = "flight_objectives_not_met"
        elif candidate is not None:
            reason = "family_limit_or_radius_exceeded"
        else:
            reason = "outside_returned_candidate_budget"
        rejected.append(RejectedCandidate(index, evaluation.status, reason, parameters))
    status = SolverStatus.SOLVED if families else inverse.status
    reason = (
        "ranked_solution_families_found" if families else inverse.termination_reason
    )
    return ImpactSolutionResult(
        request.inverse_request.problem_id,
        status,
        reason,
        len(evaluations),
        families,
        tuple(rejected),
        evaluator.model_manifest,
        (
            ("family_solver_id", "deterministic-normalized-radius-clustering"),
            ("family_solver_version", "1.0.0"),
            ("inverse_result_schema", "inverse-flight-result/v1"),
        ),
    )


__all__ = ["ImpactForwardEvaluator", "solve_impact_solution_families"]
