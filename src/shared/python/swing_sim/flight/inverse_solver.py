"""Deterministic bounded search for desired ball-flight outcomes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .inverse_contract import (
    EvaluationStatus,
    FlightObjective,
    InverseFlightRequest,
    InverseFlightResult,
    ObjectiveMode,
    ObjectiveResidual,
    ParameterValue,
    SolutionCandidate,
    SolverEvaluation,
    SolverStatus,
)

ForwardEvaluator = Callable[[dict[str, float]], SolverEvaluation]
_CONSTRAINT_PENALTY = 1_000_000.0
_PROVENANCE = (
    ("metric_schema", "ball-flight-metrics/v1"),
    ("sampler", "halton-sequence-with-initial-point"),
    ("solver_id", "deterministic-bounded-search"),
    ("solver_version", "1.0.0"),
)


def _is_prime(candidate: int) -> bool:
    divisor = 2
    while divisor * divisor <= candidate:
        if candidate % divisor == 0:
            return False
        divisor += 1
    return candidate >= 2


def _first_primes(count: int) -> tuple[int, ...]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if _is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    remaining = index
    while remaining:
        result += factor * (remaining % base)
        remaining //= base
        factor /= base
    return result


def _sample_points(request: InverseFlightRequest) -> tuple[dict[str, float], ...]:
    variables = request.variables
    initial = {item.parameter_id: item.initial_value for item in variables}
    if all(item.lower_bound == item.upper_bound for item in variables):
        return (initial,)
    points = [initial]
    seen = {tuple(initial[item.parameter_id] for item in variables)}
    bases = _first_primes(len(variables))
    index = 1
    sequence_limit = request.max_evaluations * 1024
    while len(points) < request.max_evaluations and index <= sequence_limit:
        point = {
            item.parameter_id: item.lower_bound
            + _radical_inverse(index, base) * (item.upper_bound - item.lower_bound)
            for item, base in zip(variables, bases, strict=True)
        }
        key = tuple(point[item.parameter_id] for item in variables)
        if key not in seen:
            points.append(point)
            seen.add(key)
        index += 1
    return tuple(points)


def _constraint_violation(value: float, objective: FlightObjective) -> float:
    if objective.lower_bound is not None and value < objective.lower_bound:
        return (objective.lower_bound - value) / objective.tolerance
    if objective.upper_bound is not None and value > objective.upper_bound:
        return (value - objective.upper_bound) / objective.tolerance
    return 0.0


def _residual(
    value: float, objective: FlightObjective, provenance: str
) -> ObjectiveResidual:
    if objective.mode is ObjectiveMode.TARGET:
        assert objective.target_value is not None
        normalized = (value - objective.target_value) / objective.tolerance
    elif objective.mode is ObjectiveMode.MAXIMIZE:
        normalized = -value / objective.tolerance
    else:
        normalized = value / objective.tolerance
    return ObjectiveResidual(
        objective.metric_id,
        objective.unit,
        objective.mode,
        value,
        objective.target_value,
        normalized,
        _constraint_violation(value, objective),
        provenance,
    )


def _is_feasible(residual: ObjectiveResidual, objective: FlightObjective) -> bool:
    within_target = (
        objective.mode is not ObjectiveMode.TARGET
        or abs(residual.normalized_residual) <= 1.0
    )
    return within_target and residual.constraint_violation == 0.0


def _objective_loss(residual: ObjectiveResidual, objective: FlightObjective) -> float:
    directional = (
        abs(residual.normalized_residual)
        if objective.mode is ObjectiveMode.TARGET
        else residual.normalized_residual
    )
    return objective.weight * (
        directional + _CONSTRAINT_PENALTY * residual.constraint_violation
    )


def _candidate(
    request: InverseFlightRequest,
    parameters: dict[str, float],
    evaluation: SolverEvaluation,
    evaluation_index: int,
) -> SolutionCandidate | None:
    values = {item.metric_id: item for item in evaluation.metrics}
    if any(objective.metric_id not in values for objective in request.objectives):
        return None
    residuals = tuple(
        _residual(
            values[objective.metric_id].value,
            objective,
            values[objective.metric_id].provenance,
        )
        for objective in request.objectives
    )
    feasible = all(
        _is_feasible(residual, objective)
        for residual, objective in zip(residuals, request.objectives, strict=True)
    )
    score = sum(
        _objective_loss(residual, objective)
        for residual, objective in zip(residuals, request.objectives, strict=True)
    )
    parameter_values = tuple(
        ParameterValue(item.parameter_id, item.unit, parameters[item.parameter_id])
        for item in request.variables
    )
    return SolutionCandidate(
        1, evaluation_index, feasible, score, parameter_values, residuals
    )


def _static_infeasibility(request: InverseFlightRequest) -> str | None:
    for objective in request.objectives:
        target = objective.target_value
        if target is None:
            continue
        if objective.lower_bound is not None and target < objective.lower_bound:
            return "target_outside_objective_bounds"
        if objective.upper_bound is not None and target > objective.upper_bound:
            return "target_outside_objective_bounds"
    return None


def _empty_result(
    request: InverseFlightRequest, status: SolverStatus, reason: str
) -> InverseFlightResult:
    return InverseFlightResult(
        request.problem_id, status, reason, 0, 0, 0, 0, (), _PROVENANCE
    )


def solve_inverse_flight(
    request: InverseFlightRequest, evaluator: ForwardEvaluator
) -> InverseFlightResult:
    """Search bounded inputs and rank results under explicit objective contracts.

    Preconditions:
        ``request`` has passed its DbC validation and ``evaluator`` is pure for a
        fixed parameter mapping.
    Postconditions:
        At most ``candidate_count`` candidates are returned; ranks and diagnostic
        counts are deterministic for deterministic evaluator behavior.
    """
    static_reason = _static_infeasibility(request)
    if static_reason:
        return _empty_result(request, SolverStatus.INFEASIBLE, static_reason)

    points = _sample_points(request)
    candidates: list[SolutionCandidate] = []
    completed = no_impact = failed = 0
    for evaluation_index, parameters in enumerate(points):
        try:
            raw_evaluation: object = evaluator(dict(parameters))
        except Exception:  # noqa: BLE001 - A forward-model fault is a typed failed evaluation.
            failed += 1
            continue
        if not isinstance(raw_evaluation, SolverEvaluation):
            failed += 1
            continue
        evaluation = raw_evaluation
        if evaluation.status is EvaluationStatus.NO_IMPACT:
            no_impact += 1
            continue
        if evaluation.status is not EvaluationStatus.COMPLETE:
            failed += 1
            continue
        candidate = _candidate(request, parameters, evaluation, evaluation_index)
        if candidate is None:
            failed += 1
            continue
        completed += 1
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            not item.feasible,
            sum(residual.constraint_violation for residual in item.residuals),
            item.score,
            item.evaluation_index,
        )
    )
    ranked = tuple(
        replace(candidate, rank=rank)
        for rank, candidate in enumerate(candidates[: request.candidate_count], start=1)
    )
    attempted = len(points)
    if ranked and ranked[0].feasible:
        status, reason = SolverStatus.SOLVED, "objective_tolerances_met"
    elif no_impact == attempted:
        status, reason = SolverStatus.NO_IMPACT, "all_evaluations_no_impact"
    elif completed:
        status, reason = SolverStatus.NONCONVERGED, "evaluation_budget_exhausted"
    else:
        status, reason = SolverStatus.NONCONVERGED, "no_complete_evaluations"
    return InverseFlightResult(
        request.problem_id,
        status,
        reason,
        attempted,
        completed,
        no_impact,
        failed,
        ranked,
        _PROVENANCE,
    )


__all__ = ["ForwardEvaluator", "solve_inverse_flight"]
