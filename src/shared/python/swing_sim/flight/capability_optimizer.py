"""Deterministic robust optimization over clubs and delivery capabilities."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.python.swing_sim.solver.targets import TargetRegion

from .capability_contract import (
    CapabilityObjective,
    ClubCapability,
    OptimizationAlternative,
    OptimizationRequest,
    OptimizationResult,
    PlayerCapabilityProfile,
)
from .capability_sampling import (
    sample_candidate_parameters,
    sample_perturbed_parameters,
)
from .inverse_contract import EvaluationStatus, SolverEvaluation
from .result_contract import FlightMetricId

CapabilityEvaluator = Callable[[str, dict[str, float]], SolverEvaluation]
_FAILURE_PENALTY = 1_000_000.0
_BOUNDARY_TOLERANCE = 1e-9
_PROVENANCE = (
    ("ensemble", "deterministic-correlated-low-discrepancy/v1"),
    ("flight_metrics", "ball-flight-metrics/v1"),
    ("optimizer", "capability-optimizer/v1"),
    ("target_geometry", "swing_sim.solver.targets/v1"),
)


@dataclass(frozen=True)
class _Landing:
    carry_m: float
    offline_m: float


@dataclass(frozen=True)
class _RiskMetrics:
    """Robust landing summaries consumed by objective scoring."""

    mean_carry_m: float
    expected_miss_m: float
    hold_probability: float
    dispersion_rms_m: float
    cvar_miss_m: float
    downside_carry_m: float


@dataclass(frozen=True)
class _Counts:
    completed: int = 0
    no_impact: int = 0
    failed: int = 0

    def add(self, status: EvaluationStatus) -> _Counts:
        """Return counts with one typed outcome added."""
        if status is EvaluationStatus.COMPLETE:
            return replace(self, completed=self.completed + 1)
        if status is EvaluationStatus.NO_IMPACT:
            return replace(self, no_impact=self.no_impact + 1)
        return replace(self, failed=self.failed + 1)


def _landing(evaluation: SolverEvaluation) -> _Landing | None:
    if evaluation.status is not EvaluationStatus.COMPLETE:
        return None
    metrics = {item.metric_id: item.value for item in evaluation.metrics}
    carry = metrics.get(FlightMetricId.CARRY_DISTANCE)
    offline = metrics.get(FlightMetricId.CARRY_OFFLINE)
    if (
        carry is None
        or offline is None
        or not math.isfinite(carry)
        or not math.isfinite(offline)
    ):
        return None
    return _Landing(carry, offline)


def _target(request: OptimizationRequest) -> TargetRegion:
    from shared.python.swing_sim.solver.targets import TargetRegion

    value = request.target
    return TargetRegion(
        value.kind,
        value.distance_m,
        value.radius_m,
        value.lateral_m,
        value.band_half_length_m,
        value.half_width_m,
    )


def _tail_mean(values: list[float], alpha: float, *, reverse: bool) -> float:
    count = max(1, math.ceil(len(values) * (1.0 - alpha)))
    return sum(sorted(values, reverse=reverse)[:count]) / count


def _limiting_constraints(
    club: ClubCapability,
    nominal: dict[str, float],
    success_fraction: float,
    extrapolated: bool,
    request: OptimizationRequest,
) -> tuple[str, ...]:
    limiting: list[str] = []
    for item in club.parameters:
        value = nominal[item.parameter_id]
        if abs(value - item.lower_bound) <= _BOUNDARY_TOLERANCE:
            limiting.append(f"{item.parameter_id}:lower_safe_bound")
        elif abs(value - item.upper_bound) <= _BOUNDARY_TOLERANCE:
            limiting.append(f"{item.parameter_id}:upper_safe_bound")
    if success_fraction < request.minimum_success_fraction:
        limiting.append("minimum_success_fraction")
    if extrapolated:
        limiting.append("evidence_envelope")
    return tuple(limiting)


def _score(
    objective: CapabilityObjective,
    risk: _RiskMetrics,
    target_distance: float,
) -> float:
    if objective is CapabilityObjective.MAXIMIZE_CARRY:
        return -risk.mean_carry_m
    if objective is CapabilityObjective.MINIMIZE_EXPECTED_MISS:
        return risk.expected_miss_m
    if objective is CapabilityObjective.MAXIMIZE_TARGET_HOLD:
        return -risk.hold_probability + risk.expected_miss_m * 1e-6
    if objective is CapabilityObjective.MINIMIZE_VARIABILITY:
        return risk.dispersion_rms_m
    if objective is CapabilityObjective.MINIMIZE_DOWNSIDE:
        return risk.cvar_miss_m + risk.downside_carry_m
    return abs(risk.mean_carry_m - target_distance) + risk.dispersion_rms_m


def _summarize(
    club: ClubCapability,
    nominal: dict[str, float],
    landings: list[_Landing],
    counts: _Counts,
    profile: PlayerCapabilityProfile,
    request: OptimizationRequest,
) -> OptimizationAlternative | None:
    if not landings:
        return None
    target = _target(request)
    center_carry, center_offline = target.center
    carries = [item.carry_m for item in landings]
    offlines = [item.offline_m for item in landings]
    misses = [
        math.hypot(item.carry_m - center_carry, item.offline_m - center_offline)
        for item in landings
    ]
    mean_carry = sum(carries) / len(carries)
    mean_offline = sum(offlines) / len(offlines)
    expected_miss = sum(misses) / len(misses)
    dispersion = math.sqrt(
        sum(
            (carry - mean_carry) ** 2 + (offline - mean_offline) ** 2
            for carry, offline in zip(carries, offlines, strict=True)
        )
        / len(carries)
    )
    hold = sum(
        target.contains(item.carry_m, item.offline_m) for item in landings
    ) / len(landings)
    cvar = _tail_mean(misses, request.cvar_alpha, reverse=True)
    downside = max(
        0.0, center_carry - _tail_mean(carries, request.cvar_alpha, reverse=False)
    )
    risk = _RiskMetrics(mean_carry, expected_miss, hold, dispersion, cvar, downside)
    success_fraction = counts.completed / request.ensemble_size
    failure_fraction = 1.0 - success_fraction
    extrapolated = any(
        not item.evidence_lower_bound
        <= nominal[item.parameter_id]
        <= item.evidence_upper_bound
        for item in club.parameters
    )
    confidence = (
        profile.confidence
        * club.confidence
        * success_fraction
        * (0.5 if extrapolated else 1.0)
    )
    constraints = _limiting_constraints(
        club, nominal, success_fraction, extrapolated, request
    )
    score = _score(request.objective, risk, center_carry)
    if success_fraction < request.minimum_success_fraction:
        score += _FAILURE_PENALTY * (
            request.minimum_success_fraction - success_fraction
        )
    return OptimizationAlternative(
        1,
        club.club_id,
        tuple(nominal.items()),
        score,
        mean_carry,
        expected_miss,
        dispersion,
        hold,
        cvar,
        downside,
        request.ensemble_size,
        counts.completed,
        counts.no_impact,
        counts.failed,
        failure_fraction,
        confidence,
        constraints,
        extrapolated,
        False,
    )


def _evaluate_candidate(
    club: ClubCapability,
    nominal: dict[str, float],
    profile: PlayerCapabilityProfile,
    request: OptimizationRequest,
    evaluator: CapabilityEvaluator,
) -> tuple[OptimizationAlternative | None, _Counts]:
    landings: list[_Landing] = []
    counts = _Counts()
    for sample_index in range(request.ensemble_size):
        parameters = sample_perturbed_parameters(
            club, nominal, sample_index, request.seed
        )
        try:
            raw_evaluation: object = evaluator(club.club_id, parameters)
        except Exception:
            counts = counts.add(EvaluationStatus.FAILED)
            continue
        if not isinstance(raw_evaluation, SolverEvaluation):
            counts = counts.add(EvaluationStatus.FAILED)
            continue
        evaluation = raw_evaluation
        landing = _landing(evaluation)
        if landing is None:
            status = (
                evaluation.status
                if evaluation.status is EvaluationStatus.NO_IMPACT
                else EvaluationStatus.FAILED
            )
            counts = counts.add(status)
            continue
        counts = counts.add(EvaluationStatus.COMPLETE)
        landings.append(landing)
    return _summarize(club, nominal, landings, counts, profile, request), counts


def _pareto_mark(
    alternatives: list[OptimizationAlternative], request: OptimizationRequest
) -> list[OptimizationAlternative]:
    if request.objective is not CapabilityObjective.DISTANCE_CONTROL_PARETO:
        return alternatives
    target = request.target.distance_m
    marked: list[OptimizationAlternative] = []
    for candidate in alternatives:
        distance_error = abs(candidate.mean_carry_m - target)
        dominated = any(
            other is not candidate
            and abs(other.mean_carry_m - target) <= distance_error
            and other.dispersion_rms_m <= candidate.dispersion_rms_m
            and (
                abs(other.mean_carry_m - target) < distance_error
                or other.dispersion_rms_m < candidate.dispersion_rms_m
            )
            for other in alternatives
        )
        marked.append(replace(candidate, pareto_efficient=not dominated))
    return marked


def optimize_capability(
    profile: PlayerCapabilityProfile,
    request: OptimizationRequest,
    evaluator: CapabilityEvaluator,
) -> OptimizationResult:
    """Rank robust shot alternatives using injected canonical flight evaluations.

    Preconditions:
        Requested clubs exist in ``profile`` and evaluator outputs use canonical
        carry-distance and carry-offline metrics.
    Postconditions:
        Sampling, rankings, confidence, failure diagnostics, and Pareto flags are
        deterministic for a deterministic evaluator.
    """
    clubs = [profile.club(club_id) for club_id in request.club_ids]
    alternatives: list[OptimizationAlternative] = []
    aggregate = _Counts()
    per_club_indices = {club.club_id: 0 for club in clubs}
    for evaluation_index in range(request.candidate_budget):
        club = clubs[evaluation_index % len(clubs)]
        candidate_index = per_club_indices[club.club_id]
        per_club_indices[club.club_id] += 1
        nominal = sample_candidate_parameters(club, candidate_index, request.seed)
        alternative, counts = _evaluate_candidate(
            club, nominal, profile, request, evaluator
        )
        aggregate = _Counts(
            aggregate.completed + counts.completed,
            aggregate.no_impact + counts.no_impact,
            aggregate.failed + counts.failed,
        )
        if alternative is not None:
            alternatives.append(alternative)
    alternatives = _pareto_mark(alternatives, request)
    alternatives.sort(
        key=lambda item: (
            (
                not item.pareto_efficient
                if request.objective is CapabilityObjective.DISTANCE_CONTROL_PARETO
                else False
            ),
            item.score,
            item.club_id,
            item.parameters,
        )
    )
    ranked = tuple(
        replace(item, rank=index + 1)
        for index, item in enumerate(alternatives[: request.alternatives_count])
    )
    attempted = request.candidate_budget * request.ensemble_size
    status = "solved" if ranked else "nonconverged"
    return OptimizationResult(
        request.problem_id,
        status,
        ranked,
        attempted,
        aggregate.completed,
        aggregate.no_impact,
        aggregate.failed,
        _PROVENANCE,
    )


__all__ = ["CapabilityEvaluator", "optimize_capability"]
