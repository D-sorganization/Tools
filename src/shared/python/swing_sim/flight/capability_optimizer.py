"""Deterministic robust optimization over clubs and delivery capabilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from ._capability_observation_runtime import (
    _CandidateIdentity,
    _evaluate_sample,
    _OptimizationContext,
    _SampleAttempt,
)
from ._capability_optimizer_metrics import (
    _Landing,
    _landing,
    _risk_metrics,
    _score,
)
from .capability_contract import (
    CapabilityObjective,
    ClubCapability,
    OptimizationAlternative,
    OptimizationRequest,
    OptimizationResult,
    PlayerCapabilityProfile,
)
from .capability_observation import (
    CapabilityOptimizationCancelled,
    CapabilityOptimizationHooks,
    CapabilitySampleStatus,
)
from .capability_sampling import (
    sample_candidate_parameters,
    sample_perturbed_parameters,
)
from .inverse_contract import EvaluationStatus, SolverEvaluation

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


@dataclass(frozen=True)
class _CandidateSamples:
    club: ClubCapability
    nominal: dict[str, float]
    landings: tuple[_Landing, ...]
    counts: _Counts


def _limiting_constraints(
    samples: _CandidateSamples,
    success_fraction: float,
    extrapolated: bool,
    request: OptimizationRequest,
) -> tuple[str, ...]:
    limiting: list[str] = []
    for item in samples.club.parameters:
        value = samples.nominal[item.parameter_id]
        if abs(value - item.lower_bound) <= _BOUNDARY_TOLERANCE:
            limiting.append(f"{item.parameter_id}:lower_safe_bound")
        elif abs(value - item.upper_bound) <= _BOUNDARY_TOLERANCE:
            limiting.append(f"{item.parameter_id}:upper_safe_bound")
    if success_fraction < request.minimum_success_fraction:
        limiting.append("minimum_success_fraction")
    if extrapolated:
        limiting.append("evidence_envelope")
    return tuple(limiting)


def _summarize(
    samples: _CandidateSamples, context: _OptimizationContext
) -> OptimizationAlternative | None:
    profile, request = context.profile, context.request
    if not samples.landings:
        return None
    risk = _risk_metrics(samples.landings, request)
    success_fraction = samples.counts.completed / request.ensemble_size
    failure_fraction = 1.0 - success_fraction
    extrapolated = any(
        not item.evidence_lower_bound
        <= samples.nominal[item.parameter_id]
        <= item.evidence_upper_bound
        for item in samples.club.parameters
    )
    confidence = (
        profile.confidence
        * samples.club.confidence
        * success_fraction
        * (0.5 if extrapolated else 1.0)
    )
    constraints = _limiting_constraints(
        samples, success_fraction, extrapolated, request
    )
    score = _score(request.objective, risk, request.target.distance_m)
    if success_fraction < request.minimum_success_fraction:
        score += _FAILURE_PENALTY * (
            request.minimum_success_fraction - success_fraction
        )
    return OptimizationAlternative(
        1,
        samples.club.club_id,
        tuple(samples.nominal.items()),
        score,
        risk.mean_carry_m,
        risk.expected_miss_m,
        risk.dispersion_rms_m,
        risk.hold_probability,
        risk.cvar_miss_m,
        risk.downside_carry_m,
        request.ensemble_size,
        samples.counts.completed,
        samples.counts.no_impact,
        samples.counts.failed,
        failure_fraction,
        confidence,
        constraints,
        extrapolated,
        False,
    )


def _evaluate_candidate(
    club: ClubCapability,
    nominal: dict[str, float],
    identity: _CandidateIdentity,
    context: _OptimizationContext,
) -> tuple[OptimizationAlternative | None, _Counts]:
    landings: list[_Landing] = []
    counts = _Counts()
    request = context.request
    for sample_index in range(request.ensemble_size):
        attempt_ordinal = (
            identity.candidate_ordinal * request.ensemble_size + sample_index
        )
        if context.hooks.should_cancel is not None and context.hooks.should_cancel():
            raise CapabilityOptimizationCancelled(attempt_ordinal, context.total_count)
        parameters = sample_perturbed_parameters(
            club, nominal, sample_index, request.seed
        )
        outcome = _evaluate_sample(
            _SampleAttempt(
                request.problem_id,
                attempt_ordinal,
                context.total_count,
                identity.candidate_ordinal,
                identity.club_candidate_ordinal,
                sample_index,
                club,
                nominal,
                parameters,
            ),
            context.evaluator,
            context.hooks,
        )
        evaluation = outcome.evaluation
        status = EvaluationStatus(outcome.effective_status.value)
        counts = counts.add(status)
        if outcome.effective_status is not CapabilitySampleStatus.COMPLETE:
            continue
        if evaluation is None:
            raise RuntimeError("normalized complete sample has no evaluation")
        landing = _landing(evaluation)
        if landing is None:
            raise RuntimeError("normalized complete sample has no finite landing")
        landings.append(landing)
    samples = _CandidateSamples(club, nominal, tuple(landings), counts)
    return _summarize(samples, context), counts


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


def _collect_alternatives(
    clubs: list[ClubCapability], context: _OptimizationContext
) -> tuple[list[OptimizationAlternative], _Counts]:
    alternatives: list[OptimizationAlternative] = []
    aggregate = _Counts()
    per_club_indices = {club.club_id: 0 for club in clubs}
    for candidate_ordinal in range(context.request.candidate_budget):
        club = clubs[candidate_ordinal % len(clubs)]
        club_ordinal = per_club_indices[club.club_id]
        per_club_indices[club.club_id] += 1
        nominal = sample_candidate_parameters(club, club_ordinal, context.request.seed)
        alternative, counts = _evaluate_candidate(
            club,
            nominal,
            _CandidateIdentity(candidate_ordinal, club_ordinal),
            context,
        )
        aggregate = _Counts(
            aggregate.completed + counts.completed,
            aggregate.no_impact + counts.no_impact,
            aggregate.failed + counts.failed,
        )
        if alternative is not None:
            alternatives.append(alternative)
    return alternatives, aggregate


def optimize_capability(
    profile: PlayerCapabilityProfile,
    request: OptimizationRequest,
    evaluator: CapabilityEvaluator,
    *,
    hooks: CapabilityOptimizationHooks | None = None,
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
    total_count = request.candidate_budget * request.ensemble_size
    active_hooks = hooks or CapabilityOptimizationHooks()
    context = _OptimizationContext(profile, request, evaluator, active_hooks)
    alternatives, aggregate = _collect_alternatives(clubs, context)
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
    status = "solved" if ranked else "nonconverged"
    return OptimizationResult(
        request.problem_id,
        status,
        ranked,
        total_count,
        aggregate.completed,
        aggregate.no_impact,
        aggregate.failed,
        _PROVENANCE,
    )


__all__ = ["CapabilityEvaluator", "optimize_capability"]
