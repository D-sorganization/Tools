"""Pure summary metrics for paired wind-strategy outcomes."""

from __future__ import annotations

import math

from .wind_strategy import (
    DirectionalRisk,
    StrategyAnalysisRequest,
    StrategyShotOutcome,
    StrategySummary,
    WindStrategy,
)

_BEST_COST_TOLERANCE = 1e-12


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _optional_mean(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return _mean(available) if available else None


def _directional_risk(excesses: list[float], trial_count: int) -> DirectionalRisk:
    positive = [value for value in excesses if value > 0.0]
    return DirectionalRisk(
        probability=len(positive) / trial_count,
        mean_excess_m=math.fsum(excesses) / trial_count,
        conditional_mean_excess_m=_mean(positive) if positive else 0.0,
    )


def _cvar(values: list[float], alpha: float) -> float:
    tail_count = max(1, math.ceil((1.0 - alpha) * len(values)))
    return _mean(sorted(values, reverse=True)[:tail_count])


def _effective_miss_distances(
    request: StrategyAnalysisRequest,
    cohort: list[StrategyShotOutcome],
) -> list[float]:
    failure_distance = request.analysis.miss_scale_m * math.sqrt(
        request.analysis.failure_cost
    )
    return [
        item.miss_distance_m if item.miss_distance_m is not None else failure_distance
        for item in cohort
    ]


def _directional_excesses(
    request: StrategyAnalysisRequest,
    cohort: list[StrategyShotOutcome],
) -> tuple[list[float], list[float], list[float], list[float]]:
    short: list[float] = []
    long: list[float] = []
    left: list[float] = []
    right: list[float] = []
    for item in cohort:
        if item.landing_forward_m is None or item.landing_right_m is None:
            short.append(0.0)
            long.append(0.0)
            left.append(0.0)
            right.append(0.0)
            continue
        forward_error = item.landing_forward_m - request.target.forward_m
        right_error = item.landing_right_m - request.target.right_m
        short.append(max(-forward_error, 0.0))
        long.append(max(forward_error, 0.0))
        left.append(max(-right_error, 0.0))
        right.append(max(right_error, 0.0))
    return short, long, left, right


def _best_costs(
    request: StrategyAnalysisRequest,
    outcomes: tuple[StrategyShotOutcome, ...],
) -> dict[int, float]:
    return {
        index: min(item.cost for item in outcomes if item.trial_index == index)
        for index in range(request.uncertainty.trials)
    }


def _best_credit(
    cohort: list[StrategyShotOutcome],
    outcomes: tuple[StrategyShotOutcome, ...],
    best_by_trial: dict[int, float],
) -> float:
    credit = 0.0
    for item in cohort:
        best = best_by_trial[item.trial_index]
        if abs(item.cost - best) > _BEST_COST_TOLERANCE:
            continue
        ties = sum(
            abs(peer.cost - best) <= _BEST_COST_TOLERANCE
            for peer in outcomes
            if peer.trial_index == item.trial_index
        )
        credit += 1.0 / ties
    return credit


def _summarize_strategy(
    request: StrategyAnalysisRequest,
    strategy: WindStrategy,
    outcomes: tuple[StrategyShotOutcome, ...],
    best_by_trial: dict[int, float],
) -> StrategySummary:
    cohort = [item for item in outcomes if item.strategy_id == strategy.strategy_id]
    completed = [item for item in cohort if item.status == "completed"]
    preset_regrets = [item.cost - best_by_trial[item.trial_index] for item in cohort]
    preset_probability = _best_credit(cohort, outcomes, best_by_trial) / len(cohort)
    effective_misses = _effective_miss_distances(request, cohort)
    short, long, left, right = _directional_excesses(request, cohort)
    target_holds = sum(
        item.miss_distance_m is not None
        and item.miss_distance_m <= request.analysis.target_radius_m
        for item in cohort
    )
    preset_regret = _mean(preset_regrets)
    return StrategySummary(
        strategy_id=strategy.strategy_id,
        label=strategy.label,
        completed_trials=len(completed),
        failed_trials=len(cohort) - len(completed),
        expected_cost=_mean([item.cost for item in cohort]),
        expected_perfect_information_cost=_mean(
            [item.perfect_information.cost for item in cohort]
        ),
        expected_information_cost_delta=_mean(
            [item.information_cost_delta for item in cohort]
        ),
        expected_preset_oracle_regret=preset_regret,
        preset_oracle_probability_best=preset_probability,
        expected_regret=preset_regret,
        probability_best=preset_probability,
        target_hold_probability=target_holds / len(cohort),
        miss_distance_cvar_m=_cvar(
            effective_misses, request.analysis.miss_distance_cvar_alpha
        ),
        miss_distance_cvar_alpha=request.analysis.miss_distance_cvar_alpha,
        short_risk=_directional_risk(short, len(cohort)),
        long_risk=_directional_risk(long, len(cohort)),
        left_risk=_directional_risk(left, len(cohort)),
        right_risk=_directional_risk(right, len(cohort)),
        mean_landing_forward_m=_optional_mean(
            [item.landing_forward_m for item in completed]
        ),
        mean_landing_right_m=_optional_mean(
            [item.landing_right_m for item in completed]
        ),
    )


def summarize_strategy_outcomes(
    request: StrategyAnalysisRequest,
    outcomes: tuple[StrategyShotOutcome, ...],
) -> tuple[StrategySummary, ...]:
    """Summarize paired outcomes without discarding failed trial cohorts."""
    best_by_trial = _best_costs(request, outcomes)
    return tuple(
        _summarize_strategy(request, strategy, outcomes, best_by_trial)
        for strategy in request.strategies
    )


__all__ = ["summarize_strategy_outcomes"]
