"""Paired wind-estimate strategy evaluation and regret summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from .registry import FlightModelRegistry, FlightModelType
from .types import LaunchConditions
from .wind import WindScenario
from .wind_uncertainty import WindTrial, WindUncertaintySpec, sample_wind_trials

OutcomeStatus = Literal["completed", "nonconverged", "invalid"]
_GROUND_TOLERANCE_M = 1e-5
_BEST_COST_TOLERANCE = 1e-12


def _positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class TargetPoint:
    """Landing target in app-plan coordinates: +forward and +right [m]."""

    forward_m: float
    right_m: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.forward_m) or not math.isfinite(self.right_m):
            raise ValueError("target coordinates must be finite")


@dataclass(frozen=True)
class WindStrategy:
    """A club launch plus linear estimated-crosswind aim compensation."""

    strategy_id: str
    label: str
    launch: LaunchConditions
    crosswind_aim_gain_rad_per_mps: float = 0.0

    def __post_init__(self) -> None:
        if not self.strategy_id.strip() or not self.label.strip():
            raise ValueError("strategy_id and label must be nonempty")
        if not math.isfinite(self.crosswind_aim_gain_rad_per_mps):
            raise ValueError("crosswind aim gain must be finite")
        if self.launch.wind_scenario is not None or self.launch.wind_speed != 0.0:
            raise ValueError("strategy launch must not contain wind")


@dataclass(frozen=True)
class StrategyAnalysisConfig:
    """Numerical model and dimensionless miss-cost settings."""

    model_name: str = "waterloo_penner"
    max_time_s: float = 10.0
    time_step_s: float = 0.01
    miss_scale_m: float = 20.0
    failure_cost: float = 100.0

    def __post_init__(self) -> None:
        try:
            FlightModelType(self.model_name)
        except ValueError as exc:
            raise ValueError(f"unknown flight model: {self.model_name}") from exc
        for name in ("max_time_s", "time_step_s", "miss_scale_m"):
            _positive_finite(getattr(self, name), name)
        if not math.isfinite(self.failure_cost) or self.failure_cost < 0.0:
            raise ValueError("failure_cost must be finite and nonnegative")


@dataclass(frozen=True)
class StrategyAnalysisRequest:
    """Complete immutable request for a paired strategy ensemble."""

    uncertainty: WindUncertaintySpec
    strategies: tuple[WindStrategy, ...]
    target: TargetPoint
    analysis: StrategyAnalysisConfig = StrategyAnalysisConfig()

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategies", tuple(self.strategies))
        if not self.strategies:
            raise ValueError("at least one strategy is required")
        identifiers = [strategy.strategy_id for strategy in self.strategies]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("strategy_id values must be unique")


@dataclass(frozen=True)
class StrategyShotOutcome:
    """One trial/strategy landing point, or an explicit failure cohort."""

    trial_index: int
    strategy_id: str
    status: OutcomeStatus
    true_wind: WindScenario
    estimated_wind: WindScenario
    landing_forward_m: float | None
    landing_right_m: float | None
    cost: float
    failure_reason: str | None = None


@dataclass(frozen=True)
class StrategySummary:
    """Expected cost and common-random-number regret for one strategy."""

    strategy_id: str
    label: str
    completed_trials: int
    failed_trials: int
    expected_cost: float
    expected_regret: float
    probability_best: float
    mean_landing_forward_m: float | None
    mean_landing_right_m: float | None


@dataclass(frozen=True)
class WindStrategyAnalysis:
    """Auditable wind draws, landing scatter, and strategy summaries."""

    schema_version: str
    provenance: str
    target: TargetPoint
    wind_trials: tuple[WindTrial, ...]
    outcomes: tuple[StrategyShotOutcome, ...]
    summaries: tuple[StrategySummary, ...]


@dataclass(frozen=True)
class _TrialContext:
    request: StrategyAnalysisRequest
    trial: WindTrial
    true_wind: WindScenario
    estimated_wind: WindScenario


def _aimed_launch(strategy: WindStrategy, estimate: WindScenario) -> LaunchConditions:
    estimated_crosswind_left_mps = estimate.base_velocity_mps[1]
    correction = strategy.crosswind_aim_gain_rad_per_mps * estimated_crosswind_left_mps
    return replace(
        strategy.launch,
        azimuth_angle=strategy.launch.azimuth_angle - correction,
    )


def _completed_outcome(
    context: _TrialContext,
    strategy: WindStrategy,
) -> StrategyShotOutcome:
    request = context.request
    model = FlightModelRegistry.get_model(FlightModelType(request.analysis.model_name))
    launch = replace(
        _aimed_launch(strategy, context.estimated_wind),
        wind_scenario=context.true_wind,
    )
    result = model.simulate(
        launch,
        max_time=request.analysis.max_time_s,
        dt=request.analysis.time_step_s,
    )
    if not result.trajectory or result.trajectory[-1].position[2] > _GROUND_TOLERANCE_M:
        return _failure_outcome(
            context,
            strategy,
            "nonconverged",
            "ground not reached",
        )
    landing = result.trajectory[-1].position
    forward_m = float(landing[0])
    right_m = -float(landing[1])
    miss_squared = (forward_m - request.target.forward_m) ** 2 + (
        right_m - request.target.right_m
    ) ** 2
    cost = miss_squared / request.analysis.miss_scale_m**2
    if not all(math.isfinite(value) for value in (forward_m, right_m, cost)):
        return _failure_outcome(
            context,
            strategy,
            "invalid",
            "simulation produced nonfinite landing data",
        )
    return StrategyShotOutcome(
        context.trial.trial_index,
        strategy.strategy_id,
        "completed",
        context.true_wind,
        context.estimated_wind,
        forward_m,
        right_m,
        cost,
    )


def _failure_outcome(
    context: _TrialContext,
    strategy: WindStrategy,
    status: Literal["nonconverged", "invalid"],
    reason: str,
) -> StrategyShotOutcome:
    return StrategyShotOutcome(
        context.trial.trial_index,
        strategy.strategy_id,
        status,
        context.true_wind,
        context.estimated_wind,
        None,
        None,
        context.request.analysis.failure_cost,
        reason,
    )


def _evaluate(
    request: StrategyAnalysisRequest,
    trials: tuple[WindTrial, ...],
) -> tuple[StrategyShotOutcome, ...]:
    outcomes: list[StrategyShotOutcome] = []
    for trial in trials:
        true_wind = trial.true_scenario(request.uncertainty.provenance)
        estimated_wind = trial.estimated_scenario(request.uncertainty.provenance)
        context = _TrialContext(request, trial, true_wind, estimated_wind)
        for strategy in request.strategies:
            try:
                outcome = _completed_outcome(context, strategy)
            except (ArithmeticError, RuntimeError, ValueError) as exc:
                outcome = _failure_outcome(
                    context,
                    strategy,
                    "invalid",
                    str(exc),
                )
            outcomes.append(outcome)
    return tuple(outcomes)


def _summaries(
    request: StrategyAnalysisRequest, outcomes: tuple[StrategyShotOutcome, ...]
) -> tuple[StrategySummary, ...]:
    best_by_trial = {
        index: min(item.cost for item in outcomes if item.trial_index == index)
        for index in range(request.uncertainty.trials)
    }
    summaries: list[StrategySummary] = []
    for strategy in request.strategies:
        cohort = [item for item in outcomes if item.strategy_id == strategy.strategy_id]
        completed = [item for item in cohort if item.status == "completed"]
        costs = [item.cost for item in cohort]
        regrets = [item.cost - best_by_trial[item.trial_index] for item in cohort]
        best_credit = sum(
            1.0
            / sum(
                abs(peer.cost - best_by_trial[item.trial_index]) <= _BEST_COST_TOLERANCE
                for peer in outcomes
                if peer.trial_index == item.trial_index
            )
            for item in cohort
            if abs(item.cost - best_by_trial[item.trial_index]) <= _BEST_COST_TOLERANCE
        )
        summaries.append(
            StrategySummary(
                strategy.strategy_id,
                strategy.label,
                len(completed),
                len(cohort) - len(completed),
                math.fsum(costs) / len(costs),
                math.fsum(regrets) / len(regrets),
                best_credit / len(cohort),
                _optional_mean(completed, "landing_forward_m"),
                _optional_mean(completed, "landing_right_m"),
            )
        )
    return tuple(summaries)


def _optional_mean(outcomes: list[StrategyShotOutcome], field: str) -> float | None:
    values = [getattr(outcome, field) for outcome in outcomes]
    finite_values = [value for value in values if value is not None]
    return math.fsum(finite_values) / len(finite_values) if finite_values else None


def analyze_wind_strategies(request: StrategyAnalysisRequest) -> WindStrategyAnalysis:
    """Run paired common-random-number strategy trials and summarize regret."""
    trials = sample_wind_trials(request.uncertainty)
    outcomes = _evaluate(request, trials)
    return WindStrategyAnalysis(
        schema_version="wind-strategy-analysis/v1",
        provenance=request.uncertainty.provenance,
        target=request.target,
        wind_trials=trials,
        outcomes=outcomes,
        summaries=_summaries(request, outcomes),
    )


__all__ = [
    "StrategyAnalysisConfig",
    "StrategyAnalysisRequest",
    "StrategyShotOutcome",
    "StrategySummary",
    "TargetPoint",
    "WindStrategy",
    "WindStrategyAnalysis",
    "analyze_wind_strategies",
]
