"""Paired wind-estimate strategy evaluation and risk contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from .registry import FlightModelRegistry, FlightModelType
from .types import LaunchConditions
from .wind import WindScenario
from .wind_uncertainty import WindTrial, WindUncertaintySpec, sample_wind_trials

WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION = "wind-strategy-analysis/v2"
OutcomeStatus = Literal["completed", "nonconverged", "invalid"]
_GROUND_TOLERANCE_M = 1e-5


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
    """Numerical settings, miss cost, target hold, and empirical CVaR level."""

    model_name: str = "waterloo_penner"
    max_time_s: float = 10.0
    time_step_s: float = 0.01
    miss_scale_m: float = 20.0
    failure_cost: float = 100.0
    target_radius_m: float = 10.0
    miss_distance_cvar_alpha: float = 0.9

    def __post_init__(self) -> None:
        try:
            FlightModelType(self.model_name)
        except ValueError as exc:
            raise ValueError(f"unknown flight model: {self.model_name}") from exc
        for name in ("max_time_s", "time_step_s", "miss_scale_m"):
            _positive_finite(getattr(self, name), name)
        if not math.isfinite(self.failure_cost) or self.failure_cost < 0.0:
            raise ValueError("failure_cost must be finite and nonnegative")
        if not math.isfinite(self.target_radius_m) or self.target_radius_m < 0.0:
            raise ValueError("target_radius_m must be finite and nonnegative")
        alpha = self.miss_distance_cvar_alpha
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("miss_distance_cvar_alpha must be in (0, 1)")


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
class PerfectInformationCounterfactual:
    """Same declared strategy policy with the true wind used for its decision."""

    status: OutcomeStatus
    landing_forward_m: float | None
    landing_right_m: float | None
    miss_distance_m: float | None
    cost: float
    failure_reason: str | None = None


@dataclass(frozen=True)
class StrategyShotOutcome:
    """Actual and policy-fixed perfect-information results for one trial."""

    trial_index: int
    strategy_id: str
    status: OutcomeStatus
    true_wind: WindScenario
    estimated_wind: WindScenario
    landing_forward_m: float | None
    landing_right_m: float | None
    miss_distance_m: float | None
    cost: float
    failure_reason: str | None
    perfect_information: PerfectInformationCounterfactual
    information_cost_delta: float


@dataclass(frozen=True)
class DirectionalRisk:
    """Frequency and severity of one signed landing-error direction."""

    probability: float
    mean_excess_m: float
    conditional_mean_excess_m: float


@dataclass(frozen=True)
class StrategySummary:
    """Expected performance, counterfactual, and landing-risk metrics."""

    strategy_id: str
    label: str
    completed_trials: int
    failed_trials: int
    expected_cost: float
    expected_perfect_information_cost: float
    expected_information_cost_delta: float
    expected_preset_oracle_regret: float
    preset_oracle_probability_best: float
    expected_regret: float
    probability_best: float
    target_hold_probability: float
    miss_distance_cvar_m: float
    miss_distance_cvar_alpha: float
    short_risk: DirectionalRisk
    long_risk: DirectionalRisk
    left_risk: DirectionalRisk
    right_risk: DirectionalRisk
    mean_landing_forward_m: float | None
    mean_landing_right_m: float | None


@dataclass(frozen=True)
class WindStrategyAnalysis:
    """Auditable wind draws, paired counterfactuals, and strategy summaries."""

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


@dataclass(frozen=True)
class _SimulationResult:
    status: OutcomeStatus
    landing_forward_m: float | None
    landing_right_m: float | None
    miss_distance_m: float | None
    cost: float
    failure_reason: str | None = None


def _aimed_launch(
    strategy: WindStrategy, decision_wind: WindScenario
) -> LaunchConditions:
    crosswind_left_mps = decision_wind.base_velocity_mps[1]
    correction = strategy.crosswind_aim_gain_rad_per_mps * crosswind_left_mps
    return replace(
        strategy.launch, azimuth_angle=strategy.launch.azimuth_angle - correction
    )


def _failure_result(
    request: StrategyAnalysisRequest,
    status: Literal["nonconverged", "invalid"],
    reason: str,
) -> _SimulationResult:
    return _SimulationResult(
        status, None, None, None, request.analysis.failure_cost, reason
    )


def _simulate_policy(
    context: _TrialContext,
    strategy: WindStrategy,
    decision_wind: WindScenario,
) -> _SimulationResult:
    request = context.request
    model = FlightModelRegistry.get_model(FlightModelType(request.analysis.model_name))
    launch = replace(
        _aimed_launch(strategy, decision_wind),
        wind_scenario=context.true_wind,
    )
    result = model.simulate(
        launch,
        max_time=request.analysis.max_time_s,
        dt=request.analysis.time_step_s,
    )
    if not result.trajectory or result.trajectory[-1].position[2] > _GROUND_TOLERANCE_M:
        return _failure_result(request, "nonconverged", "ground not reached")
    landing = result.trajectory[-1].position
    forward_m = float(landing[0])
    right_m = -float(landing[1])
    forward_error = forward_m - request.target.forward_m
    right_error = right_m - request.target.right_m
    miss_distance_m = math.hypot(forward_error, right_error)
    cost = (miss_distance_m / request.analysis.miss_scale_m) ** 2
    if not all(math.isfinite(value) for value in (forward_m, right_m, cost)):
        return _failure_result(
            request,
            "invalid",
            "simulation produced nonfinite landing data",
        )
    return _SimulationResult("completed", forward_m, right_m, miss_distance_m, cost)


def _safe_simulate_policy(
    context: _TrialContext,
    strategy: WindStrategy,
    decision_wind: WindScenario,
) -> _SimulationResult:
    try:
        return _simulate_policy(context, strategy, decision_wind)
    except (ArithmeticError, RuntimeError, ValueError) as exc:
        return _failure_result(context.request, "invalid", str(exc))


def _counterfactual(result: _SimulationResult) -> PerfectInformationCounterfactual:
    return PerfectInformationCounterfactual(
        result.status,
        result.landing_forward_m,
        result.landing_right_m,
        result.miss_distance_m,
        result.cost,
        result.failure_reason,
    )


def _evaluate_strategy(
    context: _TrialContext, strategy: WindStrategy
) -> StrategyShotOutcome:
    actual = _safe_simulate_policy(context, strategy, context.estimated_wind)
    perfect = _safe_simulate_policy(context, strategy, context.true_wind)
    return StrategyShotOutcome(
        context.trial.trial_index,
        strategy.strategy_id,
        actual.status,
        context.true_wind,
        context.estimated_wind,
        actual.landing_forward_m,
        actual.landing_right_m,
        actual.miss_distance_m,
        actual.cost,
        actual.failure_reason,
        _counterfactual(perfect),
        actual.cost - perfect.cost,
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
        outcomes.extend(
            _evaluate_strategy(context, item) for item in request.strategies
        )
    return tuple(outcomes)


def analyze_wind_strategies(request: StrategyAnalysisRequest) -> WindStrategyAnalysis:
    """Run paired strategy trials and summarize actual and counterfactual risk."""
    from .wind_strategy_metrics import summarize_strategy_outcomes

    trials = sample_wind_trials(request.uncertainty)
    outcomes = _evaluate(request, trials)
    return WindStrategyAnalysis(
        schema_version=WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION,
        provenance=request.uncertainty.provenance,
        target=request.target,
        wind_trials=trials,
        outcomes=outcomes,
        summaries=summarize_strategy_outcomes(request, outcomes),
    )


__all__ = [
    "WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION",
    "DirectionalRisk",
    "PerfectInformationCounterfactual",
    "StrategyAnalysisConfig",
    "StrategyAnalysisRequest",
    "StrategyShotOutcome",
    "StrategySummary",
    "TargetPoint",
    "WindStrategy",
    "WindStrategyAnalysis",
    "analyze_wind_strategies",
]
