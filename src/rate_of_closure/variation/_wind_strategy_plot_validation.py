"""Validation helpers for the wind-strategy scalar plot adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from shared.python.contracts import require
from shared.python.swing_sim.flight import (
    StrategyAnalysisRequest,
    StrategyShotOutcome,
    WindScenario,
    WindStrategyAnalysis,
    WindTrial,
    sample_wind_trials,
)
from shared.python.swing_sim.flight.wind_strategy import (
    WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION,
)

_AGREEMENT_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class _ScenarioExpectation:
    """Immutable expected values and diagnostic label for one wind scenario."""

    speed_mps: float
    bearing_deg: float
    provenance: str
    name: str


def _agrees(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=_AGREEMENT_TOLERANCE,
        abs_tol=_AGREEMENT_TOLERANCE,
    )


def validate_header(
    request: StrategyAnalysisRequest,
    analysis: WindStrategyAnalysis,
) -> None:
    """Validate analysis-level identity against its immutable request."""
    require(
        analysis.schema_version == WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION,
        "unsupported wind strategy analysis schema",
    )
    require(
        analysis.provenance == request.uncertainty.provenance,
        "request and analysis provenance must agree",
    )
    require(
        _agrees(analysis.target.forward_m, request.target.forward_m)
        and _agrees(analysis.target.right_m, request.target.right_m),
        "request and analysis target must agree",
    )


def validate_trials(
    request: StrategyAnalysisRequest,
    analysis: WindStrategyAnalysis,
) -> Mapping[int, WindTrial]:
    """Validate deterministic sampling and index the declared wind trials."""
    expected = sample_wind_trials(request.uncertainty)
    require(
        len(analysis.wind_trials) == len(expected),
        "analysis wind trial count must agree with request",
    )
    for actual, declared in zip(analysis.wind_trials, expected, strict=True):
        require(
            actual.trial_index == declared.trial_index,
            "wind trial indices must be contiguous",
        )
        pairs = zip(
            actual.to_schema_dict().values(),
            declared.to_schema_dict().values(),
            strict=True,
        )
        require(
            all(_agrees(float(left), float(right)) for left, right in pairs),
            "analysis wind trials must agree with request sampling contract",
        )
    return {trial.trial_index: trial for trial in analysis.wind_trials}


def _validate_simulation_result(
    status: str,
    values: tuple[float | None, float | None, float | None],
    failure_reason: str | None,
    name: str,
) -> None:
    complete = status == "completed"
    finite = all(value is not None and math.isfinite(value) for value in values)
    unavailable = all(value is None for value in values)
    require(
        (
            finite and failure_reason is None
            if complete
            else unavailable and bool(failure_reason and failure_reason.strip())
        ),
        f"{name} availability does not agree with status",
    )


def _validate_scenario(
    scenario: WindScenario,
    expectation: _ScenarioExpectation,
) -> None:
    expected = WindScenario.from_meteorological(
        expectation.speed_mps,
        expectation.bearing_deg,
    )
    require(
        all(
            _agrees(left, right)
            for left, right in zip(
                scenario.base_velocity_mps, expected.base_velocity_mps, strict=True
            )
        ),
        f"{expectation.name} does not agree with its wind trial",
    )
    require(
        scenario.provenance == expectation.provenance
        and scenario.shear_fraction_per_10m == 0.0
        and scenario.turbulence_intensity_mps == 0.0
        and scenario.seed == 0
        and not scenario.gusts,
        f"{expectation.name} does not agree with its deterministic scenario contract",
    )


# fmt: off
def _validate_outcome(
    outcome: StrategyShotOutcome,
    trial: WindTrial,
    provenance: str,
    statuses: frozenset[str],
) -> None:
    require(outcome.status in statuses, "unknown actual status")
    perfect = outcome.perfect_information
    require(perfect.status in statuses, "unknown perfect-information status")
    _validate_scenario(
        outcome.true_wind,
        _ScenarioExpectation(
            trial.true_speed_mps,
            trial.true_from_bearing_deg,
            f"{provenance}/true/trial-{trial.trial_index}",
            "true wind",
        ),
    )
    _validate_scenario(
        outcome.estimated_wind,
        _ScenarioExpectation(
            trial.estimated_speed_mps,
            trial.estimated_from_bearing_deg,
            f"{provenance}/estimated/trial-{trial.trial_index}",
            "estimated wind",
        ),
    )
    actual_values = (outcome.landing_forward_m, outcome.landing_right_m, outcome.miss_distance_m)  # noqa: E501
    perfect_values = (perfect.landing_forward_m, perfect.landing_right_m, perfect.miss_distance_m)  # noqa: E501
    _validate_simulation_result(outcome.status, actual_values, outcome.failure_reason, "actual outcome")  # noqa: E501
    _validate_simulation_result(perfect.status, perfect_values, perfect.failure_reason, "perfect-information outcome")  # noqa: E501
    require(math.isfinite(outcome.cost) and outcome.cost >= 0.0, "actual cost must be finite and nonnegative")  # noqa: E501
    require(math.isfinite(perfect.cost) and perfect.cost >= 0.0, "perfect-information cost must be finite and nonnegative")  # noqa: E501
    require(
        math.isfinite(outcome.information_cost_delta)
        and _agrees(outcome.information_cost_delta, outcome.cost - perfect.cost),
        "information cost delta must agree with outcome costs",
    )
# fmt: on


def validate_outcomes(
    request: StrategyAnalysisRequest,
    analysis: WindStrategyAnalysis,
    trials: Mapping[int, WindTrial],
    statuses: frozenset[str],
) -> Mapping[tuple[int, str], StrategyShotOutcome]:
    """Validate complete trial/strategy coverage and index every outcome."""
    strategy_ids = {strategy.strategy_id for strategy in request.strategies}
    expected_keys = {
        (index, strategy_id) for index in trials for strategy_id in strategy_ids
    }
    keys = tuple(
        (outcome.trial_index, outcome.strategy_id) for outcome in analysis.outcomes
    )
    require(
        len(analysis.outcomes) == len(expected_keys),
        "analysis outcome count must agree with request",
    )
    require(len(set(keys)) == len(keys), "outcome trial/strategy keys must be unique")
    require(
        all(key[1] in strategy_ids for key in keys),
        "analysis outcome references an unknown strategy",
    )
    require(
        set(keys) == expected_keys,
        "analysis outcomes must cover every strategy and trial",
    )
    for outcome in analysis.outcomes:
        _validate_outcome(
            outcome,
            trials[outcome.trial_index],
            analysis.provenance,
            statuses,
        )
    return dict(zip(keys, analysis.outcomes, strict=True))
