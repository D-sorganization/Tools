"""Pure wind-strategy adapter for the shared scalar-ensemble/v1 contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

from rate_of_closure.variation.scalar_ensemble_contract import (
    SCALAR_ENSEMBLE_SCHEMA_VERSION,
    ScalarCohortDefinition,
    ScalarEnsembleDataset,
    ScalarEnsembleProvenance,
    ScalarEnsembleRow,
    ScalarEnsembleStage,
    ScalarVariableCategory,
    ScalarVariableDefinition,
    scalar_ensemble_row_id,
)
from shared.python.contracts import require
from shared.python.swing_sim.flight import (
    LaunchConditions,
    StrategyAnalysisRequest,
    StrategyShotOutcome,
    WindScenario,
    WindStrategy,
    WindStrategyAnalysis,
    WindTrial,
    sample_wind_trials,
)
from shared.python.swing_sim.flight.wind_strategy import (
    WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION,
)

WIND_STRATEGY_PLOT_ADAPTER_ID = "wind-strategy-plot-adapter/v1"
# Declarative metadata stays tabular so Python and TypeScript are easy to diff.
# fmt: off
WIND_STRATEGY_STAGES = (
    ScalarEnsembleStage("input", "Strategy Inputs"),
    ScalarEnsembleStage("environment", "Wind and Target"),
    ScalarEnsembleStage("actual", "Estimated-Wind Decision"),
    ScalarEnsembleStage("perfect_information", "Perfect-Information Counterfactual"),  # noqa: E501
    ScalarEnsembleStage("comparison", "Information Comparison"),
)
WIND_STRATEGY_CATEGORIES = (
    ScalarVariableCategory("wind", "Wind"), ScalarVariableCategory("launch", "Launch and Aim"),  # noqa: E501
    ScalarVariableCategory("target", "Target"), ScalarVariableCategory("actual", "Actual Outcome"),  # noqa: E501
    ScalarVariableCategory("perfect_information", "Perfect-Information Outcome"),  # noqa: E501
    ScalarVariableCategory("information", "Information Value"),
)
WIND_STRATEGY_COHORTS = (
    ScalarCohortDefinition("completed", "Completed"), ScalarCohortDefinition("nonconverged", "Nonconverged"),  # noqa: E501
    ScalarCohortDefinition("invalid", "Invalid"),
)
# fmt: on
_AGREEMENT_TOLERANCE = 1e-9

# fmt: off
_STAGE_KEYS = {"I": "input", "E": "environment", "A": "actual", "P": "perfect_information", "C": "comparison"}  # noqa: E501
_CATEGORY_KEYS = {"W": "wind", "L": "launch", "T": "target", "A": "actual", "P": "perfect_information", "I": "information"}  # noqa: E501
# fmt: on
_VARIABLE_ROWS = (
    "true_wind_speed_mps|True Wind Speed|m/s|E|W",
    "true_wind_from_bearing_deg|True Wind From Bearing|deg|E|W",
    "true_wind_forward_mps|True Wind Forward Component|m/s|E|W",
    "true_wind_left_mps|True Wind Left Component|m/s|E|W",
    "true_wind_up_mps|True Wind Up Component|m/s|E|W",
    "estimated_wind_speed_mps|Estimated Wind Speed|m/s|E|W",
    "estimated_wind_from_bearing_deg|Estimated Wind From Bearing|deg|E|W",
    "estimated_wind_forward_mps|Estimated Wind Forward Component|m/s|E|W",
    "estimated_wind_left_mps|Estimated Wind Left Component|m/s|E|W",
    "estimated_wind_up_mps|Estimated Wind Up Component|m/s|E|W",
    "wind_speed_error_mps|Wind Speed Estimate Error|m/s|E|W",
    "wind_bearing_error_deg|Wind Bearing Estimate Error|deg|E|W",
    "launch_ball_speed_mps|Launch Ball Speed|m/s|I|L",
    "launch_angle_rad|Launch Angle|rad|I|L",
    "launch_azimuth_rad|Base Launch Direction|rad|I|L",
    "launch_spin_rpm|Launch Spin Rate|rpm|I|L",
    "launch_spin_axis_forward|Spin Axis Forward|1|I|L",
    "launch_spin_axis_left|Spin Axis Left|1|I|L",
    "launch_spin_axis_up|Spin Axis Up|1|I|L",
    "crosswind_aim_gain_rad_per_mps|Crosswind Aim Gain|rad/(m/s)|I|L",
    "actual_aim_azimuth_rad|Estimated-Wind Aim Direction|rad|A|L",
    "perfect_information_aim_azimuth_rad|Perfect-Information Aim Direction|rad|P|L",
    "target_forward_m|Target Forward Coordinate|m|E|T",
    "target_right_m|Target Right Coordinate|m|E|T",
    "actual_landing_forward_m|Actual Landing Forward|m|A|A",
    "actual_landing_right_m|Actual Landing Right|m|A|A",
    "actual_miss_distance_m|Actual Miss Distance|m|A|A",
    "actual_cost|Actual Strategy Cost|1|A|A",
    "perfect_information_landing_forward_m|Perfect-Information Landing Forward|m|P|P",
    "perfect_information_landing_right_m|Perfect-Information Landing Right|m|P|P",
    "perfect_information_miss_distance_m|Perfect-Information Miss Distance|m|P|P",
    "perfect_information_cost|Perfect-Information Cost|1|P|P",
    "information_cost_delta|Information Cost Delta|1|C|I",
)


def _variables() -> tuple[ScalarVariableDefinition, ...]:
    definitions = []
    for row in _VARIABLE_ROWS:
        key, label, unit, stage, category = row.split("|")
        definitions.append(
            ScalarVariableDefinition(
                key,
                label,
                unit,
                _STAGE_KEYS[stage],
                _CATEGORY_KEYS[category],
            )
        )
    return tuple(definitions)


WIND_STRATEGY_VARIABLES = _variables()


def _agrees(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=_AGREEMENT_TOLERANCE,
        abs_tol=_AGREEMENT_TOLERANCE,
    )


def _scenario_values(prefix: str, scenario: WindScenario) -> dict[str, float]:
    forward, left, up = scenario.base_velocity_mps
    return {
        f"{prefix}_wind_forward_mps": forward,
        f"{prefix}_wind_left_mps": left,
        f"{prefix}_wind_up_mps": up,
    }


def _spin_axis(launch: LaunchConditions) -> tuple[float, float, float]:
    if launch.spin_axis is not None:
        return cast(tuple[float, float, float], launch.spin_axis)
    tilt = launch.spin_axis_angle
    azimuth = launch.azimuth_angle
    return (
        math.sin(tilt) * math.sin(azimuth),
        -math.cos(tilt),
        math.sin(tilt) * math.cos(azimuth),
    )


def _aim_azimuth(strategy: WindStrategy, wind: WindScenario) -> float:
    aim = strategy.launch.azimuth_angle - (
        strategy.crosswind_aim_gain_rad_per_mps * wind.base_velocity_mps[1]
    )
    return float(aim)


def _row_values(
    request: StrategyAnalysisRequest,
    strategy: WindStrategy,
    trial: WindTrial,
    outcome: StrategyShotOutcome,
) -> dict[str, float | None]:
    launch = strategy.launch
    spin_forward, spin_left, spin_up = _spin_axis(launch)
    perfect = outcome.perfect_information
    values: dict[str, float | None] = {
        "true_wind_speed_mps": trial.true_speed_mps,
        "true_wind_from_bearing_deg": trial.true_from_bearing_deg,
        "estimated_wind_speed_mps": trial.estimated_speed_mps,
        "estimated_wind_from_bearing_deg": trial.estimated_from_bearing_deg,
        "wind_speed_error_mps": trial.speed_error_mps,
        "wind_bearing_error_deg": trial.bearing_error_deg,
        "launch_ball_speed_mps": launch.ball_speed,
        "launch_angle_rad": launch.launch_angle,
        "launch_azimuth_rad": launch.azimuth_angle,
        "launch_spin_rpm": launch.spin_rate,
        "launch_spin_axis_forward": spin_forward,
        "launch_spin_axis_left": spin_left,
        "launch_spin_axis_up": spin_up,
        "crosswind_aim_gain_rad_per_mps": strategy.crosswind_aim_gain_rad_per_mps,
        "actual_aim_azimuth_rad": _aim_azimuth(strategy, outcome.estimated_wind),
        "perfect_information_aim_azimuth_rad": _aim_azimuth(
            strategy, outcome.true_wind
        ),
        "target_forward_m": request.target.forward_m,
        "target_right_m": request.target.right_m,
        "actual_landing_forward_m": outcome.landing_forward_m,
        "actual_landing_right_m": outcome.landing_right_m,
        "actual_miss_distance_m": outcome.miss_distance_m,
        "actual_cost": outcome.cost,
        "perfect_information_landing_forward_m": perfect.landing_forward_m,
        "perfect_information_landing_right_m": perfect.landing_right_m,
        "perfect_information_miss_distance_m": perfect.miss_distance_m,
        "perfect_information_cost": perfect.cost,
        "information_cost_delta": outcome.information_cost_delta,
    }
    values.update(_scenario_values("true", outcome.true_wind))
    values.update(_scenario_values("estimated", outcome.estimated_wind))
    return values


def _row(
    request: StrategyAnalysisRequest,
    strategy: WindStrategy,
    trial: WindTrial,
    outcome: StrategyShotOutcome,
) -> ScalarEnsembleRow:
    perfect = outcome.perfect_information
    return ScalarEnsembleRow(
        row_id=scalar_ensemble_row_id(trial.trial_index, strategy.strategy_id),
        trial_index=trial.trial_index,
        series_id=strategy.strategy_id,
        cohort=outcome.status,
        values=_row_values(request, strategy, trial, outcome),
        attributes={
            "actual_status": outcome.status,
            "perfect_information_status": perfect.status,
            "actual_failure_reason": outcome.failure_reason,
            "perfect_information_failure_reason": perfect.failure_reason,
            "strategy_label": strategy.label,
        },
    )


def _validate_header(
    request: StrategyAnalysisRequest,
    analysis: WindStrategyAnalysis,
) -> None:
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


def _validate_trials(
    request: StrategyAnalysisRequest,
    analysis: WindStrategyAnalysis,
) -> Mapping[int, WindTrial]:
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
    speed_mps: float,
    bearing_deg: float,
    expected_provenance: str,
    name: str,
) -> None:
    expected = WindScenario.from_meteorological(speed_mps, bearing_deg)
    require(
        all(
            _agrees(left, right)
            for left, right in zip(
                scenario.base_velocity_mps, expected.base_velocity_mps, strict=True
            )
        ),
        f"{name} does not agree with its wind trial",
    )
    require(
        scenario.provenance == expected_provenance
        and scenario.shear_fraction_per_10m == 0.0
        and scenario.turbulence_intensity_mps == 0.0
        and scenario.seed == 0
        and not scenario.gusts,
        f"{name} does not agree with its deterministic scenario contract",
    )


# fmt: off
def _validate_outcome(
    outcome: StrategyShotOutcome, trial: WindTrial, provenance: str
) -> None:
    statuses = {item.key for item in WIND_STRATEGY_COHORTS}
    require(outcome.status in statuses, "unknown actual status")
    perfect = outcome.perfect_information
    require(perfect.status in statuses, "unknown perfect-information status")
    _validate_scenario(outcome.true_wind, trial.true_speed_mps, trial.true_from_bearing_deg, f"{provenance}/true/trial-{trial.trial_index}", "true wind")  # noqa: E501
    _validate_scenario(outcome.estimated_wind, trial.estimated_speed_mps, trial.estimated_from_bearing_deg, f"{provenance}/estimated/trial-{trial.trial_index}", "estimated wind")  # noqa: E501
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


def _validate_outcomes(
    request: StrategyAnalysisRequest,
    analysis: WindStrategyAnalysis,
    trials: Mapping[int, WindTrial],
) -> Mapping[tuple[int, str], StrategyShotOutcome]:
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
        _validate_outcome(outcome, trials[outcome.trial_index], analysis.provenance)
    return dict(zip(keys, analysis.outcomes, strict=True))


def build_wind_strategy_plot_dataset(
    request: StrategyAnalysisRequest,
    analysis: WindStrategyAnalysis,
    *,
    result_id: str | None = None,
) -> ScalarEnsembleDataset:
    """Adapt an existing analysis without rerunning flight physics."""
    require(isinstance(request, StrategyAnalysisRequest), "request type is invalid")
    require(isinstance(analysis, WindStrategyAnalysis), "analysis type is invalid")
    _validate_header(request, analysis)
    trials = _validate_trials(request, analysis)
    outcomes = _validate_outcomes(request, analysis, trials)
    rows = tuple(
        _row(request, strategy, trials[index], outcomes[(index, strategy.strategy_id)])
        for index in sorted(trials)
        for strategy in request.strategies
    )
    return ScalarEnsembleDataset(
        SCALAR_ENSEMBLE_SCHEMA_VERSION,
        result_id or f"wind-strategy:{analysis.provenance}",
        ScalarEnsembleProvenance(
            WIND_STRATEGY_PLOT_ADAPTER_ID,
            analysis.schema_version,
            analysis.provenance,
        ),
        WIND_STRATEGY_STAGES,
        WIND_STRATEGY_CATEGORIES,
        WIND_STRATEGY_VARIABLES,
        WIND_STRATEGY_COHORTS,
        rows,
    )


# fmt: off
__all__ = ["WIND_STRATEGY_CATEGORIES", "WIND_STRATEGY_COHORTS", "WIND_STRATEGY_PLOT_ADAPTER_ID", "WIND_STRATEGY_STAGES", "WIND_STRATEGY_VARIABLES", "build_wind_strategy_plot_dataset"]  # noqa: E501
# fmt: on
