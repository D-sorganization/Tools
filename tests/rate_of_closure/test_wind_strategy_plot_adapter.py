"""Tests for projecting computed wind strategy analyses into scalar plots."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

import pytest

from rate_of_closure.variation.wind_strategy_plot_adapter import (
    build_wind_strategy_plot_dataset,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.flight import (
    LaunchConditions,
    PerfectInformationCounterfactual,
    ScalarDistribution,
    StrategyAnalysisRequest,
    StrategyShotOutcome,
    TargetPoint,
    WindEstimateError,
    WindGust,
    WindScenario,
    WindStrategy,
    WindStrategyAnalysis,
    WindTrial,
    WindUncertaintySpec,
)
from shared.python.swing_sim.flight.wind_strategy import (
    WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe, pytest.mark.contract]


def _request() -> StrategyAnalysisRequest:
    return StrategyAnalysisRequest(
        uncertainty=WindUncertaintySpec(
            trials=3,
            seed=17,
            true_speed_mps=ScalarDistribution("fixed", 5.0, minimum=0.0),
            true_from_bearing_deg=ScalarDistribution("fixed", 90.0),
            estimate_error=WindEstimateError(
                speed_bias_mps=-1.0,
                bearing_bias_deg=-10.0,
            ),
            provenance="range-study",
        ),
        strategies=(
            WindStrategy(
                strategy_id="wedge",
                label="Wedge",
                launch=LaunchConditions(
                    ball_speed=30.0,
                    launch_angle=math.radians(35.0),
                    azimuth_angle=math.radians(2.0),
                    spin_rate=6500.0,
                    spin_axis_angle=math.radians(-3.0),
                ),
                crosswind_aim_gain_rad_per_mps=0.01,
            ),
        ),
        target=TargetPoint(100.0, 4.0),
    )


def _wind_trial(index: int) -> WindTrial:
    return WindTrial(index, 5.0, 90.0, 4.0, 80.0, -1.0, -10.0)


def _scenario(speed: float, bearing: float, provenance: str) -> WindScenario:
    base = WindScenario.from_meteorological(speed, bearing)
    return replace(base, provenance=provenance)


def _outcome(
    index: int,
    status: Literal["completed", "nonconverged", "invalid"],
    actual: tuple[float, float, float] | None,
    perfect: tuple[float, float, float] | None,
) -> StrategyShotOutcome:
    provenance = "range-study"
    actual_cost = 100.0 if actual is None else actual[2]
    perfect_cost = 100.0 if perfect is None else perfect[2]
    return StrategyShotOutcome(
        trial_index=index,
        strategy_id="wedge",
        status=status,
        true_wind=_scenario(5.0, 90.0, f"{provenance}/true/trial-{index}"),
        estimated_wind=_scenario(4.0, 80.0, f"{provenance}/estimated/trial-{index}"),
        landing_forward_m=None if actual is None else actual[0],
        landing_right_m=None if actual is None else actual[1],
        miss_distance_m=(
            None if actual is None else math.hypot(actual[0] - 100.0, actual[1] - 4.0)
        ),
        cost=actual_cost,
        failure_reason=None if actual is not None else f"{status} outcome",
        perfect_information=PerfectInformationCounterfactual(
            status="invalid" if perfect is None else "completed",
            landing_forward_m=None if perfect is None else perfect[0],
            landing_right_m=None if perfect is None else perfect[1],
            miss_distance_m=(
                None
                if perfect is None
                else math.hypot(perfect[0] - 100.0, perfect[1] - 4.0)
            ),
            cost=perfect_cost,
            failure_reason=None if perfect is not None else "counterfactual failed",
        ),
        information_cost_delta=actual_cost - perfect_cost,
    )


def _analysis() -> WindStrategyAnalysis:
    trials = tuple(_wind_trial(index) for index in range(3))
    outcomes = (
        _outcome(0, "completed", (98.0, 5.0, 0.0125), (100.0, 4.0, 0.0)),
        _outcome(1, "nonconverged", None, (99.0, 4.0, 0.0025)),
        _outcome(2, "invalid", None, None),
    )
    return WindStrategyAnalysis(
        schema_version=WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION,
        provenance="range-study",
        target=TargetPoint(100.0, 4.0),
        wind_trials=trials,
        outcomes=outcomes,
        summaries=(),
    )


def test_adapter_exposes_wind_launch_aim_target_and_result_variables() -> None:
    dataset = build_wind_strategy_plot_dataset(_request(), _analysis())

    assert dataset.result_id == "wind-strategy:range-study"
    assert dataset.provenance.source_provenance == "range-study"
    assert tuple(stage.key for stage in dataset.stages) == (
        "input",
        "environment",
        "actual",
        "perfect_information",
        "comparison",
    )
    assert dataset.variable("true_wind_speed_mps").unit == "m/s"
    assert dataset.variable("launch_azimuth_rad").unit == "rad"
    assert dataset.variable("actual_landing_forward_m").category_key == "actual"
    assert dataset.variable("information_cost_delta").unit == "1"


def test_adapter_preserves_all_status_cohorts_and_never_fills_failed_landings() -> None:
    dataset = build_wind_strategy_plot_dataset(_request(), _analysis())

    assert tuple(row.cohort for row in dataset.rows) == (
        "completed",
        "nonconverged",
        "invalid",
    )
    assert dataset.rows[0].row_id == "series:wedge/trial:0"
    assert dataset.rows[0].series_id == "wedge"
    assert dataset.rows[1].value("actual_landing_forward_m") is None
    assert dataset.rows[1].value("actual_miss_distance_m") is None
    assert dataset.rows[1].value("perfect_information_landing_forward_m") == 99.0
    assert dataset.rows[2].value("perfect_information_landing_forward_m") is None
    scatter = dataset.scatter("actual_landing_forward_m", "actual_landing_right_m")
    assert tuple(point.row_id for point in scatter.points) == ("series:wedge/trial:0",)
    assert scatter.availability.by_cohort["completed"].paired_finite == 1
    assert scatter.availability.by_cohort["nonconverged"].unavailable == 1
    assert scatter.availability.by_cohort["invalid"].unavailable == 1


def test_adapter_projects_declared_launch_and_estimated_wind_aim_without_physics() -> (
    None
):
    dataset = build_wind_strategy_plot_dataset(_request(), _analysis())
    row = dataset.rows[0]

    assert row.value("true_wind_speed_mps") == 5.0
    assert row.value("estimated_wind_speed_mps") == 4.0
    assert row.value("wind_speed_error_mps") == -1.0
    assert row.value("wind_bearing_error_deg") == -10.0
    assert row.value("estimated_wind_left_mps") == pytest.approx(
        4.0 * math.sin(math.radians(80.0))
    )
    assert row.value("launch_ball_speed_mps") == 30.0
    assert row.value("launch_angle_rad") == pytest.approx(math.radians(35.0))
    assert row.value("launch_azimuth_rad") == pytest.approx(math.radians(2.0))
    expected_aim = math.radians(2.0) - 0.01 * 4.0 * math.sin(math.radians(80.0))
    assert row.value("actual_aim_azimuth_rad") == pytest.approx(expected_aim)
    assert row.value("target_forward_m") == 100.0
    assert row.value("target_right_m") == 4.0
    assert row.value("actual_cost") == 0.0125
    assert row.value("perfect_information_cost") == 0.0
    assert row.value("information_cost_delta") == 0.0125


@pytest.mark.parametrize(
    ("analysis", "message"),
    [
        (replace(_analysis(), provenance="other"), "provenance"),
        (replace(_analysis(), target=TargetPoint(101.0, 4.0)), "target"),
        (
            replace(_analysis(), wind_trials=_analysis().wind_trials[:-1]),
            "wind trial count",
        ),
        (
            replace(
                _analysis(),
                outcomes=(replace(_analysis().outcomes[0], strategy_id="missing"),)
                + _analysis().outcomes[1:],
            ),
            "unknown strategy",
        ),
    ],
)
def test_adapter_fails_closed_when_request_and_analysis_disagree(
    analysis: WindStrategyAnalysis, message: str
) -> None:
    with pytest.raises(ContractViolationError, match=message):
        build_wind_strategy_plot_dataset(_request(), analysis)


def test_adapter_rejects_duplicate_or_missing_trial_strategy_rows() -> None:
    analysis = _analysis()
    duplicate = replace(
        analysis, outcomes=analysis.outcomes[:-1] + (analysis.outcomes[0],)
    )
    missing = replace(analysis, outcomes=analysis.outcomes[:-1])

    with pytest.raises(ContractViolationError, match="outcome trial/strategy keys"):
        build_wind_strategy_plot_dataset(_request(), duplicate)
    with pytest.raises(ContractViolationError, match="outcome count"):
        build_wind_strategy_plot_dataset(_request(), missing)


def test_adapter_rejects_inconsistent_trial_wind_and_outcome_availability() -> None:
    analysis = _analysis()
    wrong_wind = replace(
        analysis.outcomes[0],
        true_wind=_scenario(6.0, 90.0, "range-study/true/trial-0"),
    )
    wrong_values = replace(
        analysis.outcomes[1],
        landing_forward_m=12.0,
    )

    with pytest.raises(ContractViolationError, match="true wind"):
        build_wind_strategy_plot_dataset(
            _request(),
            replace(analysis, outcomes=(wrong_wind,) + analysis.outcomes[1:]),
        )
    with pytest.raises(ContractViolationError, match="actual outcome availability"):
        build_wind_strategy_plot_dataset(
            _request(),
            replace(
                analysis,
                outcomes=(analysis.outcomes[0], wrong_values, analysis.outcomes[2]),
            ),
        )


@pytest.mark.parametrize(
    "scenario",
    [
        replace(_scenario(5.0, 90.0, "range-study/true/trial-0"), provenance="other"),
        replace(
            _scenario(5.0, 90.0, "range-study/true/trial-0"),
            shear_fraction_per_10m=0.1,
        ),
        replace(
            _scenario(5.0, 90.0, "range-study/true/trial-0"),
            turbulence_intensity_mps=0.2,
        ),
        replace(_scenario(5.0, 90.0, "range-study/true/trial-0"), seed=1),
        replace(
            _scenario(5.0, 90.0, "range-study/true/trial-0"),
            gusts=(WindGust(0.0, 1.0, (0.0, 1.0, 0.0)),),
        ),
    ],
)
def test_adapter_rejects_noncanonical_deterministic_wind_scenarios(
    scenario: WindScenario,
) -> None:
    analysis = _analysis()
    wrong = replace(analysis.outcomes[0], true_wind=scenario)

    with pytest.raises(ContractViolationError, match="deterministic scenario"):
        build_wind_strategy_plot_dataset(
            _request(),
            replace(analysis, outcomes=(wrong,) + analysis.outcomes[1:]),
        )
