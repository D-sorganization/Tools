"""Contract tests for deterministic wind-estimate strategy analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from shared.python.swing_sim.flight import (
    FlightModelRegistry,
    LaunchConditions,
    PerfectInformationCounterfactual,
    ScalarDistribution,
    StrategyAnalysisConfig,
    StrategyAnalysisRequest,
    StrategyShotOutcome,
    TargetPoint,
    WindEstimateError,
    WindScenario,
    WindStrategy,
    WindUncertaintySpec,
    analyze_wind_strategies,
    sample_wind_trials,
)
from shared.python.swing_sim.flight.wind_strategy_metrics import (
    summarize_strategy_outcomes,
)

_FIXTURE = (
    Path(__file__).parents[5]
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "wind_uncertainty_golden_v1.json"
)
_RISK_FIXTURE = _FIXTURE.with_name("wind_strategy_risk_golden_v2.json")


def _spec(trials: int = 8) -> WindUncertaintySpec:
    return WindUncertaintySpec(
        trials=trials,
        seed=4199,
        true_speed_mps=ScalarDistribution("normal", 5.0, 1.1, minimum=0.0),
        true_from_bearing_deg=ScalarDistribution("uniform", 15.0, 20.0),
        estimate_error=WindEstimateError(
            speed_bias_mps=-0.8,
            speed_std_mps=0.7,
            bearing_bias_deg=3.0,
            bearing_std_deg=4.0,
            correlation=0.45,
        ),
        provenance="test/weather_station_plus_player_estimate",
    )


def test_seeded_samples_match_cross_language_golden_fixture() -> None:
    expected = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    samples = sample_wind_trials(_spec(len(expected["trials"])))

    actual = [sample.to_schema_dict() for sample in samples]
    assert actual == expected["trials"]


def test_samples_are_reproducible_and_expose_estimation_error() -> None:
    first = sample_wind_trials(_spec(64))
    second = sample_wind_trials(_spec(64))

    assert first == second
    assert np.mean([trial.speed_error_mps for trial in first]) < 0.0
    assert all(trial.true_speed_mps >= 0.0 for trial in first)
    assert all(-180.0 <= trial.true_from_bearing_deg < 180.0 for trial in first)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"correlation": 1.01},
        {"speed_std_mps": -0.1},
    ],
)
def test_invalid_estimate_error_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        WindEstimateError(**kwargs)


def test_strategy_analysis_returns_scatter_cost_and_common_random_regret() -> None:
    launch = LaunchConditions.from_imperial(150.0, 12.0, 2500.0)
    request = StrategyAnalysisRequest(
        uncertainty=_spec(6),
        strategies=(
            WindStrategy("straight", "Straight", launch),
            WindStrategy(
                "compensated",
                "Estimated-Crosswind Compensation",
                launch,
                crosswind_aim_gain_rad_per_mps=math.radians(0.2),
            ),
        ),
        target=TargetPoint(230.0, 0.0),
        analysis=StrategyAnalysisConfig(
            model_name="waterloo_penner",
            miss_scale_m=20.0,
            failure_cost=100.0,
            target_radius_m=10.0,
            miss_distance_cvar_alpha=0.75,
        ),
    )

    result = analyze_wind_strategies(request)

    assert len(result.wind_trials) == 6
    assert len(result.outcomes) == 12
    assert result.schema_version == "wind-strategy-analysis/v2"
    assert {summary.strategy_id for summary in result.summaries} == {
        "straight",
        "compensated",
    }
    assert all(summary.completed_trials == 6 for summary in result.summaries)
    assert all(summary.failed_trials == 0 for summary in result.summaries)
    assert all(summary.expected_cost >= 0.0 for summary in result.summaries)
    assert min(summary.expected_regret for summary in result.summaries) >= 0.0
    for summary in result.summaries:
        assert summary.expected_regret == summary.expected_preset_oracle_regret
        assert summary.probability_best == summary.preset_oracle_probability_best
        assert summary.miss_distance_cvar_alpha == 0.75
        assert summary.miss_distance_cvar_m >= 0.0
        assert 0.0 <= summary.target_hold_probability <= 1.0
        for risk in (
            summary.short_risk,
            summary.long_risk,
            summary.left_risk,
            summary.right_risk,
        ):
            assert 0.0 <= risk.probability <= 1.0
            assert risk.mean_excess_m >= 0.0
            assert risk.conditional_mean_excess_m >= 0.0
    assert all(outcome.perfect_information.cost >= 0.0 for outcome in result.outcomes)
    assert all(
        math.isfinite(outcome.information_cost_delta) for outcome in result.outcomes
    )
    for trial_index in range(6):
        paired = [item for item in result.outcomes if item.trial_index == trial_index]
        assert len({item.true_wind for item in paired}) == 1


def test_nonconverged_flights_receive_declared_failure_cost() -> None:
    launch = LaunchConditions.from_imperial(150.0, 45.0, 2500.0)
    request = StrategyAnalysisRequest(
        uncertainty=_spec(2),
        strategies=(WindStrategy("lofted", "Lofted", launch),),
        target=TargetPoint(100.0, 0.0),
        analysis=StrategyAnalysisConfig(max_time_s=0.01, failure_cost=37.0),
    )

    result = analyze_wind_strategies(request)

    assert {outcome.status for outcome in result.outcomes} == {"nonconverged"}
    assert result.summaries[0].completed_trials == 0
    assert result.summaries[0].failed_trials == 2
    assert result.summaries[0].expected_cost == 37.0


def test_numerical_failure_is_retained_as_invalid_cohort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenModel:
        def simulate(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError("deliberate integration failure")

    monkeypatch.setattr(
        FlightModelRegistry,
        "get_model",
        classmethod(lambda _cls, _model_type: BrokenModel()),
    )
    launch = LaunchConditions.from_imperial(150.0, 12.0, 2500.0)
    request = StrategyAnalysisRequest(
        uncertainty=_spec(2),
        strategies=(WindStrategy("broken", "Broken", launch),),
        target=TargetPoint(200.0, 0.0),
        analysis=StrategyAnalysisConfig(failure_cost=23.0),
    )

    result = analyze_wind_strategies(request)

    assert {outcome.status for outcome in result.outcomes} == {"invalid"}
    assert all(outcome.landing_forward_m is None for outcome in result.outcomes)
    assert all(outcome.cost == 23.0 for outcome in result.outcomes)
    assert result.summaries[0].failed_trials == 2


def test_true_wind_counterfactual_matches_actual_when_estimate_is_exact() -> None:
    uncertainty = WindUncertaintySpec(
        trials=3,
        seed=7,
        true_speed_mps=ScalarDistribution("fixed", 5.0, minimum=0.0),
        true_from_bearing_deg=ScalarDistribution("fixed", 90.0),
        provenance="test/exact-estimate",
    )
    launch = LaunchConditions.from_imperial(150.0, 12.0, 2500.0)
    result = analyze_wind_strategies(
        StrategyAnalysisRequest(
            uncertainty=uncertainty,
            strategies=(
                WindStrategy(
                    "compensated",
                    "Compensated",
                    launch,
                    crosswind_aim_gain_rad_per_mps=math.radians(0.2),
                ),
            ),
            target=TargetPoint(230.0, 0.0),
        )
    )

    assert all(
        outcome.cost == pytest.approx(outcome.perfect_information.cost)
        for outcome in result.outcomes
    )
    assert all(
        outcome.information_cost_delta == pytest.approx(0.0)
        for outcome in result.outcomes
    )
    assert result.summaries[0].expected_information_cost_delta == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_radius_m": -0.1},
        {"miss_distance_cvar_alpha": 0.0},
        {"miss_distance_cvar_alpha": 1.0},
    ],
)
def test_strategy_risk_configuration_rejects_invalid_bounds(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        StrategyAnalysisConfig(**kwargs)


def test_failure_cohorts_are_included_in_hold_and_tail_risk_denominators() -> None:
    fixture = json.loads(_RISK_FIXTURE.read_text(encoding="utf-8"))
    failure_case = fixture["failure_case"]
    expected = failure_case["expected"]
    launch = LaunchConditions.from_imperial(150.0, 45.0, 2500.0)
    result = analyze_wind_strategies(
        StrategyAnalysisRequest(
            uncertainty=_spec(2),
            strategies=(WindStrategy("lofted", "Lofted", launch),),
            target=TargetPoint(100.0, 0.0),
            analysis=StrategyAnalysisConfig(
                max_time_s=0.01,
                miss_scale_m=failure_case["miss_scale_m"],
                failure_cost=failure_case["failure_cost"],
                target_radius_m=failure_case["target_radius_m"],
                miss_distance_cvar_alpha=failure_case["miss_distance_cvar_alpha"],
            ),
        )
    )

    summary = result.summaries[0]
    assert result.schema_version == fixture["schema_version"]
    assert summary.target_hold_probability == expected["target_hold_probability"]
    assert summary.miss_distance_cvar_m == pytest.approx(
        expected["miss_distance_cvar_m"]
    )
    assert summary.expected_information_cost_delta == pytest.approx(
        expected["expected_information_cost_delta"]
    )
    assert summary.expected_preset_oracle_regret == pytest.approx(
        expected["expected_preset_oracle_regret"]
    )
    for risk in (
        summary.short_risk,
        summary.long_risk,
        summary.left_risk,
        summary.right_risk,
    ):
        assert risk.probability == expected["directional_probability"]


def test_directional_risk_matches_cross_language_golden_fixture() -> None:
    fixture = json.loads(_RISK_FIXTURE.read_text(encoding="utf-8"))
    case = fixture["directional_case"]
    target = TargetPoint(**case["target"])
    strategy = WindStrategy(
        "fixture", "Fixture", LaunchConditions.from_imperial(150.0, 12.0, 2500.0)
    )
    request = StrategyAnalysisRequest(
        uncertainty=_spec(len(case["landings"])),
        strategies=(strategy,),
        target=target,
        analysis=StrategyAnalysisConfig(
            miss_scale_m=case["miss_scale_m"],
            failure_cost=case["failure_cost"],
            target_radius_m=case["target_radius_m"],
            miss_distance_cvar_alpha=case["miss_distance_cvar_alpha"],
        ),
    )
    calm = WindScenario(provenance="test/golden-risk")
    outcomes: list[StrategyShotOutcome] = []
    for index, landing in enumerate(case["landings"]):
        if landing is None:
            status, forward, right, miss, cost = (
                "nonconverged",
                None,
                None,
                None,
                case["failure_cost"],
            )
            reason = "fixture failure"
        else:
            status, forward, right = (
                "completed",
                landing["forward_m"],
                landing["right_m"],
            )
            miss = math.hypot(forward - target.forward_m, right - target.right_m)
            cost = (miss / case["miss_scale_m"]) ** 2
            reason = None
        perfect = PerfectInformationCounterfactual(
            status, forward, right, miss, cost, reason
        )
        outcomes.append(
            StrategyShotOutcome(
                index,
                strategy.strategy_id,
                status,
                calm,
                calm,
                forward,
                right,
                miss,
                cost,
                reason,
                perfect,
                0.0,
            )
        )

    summary = summarize_strategy_outcomes(request, tuple(outcomes))[0]
    expected = case["expected"]
    assert summary.expected_cost == pytest.approx(expected["expected_cost"])
    assert summary.target_hold_probability == expected["target_hold_probability"]
    assert summary.miss_distance_cvar_m == pytest.approx(
        expected["miss_distance_cvar_m"]
    )
    for name in ("short_risk", "long_risk", "left_risk", "right_risk"):
        risk = getattr(summary, name)
        expected_risk = expected[name]
        assert risk.probability == expected_risk["probability"]
        assert risk.mean_excess_m == pytest.approx(expected_risk["mean_excess_m"])
        assert risk.conditional_mean_excess_m == pytest.approx(
            expected_risk["conditional_mean_excess_m"]
        )
