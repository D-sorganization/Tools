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
    ScalarDistribution,
    StrategyAnalysisConfig,
    StrategyAnalysisRequest,
    TargetPoint,
    WindEstimateError,
    WindStrategy,
    WindUncertaintySpec,
    analyze_wind_strategies,
    sample_wind_trials,
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
        ),
    )

    result = analyze_wind_strategies(request)

    assert len(result.wind_trials) == 6
    assert len(result.outcomes) == 12
    assert {summary.strategy_id for summary in result.summaries} == {
        "straight",
        "compensated",
    }
    assert all(summary.completed_trials == 6 for summary in result.summaries)
    assert all(summary.failed_trials == 0 for summary in result.summaries)
    assert all(summary.expected_cost >= 0.0 for summary in result.summaries)
    assert min(summary.expected_regret for summary in result.summaries) >= 0.0
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
