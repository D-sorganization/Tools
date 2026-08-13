"""Contract tests for the model-backed capability evaluator."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from shared.python.swing_sim.flight import (
    CapabilityFlightEvaluatorConfig,
    CapabilityObjective,
    CapabilityParameter,
    CapabilitySpinDefault,
    ClubCapability,
    EvaluationStatus,
    FlightMetricId,
    OptimizationRequest,
    PlayerCapabilityProfile,
    SolverEvaluation,
    TargetDefinition,
    make_capability_flight_evaluator,
    optimize_capability,
)

_FIXTURE = (
    Path(__file__).parents[5]
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "capability_optimizer_golden_v1.json"
)
_PARITY_FIXTURE = _FIXTURE.parent / "capability_flight_evaluator_parity_v1.json"
PINNED = {
    "ball_speed": 74.65568,
    "launch_angle": 10.9,
    "launch_direction": 0.0,
}


def _config(
    *,
    max_time_s: float = 10.0,
    spin_rpm: float = 2686.0,
    sample_interval_s: float = 0.01,
) -> CapabilityFlightEvaluatorConfig:
    default = CapabilitySpinDefault("driver", spin_rpm, 0.0, "test-fixture")
    return CapabilityFlightEvaluatorConfig(max_time_s, sample_interval_s, (default,))


def _parameter(
    parameter_id: str, unit: str, value: float, lower: float, upper: float
) -> CapabilityParameter:
    return CapabilityParameter(
        parameter_id, unit, lower, upper, lower, upper, value, 0.0, 0.0
    )


def _profile(*, include_spin: bool = False) -> PlayerCapabilityProfile:
    parameters = [
        _parameter("ball_speed", "m/s", PINNED["ball_speed"], 40.0, 80.0),
        _parameter("launch_angle", "deg", PINNED["launch_angle"], 0.0, 30.0),
        _parameter("launch_direction", "deg", 0.0, -15.0, 15.0),
    ]
    if include_spin:
        parameters.extend(
            (
                _parameter("total_spin", "rpm", 2686.0, 0.0, 6000.0),
                _parameter("spin_axis_tilt", "deg", 0.0, -45.0, 45.0),
            )
        )
    size = len(parameters)
    matrix = tuple(
        tuple(1.0 if row == column else 0.0 for column in range(size))
        for row in range(size)
    )
    club = ClubCapability(
        "driver", tuple(parameters), "correlation", matrix, "test", 1.0
    )
    return PlayerCapabilityProfile("player", (club,), "test", 1.0)


def _request(*, max_samples: int = 1) -> OptimizationRequest:
    return OptimizationRequest(
        "flight-evaluator",
        CapabilityObjective.MAXIMIZE_TARGET_HOLD,
        ("driver",),
        TargetDefinition("green", 247.0, 0.0, 10.0, 15.0, 15.0),
        max_samples,
        1,
        1,
        7,
        0.8,
        0.5,
    )


def _metric_map(evaluation: SolverEvaluation) -> dict[FlightMetricId, float]:
    return {item.metric_id: item.value for item in evaluation.metrics}


@pytest.mark.integration
def test_runs_waterloo_penner_with_all_scalars_and_target_values() -> None:
    evaluator = make_capability_flight_evaluator(_profile(), _request(), _config())
    evaluation = evaluator("driver", PINNED)
    metrics = _metric_map(evaluation)

    assert evaluation.status is EvaluationStatus.COMPLETE
    assert metrics[FlightMetricId.BALL_SPEED] == pytest.approx(74.65568)
    assert metrics[FlightMetricId.VERTICAL_LAUNCH_ANGLE] == pytest.approx(10.9)
    assert metrics[FlightMetricId.CARRY_DISTANCE] == pytest.approx(247.484, rel=0.01)
    assert metrics[FlightMetricId.APEX_HEIGHT] == pytest.approx(28.226, rel=0.02)
    assert FlightMetricId.TARGET_RESIDUAL in metrics
    assert FlightMetricId.INITIAL_VELOCITY not in metrics
    assert len(metrics) == 16
    assert all(
        "waterloo_penner:waterloo-penner-coefficients/v1" in item.provenance
        for item in evaluation.metrics
    )
    assert all(
        "spin:fixed_club_default:test-fixture" in item.provenance
        for item in evaluation.metrics
    )

    fixture = json.loads(_PARITY_FIXTURE.read_text(encoding="utf-8"))
    for metric_id, expected in fixture["expected_scalars"].items():
        assert metrics[FlightMetricId(metric_id)] == pytest.approx(
            expected["value"], abs=expected["absolute_tolerance"]
        )


@pytest.mark.integration
def test_preserves_positive_right_direction_and_fade_side_spin_tilt() -> None:
    evaluator = make_capability_flight_evaluator(
        _profile(include_spin=True), _request()
    )
    direction = evaluator(
        "driver",
        {
            **PINNED,
            "launch_direction": 5.0,
            "total_spin": 2686.0,
            "spin_axis_tilt": 0.0,
        },
    )
    tilt = evaluator("driver", {**PINNED, "total_spin": 2686.0, "spin_axis_tilt": 10.0})
    negative_tilt = evaluator(
        "driver", {**PINNED, "total_spin": 2686.0, "spin_axis_tilt": -10.0}
    )
    direction_metrics = _metric_map(direction)
    tilt_metrics = _metric_map(tilt)

    assert direction_metrics[FlightMetricId.LAUNCH_DIRECTION] == pytest.approx(5.0)
    assert direction_metrics[FlightMetricId.CARRY_OFFLINE] > 1.0
    assert tilt_metrics[FlightMetricId.SPIN_AXIS_TILT] == pytest.approx(10.0)
    assert tilt_metrics[FlightMetricId.CARRY_OFFLINE] > 1.0
    assert _metric_map(negative_tilt)[FlightMetricId.CARRY_OFFLINE] < -1.0


@pytest.mark.integration
def test_plugs_into_existing_profile_fixture_and_optimizer() -> None:
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    profile = PlayerCapabilityProfile.from_dict(payload["profile"])
    request = OptimizationRequest.from_dict(
        {
            **payload["request"],
            "club_ids": ["driver"],
            "candidate_budget": 1,
            "ensemble_size": 1,
            "alternatives_count": 1,
        }
    )
    evaluator = make_capability_flight_evaluator(profile, request, _config())
    result = optimize_capability(profile, request, evaluator)

    assert result.status == "solved"
    assert result.evaluations_completed == 1
    assert result.alternatives[0].mean_carry_m > 100.0


@pytest.mark.unit
def test_short_horizon_is_nonconverged_and_zero_spin_remains_complete() -> None:
    short = _config(max_time_s=0.001)
    evaluation = make_capability_flight_evaluator(_profile(), _request(), short)(
        "driver", PINNED
    )
    assert evaluation.status is EvaluationStatus.NONCONVERGED
    assert evaluation.metrics == ()
    assert evaluation.reason == "no_ground_crossing_before_max_time"

    zero_spin = _config(spin_rpm=0.0)
    completed = make_capability_flight_evaluator(_profile(), _request(), zero_spin)(
        "driver", PINNED
    )
    assert completed.status is EvaluationStatus.COMPLETE
    assert FlightMetricId.SPIN_AXIS_TILT not in _metric_map(completed)
    coarse = make_capability_flight_evaluator(
        _profile(), _request(), _config(sample_interval_s=0.1)
    )("driver", PINNED)
    assert coarse.status is EvaluationStatus.COMPLETE


@pytest.mark.unit
def test_each_club_requires_its_own_auditable_spin_default() -> None:
    driver = _profile().clubs[0]
    iron = replace(driver, club_id="iron")
    profile = PlayerCapabilityProfile("player", (driver, iron), "test", 1.0)
    request = replace(_request(), club_ids=("driver", "iron"))

    with pytest.raises(ValueError, match="iron.*explicit spin default"):
        make_capability_flight_evaluator(profile, request, _config())

    defaults = (
        CapabilitySpinDefault("driver", 2686.0, 0.0, "driver-fixture"),
        CapabilitySpinDefault("iron", 6200.0, 0.0, "iron-fixture"),
    )
    config = CapabilityFlightEvaluatorConfig(10.0, 0.01, defaults)
    evaluator = make_capability_flight_evaluator(profile, request, config)
    iron_result = evaluator("iron", PINNED)

    assert iron_result.status is EvaluationStatus.COMPLETE
    assert all(
        "spin:fixed_club_default:iron-fixture" in item.provenance
        for item in iron_result.metrics
    )


@pytest.mark.unit
def test_rejects_schema_units_bounds_and_configuration() -> None:
    evaluator = make_capability_flight_evaluator(_profile(), _request(), _config())
    with pytest.raises(ValueError, match="fields"):
        evaluator("driver", {**PINNED, "unused": 1.0})
    with pytest.raises(ValueError, match="safe bounds"):
        evaluator("driver", {**PINNED, "ball_speed": 100.0})
    bad_profile: dict[str, Any] = _profile().to_dict()
    bad_profile["clubs"][0]["parameters"][0]["unit"] = "mph"
    with pytest.raises(ValueError, match="ball_speed.*m/s"):
        make_capability_flight_evaluator(
            PlayerCapabilityProfile.from_dict(bad_profile), _request(), _config()
        )
    with pytest.raises(ValueError, match="max_time_s"):
        CapabilityFlightEvaluatorConfig(max_time_s=0.0)
    with pytest.raises(ValueError, match="explicit spin default"):
        make_capability_flight_evaluator(_profile(), _request())
    with pytest.raises(ValueError, match=r"\[0.001, 0.1\]"):
        CapabilityFlightEvaluatorConfig(10.0, 9.0, ())
    with pytest.raises(ValueError, match="align"):
        CapabilityFlightEvaluatorConfig(10.0, 0.0015, ())


@pytest.mark.unit
@pytest.mark.parametrize(
    "parameter_id, field, value, message",
    [
        ("ball_speed", "lower_bound", 0.0, "ball_speed.*greater than zero"),
        ("launch_angle", "upper_bound", 100.0, r"launch_angle.*\[-90, 90\]"),
        (
            "launch_direction",
            "upper_bound",
            300.0,
            r"launch_direction.*\[-180, 180\]",
        ),
        ("total_spin", "lower_bound", -1.0, r"total_spin.*\[0, inf\]"),
        ("spin_axis_tilt", "upper_bound", 120.0, r"spin_axis_tilt.*\[-90, 90\]"),
    ],
)
def test_rejects_profiles_outside_physical_flight_domains(
    parameter_id: str, field: str, value: float, message: str
) -> None:
    payload: dict[str, Any] = _profile(
        include_spin=parameter_id in {"total_spin", "spin_axis_tilt"}
    ).to_dict()
    parameters = payload["clubs"][0]["parameters"]
    parameter = next(
        item for item in parameters if item["parameter_id"] == parameter_id
    )
    parameter[field] = value
    if field == "upper_bound":
        parameter["evidence_upper_bound"] = value
    with pytest.raises(ValueError, match=message):
        make_capability_flight_evaluator(
            PlayerCapabilityProfile.from_dict(payload), _request(), _config()
        )


@pytest.mark.unit
def test_integrator_failure_is_typed_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shared.python.swing_sim.flight.capability_flight_evaluator as module

    def fail(*_args: object, **_kwargs: object) -> object:
        raise OverflowError("private solver detail")

    monkeypatch.setattr(module, "simulate", fail)
    evaluator = make_capability_flight_evaluator(_profile(), _request(), _config())
    first = evaluator("driver", PINNED)
    second = evaluator("driver", PINNED)
    assert first == second
    assert first.status is EvaluationStatus.FAILED
    assert first.reason == "flight_model_failure"
