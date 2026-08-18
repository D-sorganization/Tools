"""Atomic accepted-flight authority and immutability contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rate_of_closure.flight_accepted_study import (
    FlightStudyContext,
    build_accepted_flight_study,
)
from rate_of_closure.simulation import (
    compare_wind,
    explore_flight,
    launch_from_delivery,
    launch_from_direct,
)
from shared.python.swing_sim.flight import (
    LaunchDirectionConvention,
    WindScenario,
)
from shared.python.swing_sim.impact import DeliveryParameters

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _direct_context(
    scenario: WindScenario | None = None,
    *,
    inputs: tuple[tuple[str, float], ...] | None = None,
) -> FlightStudyContext:
    values = inputs or (
        ("ball_speed_mph", 150.0),
        ("launch_angle_deg", 12.0),
        ("launch_direction_deg", 3.0),
        ("spin_rpm", 2700.0),
        ("spin_axis_tilt_deg", 5.0),
    )
    base = launch_from_direct(
        *(value for _key, value in values),
        direction_convention=LaunchDirectionConvention.APP_NATIVE,
    )
    return FlightStudyContext(
        "direct",
        values,
        LaunchDirectionConvention.APP_NATIVE,
        "waterloo_penner",
        scenario,
        replace(base, wind_speed=0.0, wind_scenario=scenario),
    )


def test_direct_candidate_is_complete_immutable_and_summary_coherent() -> None:
    context = _direct_context()
    result = explore_flight(context.expected_launch, context.model_name)
    accepted = build_accepted_flight_study(1, context, result, None)
    assert accepted.plan.raw_count == len(result.times)
    assert accepted.exploration.metrics["carry_m"] == result.metrics["carry_m"]
    with pytest.raises(ValueError):
        accepted.exploration.positions.setflags(write=True)
    with pytest.raises(TypeError):
        accepted.exploration.metrics["carry_m"] = 0.0


def test_context_rederives_direct_inputs_and_rejects_wrong_authority() -> None:
    context = _direct_context()
    wrong_inputs = tuple(
        (key, 149.0 if key == "ball_speed_mph" else value)
        for key, value in context.input_values
    )
    with pytest.raises(ValueError, match="raw flight inputs"):
        replace(context, input_values=wrong_inputs)
    with pytest.raises(ValueError, match="legacy scalar wind"):
        replace(
            context, expected_launch=replace(context.expected_launch, wind_speed=1.0)
        )
    result = explore_flight(context.expected_launch, context.model_name)
    result.metrics["landing_angle_deg"] += 5.0
    with pytest.raises(ValueError, match="landing angle"):
        build_accepted_flight_study(1, context, result, None)


def test_raw_origin_landing_floor_and_launch_velocity_are_bound() -> None:
    context = _direct_context()
    translated = explore_flight(context.expected_launch, context.model_name)
    translated.positions[:] += (10.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="launch downrange position"):
        build_accepted_flight_study(1, context, translated, None)

    airborne = explore_flight(context.expected_launch, context.model_name)
    airborne.positions[-1, 1] += 1.0
    with pytest.raises(ValueError, match="landing height"):
        build_accepted_flight_study(1, context, airborne, None)

    below_ground = explore_flight(context.expected_launch, context.model_name)
    below_ground.positions[1, 1] = -1.0
    with pytest.raises(ValueError, match="ground plane"):
        build_accepted_flight_study(1, context, below_ground, None)

    forged_velocity = explore_flight(context.expected_launch, context.model_name)
    forged_velocity.velocities[0] *= 0.5
    with pytest.raises(ValueError, match="raw launch speed"):
        build_accepted_flight_study(1, context, forged_velocity, None)


def test_delivery_context_includes_lie_and_does_not_confuse_club_and_ball_speed() -> (
    None
):
    delivery = DeliveryParameters(
        clubhead_speed_mps=50.0,
        club_path_deg=1.0,
        face_angle_deg=2.0,
        attack_angle_deg=-1.0,
        dynamic_loft_deg=12.0,
        impact_offset_toe_mm=1.0,
        impact_offset_high_mm=2.0,
        lie_deg=1.5,
    )
    launch = launch_from_delivery(delivery)
    inputs = (
        ("clubhead_speed_mps", 50.0),
        ("club_path_deg", 1.0),
        ("face_angle_deg", 2.0),
        ("attack_angle_deg", -1.0),
        ("dynamic_loft_deg", 12.0),
        ("impact_offset_toe_mm", 1.0),
        ("impact_offset_high_mm", 2.0),
        ("lie_deg", 1.5),
    )
    context = FlightStudyContext(
        "delivery",
        inputs,
        LaunchDirectionConvention.APP_NATIVE,
        "waterloo_penner",
        None,
        launch,
    )
    accepted = build_accepted_flight_study(
        1, context, explore_flight(launch, "waterloo_penner"), None
    )
    assert accepted.context.input_values[-1] == ("lie_deg", 1.5)


def test_wind_pair_is_snapshotted_and_deltas_are_recomputed() -> None:
    scenario = WindScenario.from_meteorological(5.0, 90.0)
    context = _direct_context(scenario)
    comparison = compare_wind(
        replace(context.expected_launch, wind_scenario=None),
        scenario,
        context.model_name,
    )
    accepted = build_accepted_flight_study(2, context, comparison.wind, comparison)
    before = accepted.comparison.deltas["carry_m"]
    comparison.deltas["carry_m"] = 999.0
    comparison.wind.positions[0, 0] = 999.0
    assert accepted.comparison.deltas["carry_m"] == before
    assert accepted.exploration.positions[0, 0] != 999.0
    forged = compare_wind(
        replace(context.expected_launch, wind_scenario=None),
        scenario,
        context.model_name,
    )
    forged.deltas["carry_m"] += 1.0
    with pytest.raises(ValueError, match="wind carry_m"):
        build_accepted_flight_study(3, context, forged.wind, forged)


@pytest.mark.parametrize("generation", [True, 0, 2**53])
def test_generation_must_be_a_positive_safe_integer(generation: object) -> None:
    context = _direct_context()
    result = explore_flight(context.expected_launch)
    with pytest.raises(ValueError):
        build_accepted_flight_study(generation, context, result, None)
