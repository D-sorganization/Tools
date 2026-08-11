"""State-machine tests for bounded skid, roll, regions, and rest."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from shared.python.swing_sim.ground import (
    GroundPhase,
    PlanarSurfaceDomain,
    SkidRollExecution,
    SkidRollSettings,
    SkidRollTerminationReason,
    SurfaceResolver,
    simulate_skid_roll,
)

from ._support import _settled_prefix, _surface, _surface_run_request

_GOLDEN = (
    Path(__file__).parents[5]
    / "rate_of_closure/web/src/model/__fixtures__/ground_skid_roll_golden_v1.json"
)


def test_flat_skid_and_roll_match_analytic_transition_and_stop() -> None:
    surface = replace(
        _surface(),
        static_friction=0.3,
        kinetic_friction=0.2,
        rolling_resistance=0.05,
    )
    request = _surface_run_request(surface=surface)
    prefix = _settled_prefix(request)
    result = simulate_skid_roll(request, prefix)
    skid_time = 7.0 / (0.2 * 9.80665 * 3.5)
    skid_distance = 7.0 * skid_time - 0.5 * 0.2 * 9.80665 * skid_time**2
    roll_distance = 5.0**2 / (2.0 * 0.05 * 9.80665)

    assert result.termination.reason is SkidRollTerminationReason.REST
    assert result.skid_distance_m == pytest.approx(skid_distance, rel=2e-8)
    assert result.roll_distance_m == pytest.approx(roll_distance, rel=2e-8)
    assert result.trajectory[-1].phase is GroundPhase.REST
    assert result.energy.dissipation_j >= 0.0
    assert result.events[0].sequence == len(prefix.events)


def test_shared_golden_fixture_pins_flat_skid_and_roll_solution() -> None:
    fixture = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    inputs = fixture["input"]
    expected = fixture["expected"]
    surface = replace(
        _surface(),
        kinetic_friction=inputs["kinetic_friction"],
        rolling_resistance=inputs["rolling_resistance"],
    )
    request = _surface_run_request(surface=surface)
    prefix = _settled_prefix(
        request, velocity_m_s=(inputs["initial_speed_m_s"], 0.0, 0.0)
    )
    result = simulate_skid_roll(request, prefix)

    assert fixture["schema_version"] == "ground-skid-roll-golden/v1"
    assert hashlib.sha256(_GOLDEN.read_bytes()).hexdigest() == (
        "74e23ebe86c8b476a3414b0ff11e561e126810b5358337cb87bc1e35e3a1d73d"  # pragma: allowlist secret  # noqa: E501
    )
    assert result.termination.reason.value == expected["termination"]
    assert result.events[0].time_s - prefix.termination.time_s == pytest.approx(
        expected["skid_duration_s"], rel=2e-8
    )
    assert result.events[0].velocity_after_m_s[0] == pytest.approx(
        expected["skid_to_roll_speed_m_s"], rel=2e-8
    )
    assert result.events[1].time_s - result.events[0].time_s == pytest.approx(
        expected["roll_duration_s"], rel=2e-8
    )
    assert result.skid_distance_m == pytest.approx(
        expected["skid_distance_m"], rel=2e-8
    )
    assert result.roll_distance_m == pytest.approx(
        expected["roll_distance_m"], rel=2e-8
    )
    assert result.skid_distance_m + result.roll_distance_m == pytest.approx(
        expected["surface_path_distance_m"], rel=2e-8
    )


def test_zero_rolling_resistance_and_retained_axial_spin_end_at_time_limit() -> None:
    surface = replace(_surface(), rolling_resistance=0.0)
    request = _surface_run_request(surface=surface, max_time_s=0.4)
    rolling_spin = (0.0, 17.0, -2.0 / request.ball_radius_m)
    prefix = _settled_prefix(
        request,
        velocity_m_s=(2.0, 0.0, 0.0),
        angular_velocity_rad_s=rolling_spin,
    )
    result = simulate_skid_roll(request, prefix)

    assert result.termination.reason is SkidRollTerminationReason.TIME_LIMIT
    assert result.termination.elapsed_time_s == pytest.approx(request.max_time_s)
    assert result.final_state.angular_velocity_rad_s[1] == pytest.approx(17.0)
    assert all(point.phase is not GroundPhase.REST for point in result.trajectory)


def test_cancellation_and_step_limit_remain_typed_internal_outcomes() -> None:
    request = _surface_run_request(max_time_s=1.0)
    prefix = _settled_prefix(request)
    cancelled = simulate_skid_roll(
        request,
        prefix,
        SkidRollExecution(is_cancelled=lambda: True),
    )
    bounded = simulate_skid_roll(
        request,
        prefix,
        SkidRollExecution(
            settings=SkidRollSettings(integration_step_s=0.001, max_steps=1)
        ),
    )

    assert cancelled.termination.reason is SkidRollTerminationReason.CANCELLED
    assert bounded.termination.reason is SkidRollTerminationReason.STEP_LIMIT


def test_prefix_events_count_against_event_limit_at_exact_transition() -> None:
    surface = replace(_surface(), kinetic_friction=0.2, rolling_resistance=0.05)
    request = _surface_run_request(surface=surface, max_events=2)
    prefix = _settled_prefix(request)
    result = simulate_skid_roll(request, prefix)
    skid_time = 7.0 / (0.2 * 9.80665 * 3.5)

    assert result.termination.reason is SkidRollTerminationReason.EVENT_LIMIT
    assert result.events == ()
    assert result.termination.elapsed_time_s == pytest.approx(0.02 + skid_time)
    assert result.final_state.velocity_m_s[0] == pytest.approx(5.0, abs=1e-10)


def test_inclined_pure_roll_uses_arbitrary_plane_gravity_component() -> None:
    angle = math.radians(8.0)
    normal = (0.0, math.cos(angle), -math.sin(angle))
    downhill = (0.0, -math.sin(angle), -math.cos(angle))
    surface = replace(
        _surface(),
        normal_unit=normal,
        rolling_resistance=0.02,
    )
    request = _surface_run_request(surface=surface, max_time_s=0.3)
    speed = 2.0
    velocity = tuple(speed * value for value in downhill)
    prefix = _settled_prefix(
        request,
        velocity_m_s=velocity,
        angular_velocity_rad_s=(-speed / request.ball_radius_m, 0.0, 0.0),
    )
    result = simulate_skid_roll(request, prefix)
    handoff = prefix.handoff_state
    if handoff is None:
        raise RuntimeError("test prefix must expose a handoff")
    duration = request.max_time_s - prefix.termination.elapsed_time_s
    acceleration = 9.80665 * math.sin(
        angle
    ) / 1.4 - surface.rolling_resistance * 9.80665 * math.cos(angle)
    expected_path = speed * duration + 0.5 * acceleration * duration**2
    displacement = tuple(
        result.final_state.position_m[index] - handoff.position_m[index]
        for index in range(3)
    )
    actual_path = sum(
        value * direction
        for value, direction in zip(displacement, downhill, strict=True)
    )

    assert result.termination.reason is SkidRollTerminationReason.TIME_LIMIT
    assert result.termination.elapsed_time_s == pytest.approx(request.max_time_s)
    assert actual_path == pytest.approx(expected_path, rel=2e-9)


def test_integration_refinement_converges_on_the_same_static_plane() -> None:
    surface = replace(_surface(), rolling_resistance=0.05)
    request = _surface_run_request(surface=surface)
    prefix = _settled_prefix(request)
    coarse = simulate_skid_roll(
        request,
        prefix,
        SkidRollExecution(settings=SkidRollSettings(integration_step_s=0.002)),
    )
    fine = simulate_skid_roll(
        request,
        prefix,
        SkidRollExecution(settings=SkidRollSettings(integration_step_s=0.0005)),
    )

    assert fine.termination.reason is SkidRollTerminationReason.REST
    assert coarse.skid_distance_m == pytest.approx(fine.skid_distance_m, rel=2e-7)
    assert coarse.roll_distance_m == pytest.approx(fine.roll_distance_m, rel=2e-7)


def test_oblique_incline_skid_transition_converges_without_slip_overshoot() -> None:
    angle = math.radians(8.0)
    normal = (0.0, math.cos(angle), -math.sin(angle))
    downhill = (0.0, -math.sin(angle), -math.cos(angle))
    surface = replace(
        _surface(),
        normal_unit=normal,
        static_friction=0.3,
        kinetic_friction=0.2,
        rolling_resistance=0.05,
    )
    request = _surface_run_request(surface=surface, max_time_s=0.8)
    prefix = _settled_prefix(
        request,
        velocity_m_s=(1.0, 0.2 * downhill[1], 0.2 * downhill[2]),
        angular_velocity_rad_s=(0.0, 0.0, 0.0),
    )
    results = tuple(
        simulate_skid_roll(
            request,
            prefix,
            SkidRollExecution(
                settings=SkidRollSettings(
                    integration_step_s=step_s,
                    max_steps=500_000,
                )
            ),
        )
        for step_s in (0.003, 0.002, 0.00005)
    )
    transitions = tuple(
        next(
            event for event in result.events if event.event_type.value == "skid_to_roll"
        )
        for result in results
    )

    assert transitions[0].time_s == pytest.approx(transitions[-1].time_s, abs=2e-3)
    assert transitions[1].time_s == pytest.approx(transitions[-1].time_s, abs=2e-3)
    assert results[0].skid_distance_m == pytest.approx(
        results[-1].skid_distance_m, abs=2e-3
    )


def test_surface_run_rejects_prefix_from_materially_different_request() -> None:
    original = _surface_run_request()
    prefix = _settled_prefix(original)
    changed_requests = (
        replace(
            original,
            surface=replace(original.surface, normal_restitution=0.81),
        ),
        replace(original, ball_mass_kg=original.ball_mass_kg + 0.001),
        replace(original, max_time_s=original.max_time_s + 0.1),
    )

    for changed in changed_requests:
        with pytest.raises(ValueError, match="request fingerprint"):
            simulate_skid_roll(changed, prefix)


def test_zero_speed_on_unheld_slope_accelerates_downhill_without_zero_step() -> None:
    angle = math.radians(8.0)
    normal = (0.0, math.cos(angle), -math.sin(angle))
    downhill = (0.0, -math.sin(angle), -math.cos(angle))
    surface = replace(
        _surface(),
        normal_unit=normal,
        rolling_resistance=0.0,
    )
    request = _surface_run_request(surface=surface, max_time_s=0.05)
    prefix = _settled_prefix(
        request,
        velocity_m_s=(0.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, 0.0),
    )

    result = simulate_skid_roll(request, prefix)

    assert result.termination.reason is SkidRollTerminationReason.TIME_LIMIT
    assert (
        sum(
            value * direction
            for value, direction in zip(
                result.final_state.velocity_m_s, downhill, strict=True
            )
        )
        > 0.0
    )


def test_edge_start_with_outward_slope_acceleration_leaves_at_handoff() -> None:
    angle = math.radians(8.0)
    normal = (0.0, math.cos(angle), -math.sin(angle))
    downhill = (0.0, -math.sin(angle), -math.cos(angle))
    surface = replace(
        _surface(),
        normal_unit=normal,
        rolling_resistance=0.0,
    )
    request = _surface_run_request(surface=surface, max_time_s=0.05)
    prefix = _settled_prefix(
        request,
        velocity_m_s=(0.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, 0.0),
    )
    handoff = prefix.handoff_state
    if handoff is None:
        raise RuntimeError("test prefix must expose a handoff")
    coordinate = sum(
        (handoff.position_m[index]) * downhill[index] for index in range(3)
    )
    resolver = SurfaceResolver(
        PlanarSurfaceDomain(
            surface,
            axis_unit=downhill,
            upper_coordinate_m=coordinate,
        )
    )

    result = simulate_skid_roll(
        request,
        prefix,
        SkidRollExecution(resolver=resolver),
    )

    assert result.termination.reason is SkidRollTerminationReason.LEFT_SURFACE
    assert result.termination.time_s == handoff.time_s


def test_finite_region_emits_exact_left_surface_event() -> None:
    surface = replace(_surface(), rolling_resistance=0.0)
    request = _surface_run_request(surface=surface, max_time_s=1.0)
    prefix = _settled_prefix(
        request,
        velocity_m_s=(2.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, -2.0 / request.ball_radius_m),
    )
    handoff = prefix.handoff_state
    if handoff is None:
        raise RuntimeError("test prefix must expose a handoff")
    start_x = handoff.position_m[0]
    resolver = SurfaceResolver(
        PlanarSurfaceDomain(
            surface,
            lower_coordinate_m=start_x - 1.0,
            upper_coordinate_m=start_x + 0.5,
        )
    )
    result = simulate_skid_roll(request, prefix, SkidRollExecution(resolver=resolver))

    assert result.termination.reason is SkidRollTerminationReason.LEFT_SURFACE
    assert result.final_state.position_m[0] == pytest.approx(start_x + 0.5, abs=1e-10)
    assert result.events[-1].event_type.value == "left_surface"


def test_moving_surface_relative_solution_and_energy_accounting_match_fixed_case() -> (
    None
):
    fixed = replace(_surface(), rolling_resistance=0.05)
    moving = replace(fixed, surface_velocity_m_s=(1.0, 0.0, 0.0))
    fixed_request = _surface_run_request(surface=fixed, max_time_s=0.2)
    moving_request = _surface_run_request(surface=moving, max_time_s=0.2)
    fixed_prefix = _settled_prefix(fixed_request, velocity_m_s=(3.0, 0.0, 0.0))
    moving_prefix = _settled_prefix(moving_request, velocity_m_s=(4.0, 0.0, 0.0))

    fixed_result = simulate_skid_roll(fixed_request, fixed_prefix)
    moving_result = simulate_skid_roll(moving_request, moving_prefix)
    moving_handoff = moving_prefix.handoff_state
    fixed_handoff = fixed_prefix.handoff_state
    if moving_handoff is None or fixed_handoff is None:
        raise RuntimeError("test prefixes must expose handoffs")
    elapsed = moving_result.final_state.time_s - moving_handoff.time_s
    moving_relative_x = (
        moving_result.final_state.position_m[0] - moving_handoff.position_m[0] - elapsed
    )
    fixed_relative_x = (
        fixed_result.final_state.position_m[0] - fixed_handoff.position_m[0]
    )

    assert moving_relative_x == pytest.approx(fixed_relative_x, rel=1e-9)
    assert moving_result.energy.dissipation_j == pytest.approx(
        fixed_result.energy.dissipation_j, rel=1e-8
    )
    assert moving_result.energy.surface_work_j != 0.0


@pytest.mark.parametrize(
    ("velocity_m_s", "angular_velocity_rad_s"),
    (
        ((7.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((3.0, 0.0, 4.0), (12.0, 5.0, -7.0)),
        ((0.0, 0.0, 3.0), (0.0, 0.0, 0.0)),
    ),
)
def test_stationary_flat_surface_is_passive_for_general_tangential_states(
    velocity_m_s: tuple[float, float, float],
    angular_velocity_rad_s: tuple[float, float, float],
) -> None:
    surface = replace(_surface(), rolling_resistance=0.03)
    request = _surface_run_request(surface=surface, max_time_s=0.25)
    prefix = _settled_prefix(
        request,
        velocity_m_s=velocity_m_s,
        angular_velocity_rad_s=angular_velocity_rad_s,
    )
    result = simulate_skid_roll(request, prefix)
    energy = result.energy

    assert energy.surface_work_j == pytest.approx(0.0, abs=1e-14)
    assert energy.gravity_work_j == pytest.approx(0.0, abs=1e-14)
    assert energy.kinetic_after_j <= energy.kinetic_before_j + 1e-12
    assert energy.dissipation_j == pytest.approx(
        energy.kinetic_before_j - energy.kinetic_after_j,
        abs=1e-10,
    )
