"""Deterministic repeated-hop and bounded-failure tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from shared.python.swing_sim.ground import (
    BounceModelSettings,
    BounceTerminationReason,
    GroundEventType,
    GroundPhase,
    interpolate_first_contact,
    simulate_repeated_bounce,
)

from ._support import _request

_GOLDEN = (
    __import__("pathlib").Path(__file__).parents[5]
    / "rate_of_closure/web/src/model/__fixtures__/ground_impact_bounce_golden_v1.json"
)


def test_contact_bracket_interpolation_hits_exact_sphere_plane() -> None:
    request = _request()
    state = interpolate_first_contact(request)

    assert (
        request.last_separated_state.time_s
        < state.time_s
        <= request.first_penetrating_state.time_s
    )
    assert request.surface.signed_gap_m(state, request.ball_radius_m) == pytest.approx(
        0.0, abs=1e-12
    )
    assert state.velocity_m_s[0] == pytest.approx(31.0)


def test_horizontal_repeated_hops_match_analytic_event_times_and_speeds() -> None:
    request = replace(
        _request(),
        surface=replace(
            _request().surface,
            normal_restitution=0.5,
            static_friction=0.0,
            kinetic_friction=0.0,
        ),
        max_events=4,
        output_interval_s=0.05,
    )
    result = simulate_repeated_bounce(
        request, BounceModelSettings(capture_speed_m_s=0.01)
    )

    assert [event.event_type for event in result.events] == [
        GroundEventType.FIRST_CONTACT,
        GroundEventType.BOUNCE,
        GroundEventType.BOUNCE,
        GroundEventType.BOUNCE,
    ]
    first_outgoing = result.events[0].velocity_after_m_s[1]
    expected_first_hop = 2.0 * first_outgoing / 9.80665
    assert result.events[1].time_s - result.events[0].time_s == pytest.approx(
        expected_first_hop, rel=1e-10
    )
    assert result.events[1].velocity_after_m_s[1] == pytest.approx(first_outgoing * 0.5)
    assert result.termination.reason is BounceTerminationReason.EVENT_LIMIT


def test_zero_restitution_hands_off_without_zero_time_retrigger() -> None:
    request = replace(
        _request(), surface=replace(_request().surface, normal_restitution=0.0)
    )
    result = simulate_repeated_bounce(request)

    assert result.termination.reason is BounceTerminationReason.SETTLED_TO_SKID
    assert result.handoff_state is not None
    assert len(result.events) == 1
    assert result.handoff_state.velocity_m_s[1] == pytest.approx(0.0, abs=1e-12)
    assert result.trajectory[-1].phase is GroundPhase.SKID
    assert result.trajectory[-1].time_s == result.events[-1].time_s


def test_capture_band_uses_effective_zero_restitution_on_final_micro_impact() -> None:
    request = replace(
        _request(),
        surface=replace(_request().surface, normal_restitution=0.4),
        last_separated_state=replace(
            _request().last_separated_state, velocity_m_s=(5.0, -0.05, 0.0)
        ),
        first_penetrating_state=replace(
            _request().first_penetrating_state, velocity_m_s=(5.0, -0.05, 0.0)
        ),
    )
    result = simulate_repeated_bounce(
        request, BounceModelSettings(capture_speed_m_s=0.1)
    )

    assert result.termination.reason is BounceTerminationReason.SETTLED_TO_SKID
    assert result.events[0].velocity_after_m_s[1] == pytest.approx(0.0, abs=1e-12)
    assert len(result.events) == 1
    assert result.trajectory[-1].phase is GroundPhase.SKID


def test_samples_are_strictly_ordered_and_event_points_replace_grid_collisions() -> (
    None
):
    result = simulate_repeated_bounce(
        replace(_request(), output_interval_s=0.02, max_events=5)
    )
    times = [point.time_s for point in result.trajectory]

    assert all(right > left for left, right in zip(times, times[1:], strict=False))
    assert len(times) == len(set(times))
    for event in result.events:
        matching = [
            point
            for point in result.trajectory
            if abs(point.time_s - event.time_s) <= 1e-12
        ]
        assert len(matching) == 1


def test_output_grid_refinement_does_not_change_event_ledger() -> None:
    coarse = simulate_repeated_bounce(
        replace(_request(), output_interval_s=0.1, max_events=5)
    )
    fine = simulate_repeated_bounce(
        replace(_request(), output_interval_s=0.01, max_events=5)
    )

    assert [
        (event.time_s, event.velocity_after_m_s) for event in coarse.events
    ] == pytest.approx(
        [(event.time_s, event.velocity_after_m_s) for event in fine.events]
    )
    assert coarse.termination == fine.termination
    assert coarse.airborne_segments == fine.airborne_segments
    assert coarse.bounce_air_distance_m == pytest.approx(
        sum(segment.horizontal_distance_m for segment in coarse.airborne_segments)
    )


def test_time_limit_returns_only_validated_prefix() -> None:
    result = simulate_repeated_bounce(
        replace(_request(), max_time_s=0.05, output_interval_s=0.01)
    )

    assert result.termination.reason is BounceTerminationReason.TIME_LIMIT
    assert result.handoff_state is None
    assert result.trajectory[-1].phase is GroundPhase.BOUNCE
    assert result.trajectory[-1].time_s == pytest.approx(result.termination.time_s)


def test_preflight_and_midrun_cancellation_are_deterministic() -> None:
    preflight = simulate_repeated_bounce(_request(), is_cancelled=lambda: True)
    calls = 0

    def cancel_after_first_event() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    midrun = simulate_repeated_bounce(_request(), is_cancelled=cancel_after_first_event)

    assert preflight.termination.reason is BounceTerminationReason.CANCELLED
    assert preflight.trajectory == ()
    assert midrun.termination.reason is BounceTerminationReason.CANCELLED
    assert len(midrun.events) == 1


def test_noncanonical_gravity_is_rejected_before_simulation() -> None:
    with pytest.raises(ValueError, match="versioned standard gravity"):
        BounceModelSettings(gravity_m_s2=(0.1, -9.80665, 0.0))


def test_no_recontact_returns_a_typed_validated_prefix(monkeypatch) -> None:
    import shared.python.swing_sim.ground.bounce_simulation as simulation

    monkeypatch.setattr(simulation, "contact_state_after_hop", lambda *args: None)
    result = simulation.simulate_repeated_bounce(_request())

    assert result.termination.reason is BounceTerminationReason.NO_RECONTACT
    assert result.handoff_state is None


def test_airborne_distance_cannot_silently_become_horizontal_chord() -> None:
    with pytest.raises(ValueError, match="versioned standard gravity"):
        BounceModelSettings(gravity_m_s2=(0.0, -9.80665, 0.1))


def test_result_rejects_event_and_segment_discontinuities() -> None:
    result = simulate_repeated_bounce(replace(_request(), max_events=3))
    reversed_events = (result.events[1], result.events[0], *result.events[2:])
    with pytest.raises(ValueError, match="sequence|times"):
        replace(result, events=reversed_events)
    original_start = result.airborne_segments[0].start_position_m
    bad_segment = replace(
        result.airborne_segments[0],
        start_position_m=(original_start[0], 999.0, original_start[2]),
    )
    with pytest.raises(ValueError, match="distance|start position"):
        replace(
            result,
            airborne_segments=(bad_segment, *result.airborne_segments[1:]),
        )


def test_result_discloses_material_fields_not_used_by_bounce_law() -> None:
    result = simulate_repeated_bounce(_request())

    joined = " ".join(result.warnings)
    assert "firmness_pa" in joined
    assert "grass_height_m" in joined
    assert "moisture_fraction" in joined
    assert "#4271" in joined


def test_public_prefix_is_not_the_final_ground_simulation_result() -> None:
    result = simulate_repeated_bounce(_request())

    assert type(result).__name__ == "RepeatedBounceResult"
    assert not hasattr(result, "summary")
    assert not hasattr(result, "total_distance_m")


def test_shared_golden_fixture_matches_repeated_hop_event_times() -> None:
    case = json.loads(_GOLDEN.read_text(encoding="utf-8"))["bounce_case"]
    request = _request()
    radius = request.ball_radius_m
    contact_time = 1.0
    bracket_dt = 0.001
    incoming_speed = case["incoming_normal_speed_m_s"]
    request = replace(
        request,
        surface=replace(
            request.surface,
            normal_restitution=case["normal_restitution"],
            static_friction=0.0,
            kinetic_friction=0.0,
        ),
        last_separated_state=replace(
            request.last_separated_state,
            time_s=contact_time - bracket_dt,
            position_m=(0.0, radius + incoming_speed * bracket_dt, 0.0),
            velocity_m_s=(0.0, -incoming_speed, 0.0),
            angular_velocity_rad_s=(0.0, 0.0, 0.0),
        ),
        first_penetrating_state=replace(
            request.first_penetrating_state,
            time_s=contact_time + bracket_dt,
            position_m=(0.0, radius - incoming_speed * bracket_dt, 0.0),
            velocity_m_s=(0.0, -incoming_speed, 0.0),
            angular_velocity_rad_s=(0.0, 0.0, 0.0),
        ),
        max_events=4,
    )
    result = simulate_repeated_bounce(
        request, BounceModelSettings(capture_speed_m_s=0.001)
    )
    elapsed = [event.time_s - result.events[0].time_s for event in result.events]

    assert elapsed == pytest.approx(case["expected_event_elapsed_s"])


def test_contact_interpolation_converges_under_bracket_refinement() -> None:
    request = _request()
    radius = request.ball_radius_m

    def bracket(delta: float):
        return replace(
            request,
            last_separated_state=replace(
                request.last_separated_state,
                time_s=1.0 - delta,
                position_m=(2.0 - 3.0 * delta, radius + 4.0 * delta, 0.0),
                velocity_m_s=(3.0, -4.0, 0.0),
            ),
            first_penetrating_state=replace(
                request.first_penetrating_state,
                time_s=1.0 + delta,
                position_m=(2.0 + 3.0 * delta, radius - 4.0 * delta, 0.0),
                velocity_m_s=(3.0, -4.0, 0.0),
            ),
        )

    coarse = interpolate_first_contact(bracket(0.01))
    fine = interpolate_first_contact(bracket(0.0001))

    assert coarse.time_s == pytest.approx(1.0)
    assert coarse.position_m == pytest.approx(fine.position_m, abs=1e-12)


def test_unexpected_solver_value_error_returns_numerical_failure_prefix(
    monkeypatch,
) -> None:
    import shared.python.swing_sim.ground.bounce_simulation as simulation

    def fail(*args, **kwargs):
        raise ValueError("synthetic numerical failure")

    monkeypatch.setattr(simulation, "resolve_sphere_plane_impact", fail)
    result = simulation.simulate_repeated_bounce(_request())

    assert result.termination.reason is BounceTerminationReason.NUMERICAL_FAILURE
    assert result.events == ()
    assert result.trajectory == ()
