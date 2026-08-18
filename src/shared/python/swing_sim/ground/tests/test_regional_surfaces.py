"""Deterministic contracts for coplanar regional material transitions."""

from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from shared.python.swing_sim.ground import (
    GroundEventType,
    GroundPhase,
    GroundSurfaceProfile,
    PlanarSurfaceDomain,
    PlanarSurfaceRegion,
    SkidRollExecution,
    SkidRollSettings,
    SkidRollTerminationReason,
    SurfaceKinematicSegment,
    SurfaceResolver,
    compose_ground_result,
    simulate_skid_roll,
)

from ._support import _settled_prefix, _surface, _surface_run_request


def _regional_resolver(
    *,
    boundary_m: float,
    rolling_resistance: float,
) -> tuple[SurfaceResolver, GroundSurfaceProfile]:
    base = replace(_surface(), rolling_resistance=0.0)
    region_surface = replace(
        base,
        surface_id="slow-apron",
        rolling_resistance=rolling_resistance,
    )
    resolver = SurfaceResolver(
        PlanarSurfaceDomain(base),
        regions=(
            PlanarSurfaceRegion(
                "slow-apron",
                PlanarSurfaceDomain(
                    region_surface,
                    lower_coordinate_m=boundary_m,
                    upper_coordinate_m=boundary_m + 10.0,
                ),
                precedence=10,
            ),
        ),
    )
    return resolver, region_surface


def test_overlapping_region_precedence_and_exact_boundary_are_deterministic() -> None:
    base = replace(_surface(), rolling_resistance=0.0)
    first = replace(base, surface_id="first", rolling_resistance=0.02)
    second = replace(base, surface_id="second", rolling_resistance=0.08)
    resolver = SurfaceResolver(
        PlanarSurfaceDomain(base),
        regions=(
            PlanarSurfaceRegion(
                "first",
                PlanarSurfaceDomain(
                    first,
                    lower_coordinate_m=1.0,
                    upper_coordinate_m=3.0,
                ),
                precedence=1,
            ),
            PlanarSurfaceRegion(
                "second",
                PlanarSurfaceDomain(
                    second,
                    lower_coordinate_m=2.0,
                    upper_coordinate_m=4.0,
                ),
                precedence=2,
            ),
        ),
    )
    segment = SurfaceKinematicSegment(
        start_position_m=(1.5, 0.02135, 0.0),
        start_velocity_m_s=(1.0, 0.0, 0.0),
        acceleration_m_s2=(0.0, 0.0, 0.0),
        duration_s=1.0,
    )

    assert resolver.region_at(segment.start_position_m).region_id == "first"
    crossing = resolver.first_transition(segment, active_region_id="first")

    assert crossing is not None
    assert crossing.time_offset_s == pytest.approx(0.5, abs=1e-12)
    assert crossing.position_m[0] == pytest.approx(2.0, abs=1e-12)
    assert crossing.from_region_id == "first"
    assert crossing.to_region_id == "second"


def test_region_profiles_must_be_coplanar_and_kinematically_continuous() -> None:
    base = _surface()
    changed_normal = replace(base, surface_id="tilted", normal_unit=(0.0, 0.8, 0.6))
    changed_velocity = replace(
        base,
        surface_id="moving",
        surface_velocity_m_s=(1.0, 0.0, 0.0),
    )

    for changed in (changed_normal, changed_velocity):
        with pytest.raises(ValueError, match="coplanar.*surface velocity"):
            SurfaceResolver(
                PlanarSurfaceDomain(base),
                regions=(
                    PlanarSurfaceRegion(
                        "changed",
                        PlanarSurfaceDomain(
                            changed,
                            lower_coordinate_m=1.0,
                            upper_coordinate_m=2.0,
                        ),
                        precedence=1,
                    ),
                ),
            )


def test_material_transition_splits_motion_without_state_discontinuity() -> None:
    base = replace(_surface(), rolling_resistance=0.0)
    request = _surface_run_request(surface=base, max_time_s=0.4)
    prefix = _settled_prefix(
        request,
        velocity_m_s=(2.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, -2.0 / request.ball_radius_m),
    )
    handoff = prefix.handoff_state
    assert handoff is not None
    boundary = handoff.position_m[0] + 0.25
    resolver, _ = _regional_resolver(
        boundary_m=boundary,
        rolling_resistance=0.1,
    )

    result = simulate_skid_roll(
        request,
        prefix,
        SkidRollExecution(resolver=resolver),
    )
    transition = next(
        event
        for event in result.events
        if event.event_type is GroundEventType.SURFACE_TRANSITION
    )
    expected_transition_time = handoff.time_s + 0.25 / 2.0
    post_transition_duration = result.final_state.time_s - expected_transition_time
    expected_final_speed = 2.0 - 0.1 * 9.80665 * post_transition_duration

    assert result.termination.reason is SkidRollTerminationReason.TIME_LIMIT
    assert transition.time_s == pytest.approx(expected_transition_time, abs=1e-12)
    assert transition.position_m[0] == pytest.approx(boundary, abs=1e-12)
    assert transition.velocity_before_m_s == transition.velocity_after_m_s
    assert (
        transition.angular_velocity_before_rad_s
        == transition.angular_velocity_after_rad_s
    )
    assert len(result.surface_transitions) == 1
    assert result.surface_transitions[0].from_surface_id == base.surface_id
    assert result.surface_transitions[0].to_surface_id == "slow-apron"
    with pytest.raises(ValueError, match="transition ledger"):
        replace(result, surface_transitions=())
    assert result.final_state.velocity_m_s[0] == pytest.approx(
        expected_final_speed,
        rel=1e-9,
    )
    assert any(
        point.time_s == pytest.approx(transition.time_s)
        and point.phase is GroundPhase.ROLL
        for point in result.trajectory
    )

    composed = compose_ground_result(request, prefix, result)
    assert any(
        event.event_type is GroundEventType.SURFACE_TRANSITION
        for event in composed.events
    )
    assert any(warning.code == "REGIONAL_PLANAR_V1" for warning in composed.warnings)


def test_base_domain_exit_wins_when_region_ends_at_the_same_boundary() -> None:
    base = replace(_surface(), rolling_resistance=0.0)
    request = _surface_run_request(surface=base, max_time_s=1.0)
    prefix = _settled_prefix(
        request,
        velocity_m_s=(2.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, -2.0 / request.ball_radius_m),
    )
    handoff = prefix.handoff_state
    assert handoff is not None
    start = handoff.position_m[0]
    upper = start + 0.2
    region_surface = replace(base, surface_id="edge-region")
    resolver = SurfaceResolver(
        PlanarSurfaceDomain(
            base,
            lower_coordinate_m=start - 0.1,
            upper_coordinate_m=upper,
        ),
        regions=(
            PlanarSurfaceRegion(
                "edge-region",
                PlanarSurfaceDomain(
                    region_surface,
                    lower_coordinate_m=start + 0.1,
                    upper_coordinate_m=upper,
                ),
                precedence=1,
            ),
        ),
    )

    result = simulate_skid_roll(
        request,
        prefix,
        SkidRollExecution(resolver=resolver),
    )

    assert result.termination.reason is SkidRollTerminationReason.LEFT_SURFACE
    assert result.events[-1].event_type is GroundEventType.LEFT_SURFACE
    assert len(result.surface_transitions) == 1


def test_surface_transition_limit_is_typed_and_does_not_chatter() -> None:
    base = replace(_surface(), rolling_resistance=0.0)
    request = _surface_run_request(surface=base, max_time_s=1.0)
    prefix = _settled_prefix(
        request,
        velocity_m_s=(2.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, -2.0 / request.ball_radius_m),
    )
    handoff = prefix.handoff_state
    assert handoff is not None
    start = handoff.position_m[0]
    profiles = tuple(replace(base, surface_id=f"region-{index}") for index in range(2))
    resolver = SurfaceResolver(
        PlanarSurfaceDomain(base),
        regions=tuple(
            PlanarSurfaceRegion(
                f"region-{index}",
                PlanarSurfaceDomain(
                    profile,
                    lower_coordinate_m=start + 0.1 * (index + 1),
                    upper_coordinate_m=start + 0.1 * (index + 2),
                ),
                precedence=index + 1,
            )
            for index, profile in enumerate(profiles)
        ),
    )

    result = simulate_skid_roll(
        request,
        prefix,
        SkidRollExecution(
            resolver=resolver,
            settings=SkidRollSettings(max_surface_transitions=1),
        ),
    )

    assert (
        result.termination.reason is SkidRollTerminationReason.SURFACE_TRANSITION_LIMIT
    )
    assert (
        sum(
            event.event_type is GroundEventType.SURFACE_TRANSITION
            for event in result.events
        )
        == 1
    )


@hypothesis_settings(max_examples=24, deadline=None)
@given(
    boundary_offset_m=st.floats(min_value=0.05, max_value=0.4),
    rolling_resistance=st.floats(min_value=0.01, max_value=0.2),
)
def test_regional_rolling_speed_matches_piecewise_analytic_solution(
    boundary_offset_m: float,
    rolling_resistance: float,
) -> None:
    base = replace(_surface(), rolling_resistance=0.0)
    request = _surface_run_request(surface=base, max_time_s=0.5)
    initial_speed = 2.0
    prefix = _settled_prefix(
        request,
        velocity_m_s=(initial_speed, 0.0, 0.0),
        angular_velocity_rad_s=(
            0.0,
            0.0,
            -initial_speed / request.ball_radius_m,
        ),
    )
    handoff = prefix.handoff_state
    assert handoff is not None
    resolver, _ = _regional_resolver(
        boundary_m=handoff.position_m[0] + boundary_offset_m,
        rolling_resistance=rolling_resistance,
    )

    result = simulate_skid_roll(
        request,
        prefix,
        SkidRollExecution(resolver=resolver),
    )
    transition_time = handoff.time_s + boundary_offset_m / initial_speed
    duration_on_region = result.final_state.time_s - transition_time
    expected_speed = initial_speed - rolling_resistance * 9.80665 * duration_on_region

    assert result.termination.reason is SkidRollTerminationReason.TIME_LIMIT
    assert result.final_state.velocity_m_s[0] == pytest.approx(
        expected_speed,
        rel=2e-9,
        abs=1e-11,
    )
