"""Focused regressions for canonical-state passivity accounting."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from shared.python.swing_sim.ground import (
    PlanarSurfaceDomain,
    SkidRollExecution,
    SkidRollSettings,
    SkidRollTerminationReason,
    SurfaceResolver,
    simulate_skid_roll,
)
from shared.python.swing_sim.ground.impact_types import SphereProperties
from shared.python.swing_sim.ground.skid_roll_runtime import (
    SurfaceRun,
    _canonical_snap_energy_budget,
)
from shared.python.swing_sim.ground.surface_motion_types import RigidMotion

from ._support import _settled_prefix, _surface, _surface_run_request


def _run() -> SurfaceRun:
    request = _surface_run_request(max_time_s=0.1)
    prefix = _settled_prefix(
        request,
        velocity_m_s=(1.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, -1.0 / request.ball_radius_m),
        immediate=True,
    )
    handoff = prefix.handoff_state
    if handoff is None:
        raise RuntimeError("test prefix must expose a handoff")
    return SurfaceRun(
        request,
        prefix,
        SurfaceResolver(PlanarSurfaceDomain(request.surface)),
        SkidRollSettings(),
        lambda: False,
        SphereProperties(
            request.ball_radius_m,
            request.ball_mass_kg,
            request.rotational_inertia_factor,
        ),
        handoff,
        # `active_surface`, matching `simulate_skid_roll`'s own construction.
        # It arrived with regional-surface support and this helper was never
        # updated, so every passivity test raised TypeError before asserting
        # anything about passivity.
        request.surface,
        next_grid_time_s=handoff.time_s + request.output_interval_s,
    )


def _unforced_acceleration() -> RigidMotion:
    return RigidMotion(
        acceleration_m_s2=(10.0, 0.0, 0.0),
        angular_acceleration_rad_s2=(0.0, 0.0, 0.0),
        contact_slip_acceleration_m_s2=(10.0, 0.0, 0.0),
        contact_force_n=(0.0, 0.0, 0.0),
    )


def test_each_segment_rejects_unforced_energy_creation_without_masking() -> None:
    """Prior legitimate loss cannot mask a later energy-creating segment."""
    run = _run()
    run.physical_dissipation_j = 0.01

    with pytest.raises(ValueError, match="passive energy accounting"):
        run.advance(_unforced_acceleration(), 0.01)


def test_unbudgeted_canonical_endpoint_energy_creation_is_rejected() -> None:
    """Only explicitly bounded state quantization may be excluded from loss."""
    run = _run()
    run.state = replace(
        run.state,
        velocity_m_s=(10.0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="passive energy accounting"):
        run.result(SkidRollTerminationReason.TIME_LIMIT)


def test_rolling_projection_rejects_an_off_manifold_state() -> None:
    run = _run()
    run.state = replace(run.state, angular_velocity_rad_s=(0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="passive energy accounting"):
        run.rolling_projection()


def test_canonical_snap_rejects_a_component_outside_the_wire_quantum() -> None:
    run = _run()
    raw = (
        run.state.position_m,
        run.state.velocity_m_s,
        run.state.angular_velocity_rad_s,
    )
    forged = replace(
        run.state,
        position_m=(run.state.position_m[0] + 1e-6, *run.state.position_m[1:]),
    )

    with pytest.raises(ValueError, match="canonical quantization bound"):
        _canonical_snap_energy_budget(
            raw,
            forged,
            run.body,
            run.settings.gravity_m_s2,
        )


def test_moving_surface_holds_zero_relative_speed_on_resistance_dominated_slope() -> (
    None
):
    """A carried ball must not overshoot through the rolling-resistance cusp."""
    angle = 0.05
    surface = replace(
        _surface(),
        normal_unit=(0.0, math.cos(angle), math.sin(angle)),
        surface_velocity_m_s=(0.04, 0.0, 0.0),
        rolling_resistance=0.04,
    )
    request = _surface_run_request(surface=surface, max_time_s=0.1)
    prefix = _settled_prefix(
        request,
        velocity_m_s=surface.surface_velocity_m_s,
        angular_velocity_rad_s=(0.0, 0.0, 0.0),
        immediate=True,
    )

    result = simulate_skid_roll(request, prefix)

    assert result.termination.reason is SkidRollTerminationReason.TIME_LIMIT
    assert result.final_state.velocity_m_s == surface.surface_velocity_m_s
    assert result.final_state.angular_velocity_rad_s == (0.0, 0.0, 0.0)


def test_moving_surface_snaps_sub_tolerance_uphill_residual_before_holding() -> None:
    """The holding branch must not preserve a tiny energy-creating velocity."""
    angle = 0.05
    residual_m_s = 9e-10
    surface = replace(
        _surface(),
        normal_unit=(0.0, math.cos(angle), math.sin(angle)),
        surface_velocity_m_s=(0.04, 0.0, 0.0),
        rolling_resistance=0.04,
    )
    uphill = (0.0, math.sin(angle), -math.cos(angle))
    request = _surface_run_request(surface=surface, max_time_s=0.1)
    prefix = _settled_prefix(
        request,
        velocity_m_s=tuple(
            surface.surface_velocity_m_s[index] + residual_m_s * uphill[index]
            for index in range(3)
        ),
        angular_velocity_rad_s=(-residual_m_s / request.ball_radius_m, 0.0, 0.0),
        immediate=True,
    )

    result = simulate_skid_roll(request, prefix)

    assert result.termination.reason is SkidRollTerminationReason.TIME_LIMIT
    assert result.final_state.velocity_m_s == surface.surface_velocity_m_s
    assert result.final_state.angular_velocity_rad_s == (0.0, 0.0, 0.0)


def test_holding_projection_uses_the_independent_velocity_tolerance() -> None:
    """A valid hold may be wider than the independent contact-slip tolerance."""
    angle = 0.05
    residual_m_s = 5e-7
    surface = replace(
        _surface(),
        normal_unit=(0.0, math.cos(angle), math.sin(angle)),
        surface_velocity_m_s=(0.04, 0.0, 0.0),
        rolling_resistance=0.04,
    )
    uphill = (0.0, math.sin(angle), -math.cos(angle))
    request = _surface_run_request(surface=surface, max_time_s=0.1)
    prefix = _settled_prefix(
        request,
        velocity_m_s=tuple(
            surface.surface_velocity_m_s[index] + residual_m_s * uphill[index]
            for index in range(3)
        ),
        angular_velocity_rad_s=(-residual_m_s / request.ball_radius_m, 0.0, 0.0),
        immediate=True,
    )
    execution = SkidRollExecution(
        settings=SkidRollSettings(
            velocity_tolerance_m_s=1e-6,
            slip_tolerance_m_s=1e-9,
        )
    )

    result = simulate_skid_roll(request, prefix, execution)

    assert result.termination.reason is SkidRollTerminationReason.TIME_LIMIT
    assert result.final_state.velocity_m_s == surface.surface_velocity_m_s
    assert result.final_state.angular_velocity_rad_s == (0.0, 0.0, 0.0)


def test_stationary_holding_projection_reports_rest_before_the_time_horizon() -> None:
    """An exact stopped projection must emit REST before advancing a hold step."""
    speed_m_s = 9e-10
    request = _surface_run_request(max_time_s=0.01)
    prefix = _settled_prefix(
        request,
        velocity_m_s=(speed_m_s, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, -speed_m_s / request.ball_radius_m),
        immediate=True,
    )

    result = simulate_skid_roll(request, prefix)

    assert result.termination.reason is SkidRollTerminationReason.REST
    assert result.final_state.velocity_m_s == (0.0, 0.0, 0.0)
    assert result.final_state.angular_velocity_rad_s == (0.0, 0.0, 0.0)
