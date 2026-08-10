"""Analytic and invariant tests for skid and pure-roll dynamics."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from shared.python.swing_sim.ground import GroundContactState, SphereProperties
from shared.python.swing_sim.ground.skid_roll_dynamics import (
    advance_constant_motion,
    contact_slip_velocity,
    rolling_kinematics,
    rolling_state,
    skid_kinematics,
    static_rolling_feasible,
    time_to_vector_zero,
)

from ._support import _surface


def _body() -> SphereProperties:
    return SphereProperties(0.02135, 0.04593, 0.4)


def _state(
    velocity_m_s: tuple[float, float, float],
    angular_velocity_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> GroundContactState:
    return GroundContactState(
        1.0,
        _surface().frame,
        (0.0, _body().radius_m, 0.0),
        velocity_m_s,
        angular_velocity_rad_s,
    )


def test_flat_zero_spin_skid_reaches_classic_five_sevenths_speed_exactly() -> None:
    surface = replace(_surface(), kinetic_friction=0.2)
    state = _state((7.0, 0.0, 0.0))
    motion = skid_kinematics(state, surface, _body())
    transition_s = time_to_vector_zero(
        contact_slip_velocity(state, surface, _body()),
        motion.contact_slip_acceleration_m_s2,
        tolerance=1e-12,
    )

    assert transition_s is not None
    transitioned = advance_constant_motion(state, motion, transition_s)
    assert transitioned.velocity_m_s[0] == pytest.approx(5.0, abs=1e-12)
    assert contact_slip_velocity(transitioned, surface, _body()) == pytest.approx(
        (0.0, 0.0, 0.0), abs=1e-11
    )


def test_static_rolling_feasibility_accepts_exact_coulomb_boundary() -> None:
    normal = (0.0, math.cos(math.radians(10.0)), -math.sin(math.radians(10.0)))
    surface = replace(_surface(), normal_unit=normal)
    gravity_tangent = 9.80665 * math.sin(math.radians(10.0))
    normal_accel = 9.80665 * math.cos(math.radians(10.0))
    exact_mu = 0.4 / 1.4 * gravity_tangent / normal_accel
    boundary = replace(surface, static_friction=exact_mu, kinetic_friction=exact_mu)

    assert static_rolling_feasible(boundary, _body())
    assert not static_rolling_feasible(
        replace(
            boundary, static_friction=exact_mu * (1.0 - 1e-8), kinetic_friction=0.0
        ),
        _body(),
    )


def test_rolling_kinematics_matches_stimp_stop_time_distance_and_slope_drive() -> None:
    body = _body()
    flat = replace(_surface(), rolling_resistance=0.05)
    initial = rolling_state(_state((2.0, 0.0, 0.0)), flat, body)
    motion = rolling_kinematics(initial, flat, body)
    stop_s = time_to_vector_zero(
        (2.0, 0.0, 0.0), motion.acceleration_m_s2, tolerance=1e-12
    )

    assert stop_s == pytest.approx(2.0 / (0.05 * 9.80665), rel=1e-12)
    stopped = advance_constant_motion(initial, motion, stop_s)
    assert stopped.position_m[0] == pytest.approx(
        2.0**2 / (2.0 * 0.05 * 9.80665), rel=1e-12
    )

    angle = math.radians(8.0)
    normal = (0.0, math.cos(angle), -math.sin(angle))
    slope = replace(flat, normal_unit=normal, rolling_resistance=0.0)
    downhill = (0.0, -math.sin(angle), -math.cos(angle))
    rolling = rolling_state(
        _state(tuple(2.0 * value for value in downhill)), slope, body
    )
    slope_motion = rolling_kinematics(rolling, slope, body)
    expected = 9.80665 * math.sin(angle) / 1.4
    assert sum(
        a * b for a, b in zip(slope_motion.acceleration_m_s2, downhill, strict=True)
    ) == pytest.approx(expected, rel=2e-12)


def test_normal_axis_spin_is_preserved_by_skid_and_roll_motion() -> None:
    body = _body()
    surface = _surface()
    state = _state((4.0, 0.0, 0.0), (0.0, 23.0, 0.0))
    skid = skid_kinematics(state, surface, body)
    after_skid = advance_constant_motion(state, skid, 0.05)
    rolling = rolling_state(after_skid, surface, body)
    roll = rolling_kinematics(rolling, surface, body)
    after_roll = advance_constant_motion(rolling, roll, 0.05)

    assert after_skid.angular_velocity_rad_s[1] == pytest.approx(23.0, abs=1e-12)
    assert after_roll.angular_velocity_rad_s[1] == pytest.approx(23.0, abs=1e-12)
