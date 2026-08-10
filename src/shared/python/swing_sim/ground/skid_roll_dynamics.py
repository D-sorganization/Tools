"""Rigid-sphere skid and no-slip roll dynamics on one arbitrary plane."""

from __future__ import annotations

import math
from dataclasses import replace

from ._vector_math import add, cross, dot, norm, scale, subtract, unit
from .contract_types import GroundContactState, GroundSurfaceProfile, Vector3
from .impact_types import SphereProperties
from .surface_motion_types import STANDARD_GRAVITY_M_S2, RigidMotion

_ZERO: Vector3 = (0.0, 0.0, 0.0)


def tangent(vector: Vector3, normal: Vector3) -> Vector3:
    """Project a vector into a plane."""
    return subtract(vector, scale(normal, dot(vector, normal)))


def contact_slip_velocity(
    state: GroundContactState,
    surface: GroundSurfaceProfile,
    body: SphereProperties,
) -> Vector3:
    """Return tangential contact velocity relative to the moving surface."""
    arm = scale(surface.normal_unit, -body.radius_m)
    contact_velocity = add(state.velocity_m_s, cross(state.angular_velocity_rad_s, arm))
    return tangent(
        subtract(contact_velocity, surface.surface_velocity_m_s), surface.normal_unit
    )


def _normal_gravity(surface: GroundSurfaceProfile, gravity: Vector3) -> float:
    normal_acceleration = -dot(gravity, surface.normal_unit)
    if normal_acceleration <= 0.0:
        raise ValueError("surface must oppose the versioned gravity vector")
    return float(normal_acceleration)


def skid_kinematics(
    state: GroundContactState,
    surface: GroundSurfaceProfile,
    body: SphereProperties,
    gravity_m_s2: Vector3 = STANDARD_GRAVITY_M_S2,
) -> RigidMotion:
    """Return frozen-direction Coulomb skid accelerations at one state."""
    slip = contact_slip_velocity(state, surface, body)
    direction = unit(slip, tolerance=1e-15)
    normal_acceleration = _normal_gravity(surface, gravity_m_s2)
    normal_force = body.mass_kg * normal_acceleration
    friction = scale(direction, -surface.kinetic_friction * normal_force)
    gravity_tangent = tangent(gravity_m_s2, surface.normal_unit)
    acceleration = add(gravity_tangent, scale(friction, 1.0 / body.mass_kg))
    arm = scale(surface.normal_unit, -body.radius_m)
    angular_acceleration = scale(cross(arm, friction), 1.0 / body.inertia_kg_m2)
    slip_acceleration = add(acceleration, cross(angular_acceleration, arm))
    return RigidMotion(acceleration, angular_acceleration, slip_acceleration, friction)


def static_rolling_feasible(
    surface: GroundSurfaceProfile,
    body: SphereProperties,
    gravity_m_s2: Vector3 = STANDARD_GRAVITY_M_S2,
) -> bool:
    """Return whether static friction can support gravity-driven pure roll."""
    normal_acceleration = _normal_gravity(surface, gravity_m_s2)
    gravity_tangent = tangent(gravity_m_s2, surface.normal_unit)
    factor = body.rotational_inertia_factor
    required = body.mass_kg * factor / (1.0 + factor) * norm(gravity_tangent)
    available = surface.static_friction * body.mass_kg * normal_acceleration
    return bool(required <= available + 1e-12)


def rolling_state(
    state: GroundContactState,
    surface: GroundSurfaceProfile,
    body: SphereProperties,
) -> GroundContactState:
    """Project only round-off residual onto the exact no-slip rolling manifold."""
    normal = surface.normal_unit
    relative = tangent(
        subtract(state.velocity_m_s, surface.surface_velocity_m_s), normal
    )
    axial_spin = dot(state.angular_velocity_rad_s, normal)
    rolling_spin = add(
        scale(cross(normal, relative), 1.0 / body.radius_m),
        scale(normal, axial_spin),
    )
    velocity = add(surface.surface_velocity_m_s, relative)
    return replace(state, velocity_m_s=velocity, angular_velocity_rad_s=rolling_spin)


def rolling_kinematics(
    state: GroundContactState,
    surface: GroundSurfaceProfile,
    body: SphereProperties,
    gravity_m_s2: Vector3 = STANDARD_GRAVITY_M_S2,
) -> RigidMotion:
    """Return pure-roll acceleration with stimp-compatible rolling resistance."""
    if not static_rolling_feasible(surface, body, gravity_m_s2):
        raise ValueError("static friction cannot sustain pure roll on this slope")
    normal = surface.normal_unit
    gravity_tangent = tangent(gravity_m_s2, normal)
    factor = body.rotational_inertia_factor
    drive = scale(gravity_tangent, 1.0 / (1.0 + factor))
    relative = tangent(
        subtract(state.velocity_m_s, surface.surface_velocity_m_s), normal
    )
    direction = _rolling_resistance_direction(relative, drive)
    resistance = surface.rolling_resistance * _normal_gravity(surface, gravity_m_s2)
    acceleration = add(drive, scale(direction, -resistance))
    contact_force = scale(subtract(acceleration, gravity_tangent), body.mass_kg)
    angular_acceleration = scale(cross(normal, acceleration), 1.0 / body.radius_m)
    return RigidMotion(acceleration, angular_acceleration, _ZERO, contact_force)


def _rolling_resistance_direction(relative: Vector3, drive: Vector3) -> Vector3:
    if norm(relative) > 1e-15:
        return unit(relative, tolerance=1e-15)
    if norm(drive) > 1e-15:
        return unit(drive, tolerance=1e-15)
    return _ZERO


def stable_at_zero_speed(
    surface: GroundSurfaceProfile,
    body: SphereProperties,
    gravity_m_s2: Vector3 = STANDARD_GRAVITY_M_S2,
) -> bool:
    """Return whether rolling resistance can hold the zero-speed state."""
    drive = norm(tangent(gravity_m_s2, surface.normal_unit)) / (
        1.0 + body.rotational_inertia_factor
    )
    resistance = surface.rolling_resistance * _normal_gravity(surface, gravity_m_s2)
    return (
        static_rolling_feasible(surface, body, gravity_m_s2)
        and drive <= resistance + 1e-12
    )


def advance_constant_motion(
    state: GroundContactState,
    motion: RigidMotion,
    duration_s: float,
) -> GroundContactState:
    """Advance position, velocity, and spin under constant accelerations."""
    if not math.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError("duration_s must be finite and nonnegative")
    position = add(
        state.position_m,
        add(
            scale(state.velocity_m_s, duration_s),
            scale(motion.acceleration_m_s2, 0.5 * duration_s**2),
        ),
    )
    velocity = add(state.velocity_m_s, scale(motion.acceleration_m_s2, duration_s))
    spin = add(
        state.angular_velocity_rad_s,
        scale(motion.angular_acceleration_rad_s2, duration_s),
    )
    return replace(
        state,
        time_s=state.time_s + duration_s,
        position_m=position,
        velocity_m_s=velocity,
        angular_velocity_rad_s=spin,
    )


def time_to_vector_zero(
    value: Vector3,
    rate: Vector3,
    *,
    tolerance: float,
) -> float | None:
    """Return the exact positive root when a constant vector rate reaches zero."""
    denominator = dot(rate, rate)
    if denominator <= tolerance**2:
        return None
    time_s = -dot(value, rate) / denominator
    if time_s <= 0.0:
        return None
    residual = add(value, scale(rate, time_s))
    return time_s if norm(residual) <= tolerance else None


def bounded_closing_duration(
    value: Vector3,
    rate: Vector3,
    requested_duration_s: float,
) -> float:
    """Bound a closing vector step to avoid crossing its singular zero state."""
    if not math.isfinite(requested_duration_s) or requested_duration_s <= 0.0:
        raise ValueError("requested_duration_s must be finite and positive")
    magnitude = norm(value)
    rate_magnitude = norm(rate)
    if magnitude == 0.0 or rate_magnitude == 0.0 or dot(value, rate) >= 0.0:
        return requested_duration_s
    return float(min(requested_duration_s, 0.25 * magnitude / rate_magnitude))


def relative_path_distance(
    initial_velocity: Vector3,
    acceleration: Vector3,
    duration_s: float,
) -> float:
    """Integrate center speed with exact collinear and Simpson vector paths."""
    final_velocity = add(initial_velocity, scale(acceleration, duration_s))
    midpoint = add(initial_velocity, scale(acceleration, 0.5 * duration_s))
    if norm(cross(initial_velocity, final_velocity)) <= 1e-12:
        return float(0.5 * (norm(initial_velocity) + norm(final_velocity)) * duration_s)
    return float(
        duration_s
        * (norm(initial_velocity) + 4.0 * norm(midpoint) + norm(final_velocity))
        / 6.0
    )


def kinetic_energy(state: GroundContactState, body: SphereProperties) -> float:
    """Return translational plus isotropic rotational kinetic energy."""
    return float(
        0.5 * body.mass_kg * dot(state.velocity_m_s, state.velocity_m_s)
        + 0.5
        * body.inertia_kg_m2
        * dot(
            state.angular_velocity_rad_s,
            state.angular_velocity_rad_s,
        )
    )


__all__ = [
    "advance_constant_motion",
    "bounded_closing_duration",
    "contact_slip_velocity",
    "kinetic_energy",
    "relative_path_distance",
    "rolling_kinematics",
    "rolling_state",
    "skid_kinematics",
    "stable_at_zero_speed",
    "static_rolling_feasible",
    "tangent",
    "time_to_vector_zero",
]
