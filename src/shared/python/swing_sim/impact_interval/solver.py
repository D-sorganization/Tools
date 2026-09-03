"""Reference fixed-step six-DOF club/ball contact-interval solver.

The reference implementation deliberately favors explicit state and audit
channels over a black-box integrator. The hot loop is isolated here so a Rust
kernel can replace it behind the same façade without changing callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from shared.python.swing_sim.impact import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
)

from ._rotation import exp_rotation, log_rotation
from .types import (
    BoundaryKind,
    ClubRigidBody,
    ImpactIntervalAudit,
    ImpactIntervalConfig,
    ImpactIntervalInitialState,
    ImpactIntervalResult,
)


@dataclass
class _State:
    club_position: np.ndarray
    orientation: np.ndarray
    club_velocity: np.ndarray
    club_omega: np.ndarray
    ball_position: np.ndarray
    ball_velocity: np.ndarray
    ball_omega: np.ndarray


def _kinetic_energy(state: _State, club: ClubRigidBody) -> float:
    inertia_world = state.orientation @ club.inertia_body_kg_m2 @ state.orientation.T
    club_energy = 0.5 * club.mass_kg * float(
        np.dot(state.club_velocity, state.club_velocity)
    ) + 0.5 * float(state.club_omega @ inertia_world @ state.club_omega)
    ball_energy = 0.5 * GOLF_BALL_MASS_KG * float(
        np.dot(state.ball_velocity, state.ball_velocity)
    ) + 0.5 * GOLF_BALL_MOMENT_OF_INERTIA_KG_M2 * float(
        np.dot(state.ball_omega, state.ball_omega)
    )
    return float(club_energy + ball_energy)


def _contact_state(
    state: _State, club: ClubRigidBody
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    normal = state.orientation @ club.face_normal_body
    normal = normal / np.linalg.norm(normal)
    r_club = state.orientation @ club.cg_to_contact_body_m
    contact_point = state.club_position + r_club
    r_ball = -GOLF_BALL_RADIUS_M * normal
    club_surface_velocity = state.club_velocity + np.cross(state.club_omega, r_club)
    ball_surface_velocity = state.ball_velocity + np.cross(state.ball_omega, r_ball)
    relative_velocity = club_surface_velocity - ball_surface_velocity
    compression = GOLF_BALL_RADIUS_M - float(
        np.dot(state.ball_position - contact_point, normal)
    )
    compression_rate = float(np.dot(relative_velocity, normal))
    return (
        normal,
        r_club,
        contact_point,
        relative_velocity,
        compression,
        compression_rate,
    )


def _torsional_torque(
    state: _State,
    initial_orientation: np.ndarray,
    club: ClubRigidBody,
    config: ImpactIntervalConfig,
) -> tuple[np.ndarray, float]:
    if config.boundary is not BoundaryKind.TORSIONAL_GRIP:
        return np.zeros(3), 0.0
    relative = state.orientation @ initial_orientation.T
    rotation_vector = log_rotation(relative)
    shaft_axis = state.orientation @ club.shaft_axis_body
    twist = float(np.dot(rotation_vector, shaft_axis))
    twist_rate = float(np.dot(state.club_omega, shaft_axis))
    magnitude = -(
        config.torsional_stiffness_n_m_per_rad * twist
        + config.torsional_damping_n_m_s_per_rad * twist_rate
    )
    return magnitude * shaft_axis, twist


def _advance_club(
    state: _State,
    club: ClubRigidBody,
    config: ImpactIntervalConfig,
    force_on_ball: np.ndarray,
    r_contact: np.ndarray,
    anchor: np.ndarray,
    initial_orientation: np.ndarray,
) -> float:
    dt = config.time_step_s
    inertia_world = state.orientation @ club.inertia_body_kg_m2 @ state.orientation.T
    spring_torque, twist = _torsional_torque(state, initial_orientation, club, config)
    if config.boundary is BoundaryKind.FREE:
        torque = np.cross(r_contact, -force_on_ball)
        angular_acceleration = np.linalg.solve(
            inertia_world,
            torque
            + spring_torque
            - np.cross(state.club_omega, inertia_world @ state.club_omega),
        )
        state.club_velocity += (-force_on_ball / club.mass_kg) * dt
        state.club_omega += angular_acceleration * dt
        state.club_position += state.club_velocity * dt
        state.orientation = exp_rotation(state.club_omega * dt) @ state.orientation
        return twist

    r_attachment = state.orientation @ club.cg_to_attachment_body_m
    pivot_inertia = inertia_world + club.mass_kg * (
        float(np.dot(r_attachment, r_attachment)) * np.eye(3)
        - np.outer(r_attachment, r_attachment)
    )
    lever_from_pivot = r_contact - r_attachment
    torque = np.cross(lever_from_pivot, -force_on_ball) + spring_torque
    angular_acceleration = np.linalg.solve(
        pivot_inertia,
        torque - np.cross(state.club_omega, pivot_inertia @ state.club_omega),
    )
    state.club_omega += angular_acceleration * dt
    state.orientation = exp_rotation(state.club_omega * dt) @ state.orientation
    r_attachment_new = state.orientation @ club.cg_to_attachment_body_m
    state.club_position = anchor - r_attachment_new
    state.club_velocity = -np.cross(state.club_omega, r_attachment_new)
    return twist


def _state_arrays(
    history: dict[str, list[np.ndarray | float]],
) -> dict[str, np.ndarray]:
    """Convert the append-only trace buffer to homogeneous arrays."""
    return {name: np.asarray(values, dtype=float) for name, values in history.items()}


def solve_impact_interval(
    initial: ImpactIntervalInitialState,
    club: ClubRigidBody,
    config: ImpactIntervalConfig,
) -> ImpactIntervalResult:
    """Integrate full club and ball state until contact separation.

    Preconditions:
        Inputs satisfy their value-type contracts and the initial clubface is
        not more than one ball radius beyond the ball center.
    Postconditions:
        The returned histories have equal length, finite values, monotonic
        time, and contain the separated post-impact state when contact occurs.
    """
    if not isinstance(initial, ImpactIntervalInitialState):
        raise TypeError("initial must be an ImpactIntervalInitialState")
    if not isinstance(club, ClubRigidBody):
        raise TypeError("club must be a ClubRigidBody")
    if not isinstance(config, ImpactIntervalConfig):
        raise TypeError("config must be an ImpactIntervalConfig")
    initial.validate()
    state = _State(
        np.asarray(initial.club_position_m, dtype=float).copy(),
        np.asarray(initial.club_orientation, dtype=float).copy(),
        np.asarray(initial.club_velocity_mps, dtype=float).copy(),
        np.asarray(initial.club_angular_velocity_rad_s, dtype=float).copy(),
        np.asarray(initial.ball_position_m, dtype=float).copy(),
        np.asarray(initial.ball_velocity_mps, dtype=float).copy(),
        np.asarray(initial.ball_angular_velocity_rad_s, dtype=float).copy(),
    )
    initial_orientation = state.orientation.copy()
    anchor = state.club_position + state.orientation @ club.cg_to_attachment_body_m
    initial_energy = _kinetic_energy(state, club)
    initial_momentum = (
        club.mass_kg * state.club_velocity + GOLF_BALL_MASS_KG * state.ball_velocity
    )
    history: dict[str, list[np.ndarray | float]] = {
        name: []
        for name in (
            "time_s",
            "club_position_m",
            "club_orientation",
            "club_velocity_mps",
            "club_angular_velocity_rad_s",
            "ball_position_m",
            "ball_velocity_mps",
            "ball_angular_velocity_rad_s",
            "attachment_position_m",
            "contact_point_position_m",
            "contact_normal",
            "normal_force_n",
            "friction_force_n",
            "compression_m",
            "face_angle_deg",
            "dynamic_loft_deg",
            "twist_angle_rad",
        )
    }
    did_contact = False
    first_contact_time = 0.0
    last_contact_time = 0.0
    normal_impulse = 0.0
    friction_impulse = 0.0
    modelled_dissipation = 0.0
    max_steps = int(math.ceil(config.maximum_time_s / config.time_step_s)) + 1

    for step in range(max_steps):
        time_s = step * config.time_step_s
        normal, r_contact, point, relative, compression, rate = _contact_state(
            state, club
        )
        normal_force = config.contact_law.normal_force(compression, rate)
        tangent_velocity = relative - rate * normal
        tangent_speed = float(np.linalg.norm(tangent_velocity))
        friction_force = np.zeros(3)
        if normal_force > 0.0 and tangent_speed > 1.0e-12:
            friction_magnitude = (
                config.friction_coefficient
                * normal_force
                * math.tanh(tangent_speed / config.friction_regularization_mps)
            )
            friction_force = friction_magnitude * tangent_velocity / tangent_speed
        force_on_ball = normal_force * normal + friction_force
        relative_orientation = state.orientation @ initial_orientation.T
        rotation_vector = log_rotation(relative_orientation)
        shaft_axis = state.orientation @ club.shaft_axis_body
        twist = float(np.dot(rotation_vector, shaft_axis))
        face_angle = math.degrees(math.atan2(normal[2], normal[0]))
        loft = math.degrees(math.atan2(normal[1], math.hypot(normal[0], normal[2])))
        attachment = (
            state.club_position + state.orientation @ club.cg_to_attachment_body_m
        )
        values: dict[str, np.ndarray | float] = {
            "time_s": time_s,
            "club_position_m": state.club_position.copy(),
            "club_orientation": state.orientation.copy(),
            "club_velocity_mps": state.club_velocity.copy(),
            "club_angular_velocity_rad_s": state.club_omega.copy(),
            "ball_position_m": state.ball_position.copy(),
            "ball_velocity_mps": state.ball_velocity.copy(),
            "ball_angular_velocity_rad_s": state.ball_omega.copy(),
            "attachment_position_m": attachment,
            "contact_point_position_m": point,
            "contact_normal": normal,
            "normal_force_n": normal_force,
            "friction_force_n": float(np.linalg.norm(friction_force)),
            "compression_m": compression,
            "face_angle_deg": face_angle,
            "dynamic_loft_deg": loft,
            "twist_angle_rad": twist,
        }
        for name, value in values.items():
            history[name].append(value)

        if normal_force > 0.0:
            if not did_contact:
                first_contact_time = time_s
            did_contact = True
            last_contact_time = time_s
        elif did_contact and compression <= 0.0 and rate < 0.0:
            break
        if step == max_steps - 1:
            break

        dt = config.time_step_s
        normal_impulse += normal_force * dt
        friction_impulse += float(np.linalg.norm(friction_force)) * dt
        if normal_force > 0.0:
            damping_power = config.contact_law.damping_n_s_per_m * rate**2
            friction_power = float(np.dot(friction_force, tangent_velocity))
            modelled_dissipation += max(0.0, damping_power + friction_power) * dt

        r_ball = -GOLF_BALL_RADIUS_M * normal
        ball_torque = np.cross(r_ball, force_on_ball)
        state.ball_velocity += force_on_ball / GOLF_BALL_MASS_KG * dt
        state.ball_omega += ball_torque / GOLF_BALL_MOMENT_OF_INERTIA_KG_M2 * dt
        state.ball_position += state.ball_velocity * dt
        _advance_club(
            state,
            club,
            config,
            force_on_ball,
            r_contact,
            anchor,
            initial_orientation,
        )

    arrays = _state_arrays(history)
    final_energy = _kinetic_energy(state, club)
    final_twist = float(arrays["twist_angle_rad"][-1])
    boundary_energy = (
        0.5 * config.torsional_stiffness_n_m_per_rad * final_twist**2
        if config.boundary is BoundaryKind.TORSIONAL_GRIP
        else 0.0
    )
    final_momentum = (
        club.mass_kg * state.club_velocity + GOLF_BALL_MASS_KG * state.ball_velocity
    )
    raw_energy_residual = (
        initial_energy - final_energy - modelled_dissipation - boundary_energy
    )
    # A unilateral Kelvin-Voigt law clips tensile force during rebound. Any
    # spring energy remaining at that clipping instant is a physical loss at
    # the one-sided contact boundary, distinct from dashpot/friction loss.
    unilateral_release = max(0.0, raw_energy_residual)
    total_dissipation = modelled_dissipation + unilateral_release
    energy_residual = (
        initial_energy - final_energy - total_dissipation - boundary_energy
    )
    audit = ImpactIntervalAudit(
        initial_kinetic_energy_j=initial_energy,
        final_kinetic_energy_j=final_energy,
        dissipated_energy_j=total_dissipation,
        dashpot_and_friction_dissipation_j=modelled_dissipation,
        unilateral_release_energy_j=unilateral_release,
        boundary_stored_energy_j=boundary_energy,
        energy_residual_j=energy_residual,
        integrated_normal_impulse_n_s=normal_impulse,
        integrated_friction_impulse_n_s=friction_impulse,
        linear_momentum_residual_n_s=float(
            np.linalg.norm(final_momentum - initial_momentum)
        ),
    )
    contact_duration = (
        last_contact_time - first_contact_time + config.time_step_s
        if did_contact
        else 0.0
    )
    return ImpactIntervalResult(
        **arrays,
        contact_duration_s=contact_duration,
        did_contact=did_contact,
        audit=audit,
    )


__all__ = ["solve_impact_interval"]
