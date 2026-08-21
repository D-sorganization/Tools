"""Constrained mechanics for the rotating-base, two-hand club tier."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._numeric import (
    N_CONSTRAINTS,
    N_COORDINATES,
    FloatArray,
    _direction,
    _finite_vector,
    _rotate,
)
from .kinematics import _body_jacobians, _points, constraint_jacobian, constraint_vector
from .types import (
    DynamicsSolution,
    RotatingBaseParams,
    RotatingBaseState,
    TorsoTwoHandControl,
)


def mass_matrix(q: object, params: RotatingBaseParams) -> FloatArray:
    """Return the symmetric positive-definite generalized mass matrix."""
    state = _finite_vector("q", q, (N_COORDINATES,))
    matrix = np.zeros((N_COORDINATES, N_COORDINATES))
    matrix[0, 0] += params.torso_inertia_kg_m2
    inertias = (
        params.arm_inertia_kg_m2,
        params.arm_inertia_kg_m2,
        params.proximal_club_inertia_kg_m2,
        params.distal_club_inertia_kg_m2,
    )
    for (mass, linear, angular), inertia in zip(
        _body_jacobians(state, params), inertias, strict=True
    ):
        matrix += mass * linear.T @ linear + inertia * np.outer(angular, angular)
    return 0.5 * (matrix + matrix.T)


def potential_energy(q: object, params: RotatingBaseParams) -> float:
    """Return gravitational, torso-spring, and shaft-spring energy."""
    state = _finite_vector("q", q, (N_COORDINATES,))
    points = _points(state, params)
    phi, lead_relative, trail_relative, _x, _y, alpha, beta = state
    lead_com = points["lead_shoulder"] + 0.5 * params.arm_length_m * _direction(
        phi + lead_relative
    )
    trail_com = points["trail_shoulder"] + 0.5 * params.arm_length_m * _direction(
        phi + trail_relative
    )
    center = points["grip_center"]
    proximal_com = center + 0.5 * params.proximal_club_length_m * _direction(alpha)
    distal_com = points["flex_joint"] + 0.5 * params.distal_club_length_m * _direction(
        alpha + beta
    )
    gravity = params.gravity_m_s2 * (
        params.arm_mass_kg * (lead_com[1] + trail_com[1])
        + params.proximal_club_mass_kg * proximal_com[1]
        + params.distal_club_mass_kg * distal_com[1]
    )
    elastic = 0.5 * params.torso_stiffness_nm_rad * phi**2
    elastic += 0.5 * params.shaft_stiffness_nm_rad * beta**2
    return float(gravity + elastic)


def mechanical_energy(state: RotatingBaseState, params: RotatingBaseParams) -> float:
    """Return kinetic plus conservative potential energy."""
    return float(
        0.5 * state.qdot @ mass_matrix(state.q, params) @ state.qdot
        + potential_energy(state.q, params)
    )


def distal_segment_kinetic_energy(
    state: RotatingBaseState, params: RotatingBaseParams
) -> float:
    """Return distal-club translational plus rotational kinetic energy."""
    distal = _body_jacobians(state.q, params)[3]
    _mass, linear_jacobian, angular_jacobian = distal
    linear_velocity = linear_jacobian @ state.qdot
    angular_velocity = float(angular_jacobian @ state.qdot)
    return float(
        0.5 * params.distal_club_mass_kg * linear_velocity @ linear_velocity
        + 0.5 * params.distal_club_inertia_kg_m2 * angular_velocity**2
    )


def _gradient(function: Callable[[FloatArray], float], q: FloatArray) -> FloatArray:
    step = 2e-6
    result = np.empty(N_COORDINATES)
    for index in range(N_COORDINATES):
        offset = np.zeros(N_COORDINATES)
        offset[index] = step
        result[index] = (function(q + offset) - function(q - offset)) / (2.0 * step)
    return result


def _bias_force(
    q: FloatArray, qdot: FloatArray, params: RotatingBaseParams
) -> FloatArray:
    step = 2e-6
    derivatives = np.empty((N_COORDINATES, N_COORDINATES, N_COORDINATES))
    for index in range(N_COORDINATES):
        offset = np.zeros(N_COORDINATES)
        offset[index] = step
        derivatives[index] = (
            mass_matrix(q + offset, params) - mass_matrix(q - offset, params)
        ) / (2.0 * step)
    coriolis = np.zeros(N_COORDINATES)
    for i in range(N_COORDINATES):
        for j in range(N_COORDINATES):
            for k in range(N_COORDINATES):
                christoffel = 0.5 * (
                    derivatives[k, i, j] + derivatives[j, i, k] - derivatives[i, j, k]
                )
                coriolis[i] += christoffel * qdot[j] * qdot[k]
    return coriolis + _gradient(lambda value: potential_energy(value, params), q)


def _damping_force(qdot: FloatArray, params: RotatingBaseParams) -> FloatArray:
    force = np.zeros(N_COORDINATES)
    force[0] = -params.torso_damping_nms_rad * qdot[0]
    force[1:3] = -params.arm_damping_nms_rad * qdot[1:3]
    force[6] = -params.shaft_damping_nms_rad * qdot[6]
    return force


def control_generalized_force(control: TorsoTwoHandControl) -> FloatArray:
    """Map actuator moments to generalized forces by virtual work."""
    force = np.zeros(N_COORDINATES)
    force[0] += control.torso_nm
    force[1] += control.lead_arm_nm
    force[2] += control.trail_arm_nm
    for relative_index, wrist in (
        (1, control.lead_wrist_nm),
        (2, control.trail_wrist_nm),
    ):
        force[0] -= wrist
        force[relative_index] -= wrist
        force[5] += wrist
    return force


def _constraint_acceleration_bias(
    q: FloatArray, qdot: FloatArray, params: RotatingBaseParams
) -> FloatArray:
    phi, _lead, _trail, _x, _y, alpha, _beta = q
    result = np.empty(N_CONSTRAINTS)
    for row, relative_index, shoulder_offset, grip_offset in (
        (0, 1, params.lead_shoulder_offset_m, params.lead_grip_offset_m),
        (2, 2, params.trail_shoulder_offset_m, params.trail_grip_offset_m),
    ):
        absolute = phi + q[relative_index]
        absolute_rate = qdot[0] + qdot[relative_index]
        hand_bias = -_rotate(shoulder_offset, phi) * qdot[0] ** 2
        hand_bias -= params.arm_length_m * _direction(absolute) * absolute_rate**2
        grip_bias = -grip_offset * _direction(alpha) * qdot[5] ** 2
        result[row : row + 2] = hand_bias - grip_bias
    return result


def solve_constrained_dynamics(
    state: RotatingBaseState,
    control: TorsoTwoHandControl,
    params: RotatingBaseParams,
) -> DynamicsSolution:
    """Solve the full-rank KKT system and return bilateral reactions."""
    position_residual = constraint_vector(state.q, params)
    if np.linalg.norm(position_residual) > 1e-7:
        raise ValueError("state violates bilateral position constraints")
    jacobian = constraint_jacobian(state.q, params)
    rank = int(np.linalg.matrix_rank(jacobian, tol=params.rank_tolerance))
    if rank != N_CONSTRAINTS:
        raise ValueError("bilateral constraint Jacobian is rank deficient")
    matrix = mass_matrix(state.q, params)
    bias = _bias_force(state.q, state.qdot, params)
    applied = control_generalized_force(control) + _damping_force(state.qdot, params)
    gamma = _constraint_acceleration_bias(state.q, state.qdot, params)
    kkt = np.block([[matrix, -jacobian.T], [jacobian, np.zeros((N_CONSTRAINTS,) * 2)]])
    rhs = np.concatenate((applied - bias, -gamma))
    solved = np.linalg.solve(kkt, rhs)
    qddot, multipliers = solved[:N_COORDINATES], solved[N_COORDINATES:]
    force_hands = multipliers.reshape(2, 2)
    force_club = -force_hands
    offsets = np.array([params.lead_grip_offset_m, params.trail_grip_offset_m])[
        :, None
    ] * _direction(state.q[5])
    couple = float(
        np.sum(offsets[:, 0] * force_club[:, 1] - offsets[:, 1] * force_club[:, 0])
    )
    residual = kkt @ solved - rhs
    acceleration_residual = jacobian @ qddot + gamma
    return DynamicsSolution(
        qddot=qddot,
        multipliers_n=multipliers,
        force_on_hands_n=force_hands,
        force_on_club_n=force_club,
        force_generated_couple_nm=couple,
        constraint_rank=rank,
        kkt_residual_norm=float(np.linalg.norm(residual)),
        acceleration_constraint_residual_norm=float(
            np.linalg.norm(acceleration_residual)
        ),
    )
