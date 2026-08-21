"""Planar bilateral-loop kinematics for the rotating-base tier."""

from __future__ import annotations

import numpy as np

from ._numeric import (
    N_CONSTRAINTS,
    N_COORDINATES,
    FloatArray,
    _direction,
    _direction_derivative,
    _finite_vector,
    _rotate,
    _rotate_derivative,
)
from .types import RotatingBaseParams


def _points(q: FloatArray, params: RotatingBaseParams) -> dict[str, FloatArray]:
    phi, lead_relative, trail_relative, x_grip, y_grip, alpha, beta = q
    center = np.array([x_grip, y_grip])
    lead_shoulder = _rotate(params.lead_shoulder_offset_m, phi)
    trail_shoulder = _rotate(params.trail_shoulder_offset_m, phi)
    lead_hand = lead_shoulder + params.arm_length_m * _direction(phi + lead_relative)
    trail_hand = trail_shoulder + params.arm_length_m * _direction(phi + trail_relative)
    lead_grip = center + params.lead_grip_offset_m * _direction(alpha)
    trail_grip = center + params.trail_grip_offset_m * _direction(alpha)
    flex_joint = center + params.proximal_club_length_m * _direction(alpha)
    clubhead = flex_joint + params.distal_club_length_m * _direction(alpha + beta)
    return {
        "lead_shoulder": lead_shoulder,
        "trail_shoulder": trail_shoulder,
        "lead_hand": lead_hand,
        "trail_hand": trail_hand,
        "grip_center": center,
        "lead_grip": lead_grip,
        "trail_grip": trail_grip,
        "flex_joint": flex_joint,
        "clubhead": clubhead,
    }


def kinematics(q: object, params: RotatingBaseParams) -> dict[str, FloatArray]:
    """Return all declared points in the planar world frame."""
    return _points(_finite_vector("q", q, (N_COORDINATES,)), params)


def constraint_vector(q: object, params: RotatingBaseParams) -> FloatArray:
    """Return lead and trail hand-minus-grip position residuals."""
    points = kinematics(q, params)
    return np.concatenate(
        (
            points["lead_hand"] - points["lead_grip"],
            points["trail_hand"] - points["trail_grip"],
        )
    )


def constraint_jacobian(q: object, params: RotatingBaseParams) -> FloatArray:
    """Return the analytic four-by-seven loop-closure Jacobian."""
    state = _finite_vector("q", q, (N_COORDINATES,))
    phi, lead_relative, trail_relative, _x, _y, alpha, _beta = state
    matrix = np.zeros((N_CONSTRAINTS, N_COORDINATES))
    for row, relative_index, shoulder_offset, grip_offset in (
        (0, 1, params.lead_shoulder_offset_m, params.lead_grip_offset_m),
        (2, 2, params.trail_shoulder_offset_m, params.trail_grip_offset_m),
    ):
        absolute = phi + state[relative_index]
        arm_derivative = params.arm_length_m * _direction_derivative(absolute)
        matrix[row : row + 2, 0] = (
            _rotate_derivative(shoulder_offset, phi) + arm_derivative
        )
        matrix[row : row + 2, relative_index] = arm_derivative
        matrix[row : row + 2, 3:5] = -np.eye(2)
        matrix[row : row + 2, 5] = -grip_offset * _direction_derivative(alpha)
    return matrix


def _body_jacobians(
    q: FloatArray, params: RotatingBaseParams
) -> tuple[tuple[float, FloatArray, FloatArray], ...]:
    phi, lead_relative, trail_relative, _x, _y, alpha, beta = q
    bodies: list[tuple[float, FloatArray, FloatArray]] = []
    for relative_index, shoulder_offset in (
        (1, params.lead_shoulder_offset_m),
        (2, params.trail_shoulder_offset_m),
    ):
        absolute = phi + q[relative_index]
        arm_term = 0.5 * params.arm_length_m * _direction_derivative(absolute)
        jacobian = np.zeros((2, N_COORDINATES))
        jacobian[:, 0] = _rotate_derivative(shoulder_offset, phi) + arm_term
        jacobian[:, relative_index] = arm_term
        angular = np.zeros(N_COORDINATES)
        angular[[0, relative_index]] = 1.0
        bodies.append((params.arm_mass_kg, jacobian, angular))
    proximal = np.zeros((2, N_COORDINATES))
    proximal[:, 3:5] = np.eye(2)
    proximal[:, 5] = 0.5 * params.proximal_club_length_m * _direction_derivative(alpha)
    proximal_angular = np.zeros(N_COORDINATES)
    proximal_angular[5] = 1.0
    bodies.append((params.proximal_club_mass_kg, proximal, proximal_angular))
    distal = np.zeros((2, N_COORDINATES))
    distal[:, 3:5] = np.eye(2)
    distal[:, 5] = params.proximal_club_length_m * _direction_derivative(
        alpha
    ) + 0.5 * params.distal_club_length_m * _direction_derivative(alpha + beta)
    distal[:, 6] = (
        0.5 * params.distal_club_length_m * _direction_derivative(alpha + beta)
    )
    distal_angular = np.zeros(N_COORDINATES)
    distal_angular[[5, 6]] = 1.0
    bodies.append((params.distal_club_mass_kg, distal, distal_angular))
    return tuple(bodies)
