"""Canonical small-rotation kinetic Hessians; all quantities in observer axes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import ComponentMassProperties


def cross_matrix(vector: np.ndarray) -> np.ndarray:
    """Cross-product matrix of a validated real three-vector."""
    x, y, z = map(float, vector)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_hessian(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hessian at zero of left dot Exp(theta) right."""
    return np.asarray(
        (np.outer(left, right) + np.outer(right, left)) / 2 - (left @ right) * np.eye(3)
    )


@dataclass(frozen=True)
class BodyGeometry:
    """SI mass, COM inertia and reference geometry used by kinetic derivatives."""

    mass: float
    inertia: np.ndarray
    offset: np.ndarray
    com: np.ndarray
    offset_cross: np.ndarray


def body_geometry(body: ComponentMassProperties, reference: np.ndarray) -> BodyGeometry:
    offset = np.asarray(body.center_of_mass_m)
    return BodyGeometry(
        body.mass_kg,
        np.asarray(body.inertia_at_com_kg_m2),
        offset,
        reference + offset,
        cross_matrix(offset),
    )


def velocity_position(data: BodyGeometry, omega: np.ndarray) -> np.ndarray:
    """Mixed Hessian d²T/d(qdot)dq using the left Jacobian of Exp(theta)."""
    cross = cross_matrix(omega)
    offset_cross = data.offset_cross
    com_velocity = cross @ data.com
    result = np.zeros((6, 6))
    result[:3, :3] = data.mass * cross
    result[:3, 3:] = -data.mass * cross @ offset_cross
    result[3:, :3] = data.mass * offset_cross @ cross
    result[3:, 3:] = (
        data.mass
        * (
            0.5 * cross_matrix(offset_cross @ com_velocity)
            + np.outer(data.offset, com_velocity)
            - (data.offset @ com_velocity) * np.eye(3)
            - offset_cross @ cross @ offset_cross
        )
        + data.inertia @ cross
        - 0.5 * cross_matrix(data.inertia @ omega)
    )
    return np.asarray(result)


def kinetic_position_hessian(data: BodyGeometry, omega: np.ndarray) -> np.ndarray:
    """Hessian of centrifugal kinetic energy at zero relative velocity."""
    cross = cross_matrix(omega)
    metric = cross.T @ cross
    com_map = np.column_stack((np.eye(3), -data.offset_cross))
    result = data.mass * com_map.T @ metric @ com_map
    result[3:, 3:] += (
        data.mass * rotation_hessian(metric @ data.com, data.offset)
        + cross.T @ data.inertia @ cross
        + rotation_hessian(data.inertia @ omega, omega)
    )
    return np.asarray(result)


def equilibrium_force(data: BodyGeometry, motion: tuple[np.ndarray, ...]) -> np.ndarray:
    """Opposite of full inertial body wrench required at zero relative state."""
    omega, alpha, origin_acceleration = motion
    acceleration = (
        origin_acceleration
        + np.cross(alpha, data.com)
        + np.cross(omega, np.cross(omega, data.com))
    )
    force = -data.mass * acceleration
    torque = (
        np.cross(data.offset, force)
        - data.inertia @ alpha
        - np.cross(omega, data.inertia @ omega)
    )
    return np.asarray(np.r_[force, torque])
