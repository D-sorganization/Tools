"""Finite-grip preload tangent at relative rest, with the anchor prescribed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import finite_array
from ._grip_finite_response import (
    FiniteGripResponse,
    FinitePoseGrip,
    finite_grip_response,
)
from ._grip_moving_kinematics import MaterialPointMotion, moving_grip_kinematics
from ._shaft_se3 import right_jacobian_derivative


@dataclass(frozen=True)
class StationaryGripOperators:
    """Fresh material coefficients; stiffness differentiates the left-side force.

    Root/anchor relative twists and rates vanish in the common observer.
    The anchor is prescribed. These coefficients do not qualify stability,
    bandwidth or a trajectory and do not include absolute hand body inertia.
    """

    response: FiniteGripResponse
    mass: np.ndarray
    damping: np.ndarray
    stiffness: np.ndarray


def require_stationary_motion(motion: MaterialPointMotion) -> None:
    """Refuse a nonstationary boundary instead of dropping its rate terms."""
    if not isinstance(motion, MaterialPointMotion):
        raise TypeError("motion must be MaterialPointMotion")
    if np.any(motion.twist) or np.any(motion.twist_rate):
        raise ValueError("stationary grip operators require zero relative twist/rate")


def _map_derivative(
    mapping: np.ndarray, phi: np.ndarray, axis: np.ndarray
) -> np.ndarray:
    inverse = mapping[3:, 3:]
    jacobian_rate = right_jacobian_derivative(
        np.r_[np.zeros(3), phi], np.r_[np.zeros(3), inverse @ axis]
    )[3:, 3:]
    derivative = np.zeros((6, 6))
    derivative[:3, :3] = mapping[:3, :3] @ np.cross(axis, np.eye(3)).T
    derivative[3:, 3:] = -inverse @ jacobian_rate @ inverse
    return derivative


def stationary_grip_operators(
    grip: FinitePoseGrip, root: MaterialPointMotion, anchor: MaterialPointMotion
) -> StationaryGripOperators:
    """Retain d(Ar.T)g in addition to Ar.T K Ar at finite preload.

    K here is a constant coordinate coefficient. The returned stiffness is
    the moving material derivative, not a fixed-chart Hessian. At relative
    rest mass and damping pull back by Ar.T M Ar and Ar.T C Ar respectively.
    """
    require_stationary_motion(root)
    require_stationary_motion(anchor)
    response = finite_grip_response(grip, root, anchor)
    motion = moving_grip_kinematics(root, anchor)
    mapping, phi = motion.root_motion_map, motion.displacement[3:]
    matrices = []
    with np.errstate(over="ignore", invalid="ignore"):
        for factor in (
            grip.inertance_factor,
            grip.damping_factor,
            grip.stiffness_factor,
        ):
            mapped = np.asarray(factor) @ mapping
            matrices.append(mapped.T @ mapped)
        for index, axis in enumerate(np.eye(3), start=3):
            matrices[2][:, index] += (
                _map_derivative(mapping, phi, axis).T @ response.storage.effort
            )
    mass, damping, stiffness = (
        finite_array(matrix, (6, 6), "stationary grip coefficient")
        for matrix in matrices
    )
    return StationaryGripOperators(response, mass, damping, stiffness)


__all__ = ()
