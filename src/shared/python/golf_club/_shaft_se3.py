"""Objective section interpolation for the candidate loaded-shaft element.

Twists are linear-first [translation generator in m, rotation in rad].
Section poses map material axes into one common observer frame. This chart
is kinematics only: it does not establish strain limits, equilibrium or a
dynamic tangent. The principal relative rotation must remain below pi.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.spatial.transform import Rotation

from ._grip_contracts import finite_array
from ._validation import require_finite_float, require_rotation

# A numerical chart guard, not an allowable material rotation/strain claim.
_LOG_BRANCH_MARGIN_RAD = 1e-6


def _rotation_chart(vector: np.ndarray) -> None:
    if np.linalg.norm(vector) >= np.pi - _LOG_BRANCH_MARGIN_RAD:
        raise ValueError("rotation reaches the principal-logarithm branch boundary")


def _phi1_generator(matrix: np.ndarray) -> np.ndarray:
    size = len(matrix)
    generator = np.zeros((2 * size, 2 * size))
    generator[:size, :size] = matrix
    generator[:size, size:] = np.eye(size)
    return generator


def _left_rotation_jacobian(vector: np.ndarray) -> np.ndarray:
    """Integrate Exp(t*[rotation]x), retaining the exact zero-angle limit.

    The upper block of Exp([[A, I], [0, 0]]) is phi_1(A). Using SciPy's
    matrix exponential avoids divisions by small angles and a dead zone.
    """
    generator = _phi1_generator(np.cross(vector, np.eye(3)).T)
    return np.asarray(expm(generator)[:3, 3:])


def twist_ad(value: object) -> np.ndarray:
    """Return the Lie bracket matrix for a finite linear-first twist."""
    twist = finite_array(value, (6,), "section twist")
    angular = np.cross(twist[3:], np.eye(3)).T
    linear = np.cross(twist[:3], np.eye(3)).T
    return np.block([[angular, linear], [np.zeros((3, 3)), angular]])


def right_jacobian(value: object) -> np.ndarray:
    """Map exponential-coordinate rates to material twists: phi_1(-ad(q))."""
    generator = _phi1_generator(-twist_ad(value))
    return finite_array(expm(generator)[:6, 6:], (6, 6), "right Jacobian")


def right_jacobian_derivative(value: object, direction: object) -> np.ndarray:
    """Differentiate the right Jacobian without differencing or angle cutoffs.

    The upper block of the matrix-exponential Frechet derivative differentiates
    the integral defining phi_1. SciPy uses scaling, Pade and squaring.
    """
    generator = _phi1_generator(-twist_ad(value))
    derivative = np.zeros_like(generator)
    derivative[:6, :6] = -twist_ad(direction)
    result = expm_frechet(generator, derivative, compute_expm=False)
    return finite_array(result[:6, 6:], (6, 6), "right Jacobian derivative")


def _rigid_pose(value: object) -> np.ndarray:
    pose = finite_array(value, (4, 4), "section pose")
    require_rotation(pose[:3, :3])
    if not np.array_equal(pose[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("section pose must have homogeneous bottom row [0, 0, 0, 1]")
    return pose


def exp_twist(value: object) -> np.ndarray:
    """Map a finite linear-first local twist into a fresh rigid transform.

    Rotations at or within 1e-6 rad of pi are outside this local chart.
    No coercion from strings, booleans or complex values is permitted.
    """
    twist = finite_array(value, (6,), "section twist")
    angular = twist[3:]
    _rotation_chart(angular)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(angular).as_matrix()
    with np.errstate(over="ignore", invalid="ignore"):
        pose[:3, 3] = _left_rotation_jacobian(angular) @ twist[:3]
    return _rigid_pose(pose)


def log_pose(value: object) -> np.ndarray:
    """Return the principal linear-first twist of a proper rigid transform.

    The rotation branch margin is explicit. Translation is the exponential
    generator, not the endpoint position. Input arrays are never modified.
    """
    pose = _rigid_pose(value)
    angular = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
    _rotation_chart(angular)
    linear = np.linalg.solve(_left_rotation_jacobian(angular), pose[:3, 3])
    return finite_array(np.r_[linear, angular], (6,), "section logarithm")


def _material_fraction(value: object) -> float:
    coordinate = require_finite_float(value, "material fraction")
    if not 0.0 <= coordinate <= 1.0:
        raise ValueError("material fraction must lie in [0, 1]")
    return float(coordinate)


def _relative_maps(relative: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the relative-log derivative and its left/right inverse Jacobians."""
    left = np.linalg.solve(right_jacobian(-relative), np.eye(6))
    right = np.linalg.solve(right_jacobian(relative), np.eye(6))
    return np.hstack((-left, right)), left, right


def _velocity_inputs(
    relative: object, fraction: object
) -> tuple[np.ndarray, float, np.ndarray]:
    twist = finite_array(relative, (6,), "relative section twist")
    _rotation_chart(twist[3:])
    coordinate = _material_fraction(fraction)
    inverse = np.linalg.solve(right_jacobian(twist), np.eye(6))
    return twist, coordinate, inverse


def section_velocity_map(relative: object, fraction: object) -> np.ndarray:
    """Map twelve nodal material velocities to one interpolated material twist.

    For d=Log(H_left^-1 H_right) and alpha in [0,1], Q=[I-B,B],
    B=alpha Jr(alpha*d) Jr(d)^-1. The same map transfers virtual work.
    Principal-chart limits are numerical, not a material-domain qualification.
    """
    twist, coordinate, inverse = _velocity_inputs(relative, fraction)
    right = coordinate * right_jacobian(coordinate * twist) @ inverse
    return finite_array(
        np.hstack((np.eye(6) - right, right)), (6, 12), "section velocity map"
    )


def section_velocity_map_derivative(
    relative: object, fraction: object, direction: object
) -> np.ndarray:
    """Differentiate Q with respect to its relative logarithm in one direction.

    Direction is d_dot or a relative-log variation, not an unconverted nodal
    material velocity. Matrix-exponential Frechet derivatives retain zero and
    tiny rotations; no finite differences or imposed symmetry are used.
    """
    twist, coordinate, inverse = _velocity_inputs(relative, fraction)
    delta = finite_array(direction, (6,), "relative twist direction")
    jacobian = right_jacobian(coordinate * twist)
    derivative = coordinate * (
        right_jacobian_derivative(coordinate * twist, coordinate * delta) @ inverse
        - jacobian @ inverse @ right_jacobian_derivative(twist, delta) @ inverse
    )
    return finite_array(
        np.hstack((-derivative, derivative)), (6, 12), "section velocity-map derivative"
    )


def interpolate_pose(left: object, right: object, fraction: object) -> np.ndarray:
    """Evaluate H_left Exp(fraction Log(H_left^-1 H_right)).

    Both poses must use the same observer frame and fraction must be a finite
    real number in [0, 1]. Constant material strain yields a circular centerline
    under pure bending; a common rigid motion leaves relative strain unchanged.
    """
    coordinate = _material_fraction(fraction)
    left_pose, right_pose = _rigid_pose(left), _rigid_pose(right)
    relative = np.linalg.solve(left_pose, right_pose)
    return _rigid_pose(left_pose @ exp_twist(coordinate * log_pose(relative)))


__all__ = ()
