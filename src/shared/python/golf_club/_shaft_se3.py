"""Objective section interpolation for the candidate loaded-shaft element.

Twists are linear-first [translation generator in m, rotation in rad].
Section poses map material axes into one common observer frame. This chart
is kinematics only: it does not establish strain limits, equilibrium or a
dynamic tangent. The principal relative rotation must remain below pi.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from ._grip_contracts import finite_array
from ._validation import require_finite_float, require_rotation

# A numerical chart guard, not an allowable material rotation/strain claim.
_LOG_BRANCH_MARGIN_RAD = 1e-6


def _rotation_chart(vector: np.ndarray) -> None:
    if np.linalg.norm(vector) >= np.pi - _LOG_BRANCH_MARGIN_RAD:
        raise ValueError("rotation reaches the principal-logarithm branch boundary")


def _left_rotation_jacobian(vector: np.ndarray) -> np.ndarray:
    """Integrate Exp(t*[rotation]x), retaining the exact zero-angle limit.

    The upper block of Exp([[A, I], [0, 0]]) is phi_1(A). Using SciPy's
    matrix exponential avoids divisions by small angles and a dead zone.
    """
    generator = np.zeros((6, 6))
    generator[:3, :3] = np.cross(vector, np.eye(3)).T
    generator[:3, 3:] = np.eye(3)
    return np.asarray(expm(generator)[:3, 3:])


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


def interpolate_pose(left: object, right: object, fraction: object) -> np.ndarray:
    """Evaluate H_left Exp(fraction Log(H_left^-1 H_right)).

    Both poses must use the same observer frame and fraction must be a finite
    real number in [0, 1]. Constant material strain yields a circular centerline
    under pure bending; a common rigid motion leaves relative strain unchanged.
    """
    coordinate = require_finite_float(fraction, "material fraction")
    if not 0.0 <= coordinate <= 1.0:
        raise ValueError("material fraction must lie in [0, 1]")
    left_pose, right_pose = _rigid_pose(left), _rigid_pose(right)
    relative = np.linalg.solve(left_pose, right_pose)
    return _rigid_pose(left_pose @ exp_twist(coordinate * log_pose(relative)))


__all__ = ()
