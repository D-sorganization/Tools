"""SO(3) — 3D Rotation helpers.

Functions follow Lynch & Park textbook naming conventions:
- VecToso3, so3ToVec, MatrixExp3, MatrixLog3
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rotation_converter._contracts import ensure, require, require_finite

from ._helpers import _near_zero


def VecToso3(omega: Any) -> np.ndarray:
    """Convert a 3-vector to a 3x3 skew-symmetric matrix [omega]."""
    omega = np.asarray(omega, dtype=float)
    require(omega.shape == (3,), "omega must have 3 elements", omega.shape)
    return np.array(
        [
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0],
        ]
    )


def so3ToVec(so3mat: Any) -> np.ndarray:
    """Extract the 3-vector from a 3x3 skew-symmetric matrix."""
    so3mat = np.asarray(so3mat, dtype=float)
    require(so3mat.shape == (3, 3), "so3 matrix must be 3x3", so3mat.shape)
    return np.array([so3mat[2, 1], so3mat[0, 2], so3mat[1, 0]])


def MatrixExp3(so3mat: Any) -> np.ndarray:
    """Compute the matrix exponential of an so(3) matrix -> SO(3).

    Implements Rodrigues' formula.
    """
    so3mat = np.asarray(so3mat, dtype=float)
    require(so3mat.shape == (3, 3), "so(3) matrix must be 3x3")
    require_finite(so3mat, "so(3) matrix")

    omega_vec = so3ToVec(so3mat)
    theta = float(np.linalg.norm(omega_vec))

    if _near_zero(theta):
        return np.eye(3)

    omega_hat = so3mat / theta
    R = (
        np.eye(3)
        + math.sin(theta) * omega_hat
        + (1.0 - math.cos(theta)) * (omega_hat @ omega_hat)
    )

    ensure(abs(np.linalg.det(R) - 1.0) < 1e-9, "result must be SO(3)")
    return R  # type: ignore[no-any-return]


def MatrixLog3(R: Any) -> np.ndarray:
    """Compute the matrix logarithm of SO(3) -> so(3)."""
    R = np.asarray(R, dtype=float)
    require(R.shape == (3, 3), "rotation matrix must be 3x3")
    require_finite(R, "rotation matrix")

    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)

    if _near_zero(cos_theta - 1.0):
        return np.zeros((3, 3))

    if _near_zero(cos_theta + 1.0):
        theta = math.pi
        RpI = R + np.eye(3)
        col_norms = [np.linalg.norm(RpI[:, i]) for i in range(3)]
        best_col = int(np.argmax(col_norms))
        omega = RpI[:, best_col] / np.linalg.norm(RpI[:, best_col])
        return VecToso3(omega * theta)

    theta = math.acos(cos_theta)
    omega_hat = (R - R.T) / (2.0 * math.sin(theta))
    return omega_hat * theta  # type: ignore[no-any-return]
