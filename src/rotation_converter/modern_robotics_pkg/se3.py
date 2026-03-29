"""SE(3) — Rigid body transformation helpers.

Functions follow Lynch & Park textbook naming conventions:
- VecTose3, se3ToVec, TransToRp, RpToTrans, TransInv, Adjoint
- MatrixExp6, MatrixLog6
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rotation_converter._contracts import ensure, require, require_finite

from ._helpers import _near_zero
from .so3 import MatrixExp3, MatrixLog3, VecToso3, so3ToVec


def VecTose3(V: Any) -> np.ndarray:
    """Convert a 6-vector spatial velocity to a 4x4 se(3) matrix."""
    V = np.asarray(V, dtype=float)
    require(V.shape == (6,), "spatial velocity must have 6 elements", V.shape)
    M = np.zeros((4, 4))
    M[:3, :3] = VecToso3(V[:3])
    M[:3, 3] = V[3:]
    return M


def se3ToVec(se3mat: Any) -> np.ndarray:
    """Extract the 6-vector from a 4x4 se(3) matrix."""
    se3mat = np.asarray(se3mat, dtype=float)
    require(se3mat.shape == (4, 4), "se(3) matrix must be 4x4")
    return np.concatenate([so3ToVec(se3mat[:3, :3]), se3mat[:3, 3]])


def TransToRp(T: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract rotation matrix R and position vector p from SE(3) matrix T."""
    T = np.asarray(T, dtype=float)
    require(T.shape == (4, 4), "transform must be 4x4")
    return T[:3, :3].copy(), T[:3, 3].copy()


def RpToTrans(R: Any, p: Any) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from rotation matrix R and position p."""
    R = np.asarray(R, dtype=float)
    p = np.asarray(p, dtype=float)
    require(R.shape == (3, 3), "R must be 3x3")
    require(p.shape == (3,), "p must have 3 elements")
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def TransInv(T: Any) -> np.ndarray:
    """Compute the inverse of an SE(3) transformation matrix."""
    T = np.asarray(T, dtype=float)
    require(T.shape == (4, 4), "transform must be 4x4")
    R, p = TransToRp(T)
    Rt = R.T
    T_inv = np.eye(4)
    T_inv[:3, :3] = Rt
    T_inv[:3, 3] = -Rt @ p
    return T_inv


def _Adjoint(T: Any) -> np.ndarray:
    """6x6 adjoint representation of SE(3) matrix T (internal helper)."""
    T = np.asarray(T, dtype=float)
    R, p = TransToRp(T)
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = VecToso3(p) @ R
    return Ad


def Adjoint(T: Any) -> np.ndarray:
    """Computes the adjoint representation of a homogeneous transformation matrix.

    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T
    """
    R, p = TransToRp(T)
    return np.r_[np.c_[R, np.zeros((3, 3))], np.c_[np.dot(VecToso3(p), R), R]]


def MatrixExp6(se3mat: Any) -> np.ndarray:
    """Compute the matrix exponential of an se(3) matrix -> SE(3)."""
    se3mat = np.asarray(se3mat, dtype=float)
    require(se3mat.shape == (4, 4), "se(3) matrix must be 4x4")
    require_finite(se3mat, "se(3) matrix")

    omega_mat = se3mat[:3, :3]
    omega_vec = so3ToVec(omega_mat)
    v = se3mat[:3, 3]
    theta = float(np.linalg.norm(omega_vec))

    T = np.eye(4)

    if _near_zero(theta):
        T[:3, 3] = v
        return T

    omega_hat = omega_mat / theta
    v_unit = v / theta

    R = MatrixExp3(omega_mat)
    G = (
        np.eye(3) * theta
        + (1.0 - math.cos(theta)) * omega_hat
        + (theta - math.sin(theta)) * (omega_hat @ omega_hat)
    )
    T[:3, :3] = R
    T[:3, 3] = G @ v_unit

    ensure(abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-9, "result must be SE(3)")
    return T


def MatrixLog6(T: Any) -> np.ndarray:
    """Compute the matrix logarithm of SE(3) -> se(3)."""
    T = np.asarray(T, dtype=float)
    require(T.shape == (4, 4), "SE(3) matrix must be 4x4")
    require_finite(T, "SE(3) matrix")

    R, p = TransToRp(T)
    omega_mat = MatrixLog3(R)
    omega_vec = so3ToVec(omega_mat)
    theta = float(np.linalg.norm(omega_vec))

    result = np.zeros((4, 4))

    if _near_zero(theta):
        result[:3, 3] = p
        return result

    omega_hat = omega_mat / theta
    G_inv = (
        np.eye(3) / theta
        - omega_hat / 2.0
        + (1.0 / theta - 1.0 / (2.0 * math.tan(theta / 2.0))) * (omega_hat @ omega_hat)
    )

    result[:3, :3] = omega_mat
    result[:3, 3] = (G_inv @ p) * theta
    return result
