"""Utility functions — normalization, projection, distance, and tests.

Normalize, RotInv, AxisAng3, AxisAng6, ScrewToAxis,
ProjectToSO3, ProjectToSE3, DistanceToSO3, DistanceToSE3,
TestIfSO3, TestIfSE3.
"""

from __future__ import annotations

import numpy as np

from ._helpers import _near_zero
from .se3 import RpToTrans, TransToRp
from .so3 import VecToso3


def Normalize(V: np.ndarray) -> np.ndarray:
    """Normalizes a vector."""
    return V / np.linalg.norm(V)


def RotInv(R: np.ndarray) -> np.ndarray:
    """Inverts a rotation matrix."""
    return np.array(R).T


def AxisAng3(expc3: np.ndarray) -> tuple[np.ndarray, float]:
    """Converts a 3-vector of exponential coordinates into axis-angle form."""
    return (Normalize(expc3), np.linalg.norm(expc3))


def ScrewToAxis(q: np.ndarray, s: np.ndarray, h: float) -> np.ndarray:
    """Converts a parametric screw axis description to a normalized screw axis."""
    return np.r_[s, np.cross(q, s) + np.dot(h, s)]


def AxisAng6(expc6: np.ndarray) -> tuple[np.ndarray, float]:
    """Converts a 6-vector of exponential coordinates into screw axis-angle form."""
    theta = np.linalg.norm([expc6[0], expc6[1], expc6[2]])
    if _near_zero(theta):
        theta = np.linalg.norm([expc6[3], expc6[4], expc6[5]])
    return (np.array(expc6 / theta), theta)


def ProjectToSO3(mat: np.ndarray) -> np.ndarray:
    """Returns a projection of mat into SO(3) via SVD."""
    U, s, Vh = np.linalg.svd(mat)
    R = np.dot(U, Vh)
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    return R


def ProjectToSE3(mat: np.ndarray) -> np.ndarray:
    """Returns a projection of mat into SE(3) via SVD."""
    mat = np.array(mat)
    return RpToTrans(ProjectToSO3(mat[:3, :3]), mat[:3, 3])


def DistanceToSO3(mat: np.ndarray) -> float:
    """Returns the Frobenius norm distance of mat from the SO(3) manifold."""
    if np.linalg.det(mat) > 0:
        return np.linalg.norm(np.dot(np.array(mat).T, mat) - np.eye(3))
    else:
        return 1e9


def DistanceToSE3(mat: np.ndarray) -> float:
    """Returns the Frobenius norm distance of mat from the SE(3) manifold."""
    matR = np.array(mat)[0:3, 0:3]
    if np.linalg.det(matR) > 0:
        return np.linalg.norm(
            np.r_[
                np.c_[np.dot(np.transpose(matR), matR), np.zeros((3, 1))],
                [np.array(mat)[3, :]],
            ]
            - np.eye(4)
        )
    else:
        return 1e9


def TestIfSO3(mat: np.ndarray) -> bool:
    """Returns true if mat is close to or on the manifold SO(3)."""
    return abs(DistanceToSO3(mat)) < 1e-3


def TestIfSE3(mat: np.ndarray) -> bool:
    """Returns true if mat is close to or on the manifold SE(3)."""
    return abs(DistanceToSE3(mat)) < 1e-3
