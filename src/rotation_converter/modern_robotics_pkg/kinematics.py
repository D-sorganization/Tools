"""Forward/Inverse Kinematics and Jacobians.

Product of Exponentials FK, Newton-Raphson IK, Space/Body Jacobians.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rotation_converter._contracts import ensure, require, require_finite

from .se3 import (
    Adjoint,
    MatrixExp6,
    MatrixLog6,
    TransInv,
    VecTose3,
    _Adjoint,
    se3ToVec,
)
from .so3 import so3ToVec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forward Kinematics — Product of Exponentials
# ---------------------------------------------------------------------------


def FKinSpace(M: Any, Slist: Any, thetalist: Any) -> np.ndarray:
    """Forward kinematics in the space frame (product of exponentials)."""
    M = np.asarray(M, dtype=float)
    Slist = np.asarray(Slist, dtype=float)
    thetalist = np.asarray(thetalist, dtype=float)
    require(M.shape == (4, 4), "M must be 4x4")
    require_finite(M, "M")
    require(Slist.shape[0] == 6, "Slist must have 6 rows")
    require_finite(Slist, "Slist")
    n = Slist.shape[1]
    require(thetalist.shape == (n,), f"thetalist must have {n} elements")
    require_finite(thetalist, "thetalist")

    T = np.eye(4)
    for i in range(n):
        se3 = VecTose3(Slist[:, i]) * thetalist[i]
        T = T @ MatrixExp6(se3)
    T = T @ M

    ensure(abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-9, "result must be SE(3)")
    return T


def FKinBody(M: Any, Blist: Any, thetalist: Any) -> np.ndarray:
    """Forward kinematics in the body frame (product of exponentials)."""
    M = np.asarray(M, dtype=float)
    Blist = np.asarray(Blist, dtype=float)
    thetalist = np.asarray(thetalist, dtype=float)
    require(M.shape == (4, 4), "M must be 4x4")
    require_finite(M, "M")
    require(Blist.shape[0] == 6, "Blist must have 6 rows")
    require_finite(Blist, "Blist")
    n = Blist.shape[1]
    require(thetalist.shape == (n,), f"thetalist must have {n} elements")
    require_finite(thetalist, "thetalist")

    T = M.copy()
    for i in range(n):
        se3 = VecTose3(Blist[:, i]) * thetalist[i]
        T = T @ MatrixExp6(se3)

    ensure(abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-9, "result must be SE(3)")
    return T  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Jacobians
# ---------------------------------------------------------------------------


def JacobianSpace(Slist: Any, thetalist: Any) -> np.ndarray:
    """Compute the space Jacobian for a serial chain."""
    Slist = np.asarray(Slist, dtype=float)
    thetalist = np.asarray(thetalist, dtype=float)
    require(Slist.ndim == 2 and Slist.shape[0] == 6, "Slist must be 6xn")
    require_finite(Slist, "Slist")
    n = Slist.shape[1]
    require(thetalist.shape == (n,), f"thetalist must have {n} elements")
    require_finite(thetalist, "thetalist")

    Js = np.copy(Slist)
    T = np.eye(4)
    for i in range(1, n):
        se3 = VecTose3(Slist[:, i - 1]) * thetalist[i - 1]
        T = T @ MatrixExp6(se3)
        Js[:, i] = _Adjoint(T) @ Slist[:, i]

    ensure(Js.shape == (6, n), "Jacobian must be 6xn")
    return Js


def JacobianBody(Blist: Any, thetalist: Any) -> np.ndarray:
    """Compute the body Jacobian for a serial chain."""
    Blist = np.asarray(Blist, dtype=float)
    thetalist = np.asarray(thetalist, dtype=float)
    require(Blist.ndim == 2 and Blist.shape[0] == 6, "Blist must be 6xn")
    require_finite(Blist, "Blist")
    n = Blist.shape[1]
    require(thetalist.shape == (n,), f"thetalist must have {n} elements")
    require_finite(thetalist, "thetalist")

    Jb = np.copy(Blist)
    T = np.eye(4)
    for i in range(n - 2, -1, -1):
        se3 = VecTose3(-Blist[:, i + 1]) * thetalist[i + 1]
        T = T @ MatrixExp6(se3)
        Jb[:, i] = _Adjoint(T) @ Blist[:, i]

    ensure(Jb.shape == (6, n), "Jacobian must be 6xn")
    return Jb


# ---------------------------------------------------------------------------
# Inverse Kinematics — Newton-Raphson
# ---------------------------------------------------------------------------


def IKinBody(
    Blist: Any,
    M: Any,
    T_desired: Any,
    thetalist0: Any,
    eomg: float = 1e-4,
    ev: float = 1e-4,
    max_iter: int = 100,
) -> tuple[np.ndarray, bool]:
    """Iterative inverse kinematics using Newton-Raphson in the body frame."""
    if not (eomg is not None):
        raise ValueError("eomg must be provided")
    Blist = np.asarray(Blist, dtype=float)
    M = np.asarray(M, dtype=float)
    T_desired = np.asarray(T_desired, dtype=float)
    thetalist = np.asarray(thetalist0, dtype=float).copy()

    require(M.shape == (4, 4), "M must be 4x4")
    require_finite(M, "M")
    require(T_desired.shape == (4, 4), "T_desired must be 4x4")
    require_finite(T_desired, "T_desired")
    require(Blist.ndim == 2 and Blist.shape[0] == 6, "Blist must be 6xn")
    require_finite(Blist, "Blist")
    require_finite(thetalist, "thetalist0")
    require(eomg > 0, "angular tolerance must be positive", eomg)
    require(ev > 0, "linear tolerance must be positive", ev)
    require(max_iter > 0, "max_iter must be positive", max_iter)

    for _ in range(max_iter):
        T_current = FKinBody(M, Blist, thetalist)
        T_error = TransInv(T_current) @ T_desired
        Vb = se3ToVec(MatrixLog6(T_error))
        omega_err = np.linalg.norm(Vb[:3])
        v_err = np.linalg.norm(Vb[3:])

        if omega_err < eomg and v_err < ev:
            return thetalist, True

        Jb = JacobianBody(Blist, thetalist)
        thetalist = thetalist + np.linalg.lstsq(Jb, Vb, rcond=None)[0]

    return thetalist, False


def IKinSpace(Slist: Any, M: Any, T: Any, thetalist0: Any, eomg: float, ev: float) -> tuple[np.ndarray, bool]:
    """Computes inverse kinematics in the space frame for an open chain robot."""
    if not (Slist is not None):
        raise ValueError("Slist must be provided")
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Tsb = FKinSpace(M, Slist, thetalist)
    Vs = np.dot(Adjoint(Tsb), se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T))))
    err = (
        np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg
        or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    )
    while err and i < maxiterations:
        thetalist = thetalist + np.dot(
            np.linalg.pinv(JacobianSpace(Slist, thetalist)), Vs
        )
        i = i + 1
        Tsb = FKinSpace(M, Slist, thetalist)
        Vs = np.dot(Adjoint(Tsb), se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T))))
        err = (
            np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg
            or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
        )
    return (thetalist, not err)
