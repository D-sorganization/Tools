"""Modern Robotics (Lynch & Park) core algorithms.

Implements key functions from "Modern Robotics: Mechanics, Planning,
and Control" by Kevin Lynch and Frank Park.

Functions follow the textbook naming conventions for discoverability:
- SO(3): VecToso3, so3ToVec, MatrixExp3, MatrixLog3
- SE(3): VecTose3, se3ToVec, MatrixExp6, MatrixLog6, TransToRp, RpToTrans, TransInv
- FK: FKinSpace, FKinBody (product of exponentials)
- IK: IKinBody (iterative Newton-Raphson)
- Jacobians: JacobianSpace, JacobianBody
- Trajectory: ScrewTrajectory

Architecture:
- Reuses rotation_converter.core helpers where possible (DRY)
- DbC preconditions on all public functions
- Postconditions verify SE(3)/SO(3) membership

References:
    Lynch, K.M. & Park, F.C. (2017). Modern Robotics: Mechanics,
    Planning, and Control. Cambridge University Press.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rotation_converter._contracts import ensure, require, require_finite

# ---------------------------------------------------------------------------
# Internal helpers (DRY — shared across multiple functions)
# ---------------------------------------------------------------------------


def _near_zero(val: float, tol: float = 1e-12) -> bool:
    """Check if a scalar is effectively zero."""
    return abs(val) < tol


# ===========================================================================
# SO(3) — 3D Rotation helpers
# ===========================================================================


def VecToso3(omega: Any) -> np.ndarray:
    """Convert a 3-vector to a 3x3 skew-symmetric matrix [omega].

    Args:
        omega: 3-vector angular velocity.

    Returns:
        3x3 skew-symmetric matrix in so(3).
    """
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
    """Extract the 3-vector from a 3x3 skew-symmetric matrix.

    Args:
        so3mat: 3x3 skew-symmetric matrix.

    Returns:
        3-vector omega.
    """
    so3mat = np.asarray(so3mat, dtype=float)
    require(so3mat.shape == (3, 3), "so3 matrix must be 3x3", so3mat.shape)
    return np.array([so3mat[2, 1], so3mat[0, 2], so3mat[1, 0]])


def MatrixExp3(so3mat: Any) -> np.ndarray:
    """Compute the matrix exponential of an so(3) matrix -> SO(3).

    Implements Rodrigues' formula: if so3mat = [omega_hat]*theta,
    R = I + sin(theta)*[omega_hat] + (1-cos(theta))*[omega_hat]^2

    Args:
        so3mat: 3x3 so(3) matrix (skew-symmetric * angle).

    Returns:
        3x3 rotation matrix in SO(3).
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
    """Compute the matrix logarithm of SO(3) -> so(3).

    Args:
        R: 3x3 rotation matrix in SO(3).

    Returns:
        3x3 skew-symmetric matrix in so(3) such that exp(result) = R.
    """
    R = np.asarray(R, dtype=float)
    require(R.shape == (3, 3), "rotation matrix must be 3x3")
    require_finite(R, "rotation matrix")

    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)

    if _near_zero(cos_theta - 1.0):
        # theta ≈ 0 -> near identity
        return np.zeros((3, 3))

    if _near_zero(cos_theta + 1.0):
        # theta ≈ pi
        theta = math.pi
        # Find the column of R + I with largest norm
        RpI = R + np.eye(3)
        col_norms = [np.linalg.norm(RpI[:, i]) for i in range(3)]
        best_col = int(np.argmax(col_norms))
        omega = RpI[:, best_col] / np.linalg.norm(RpI[:, best_col])
        return VecToso3(omega * theta)

    theta = math.acos(cos_theta)
    omega_hat = (R - R.T) / (2.0 * math.sin(theta))
    return omega_hat * theta  # type: ignore[no-any-return]


# ===========================================================================
# SE(3) — Rigid body transformation helpers
# ===========================================================================


def VecTose3(V: Any) -> np.ndarray:
    """Convert a 6-vector spatial velocity to a 4x4 se(3) matrix.

    V = [omega; v] -> [[omega] v; 0 0]

    Args:
        V: 6-vector [omega_1, omega_2, omega_3, v_1, v_2, v_3].

    Returns:
        4x4 matrix in se(3).
    """
    V = np.asarray(V, dtype=float)
    require(V.shape == (6,), "spatial velocity must have 6 elements", V.shape)
    M = np.zeros((4, 4))
    M[:3, :3] = VecToso3(V[:3])
    M[:3, 3] = V[3:]
    return M


def se3ToVec(se3mat: Any) -> np.ndarray:
    """Extract the 6-vector from a 4x4 se(3) matrix.

    Args:
        se3mat: 4x4 matrix in se(3).

    Returns:
        6-vector [omega; v].
    """
    se3mat = np.asarray(se3mat, dtype=float)
    require(se3mat.shape == (4, 4), "se(3) matrix must be 4x4")
    return np.concatenate([so3ToVec(se3mat[:3, :3]), se3mat[:3, 3]])


def TransToRp(T: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract rotation matrix R and position vector p from SE(3) matrix T.

    Args:
        T: 4x4 homogeneous transformation matrix.

    Returns:
        Tuple of (R, p) where R is 3x3 and p is 3-vector.
    """
    T = np.asarray(T, dtype=float)
    require(T.shape == (4, 4), "transform must be 4x4")
    return T[:3, :3].copy(), T[:3, 3].copy()


def RpToTrans(R: Any, p: Any) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from rotation matrix R and position p.

    Args:
        R: 3x3 rotation matrix.
        p: 3-vector position.

    Returns:
        4x4 homogeneous transformation matrix.
    """
    R = np.asarray(R, dtype=float)
    p = np.asarray(p, dtype=float)
    require(R.shape == (3, 3), "R must be 3x3")
    require(p.shape == (3,), "p must have 3 elements")
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def TransInv(T: Any) -> np.ndarray:
    """Compute the inverse of an SE(3) transformation matrix.

    Uses the efficient formula: T^-1 = [R^T  -R^T*p; 0 1].

    Args:
        T: 4x4 SE(3) matrix.

    Returns:
        4x4 inverse transformation.
    """
    T = np.asarray(T, dtype=float)
    require(T.shape == (4, 4), "transform must be 4x4")
    R, p = TransToRp(T)
    Rt = R.T
    T_inv = np.eye(4)
    T_inv[:3, :3] = Rt
    T_inv[:3, 3] = -Rt @ p
    return T_inv


def _Adjoint(T: Any) -> np.ndarray:
    """6x6 adjoint representation of SE(3) matrix T.

    Ad_T = [R  0; [p]R  R]

    Used internally for Jacobian computation.
    """
    T = np.asarray(T, dtype=float)
    R, p = TransToRp(T)
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = VecToso3(p) @ R
    return Ad


def MatrixExp6(se3mat: Any) -> np.ndarray:
    """Compute the matrix exponential of an se(3) matrix -> SE(3).

    If se3mat = [S]*theta where S = [omega; v] with ||omega||=1:
        T = [exp([omega]*theta)   G(theta)*v; 0  1]
    If omega = 0 (pure translation):
        T = [I  v*theta; 0  1]

    Args:
        se3mat: 4x4 matrix in se(3).

    Returns:
        4x4 matrix in SE(3).
    """
    se3mat = np.asarray(se3mat, dtype=float)
    require(se3mat.shape == (4, 4), "se(3) matrix must be 4x4")
    require_finite(se3mat, "se(3) matrix")

    omega_mat = se3mat[:3, :3]
    omega_vec = so3ToVec(omega_mat)
    v = se3mat[:3, 3]
    theta = float(np.linalg.norm(omega_vec))

    T = np.eye(4)

    if _near_zero(theta):
        # Pure translation
        T[:3, 3] = v
        return T

    # omega_mat = [omega_hat] * theta, v_full = v * theta
    omega_hat = omega_mat / theta  # unit skew-symmetric
    v_unit = v / theta  # v component of the unit twist

    R = MatrixExp3(omega_mat)
    # G(theta) from Lynch & Park Eq. 3.84
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
    """Compute the matrix logarithm of SE(3) -> se(3).

    Args:
        T: 4x4 SE(3) matrix.

    Returns:
        4x4 matrix in se(3) such that exp(result) = T.
    """
    T = np.asarray(T, dtype=float)
    require(T.shape == (4, 4), "SE(3) matrix must be 4x4")
    require_finite(T, "SE(3) matrix")

    R, p = TransToRp(T)
    omega_mat = MatrixLog3(R)
    omega_vec = so3ToVec(omega_mat)
    theta = float(np.linalg.norm(omega_vec))

    result = np.zeros((4, 4))

    if _near_zero(theta):
        # Pure translation
        result[:3, 3] = p
        return result

    omega_hat = omega_mat / theta
    G_inv = (
        np.eye(3) / theta
        - omega_hat / 2.0
        + (1.0 / theta - 1.0 / (2.0 * math.tan(theta / 2.0))) * (omega_hat @ omega_hat)
    )

    result[:3, :3] = omega_mat
    # G_inv @ p = v_unit; the se(3) matrix stores v_unit * theta
    result[:3, 3] = (G_inv @ p) * theta
    return result


# ===========================================================================
# Forward Kinematics — Product of Exponentials
# ===========================================================================


def FKinSpace(M: Any, Slist: Any, thetalist: Any) -> np.ndarray:
    """Forward kinematics in the space frame (product of exponentials).

    T = exp([S1]*θ1) * exp([S2]*θ2) * ... * exp([Sn]*θn) * M

    Args:
        M: 4x4 home configuration of end-effector.
        Slist: 6×n matrix of space-frame screw axes (columns).
        thetalist: n-vector of joint angles.

    Returns:
        4x4 SE(3) end-effector configuration.
    """
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
    """Forward kinematics in the body frame (product of exponentials).

    T = M * exp([B1]*θ1) * exp([B2]*θ2) * ... * exp([Bn]*θn)

    Args:
        M: 4x4 home configuration of end-effector.
        Blist: 6×n matrix of body-frame screw axes (columns).
        thetalist: n-vector of joint angles.

    Returns:
        4x4 SE(3) end-effector configuration.
    """
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


# ===========================================================================
# Jacobians
# ===========================================================================


def JacobianSpace(Slist: Any, thetalist: Any) -> np.ndarray:
    """Compute the space Jacobian for a serial chain.

    J_s[:,i] = Ad_{exp([S1]θ1)...exp([Si-1]θi-1)} * Si

    Args:
        Slist: 6×n space screw axes.
        thetalist: n joint angles.

    Returns:
        6×n space Jacobian matrix.
    """
    Slist = np.asarray(Slist, dtype=float)
    thetalist = np.asarray(thetalist, dtype=float)
    require(Slist.ndim == 2 and Slist.shape[0] == 6, "Slist must be 6×n")
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

    ensure(Js.shape == (6, n), "Jacobian must be 6×n")
    return Js


def JacobianBody(Blist: Any, thetalist: Any) -> np.ndarray:
    """Compute the body Jacobian for a serial chain.

    J_b[:,i] = Ad_{exp(-[Bn]θn)...exp(-[Bi+1]θi+1)} * Bi

    Args:
        Blist: 6×n body screw axes.
        thetalist: n joint angles.

    Returns:
        6×n body Jacobian matrix.
    """
    Blist = np.asarray(Blist, dtype=float)
    thetalist = np.asarray(thetalist, dtype=float)
    require(Blist.ndim == 2 and Blist.shape[0] == 6, "Blist must be 6×n")
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

    ensure(Jb.shape == (6, n), "Jacobian must be 6×n")
    return Jb


# ===========================================================================
# Inverse Kinematics — Newton-Raphson
# ===========================================================================


def IKinBody(
    Blist: Any,
    M: Any,
    T_desired: Any,
    thetalist0: Any,
    eomg: float = 1e-4,
    ev: float = 1e-4,
    max_iter: int = 100,
) -> tuple[np.ndarray, bool]:
    """Iterative inverse kinematics using Newton-Raphson in the body frame.

    Solves for joint angles that achieve a desired end-effector pose.

    Args:
        Blist: 6×n body screw axes.
        M: 4x4 home configuration.
        T_desired: 4x4 desired end-effector SE(3) pose.
        thetalist0: n-vector initial guess for joint angles.
        eomg: Angular error tolerance (rad).
        ev: Linear error tolerance.
        max_iter: Maximum iterations.

    Returns:
        Tuple of (thetalist, success) where success indicates convergence.
    """
    Blist = np.asarray(Blist, dtype=float)
    M = np.asarray(M, dtype=float)
    T_desired = np.asarray(T_desired, dtype=float)
    thetalist = np.asarray(thetalist0, dtype=float).copy()

    require(M.shape == (4, 4), "M must be 4x4")
    require_finite(M, "M")
    require(T_desired.shape == (4, 4), "T_desired must be 4x4")
    require_finite(T_desired, "T_desired")
    require(Blist.ndim == 2 and Blist.shape[0] == 6, "Blist must be 6×n")
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
        # Damped least-squares for robustness near singularities
        thetalist = thetalist + np.linalg.lstsq(Jb, Vb, rcond=None)[0]

    return thetalist, False


# ===========================================================================
# Trajectory Generation
# ===========================================================================


def _cubic_time_scaling(t: float) -> float:
    """Cubic polynomial time scaling s(t) with s(0)=0, s(1)=1."""
    return 3.0 * t**2 - 2.0 * t**3


def _quintic_time_scaling(t: float) -> float:
    """Quintic polynomial time scaling s(t) with s(0)=0, s(1)=1."""
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def ScrewTrajectory(
    Xstart: Any,
    Xend: Any,
    Tf: float,
    N: int,
    method: int = 3,
) -> list[np.ndarray]:
    """Generate a trajectory as a list of SE(3) matrices via screw motion.

    Interpolates along the screw axis between Xstart and Xend using
    the matrix exponential and logarithm.

    Args:
        Xstart: 4x4 starting SE(3) configuration.
        Xend: 4x4 ending SE(3) configuration.
        Tf: Total time of trajectory.
        N: Number of trajectory points (including start and end).
        method: 3 for cubic time scaling, 5 for quintic.

    Returns:
        List of N 4x4 SE(3) matrices along the trajectory.
    """
    Xstart = np.asarray(Xstart, dtype=float)
    Xend = np.asarray(Xend, dtype=float)
    require(Xstart.shape == (4, 4), "Xstart must be 4x4")
    require(Xend.shape == (4, 4), "Xend must be 4x4")
    require(N >= 2, "N must be >= 2")
    require(Tf > 0, "Tf must be positive")

    time_scale = _cubic_time_scaling if method == 3 else _quintic_time_scaling
    log_delta = MatrixLog6(TransInv(Xstart) @ Xend)

    trajectory: list[np.ndarray] = []
    for i in range(N):
        t_normalized = i / (N - 1)
        s = time_scale(t_normalized)
        T = Xstart @ MatrixExp6(log_delta * s)
        trajectory.append(T)

    ensure(len(trajectory) == N, "trajectory must have N points")
    return trajectory
