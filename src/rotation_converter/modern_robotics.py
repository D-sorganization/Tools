# mypy: ignore-errors
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

import logging
import math
from typing import Any

import numpy as np

from rotation_converter._contracts import ensure, require, require_finite

_logger = logging.getLogger(__name__)

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
    assert eomg is not None, "eomg must be provided"
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
    assert Tf is not None, "Tf must be provided"
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


# ===========================================================================
# Automatic Imported Missing Functions
# ===========================================================================


def Normalize(V):
    """Normalizes a vector

    :param V: A vector
    :return: A unit vector pointing in the same direction as z

    Example Input:
        V = np.array([1, 2, 3])
    Output:
        np.array([0.26726124, 0.53452248, 0.80178373])
    """
    return V / np.linalg.norm(V)


def RotInv(R):
    """Inverts a rotation matrix

    :param R: A rotation matrix
    :return: The inverse of R

    Example Input:
        R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    Output:
        np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]])
    """
    return np.array(R).T


def AxisAng3(expc3):
    """Converts a 3-vector of exponential coordinates for rotation into
    axis-angle form

    :param expc3: A 3-vector of exponential coordinates for rotation
    :return omghat: A unit rotation axis
    :return theta: The corresponding rotation angle

    Example Input:
        expc3 = np.array([1, 2, 3])
    Output:
        (np.array([0.26726124, 0.53452248, 0.80178373]), 3.7416573867739413)
    """
    return (Normalize(expc3), np.linalg.norm(expc3))


def Adjoint(T):
    """Computes the adjoint representation of a homogeneous transformation
    matrix

    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1, 0,  0, 0, 0,  0],
                  [0, 0, -1, 0, 0,  0],
                  [0, 1,  0, 0, 0,  0],
                  [0, 0,  3, 1, 0,  0],
                  [3, 0,  0, 0, 0, -1],
                  [0, 0,  0, 0, 1,  0]])
    """
    R, p = TransToRp(T)
    return np.r_[np.c_[R, np.zeros((3, 3))], np.c_[np.dot(VecToso3(p), R), R]]


def ScrewToAxis(q, s, h):
    """Takes a parametric description of a screw axis and converts it to a
    normalized screw axis

    :param q: A point lying on the screw axis
    :param s: A unit vector in the direction of the screw axis
    :param h: The pitch of the screw axis
    :return: A normalized screw axis described by the inputs

    Example Input:
        q = np.array([3, 0, 0])
        s = np.array([0, 0, 1])
        h = 2
    Output:
        np.array([0, 0, 1, 0, -3, 2])
    """
    return np.r_[s, np.cross(q, s) + np.dot(h, s)]


def AxisAng6(expc6):
    """Converts a 6-vector of exponential coordinates into screw axis-angle
    form

    :param expc6: A 6-vector of exponential coordinates for rigid-body motion
                  S*theta
    :return S: The corresponding normalized screw axis
    :return theta: The distance traveled along/about S

    Example Input:
        expc6 = np.array([1, 0, 0, 1, 2, 3])
    Output:
        (np.array([1.0, 0.0, 0.0, 1.0, 2.0, 3.0]), 1.0)
    """
    theta = np.linalg.norm([expc6[0], expc6[1], expc6[2]])
    if _near_zero(theta):
        theta = np.linalg.norm([expc6[3], expc6[4], expc6[5]])
    return (np.array(expc6 / theta), theta)


def ProjectToSO3(mat):
    """Returns a projection of mat into SO(3)

    :param mat: A matrix near SO(3) to project to SO(3)
    :return: The closest matrix to R that is in SO(3)
    Projects a matrix mat to the closest matrix in SO(3) using singular-value
    decomposition (see
    http://hades.mech.northwestern.edu/index.php/Modern_Robotics_Linear_Algebra_Review).
    This function is only appropriate for matrices close to SO(3).

    Example Input:
        mat = np.array([[ 0.675,  0.150,  0.720],
                        [ 0.370,  0.771, -0.511],
                        [-0.630,  0.619,  0.472]])
    Output:
        np.array([[ 0.67901136,  0.14894516,  0.71885945],
                  [ 0.37320708,  0.77319584, -0.51272279],
                  [-0.63218672,  0.61642804,  0.46942137]])
    """
    U, s, Vh = np.linalg.svd(mat)
    R = np.dot(U, Vh)
    if np.linalg.det(R) < 0:
        # In this case the result may be far from mat.
        R[:, 2] = -R[:, 2]
    return R


def ProjectToSE3(mat):
    """Returns a projection of mat into SE(3)

    :param mat: A 4x4 matrix to project to SE(3)
    :return: The closest matrix to T that is in SE(3)
    Projects a matrix mat to the closest matrix in SE(3) using singular-value
    decomposition (see
    http://hades.mech.northwestern.edu/index.php/Modern_Robotics_Linear_Algebra_Review).
    This function is only appropriate for matrices close to SE(3).

    Example Input:
        mat = np.array([[ 0.675,  0.150,  0.720,  1.2],
                        [ 0.370,  0.771, -0.511,  5.4],
                        [-0.630,  0.619,  0.472,  3.6],
                        [ 0.003,  0.002,  0.010,  0.9]])
    Output:
        np.array([[ 0.67901136,  0.14894516,  0.71885945,  1.2 ],
                  [ 0.37320708,  0.77319584, -0.51272279,  5.4 ],
                  [-0.63218672,  0.61642804,  0.46942137,  3.6 ],
                  [ 0.        ,  0.        ,  0.        ,  1.  ]])
    """
    mat = np.array(mat)
    return RpToTrans(ProjectToSO3(mat[:3, :3]), mat[:3, 3])


def DistanceToSO3(mat):
    """Returns the Frobenius norm to describe the distance of mat from the
    SO(3) manifold

    :param mat: A 3x3 matrix
    :return: A quantity describing the distance of mat from the SO(3)
             manifold
    Computes the distance from mat to the SO(3) manifold using the following
    method:
    If det(mat) <= 0, return a large number.
    If det(mat) > 0, return norm(mat^T.mat - I).

    Example Input:
        mat = np.array([[ 1.0,  0.0,   0.0 ],
                        [ 0.0,  0.1,  -0.95],
                        [ 0.0,  1.0,   0.1 ]])
    Output:
        0.08835
    """
    if np.linalg.det(mat) > 0:
        return np.linalg.norm(np.dot(np.array(mat).T, mat) - np.eye(3))
    else:
        return 1e9


def DistanceToSE3(mat):
    """Returns the Frobenius norm to describe the distance of mat from the
    SE(3) manifold

    :param mat: A 4x4 matrix
    :return: A quantity describing the distance of mat from the SE(3)
              manifold
    Computes the distance from mat to the SE(3) manifold using the following
    method:
    Compute the determinant of matR, the top 3x3 submatrix of mat.
    If det(matR) <= 0, return a large number.
    If det(matR) > 0, replace the top 3x3 submatrix of mat with matR^T.matR,
    and set the first three entries of the fourth column of mat to zero. Then
    return norm(mat - I).

    Example Input:
        mat = np.array([[ 1.0,  0.0,   0.0,   1.2 ],
                        [ 0.0,  0.1,  -0.95,  1.5 ],
                        [ 0.0,  1.0,   0.1,  -0.9 ],
                        [ 0.0,  0.0,   0.1,   0.98 ]])
    Output:
        0.134931
    """
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


def TestIfSO3(mat):
    """Returns true if mat is close to or on the manifold SO(3)

    :param mat: A 3x3 matrix
    :return: True if mat is very close to or in SO(3), false otherwise
    Computes the distance d from mat to the SO(3) manifold using the
    following method:
    If det(mat) <= 0, d = a large number.
    If det(mat) > 0, d = norm(mat^T.mat - I).
    If d is close to zero, return true. Otherwise, return false.

    Example Input:
        mat = np.array([[1.0, 0.0,  0.0 ],
                        [0.0, 0.1, -0.95],
                        [0.0, 1.0,  0.1 ]])
    Output:
        False
    """
    return abs(DistanceToSO3(mat)) < 1e-3


def TestIfSE3(mat):
    """Returns true if mat is close to or on the manifold SE(3)

    :param mat: A 4x4 matrix
    :return: True if mat is very close to or in SE(3), false otherwise
    Computes the distance d from mat to the SE(3) manifold using the
    following method:
    Compute the determinant of the top 3x3 submatrix of mat.
    If det(mat) <= 0, d = a large number.
    If det(mat) > 0, replace the top 3x3 submatrix of mat with mat^T.mat, and
    set the first three entries of the fourth column of mat to zero.
    Then d = norm(T - I).
    If d is close to zero, return true. Otherwise, return false.

    Example Input:
        mat = np.array([[1.0, 0.0,   0.0,  1.2],
                        [0.0, 0.1, -0.95,  1.5],
                        [0.0, 1.0,   0.1, -0.9],
                        [0.0, 0.0,   0.1, 0.98]])
    Output:
        False
    """
    return abs(DistanceToSE3(mat)) < 1e-3


def IKinSpace(Slist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the space frame for an open chain robot

    :param Slist: The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.

    Example Input:
        Slist = np.array([[0, 0,  1,  4, 0,    0],
                          [0, 0,  0,  0, 1,    0],
                          [0, 0, -1, -6, 0, -0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        (np.array([ 1.57073783,  2.99966384,  3.1415342 ]), True)
    """
    assert Slist is not None, "Slist must be provided"
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


def ad(V):
    """Calculate the 6x6 matrix [adV] of the given 6-vector

    :param V: A 6-vector spatial velocity
    :return: The corresponding 6x6 matrix [adV]

    Used to calculate the Lie bracket [V1, V2] = [adV1]V2

    Example Input:
        V = np.array([1, 2, 3, 4, 5, 6])
    Output:
        np.array([[ 0, -3,  2,  0,  0,  0],
                  [ 3,  0, -1,  0,  0,  0],
                  [-2,  1,  0,  0,  0,  0],
                  [ 0, -6,  5,  0, -3,  2],
                  [ 6,  0, -4,  3,  0, -1],
                  [-5,  4,  0, -2,  1,  0]])
    """
    omgmat = VecToso3([V[0], V[1], V[2]])
    return np.r_[
        np.c_[omgmat, np.zeros((3, 3))], np.c_[VecToso3([V[3], V[4], V[5]]), omgmat]
    ]


def InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist):
    """Computes inverse dynamics in the space frame for an open chain robot

    :param thetalist: n-vector of joint variables
    :param dthetalist: n-vector of joint rates
    :param ddthetalist: n-vector of joint accelerations
    :param g: Gravity vector g
    :param Ftip: Spatial force applied by the end-effector expressed in frame
                 {n+1}
    :param Mlist: List of link frames {i} relative to {i-1} at the home
                  position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The n-vector of required joint forces/torques
    This function uses forward-backward Newton-Euler iterations to solve the
    equation:
    taulist = Mlist(thetalist)ddthetalist + c(thetalist,dthetalist) \
              + g(thetalist) + Jtr(thetalist)Ftip

    Example Input (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        ddthetalist = np.array([2, 1.5, 1])
        g = np.array([0, 0, -9.8])
        Ftip = np.array([1, 1, 1, 1, 1, 1])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
    Output:
        np.array([74.69616155, -33.06766016, -3.23057314])
    """
    assert thetalist is not None, "thetalist must be provided"
    n = len(thetalist)
    Mi = np.eye(4)
    Ai = np.zeros((6, n))
    AdTi = [[None]] * (n + 1)
    Vi = np.zeros((6, n + 1))
    Vdi = np.zeros((6, n + 1))
    Vdi[:, 0] = np.r_[[0, 0, 0], -np.array(g)]
    AdTi[n] = Adjoint(TransInv(Mlist[n]))
    Fi = np.array(Ftip).copy()
    taulist = np.zeros(n)
    for i in range(n):
        Mi = np.dot(Mi, Mlist[i])
        Ai[:, i] = np.dot(Adjoint(TransInv(Mi)), np.array(Slist)[:, i])
        AdTi[i] = Adjoint(
            np.dot(MatrixExp6(VecTose3(Ai[:, i] * -thetalist[i])), TransInv(Mlist[i]))
        )
        Vi[:, i + 1] = np.dot(AdTi[i], Vi[:, i]) + Ai[:, i] * dthetalist[i]
        Vdi[:, i + 1] = (
            np.dot(AdTi[i], Vdi[:, i])
            + Ai[:, i] * ddthetalist[i]
            + np.dot(ad(Vi[:, i + 1]), Ai[:, i]) * dthetalist[i]
        )
    for i in range(n - 1, -1, -1):
        Fi = (
            np.dot(np.array(AdTi[i + 1]).T, Fi)
            + np.dot(np.array(Glist[i]), Vdi[:, i + 1])
            - np.dot(
                np.array(ad(Vi[:, i + 1])).T, np.dot(np.array(Glist[i]), Vi[:, i + 1])
            )
        )
        taulist[i] = np.dot(np.array(Fi).T, Ai[:, i])
    return taulist


def MassMatrix(thetalist, Mlist, Glist, Slist):
    """Computes the mass matrix of an open chain robot based on the given
    configuration

    :param thetalist: A list of joint variables
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The numerical inertia matrix M(thetalist) of an n-joint serial
             chain at the given configuration thetalist
    This function calls InverseDynamics n times, each time passing a
    ddthetalist vector with a single element equal to one and all other
    inputs set to zero.
    Each call of InverseDynamics generates a single column, and these columns
    are assembled to create the inertia matrix.

    Example Input (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
    Output:
        np.array([[ 2.25433380e+01, -3.07146754e-01, -7.18426391e-03]
                  [-3.07146754e-01,  1.96850717e+00,  4.32157368e-01]
                  [-7.18426391e-03,  4.32157368e-01,  1.91630858e-01]])
    """
    assert thetalist is not None, "thetalist must be provided"
    n = len(thetalist)
    M = np.zeros((n, n))
    for i in range(n):
        ddthetalist = [0] * n
        ddthetalist[i] = 1
        M[:, i] = InverseDynamics(
            thetalist,
            [0] * n,
            ddthetalist,
            [0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            Mlist,
            Glist,
            Slist,
        )
    return M


def VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist):
    """Computes the Coriolis and centripetal terms in the inverse dynamics of
    an open chain robot

    :param thetalist: A list of joint variables,
    :param dthetalist: A list of joint rates,
    :param Mlist: List of link frames i relative to i-1 at the home position,
    :param Glist: Spatial inertia matrices Gi of the links,
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns.
    :return: The vector c(thetalist,dthetalist) of Coriolis and centripetal
             terms for a given thetalist and dthetalist.
    This function calls InverseDynamics with g = 0, Ftip = 0, and
    ddthetalist = 0.

    Example Input (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
    Output:
        np.array([0.26453118, -0.05505157, -0.00689132])
    """
    return InverseDynamics(
        thetalist,
        dthetalist,
        [0] * len(thetalist),
        [0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        Mlist,
        Glist,
        Slist,
    )


def GravityForces(thetalist, g, Mlist, Glist, Slist):
    """Computes the joint forces/torques an open chain robot requires to
    overcome gravity at its configuration

    :param thetalist: A list of joint variables
    :param g: 3-vector for gravitational acceleration
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return grav: The joint forces/torques required to overcome gravity at
                  thetalist
    This function calls InverseDynamics with Ftip = 0, dthetalist = 0, and
    ddthetalist = 0.

    Example Inputs (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        g = np.array([0, 0, -9.8])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
    Output:
        np.array([28.40331262, -37.64094817, -5.4415892])
    """
    assert thetalist is not None, "thetalist must be provided"
    n = len(thetalist)
    return InverseDynamics(
        thetalist, [0] * n, [0] * n, g, [0, 0, 0, 0, 0, 0], Mlist, Glist, Slist
    )


def EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist):
    """Computes the joint forces/torques an open chain robot requires only to
    create the end-effector force Ftip

    :param thetalist: A list of joint variables
    :param Ftip: Spatial force applied by the end-effector expressed in frame
                 {n+1}
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The joint forces and torques required only to create the
             end-effector force Ftip
    This function calls InverseDynamics with g = 0, dthetalist = 0, and
    ddthetalist = 0.

    Example Input (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        Ftip = np.array([1, 1, 1, 1, 1, 1])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
    Output:
        np.array([1.40954608, 1.85771497, 1.392409])
    """
    assert thetalist is not None, "thetalist must be provided"
    n = len(thetalist)
    return InverseDynamics(
        thetalist, [0] * n, [0] * n, [0, 0, 0], Ftip, Mlist, Glist, Slist
    )


def ForwardDynamics(thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist):
    """Computes forward dynamics in the space frame for an open chain robot

    :param thetalist: A list of joint variables
    :param dthetalist: A list of joint rates
    :param taulist: An n-vector of joint forces/torques
    :param g: Gravity vector g
    :param Ftip: Spatial force applied by the end-effector expressed in frame
                 {n+1}
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The resulting joint accelerations
    This function computes ddthetalist by solving:
    Mlist(thetalist) * ddthetalist = taulist - c(thetalist,dthetalist) \
                                     - g(thetalist) - Jtr(thetalist) * Ftip

    Example Input (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        taulist = np.array([0.5, 0.6, 0.7])
        g = np.array([0, 0, -9.8])
        Ftip = np.array([1, 1, 1, 1, 1, 1])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
    Output:
        np.array([-0.97392907, 25.58466784, -32.91499212])
    """
    return np.dot(
        np.linalg.inv(MassMatrix(thetalist, Mlist, Glist, Slist)),
        np.array(taulist)
        - VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist)
        - GravityForces(thetalist, g, Mlist, Glist, Slist)
        - EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist),
    )


def EulerStep(thetalist, dthetalist, ddthetalist, dt):
    """Compute the joint angles and velocities at the next timestep using            from here
    first order Euler integration

    :param thetalist: n-vector of joint variables
    :param dthetalist: n-vector of joint rates
    :param ddthetalist: n-vector of joint accelerations
    :param dt: The timestep delta t
    :return thetalistNext: Vector of joint variables after dt from first
                           order Euler integration
    :return dthetalistNext: Vector of joint rates after dt from first order
                            Euler integration

    Example Inputs (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        ddthetalist = np.array([2, 1.5, 1])
        dt = 0.1
    Output:
        thetalistNext:
        array([ 0.11,  0.12,  0.13])
        dthetalistNext:
        array([ 0.3 ,  0.35,  0.4 ])
    """
    return thetalist + dt * np.array(dthetalist), dthetalist + dt * np.array(
        ddthetalist
    )


def InverseDynamicsTrajectory(
    thetamat, dthetamat, ddthetamat, g, Ftipmat, Mlist, Glist, Slist
):
    """Calculates the joint forces/torques required to move the serial chain
    along the given trajectory using inverse dynamics

    :param thetamat: An N x n matrix of robot joint variables
    :param dthetamat: An N x n matrix of robot joint velocities
    :param ddthetamat: An N x n matrix of robot joint accelerations
    :param g: Gravity vector g
    :param Ftipmat: An N x 6 matrix of spatial forces applied by the end-
                    effector (If there are no tip forces the user should
                    input a zero and a zero matrix will be used)
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The N x n matrix of joint forces/torques for the specified
             trajectory, where each of the N rows is the vector of joint
             forces/torques at each time step

    Example Inputs (3 Link Robot):
        from __future__ import print_function
        import numpy as np
        import modern_robotics as mr
        # Create a trajectory to follow using functions from Chapter 9
        thetastart =  np.array([0, 0, 0])
        thetaend =  np.array([np.pi / 2, np.pi / 2, np.pi / 2])
        Tf = 3
        N= 1000
        method = 5
        traj = mr.JointTrajectory(thetastart, thetaend, Tf, N, method)
        thetamat = np.array(traj).copy()
        dthetamat = np.zeros((1000,3 ))
        ddthetamat = np.zeros((1000, 3))
        dt = Tf / (N - 1.0)
        for i in range(np.array(traj).shape[0] - 1):
            dthetamat[i + 1, :] = (thetamat[i + 1, :] - thetamat[i, :]) / dt
            ddthetamat[i + 1, :] \
            = (dthetamat[i + 1, :] - dthetamat[i, :]) / dt
        # Initialize robot description (Example with 3 links)
        g =  np.array([0, 0, -9.8])
        Ftipmat = np.ones((N, 6))
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
        taumat \
        = mr.InverseDynamicsTrajectory(thetamat, dthetamat, ddthetamat, g, \
                                       Ftipmat, Mlist, Glist, Slist)
    # Output using matplotlib to plot the joint forces/torques
        Tau1 = taumat[:, 0]
        Tau2 = taumat[:, 1]
        Tau3 = taumat[:, 2]
        timestamp = np.linspace(0, Tf, N)
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            _logger.warning('The result will not be plotted due to a lack of package matplotlib')
        else:
            plt.plot(timestamp, Tau1, label = "Tau1")
            plt.plot(timestamp, Tau2, label = "Tau2")
            plt.plot(timestamp, Tau3, label = "Tau3")
            plt.ylim (-40, 120)
            plt.legend(loc = 'lower right')
            plt.xlabel("Time")
            plt.ylabel("Torque")
            plt.title("Plot of Torque Trajectories")
            plt.show()
    """
    assert thetamat is not None, "thetamat must be provided"
    thetamat = np.array(thetamat).T
    dthetamat = np.array(dthetamat).T
    ddthetamat = np.array(ddthetamat).T
    Ftipmat = np.array(Ftipmat).T
    taumat = np.array(thetamat).copy()
    for i in range(np.array(thetamat).shape[1]):
        taumat[:, i] = InverseDynamics(
            thetamat[:, i],
            dthetamat[:, i],
            ddthetamat[:, i],
            g,
            Ftipmat[:, i],
            Mlist,
            Glist,
            Slist,
        )
    taumat = np.array(taumat).T
    return taumat


def ForwardDynamicsTrajectory(
    thetalist, dthetalist, taumat, g, Ftipmat, Mlist, Glist, Slist, dt, intRes
):
    """Simulates the motion of a serial chain given an open-loop history of
    joint forces/torques

    :param thetalist: n-vector of initial joint variables
    :param dthetalist: n-vector of initial joint rates
    :param taumat: An N x n matrix of joint forces/torques, where each row is
                   the joint effort at any time step
    :param g: Gravity vector g
    :param Ftipmat: An N x 6 matrix of spatial forces applied by the end-
                    effector (If there are no tip forces the user should
                    input a zero and a zero matrix will be used)
    :param Mlist: List of link frames {i} relative to {i-1} at the home
                  position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :param dt: The timestep between consecutive joint forces/torques
    :param intRes: Integration resolution is the number of times integration
                   (Euler) takes places between each time step. Must be an
                   integer value greater than or equal to 1
    :return thetamat: The N x n matrix of robot joint angles resulting from
                      the specified joint forces/torques
    :return dthetamat: The N x n matrix of robot joint velocities
    This function calls a numerical integration procedure that uses
    ForwardDynamics.

    Example Inputs (3 Link Robot):
        from __future__ import print_function
        import numpy as np
        import modern_robotics as mr
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        taumat = np.array([[3.63, -6.58, -5.57], [3.74, -5.55,  -5.5],
                           [4.31, -0.68, -5.19], [5.18,  5.63, -4.31],
                           [5.85,  8.17, -2.59], [5.78,  2.79,  -1.7],
                           [4.99,  -5.3, -1.19], [4.08, -9.41,  0.07],
                           [3.56, -10.1,  0.97], [3.49, -9.41,  1.23]])
        # Initialize robot description (Example with 3 links)
        g = np.array([0, 0, -9.8])
        Ftipmat = np.ones((np.array(taumat).shape[0], 6))
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
        dt = 0.1
        intRes = 8
        thetamat,dthetamat \
        = mr.ForwardDynamicsTrajectory(thetalist, dthetalist, taumat, g, \
                                       Ftipmat, Mlist, Glist, Slist, dt, \
                                       intRes)
    # Output using matplotlib to plot the joint angle/velocities
        theta1 = thetamat[:, 0]
        theta2 = thetamat[:, 1]
        theta3 = thetamat[:, 2]
        dtheta1 = dthetamat[:, 0]
        dtheta2 = dthetamat[:, 1]
        dtheta3 = dthetamat[:, 2]
        N = np.array(taumat).shape[0]
        Tf = np.array(taumat).shape[0] * dt
            timestamp = np.linspace(0, Tf, N)
            try:
                import matplotlib.pyplot as plt
        except ImportError:
            _logger.warning('The result will not be plotted due to a lack of package matplotlib')
        else:
            plt.plot(timestamp, theta1, label = "Theta1")
            plt.plot(timestamp, theta2, label = "Theta2")
            plt.plot(timestamp, theta3, label = "Theta3")
            plt.plot(timestamp, dtheta1, label = "DTheta1")
            plt.plot(timestamp, dtheta2, label = "DTheta2")
            plt.plot(timestamp, dtheta3, label = "DTheta3")
            plt.ylim (-12, 10)
            plt.legend(loc = 'lower right')
            plt.xlabel("Time")
            plt.ylabel("Joint Angles/Velocities")
            plt.title("Plot of Joint Angles and Joint Velocities")
            plt.show()
    """
    assert thetalist is not None, "thetalist must be provided"
    taumat = np.array(taumat).T
    Ftipmat = np.array(Ftipmat).T
    thetamat = taumat.copy().astype(float)
    thetamat[:, 0] = thetalist
    dthetamat = taumat.copy().astype(float)
    dthetamat[:, 0] = dthetalist
    for i in range(np.array(taumat).shape[1] - 1):
        for _j in range(intRes):
            ddthetalist = ForwardDynamics(
                thetalist,
                dthetalist,
                taumat[:, i],
                g,
                Ftipmat[:, i],
                Mlist,
                Glist,
                Slist,
            )
            thetalist, dthetalist = EulerStep(
                thetalist, dthetalist, ddthetalist, 1.0 * dt / intRes
            )
        thetamat[:, i + 1] = thetalist
        dthetamat[:, i + 1] = dthetalist
    thetamat = np.array(thetamat).T
    dthetamat = np.array(dthetamat).T
    return thetamat, dthetamat


def CubicTimeScaling(Tf, t):
    """Computes s(t) for a cubic time scaling

    :param Tf: Total time of the motion in seconds from rest to rest
    :param t: The current time t satisfying 0 < t < Tf
    :return: The path parameter s(t) corresponding to a third-order
             polynomial motion that begins and ends at zero velocity

    Example Input:
        Tf = 2
        t = 0.6
    Output:
        0.216
    """
    return 3 * (1.0 * t / Tf) ** 2 - 2 * (1.0 * t / Tf) ** 3


def QuinticTimeScaling(Tf, t):
    """Computes s(t) for a quintic time scaling

    :param Tf: Total time of the motion in seconds from rest to rest
    :param t: The current time t satisfying 0 < t < Tf
    :return: The path parameter s(t) corresponding to a fifth-order
             polynomial motion that begins and ends at zero velocity and zero
             acceleration

    Example Input:
        Tf = 2
        t = 0.6
    Output:
        0.16308
    """
    return 10 * (1.0 * t / Tf) ** 3 - 15 * (1.0 * t / Tf) ** 4 + 6 * (1.0 * t / Tf) ** 5


def JointTrajectory(thetastart, thetaend, Tf, N, method):
    """Computes a straight-line trajectory in joint space

    :param thetastart: The initial joint variables
    :param thetaend: The final joint variables
    :param Tf: Total time of the motion in seconds from rest to rest
    :param N: The number of points N > 1 (Start and stop) in the discrete
              representation of the trajectory
    :param method: The time-scaling method, where 3 indicates cubic (third-
                   order polynomial) time scaling and 5 indicates quintic
                   (fifth-order polynomial) time scaling
    :return: A trajectory as an N x n matrix, where each row is an n-vector
             of joint variables at an instant in time. The first row is
             thetastart and the Nth row is thetaend . The elapsed time
             between each row is Tf / (N - 1)

    Example Input:
        thetastart = np.array([1, 0, 0, 1, 1, 0.2, 0,1])
        thetaend = np.array([1.2, 0.5, 0.6, 1.1, 2, 2, 0.9, 1])
        Tf = 4
        N = 6
        method = 3
    Output:
        np.array([[     1,     0,      0,      1,     1,    0.2,      0, 1]
                  [1.0208, 0.052, 0.0624, 1.0104, 1.104, 0.3872, 0.0936, 1]
                  [1.0704, 0.176, 0.2112, 1.0352, 1.352, 0.8336, 0.3168, 1]
                  [1.1296, 0.324, 0.3888, 1.0648, 1.648, 1.3664, 0.5832, 1]
                  [1.1792, 0.448, 0.5376, 1.0896, 1.896, 1.8128, 0.8064, 1]
                  [   1.2,   0.5,    0.6,    1.1,     2,      2,    0.9, 1]])
    """
    assert thetastart is not None, "thetastart must be provided"
    N = int(N)
    timegap = Tf / (N - 1.0)
    traj = np.zeros((len(thetastart), N))
    for i in range(N):
        if method == 3:
            s = CubicTimeScaling(Tf, timegap * i)
        else:
            s = QuinticTimeScaling(Tf, timegap * i)
        traj[:, i] = s * np.array(thetaend) + (1 - s) * np.array(thetastart)
    traj = np.array(traj).T
    return traj


def CartesianTrajectory(Xstart, Xend, Tf, N, method):
    """Computes a trajectory as a list of N SE(3) matrices corresponding to
    the origin of the end-effector frame following a straight line

    :param Xstart: The initial end-effector configuration
    :param Xend: The final end-effector configuration
    :param Tf: Total time of the motion in seconds from rest to rest
    :param N: The number of points N > 1 (Start and stop) in the discrete
              representation of the trajectory
    :param method: The time-scaling method, where 3 indicates cubic (third-
                   order polynomial) time scaling and 5 indicates quintic
                   (fifth-order polynomial) time scaling
    :return: The discretized trajectory as a list of N matrices in SE(3)
             separated in time by Tf/(N-1). The first in the list is Xstart
             and the Nth is Xend
    This function is similar to ScrewTrajectory, except the origin of the
    end-effector frame follows a straight line, decoupled from the rotational
    motion.

    Example Input:
        Xstart = np.array([[1, 0, 0, 1],
                           [0, 1, 0, 0],
                           [0, 0, 1, 1],
                           [0, 0, 0, 1]])
        Xend = np.array([[0, 0, 1, 0.1],
                         [1, 0, 0,   0],
                         [0, 1, 0, 4.1],
                         [0, 0, 0,   1]])
        Tf = 5
        N = 4
        method = 5
    Output:
        [np.array([[1, 0, 0, 1]
                   [0, 1, 0, 0]
                   [0, 0, 1, 1]
                   [0, 0, 0, 1]]),
         np.array([[ 0.937, -0.214,  0.277, 0.811]
                   [ 0.277,  0.937, -0.214,     0]
                   [-0.214,  0.277,  0.937, 1.651]
                   [     0,      0,      0,     1]]),
         np.array([[ 0.277, -0.214,  0.937, 0.289]
                   [ 0.937,  0.277, -0.214,     0]
                   [-0.214,  0.937,  0.277, 3.449]
                   [     0,      0,      0,     1]]),
         np.array([[0, 0, 1, 0.1]
                   [1, 0, 0,   0]
                   [0, 1, 0, 4.1]
                   [0, 0, 0,   1]])]
    """
    assert Xstart is not None, "Xstart must be provided"
    N = int(N)
    timegap = Tf / (N - 1.0)
    traj = [[None]] * N
    Rstart, pstart = TransToRp(Xstart)
    Rend, pend = TransToRp(Xend)
    for i in range(N):
        if method == 3:
            s = CubicTimeScaling(Tf, timegap * i)
        else:
            s = QuinticTimeScaling(Tf, timegap * i)
        traj[i] = np.r_[
            np.c_[
                np.dot(
                    Rstart, MatrixExp3(MatrixLog3(np.dot(np.array(Rstart).T, Rend)) * s)
                ),
                s * np.array(pend) + (1 - s) * np.array(pstart),
            ],
            [[0, 0, 0, 1]],
        ]
    return traj


def ComputedTorque(
    thetalist,
    dthetalist,
    eint,
    g,
    Mlist,
    Glist,
    Slist,
    thetalistd,
    dthetalistd,
    ddthetalistd,
    Kp,
    Ki,
    Kd,
):
    """Computes the joint control torques at a particular time instant

    :param thetalist: n-vector of joint variables
    :param dthetalist: n-vector of joint rates
    :param eint: n-vector of the time-integral of joint errors
    :param g: Gravity vector g
    :param Mlist: List of link frames {i} relative to {i-1} at the home
                  position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :param thetalistd: n-vector of reference joint variables
    :param dthetalistd: n-vector of reference joint velocities
    :param ddthetalistd: n-vector of reference joint accelerations
    :param Kp: The feedback proportional gain (identical for each joint)
    :param Ki: The feedback integral gain (identical for each joint)
    :param Kd: The feedback derivative gain (identical for each joint)
    :return: The vector of joint forces/torques computed by the feedback
             linearizing controller at the current instant

    Example Input:
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        eint = np.array([0.2, 0.2, 0.2])
        g = np.array([0, 0, -9.8])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
        thetalistd = np.array([1.0, 1.0, 1.0])
        dthetalistd = np.array([2, 1.2, 2])
        ddthetalistd = np.array([0.1, 0.1, 0.1])
        Kp = 1.3
        Ki = 1.2
        Kd = 1.1
    Output:
        np.array([133.00525246, -29.94223324, -3.03276856])
    """
    assert thetalist is not None, "thetalist must be provided"
    e = np.subtract(thetalistd, thetalist)
    return np.dot(
        MassMatrix(thetalist, Mlist, Glist, Slist),
        Kp * e + Ki * (np.array(eint) + e) + Kd * np.subtract(dthetalistd, dthetalist),
    ) + InverseDynamics(
        thetalist, dthetalist, ddthetalistd, g, [0, 0, 0, 0, 0, 0], Mlist, Glist, Slist
    )


def SimulateControl(
    thetalist,
    dthetalist,
    g,
    Ftipmat,
    Mlist,
    Glist,
    Slist,
    thetamatd,
    dthetamatd,
    ddthetamatd,
    gtilde,
    Mtildelist,
    Gtildelist,
    Kp,
    Ki,
    Kd,
    dt,
    intRes,
):
    """Simulates the computed torque controller over a given desired
    trajectory

    :param thetalist: n-vector of initial joint variables
    :param dthetalist: n-vector of initial joint velocities
    :param g: Actual gravity vector g
    :param Ftipmat: An N x 6 matrix of spatial forces applied by the end-
                    effector (If there are no tip forces the user should
                    input a zero and a zero matrix will be used)
    :param Mlist: Actual list of link frames i relative to i-1 at the home
                  position
    :param Glist: Actual spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :param thetamatd: An Nxn matrix of desired joint variables from the
                      reference trajectory
    :param dthetamatd: An Nxn matrix of desired joint velocities
    :param ddthetamatd: An Nxn matrix of desired joint accelerations
    :param gtilde: The gravity vector based on the model of the actual robot
                   (actual values given above)
    :param Mtildelist: The link frame locations based on the model of the
                       actual robot (actual values given above)
    :param Gtildelist: The link spatial inertias based on the model of the
                       actual robot (actual values given above)
    :param Kp: The feedback proportional gain (identical for each joint)
    :param Ki: The feedback integral gain (identical for each joint)
    :param Kd: The feedback derivative gain (identical for each joint)
    :param dt: The timestep between points on the reference trajectory
    :param intRes: Integration resolution is the number of times integration
                   (Euler) takes places between each time step. Must be an
                   integer value greater than or equal to 1
    :return taumat: An Nxn matrix of the controllers commanded joint forces/
                    torques, where each row of n forces/torques corresponds
                    to a single time instant
    :return thetamat: An Nxn matrix of actual joint angles
    The end of this function plots all the actual and desired joint angles
    using matplotlib and random libraries.

    Example Input:
        from __future__ import print_function
        import numpy as np
        from modern_robotics import JointTrajectory
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        # Initialize robot description (Example with 3 links)
        g = np.array([0, 0, -9.8])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
        dt = 0.01
        # Create a trajectory to follow
        thetaend = np.array([np.pi / 2, np.pi, 1.5 * np.pi])
        Tf = 1
        N = int(1.0 * Tf / dt)
        method = 5
        traj = mr.JointTrajectory(thetalist, thetaend, Tf, N, method)
        thetamatd = np.array(traj).copy()
        dthetamatd = np.zeros((N, 3))
        ddthetamatd = np.zeros((N, 3))
        dt = Tf / (N - 1.0)
        for i in range(np.array(traj).shape[0] - 1):
            dthetamatd[i + 1, :] \
            = (thetamatd[i + 1, :] - thetamatd[i, :]) / dt
            ddthetamatd[i + 1, :] \
            = (dthetamatd[i + 1, :] - dthetamatd[i, :]) / dt
        # Possibly wrong robot description (Example with 3 links)
        gtilde = np.array([0.8, 0.2, -8.8])
        Mhat01 = np.array([[1, 0, 0,   0],
                           [0, 1, 0,   0],
                           [0, 0, 1, 0.1],
                           [0, 0, 0,   1]])
        Mhat12 = np.array([[ 0, 0, 1, 0.3],
                           [ 0, 1, 0, 0.2],
                           [-1, 0, 0,   0],
                           [ 0, 0, 0,   1]])
        Mhat23 = np.array([[1, 0, 0,    0],
                           [0, 1, 0, -0.2],
                           [0, 0, 1,  0.4],
                           [0, 0, 0,    1]])
        Mhat34 = np.array([[1, 0, 0,   0],
                           [0, 1, 0,   0],
                           [0, 0, 1, 0.2],
                           [0, 0, 0,   1]])
        Ghat1 = np.diag([0.1, 0.1, 0.1, 4, 4, 4])
        Ghat2 = np.diag([0.3, 0.3, 0.1, 9, 9, 9])
        Ghat3 = np.diag([0.1, 0.1, 0.1, 3, 3, 3])
        Gtildelist = np.array([Ghat1, Ghat2, Ghat3])
        Mtildelist = np.array([Mhat01, Mhat12, Mhat23, Mhat34])
        Ftipmat = np.ones((np.array(traj).shape[0], 6))
        Kp = 20
        Ki = 10
        Kd = 18
        intRes = 8
        taumat,thetamat \
        = mr.SimulateControl(thetalist, dthetalist, g, Ftipmat, Mlist, \
                             Glist, Slist, thetamatd, dthetamatd, \
                             ddthetamatd, gtilde, Mtildelist, Gtildelist, \
                             Kp, Ki, Kd, dt, intRes)
    """
    assert thetalist is not None, "thetalist must be provided"
    Ftipmat = np.array(Ftipmat).T
    thetamatd = np.array(thetamatd).T
    dthetamatd = np.array(dthetamatd).T
    ddthetamatd = np.array(ddthetamatd).T
    m, n = np.array(thetamatd).shape
    thetacurrent = np.array(thetalist).copy()
    dthetacurrent = np.array(dthetalist).copy()
    eint = np.zeros((m, 1)).reshape(
        m,
    )
    taumat = np.zeros(np.array(thetamatd).shape)
    thetamat = np.zeros(np.array(thetamatd).shape)
    for i in range(n):
        taulist = ComputedTorque(
            thetacurrent,
            dthetacurrent,
            eint,
            gtilde,
            Mtildelist,
            Gtildelist,
            Slist,
            thetamatd[:, i],
            dthetamatd[:, i],
            ddthetamatd[:, i],
            Kp,
            Ki,
            Kd,
        )
        for _j in range(intRes):
            ddthetalist = ForwardDynamics(
                thetacurrent,
                dthetacurrent,
                taulist,
                g,
                Ftipmat[:, i],
                Mlist,
                Glist,
                Slist,
            )
            thetacurrent, dthetacurrent = EulerStep(
                thetacurrent, dthetacurrent, ddthetalist, 1.0 * dt / intRes
            )
        taumat[:, i] = taulist
        thetamat[:, i] = thetacurrent
        eint = np.add(eint, dt * np.subtract(thetamatd[:, i], thetacurrent))
    # Output using matplotlib to plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        links = np.array(thetamat).shape[0]
        N = np.array(thetamat).shape[1]
        Tf = N * dt
        timestamp = np.linspace(0, Tf, N)
        for i in range(links):
            col = [
                np.random.uniform(0, 1),
                np.random.uniform(0, 1),
                np.random.uniform(0, 1),
            ]
            plt.plot(
                timestamp,
                thetamat[i, :],
                "-",
                color=col,
                label=("ActualTheta" + str(i + 1)),
            )
            plt.plot(
                timestamp,
                thetamatd[i, :],
                ".",
                color=col,
                label=("DesiredTheta" + str(i + 1)),
            )
        plt.legend(loc="upper left")
        plt.xlabel("Time")
        plt.ylabel("Joint Angles")
        plt.title("Plot of Actual and Desired Joint Angles")
        plt.show()
    taumat = np.array(taumat).T
    thetamat = np.array(thetamat).T
    return (taumat, thetamat)
