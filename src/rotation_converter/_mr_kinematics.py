# mypy: ignore-errors
"""Kinematics algorithms (Modern Robotics — Lynch & Park).

Forward kinematics, Jacobians, inverse kinematics, and trajectory generation.
Internal submodule extracted from modern_robotics.py to keep file size within
the 1200-line budget.  Import these symbols via ``rotation_converter.modern_robotics``
(the public shim) rather than directly from this private module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rotation_converter._contracts import ensure, require, require_finite
from rotation_converter._mr_rotation_matrices import (
    Adjoint,
    MatrixExp3,
    MatrixExp6,
    MatrixLog3,
    MatrixLog6,
    TransInv,
    TransToRp,
    VecTose3,
    _Adjoint,
    se3ToVec,
)

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
         ...]
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
