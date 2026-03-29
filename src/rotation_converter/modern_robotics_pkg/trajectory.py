"""Trajectory Generation — time scaling and interpolation.

CubicTimeScaling, QuinticTimeScaling, JointTrajectory, CartesianTrajectory,
ScrewTrajectory.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rotation_converter._contracts import ensure, require

from .se3 import MatrixExp6, MatrixLog6, TransInv, TransToRp, RpToTrans
from .so3 import MatrixExp3, MatrixLog3


# ---------------------------------------------------------------------------
# Time-scaling functions
# ---------------------------------------------------------------------------


def _cubic_time_scaling(t: float) -> float:
    """Cubic polynomial time scaling s(t) with s(0)=0, s(1)=1."""
    return 3.0 * t**2 - 2.0 * t**3


def _quintic_time_scaling(t: float) -> float:
    """Quintic polynomial time scaling s(t) with s(0)=0, s(1)=1."""
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def CubicTimeScaling(Tf: float, t: float) -> float:
    """Computes s(t) for a cubic time scaling."""
    return 3 * (1.0 * t / Tf) ** 2 - 2 * (1.0 * t / Tf) ** 3


def QuinticTimeScaling(Tf: float, t: float) -> float:
    """Computes s(t) for a quintic time scaling."""
    return 10 * (1.0 * t / Tf) ** 3 - 15 * (1.0 * t / Tf) ** 4 + 6 * (1.0 * t / Tf) ** 5


# ---------------------------------------------------------------------------
# Trajectory generators
# ---------------------------------------------------------------------------


def ScrewTrajectory(
    Xstart: Any,
    Xend: Any,
    Tf: float,
    N: int,
    method: int = 3,
) -> list[np.ndarray]:
    """Generate a trajectory as a list of SE(3) matrices via screw motion."""
    if not (Tf is not None):
        raise ValueError("Tf must be provided")
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


def JointTrajectory(
    thetastart: Any,
    thetaend: Any,
    Tf: float,
    N: int,
    method: int,
) -> np.ndarray:
    """Computes a straight-line trajectory in joint space."""
    if not (thetastart is not None):
        raise ValueError("thetastart must be provided")
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


def CartesianTrajectory(
    Xstart: Any,
    Xend: Any,
    Tf: float,
    N: int,
    method: int,
) -> list:
    """Computes a trajectory as a list of N SE(3) matrices (straight-line origin path)."""
    if not (Xstart is not None):
        raise ValueError("Xstart must be provided")
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
