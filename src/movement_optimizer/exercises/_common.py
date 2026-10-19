# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Shared helpers for planar exercise configuration factories."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from ..constants import PLATE_RADIUS_STD_M
from ..models import BodyModel, LagrangianDynamics, balance_pose


def pose_deg(ankle: float, knee: float, hip: float) -> NDArray:
    """Return a 3-angle pose vector built from degree inputs."""
    return np.radians(np.array([ankle, knee, hip], dtype=float))


def balance_config_pose(
    dynamics: LagrangianDynamics,
    raw_pose: NDArray,
    exercise_type: str,
    bar_mass: float,
    *,
    adjust_joint: int,
) -> NDArray:
    """Balance a raw pose using the shared planar balance helper."""
    return balance_pose(dynamics, raw_pose, exercise_type, bar_mass, adjust_joint=adjust_joint)


def default_bounds_deg(
    lower: tuple[float, float], middle: tuple[float, float], upper: tuple[float, float]
) -> NDArray:
    """Return a `(3, 2)` array of joint bounds built from degree tuples."""
    return np.radians(np.array([lower, middle, upper], dtype=float))


def pull_start_angles(body: BodyModel, q2_deg: float) -> NDArray:
    """Compute starting joint angles for a pulling exercise (bar at plate height).

    Shared by deadlift, clean, and snatch.  The hip-flexion angle *q2_deg*
    (in degrees) varies by exercise -- wider grips use a smaller value.
    """
    target_shoulder_h = PLATE_RADIUS_STD_M + body.L_arm
    q0 = np.radians(15)
    q2 = np.radians(q2_deg)
    needed = target_shoulder_h - body.L[0] * np.cos(q0) - body.L[2] * np.cos(q2)
    cos_q1 = np.clip(needed / body.L[1], -1, 1)
    q1 = -np.arccos(cos_q1)
    return np.array([q0, q1, q2])


def balanced_shank_pose(
    dynamics: LagrangianDynamics,
    knee_flex_deg: float,
    trunk_deg: float,
    exercise_type: str,
    bar_mass: float,
    ankle_bounds_deg: tuple[float, float],
) -> NDArray:
    """Pose with a held knee flexion and trunk lean, balanced by the shank.

    The shank angle is solved within *ankle_bounds_deg* so the whole-body
    COM sits over the centre of the inner base of support; the knee flexion
    (thigh minus shank) and the absolute trunk angle are held.

    Raises:
        ValueError: If no shank angle within the bounds balances the pose.
    """

    def residual(shank_deg: float) -> float:
        q = pose_deg(shank_deg, shank_deg + knee_flex_deg, trunk_deg)
        com_x = dynamics.com_position(q, exercise_type, bar_mass)[0]
        return float(com_x - dynamics.body.inner_center)

    lo, hi = ankle_bounds_deg
    if residual(lo) * residual(hi) > 0.0:
        raise ValueError("no shank angle within the ankle bounds balances the pose")
    shank = brentq(residual, lo, hi, xtol=1e-10)
    return pose_deg(shank, shank + knee_flex_deg, trunk_deg)


def smoothstep(t_frac: NDArray, t0: float, t1: float) -> NDArray:
    """0 before *t0*, 1 after *t1* and a C1 smoothstep in between.

    Preconditions:
        t0 < t1.
    """
    if not t0 < t1:
        raise ValueError("smoothstep needs t0 < t1")
    s = np.clip((np.asarray(t_frac, dtype=float) - t0) / (t1 - t0), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)
