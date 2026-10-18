# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Clean and jerk full-lift configuration.

The clean and jerk lifts the bar from the floor to the shoulders (clean)
and then drives it overhead to lockout (jerk):
1. Setup: bar on floor over mid-foot in clean grip.
2. First pull: bar leaves the floor until it passes the knees.
3. Second pull: explosive extension with bar close to body.
4. Rack catch: bar received on shoulders in a front squat.
5. Rack recovery: standing with bar in front rack.
6. Jerk dip/drive: short dip and drive of legs launching bar overhead.
7. Overhead catch: bar received overhead with arms locked and knees bent.
8. Overhead recovery: standing with bar locked out overhead.

Each phase starts at the time fraction of the biomechanical parity
standard (UpstreamDrift ``biomech_parity_standard.json``,
``exercises.clean_and_jerk.phases``), recorded here as
:data:`CLEAN_AND_JERK_PHASES`. The configuration pins via poses at each
intermediate phase start through a :class:`~movement_optimizer.trajectory.ViaSchedule`.

The bar model (:func:`clean_and_jerk_bar_path`) is phase-correct: hanging
one arm length below the shoulder during the pull, resting at the shoulder
in the front rack, and locked out one arm length overhead from the jerk catch on.

Segment angles are absolute, measured from vertical (shank, thigh, trunk;
positive forward), as in :mod:`movement_optimizer.models`.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from ..models import BodyModel, LagrangianDynamics
from ..trajectory.optimizer_via import ViaPoint, ViaSchedule
from ._common import (
    balance_config_pose,
    default_bounds_deg,
    pose_deg,
    pull_start_angles,
)

#: Phase start times as fractions of the lift (parity standard, clean_and_jerk).
CLEAN_AND_JERK_PHASES: Mapping[str, float] = MappingProxyType(
    {
        "setup": 0.0,
        "first_pull": 0.08,
        "second_pull": 0.2,
        "rack_catch": 0.3,
        "rack_recovery": 0.4,
        "jerk_dip_drive": 0.55,
        "overhead_catch": 0.7,
        "overhead_recovery": 0.85,
    }
)

#: Rack catch trunk lean from vertical: the standard's catch pelvis pitch
#: (``via_poses.catch.root.orientation_rpy_rad[1]`` = 0.3195 rad).
CLEAN_AND_JERK_RACK_CATCH_TRUNK_DEG: float = float(np.degrees(0.3195))

#: Rack catch knee flexion: the standard's ``knee_flex`` at the catch (-1.6964 rad).
CLEAN_AND_JERK_RACK_CATCH_KNEE_FLEX_DEG: float = float(np.degrees(-1.6964))

#: Lockout knee flexion: the standard's ``knee_flex`` at recovery (-0.05 rad).
CLEAN_AND_JERK_LOCKOUT_KNEE_FLEX_DEG: float = float(np.degrees(-0.05))

#: Shank-angle search range for balancing the catch (clean and jerk ankle bounds).
_ANKLE_BOUNDS_DEG: tuple[float, float] = (-5.0, 55.0)


def _knee_pass_angles() -> NDArray:
    """Start of the second pull: bar just above the knees.

    Knees moved back out of the bar path (shank near vertical) while the
    trunk keeps roughly its setup inclination.
    """
    return pose_deg(5, -35, 45)


def _front_rack_angles() -> NDArray:
    """Front rack position: standing with bar at shoulders."""
    return pose_deg(5, -8, 5)


def _jerk_dip_angles() -> NDArray:
    """Dip position: quarter-squat before the explosive jerk drive."""
    return pose_deg(12, -40, 8)


def _overhead_catch_angles() -> NDArray:
    """Overhead catch position: bar caught overhead with knees bent."""
    return pose_deg(12, -40, 8)


def _lockout_angles() -> NDArray:
    """Standing with the bar locked out overhead (the standard's knee)."""
    return pose_deg(8, 8 + CLEAN_AND_JERK_LOCKOUT_KNEE_FLEX_DEG, 0)


def _balanced_rack_catch(dyn: LagrangianDynamics, bar_mass: float) -> NDArray:
    """Front-squat catch with the standard's trunk lean and knee flexion.

    The shank angle is solved so the whole-body COM sits over the centre of
    the inner base of support; the knee flexion and trunk lean are held at
    the standard's catch values.

    Raises:
        ValueError: If no shank angle within the ankle bounds balances it.
    """
    knee, trunk = CLEAN_AND_JERK_RACK_CATCH_KNEE_FLEX_DEG, CLEAN_AND_JERK_RACK_CATCH_TRUNK_DEG

    def residual(shank_deg: float) -> float:
        q = pose_deg(shank_deg, shank_deg + knee, trunk)
        return float(dyn.com_position(q, "clean", bar_mass)[0] - dyn.body.inner_center)

    lo, hi = _ANKLE_BOUNDS_DEG
    if residual(lo) * residual(hi) > 0.0:
        raise ValueError("no shank angle within the ankle bounds balances the catch")
    shank = brentq(residual, lo, hi, xtol=1e-10)
    return pose_deg(shank, shank + knee, trunk)


def clean_and_jerk_via_schedule(
    dyn: LagrangianDynamics, q_start: NDArray, bar_mass: float
) -> ViaSchedule:
    """Via poses at the parity standard's phase times.

    The bar leaves the floor at ``first_pull`` (the setup is held until then),
    reaches knee pass at ``second_pull``, is caught in a front squat at
    ``rack_catch``, stood up to ``rack_recovery``, dipped at ``jerk_dip_drive``,
    caught overhead at ``overhead_catch``, and stood up to ``overhead_recovery``.
    Bar-knee clearance ends at ``second_pull``.
    """
    rack_catch = _balanced_rack_catch(dyn, bar_mass)

    def balanced(raw: NDArray, exercise: str, joint: int) -> NDArray:
        return balance_config_pose(dyn, raw, exercise, bar_mass, adjust_joint=joint)

    knee_pass = balanced(_knee_pass_angles(), "deadlift", 0)
    rack_recovery = balanced(_front_rack_angles(), "clean", 0)
    jerk_dip = balanced(_jerk_dip_angles(), "clean", 0)
    overhead_catch = balanced(_overhead_catch_angles(), "snatch", 0)
    overhead_recovery = balanced(_lockout_angles(), "snatch", 2)

    poses = {
        "first_pull": q_start,
        "second_pull": knee_pass,
        "rack_catch": rack_catch,
        "rack_recovery": rack_recovery,
        "jerk_dip_drive": jerk_dip,
        "overhead_catch": overhead_catch,
        "overhead_recovery": overhead_recovery,
    }
    points = tuple(ViaPoint(CLEAN_AND_JERK_PHASES[k], q, k) for k, q in poses.items())
    return ViaSchedule(points, clearance_end_frac=CLEAN_AND_JERK_PHASES["second_pull"])


def clean_and_jerk_bar_weights(
    t_frac: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Fraction of each bar regime (hanging, rack, overhead) at each time fraction.

    - Hanging (deadlift bar model) through the pulls (up to second_pull).
    - Front rack (clean bar model) from rack_catch through jerk_dip_drive.
    - Overhead (snatch bar model) from overhead_catch through recovery.
    - Smoothstep transitions across second_pull -> rack_catch and jerk_dip_drive -> overhead_catch.
    """
    t = np.asarray(t_frac, dtype=float)

    # 1. Pull to rack catch transition
    t0_rack = CLEAN_AND_JERK_PHASES["second_pull"]
    t1_rack = CLEAN_AND_JERK_PHASES["rack_catch"]
    s_rack = np.clip((t - t0_rack) / (t1_rack - t0_rack), 0.0, 1.0)
    w_rack_trans = s_rack * s_rack * (3.0 - 2.0 * s_rack)

    # 2. Jerk dip drive to overhead catch transition
    t0_oh = CLEAN_AND_JERK_PHASES["jerk_dip_drive"]
    t1_oh = CLEAN_AND_JERK_PHASES["overhead_catch"]
    s_oh = np.clip((t - t0_oh) / (t1_oh - t0_oh), 0.0, 1.0)
    w_oh_trans = s_oh * s_oh * (3.0 - 2.0 * s_oh)

    w_hang = 1.0 - w_rack_trans
    w_rack = w_rack_trans * (1.0 - w_oh_trans)
    w_overhead = w_oh_trans
    return w_hang, w_rack, w_overhead


def clean_and_jerk_bar_path(dyn: LagrangianDynamics, q: NDArray, t_frac: NDArray) -> NDArray:
    """Phase-correct bar positions along a clean-and-jerk trajectory.

    The bar hangs one arm length below the shoulder (the ``deadlift`` bar
    model) through both pulls, rests at the shoulder (the ``clean`` front rack
    model) from the rack catch through the jerk dip/drive, and is one arm
    length above the shoulder (the ``snatch`` overhead model) from the jerk
    catch through recovery.

    Args:
        dyn: Dynamics providing :meth:`bar_position`.
        q: Joint-angle trajectory, shape ``(n, 3)``.
        t_frac: Time of each sample as a fraction of the lift, shape ``(n,)``.

    Returns:
        Bar positions, shape ``(n, 2)``.
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    t_frac = np.asarray(t_frac, dtype=float)
    if t_frac.shape != (q.shape[0],):
        raise ValueError("t_frac must have one entry per trajectory sample")
    hang = np.array([dyn.bar_position(row, "deadlift") for row in q])
    rack = np.array([dyn.bar_position(row, "clean") for row in q])
    overhead = np.array([dyn.bar_position(row, "snatch") for row in q])
    w_hang, w_rack, w_overhead = clean_and_jerk_bar_weights(t_frac)
    path: NDArray = w_hang[:, None] * hang + w_rack[:, None] * rack + w_overhead[:, None] * overhead
    return path


def make_clean_and_jerk_config(
    body: BodyModel, bar_mass: float
) -> tuple[LagrangianDynamics, NDArray, NDArray, NDArray, ViaSchedule]:
    """Create dynamics and trajectory config for the clean and jerk.

    The deadlift-style dynamics (arms hanging, load ``m_arms + bar_mass`` at
    the shoulder) are used for the whole lift. In this planar model the bar
    sits directly at, above, or below the shoulder, so all three bar models
    share one horizontal position and the COM balance is the same for all of
    them; only the reported bar height (:func:`clean_and_jerk_bar_path`)
    changes with the phase.

    Returns:
        ``(dynamics, q_start, q_end, q_bounds, q_via)`` where ``q_via`` is a
        :class:`~movement_optimizer.trajectory.ViaSchedule` accepted by
        :class:`~movement_optimizer.trajectory.TrajectoryOptimizer`.
    """
    load = body.m_arms + bar_mass
    dyn = LagrangianDynamics(body, body.m_deadlift.copy(), body.I_deadlift.copy(), load)

    q_start_raw = pull_start_angles(body, q2_deg=52)
    q_start = balance_config_pose(dyn, q_start_raw, "deadlift", bar_mass, adjust_joint=0)
    q_end = balance_config_pose(dyn, _lockout_angles(), "snatch", bar_mass, adjust_joint=2)
    schedule = clean_and_jerk_via_schedule(dyn, q_start, bar_mass)
    q_bounds = default_bounds_deg(_ANKLE_BOUNDS_DEG, (-110, 10), (-10, 80))
    return dyn, q_start, q_end, q_bounds, schedule
