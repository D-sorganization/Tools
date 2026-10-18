# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Clean-and-jerk exercise configuration: the whole lift, floor to lockout.

The clean and jerk is two lifts in one: the clean (first pull, second pull,
turnover and a front-squat catch with the bar on the shoulders, then the
recovery to a standing front rack) and the jerk (a short dip and drive of the
legs that launches the bar, a catch overhead with the knees bent, then the
recovery to a standing lockout).

Each phase starts at the time fraction of the biomechanical parity standard
(UpstreamDrift ``biomech_parity_standard.json``,
``exercises.clean_and_jerk.phases``), recorded here as
:data:`CLEAN_AND_JERK_PHASES`.  Three events inside those phases are pinned
too (:data:`CLEAN_AND_JERK_EVENTS`): the full extension that ends the second
pull and starts the turnover, the bottom of the jerk dip and the full
extension of the jerk drive.  The configuration pins one via pose at each
through a :class:`~movement_optimizer.trajectory.ViaSchedule`.

The standard's catch pose sets the front-squat catch (knee flexion and
trunk lean); its lockout pose sets the standing lockout.  It has no pose for
the overhead catch: its symmetric stance makes the jerk a power jerk (no
split), received with the knees bent to the depth of the dip of
:mod:`.jerk` and the trunk at the lockout's lean.

The bar model is phase-correct (:func:`clean_and_jerk_bar_path`): hanging in
the hands through the pulls, on the shoulders (front rack) from the catch to
the jerk drive, and overhead from the overhead catch on.

Segment angles are absolute, measured from vertical (shank, thigh, trunk;
positive forward), as in :mod:`movement_optimizer.models`.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from ..models import BodyModel, LagrangianDynamics
from ..trajectory.optimizer_via import ViaPoint, ViaSchedule
from ._common import (
    balance_config_pose,
    balanced_shank_pose,
    default_bounds_deg,
    pose_deg,
    pull_start_angles,
    smoothstep,
)
from .jerk import jerk_dip_angles

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

#: Events inside the standard's phases, as fractions of the lift: the full
#: extension (``turnover``) halfway through the second pull and turnover, and
#: the dip bottom and drive extension splitting the dip-and-drive in thirds.
CLEAN_AND_JERK_EVENTS: Mapping[str, float] = MappingProxyType(
    {
        "turnover": 0.25,
        "dip_bottom": 0.6,
        "drive_extension": 0.65,
    }
)

#: Rack-catch trunk lean from vertical: the standard's catch pelvis pitch
#: (``via_poses.catch.root.orientation_rpy_rad[1]`` = 0.3195 rad).
RACK_CATCH_TRUNK_DEG: float = float(np.degrees(0.3195))

#: Rack-catch knee flexion: the standard's catch ``knee_flex`` (-1.6964 rad).
RACK_CATCH_KNEE_FLEX_DEG: float = float(np.degrees(-1.6964))

#: Lockout knee flexion: the standard's lockout ``knee_flex`` (-0.05 rad).
LOCKOUT_KNEE_FLEX_DEG: float = float(np.degrees(-0.05))

#: Overhead-catch trunk lean: the standard's lockout pelvis pitch
#: (``via_poses.lockout.root.orientation_rpy_rad[1]`` = -0.0009 rad).
OVERHEAD_CATCH_TRUNK_DEG: float = float(np.degrees(-0.0009))

#: Overhead-catch (power jerk) knee flexion: the dip depth of :mod:`.jerk`.
OVERHEAD_CATCH_KNEE_FLEX_DEG: float = float(np.degrees(jerk_dip_angles()[1] - jerk_dip_angles()[0]))

#: Jerk-dip trunk lean: the dip pose of :mod:`.jerk`.
_DIP_TRUNK_DEG: float = float(np.degrees(jerk_dip_angles()[2]))

#: Shank-angle range, up to the model's ankle limit (``JOINT_LIMITS``, 50 deg):
#: the planar front-squat catch balances at a 46 deg shank (the bar sits over
#: the shoulder, not in front of it as in the standard's 3-D catch).
_ANKLE_BOUNDS_DEG: tuple[float, float] = (-5.0, 50.0)

#: Bar conventions of :meth:`LagrangianDynamics.bar_position` per bar phase.
_HANG, _RACK, _OVERHEAD = "deadlift", "clean_and_jerk", "jerk"


def _times() -> dict[str, float]:
    return {**CLEAN_AND_JERK_PHASES, **CLEAN_AND_JERK_EVENTS}


def _knee_pass_angles() -> NDArray:
    """Start of the second pull: bar just above the knees, shank near vertical."""
    return pose_deg(5, -35, 48)


def _extension_angles() -> NDArray:
    """Full extension (second pull, jerk drive): knees at the lockout flexion."""
    return pose_deg(5, 5 + LOCKOUT_KNEE_FLEX_DEG, 0)


def _standing_angles() -> NDArray:
    """Standing (front rack or lockout) at the standard's lockout knee."""
    return pose_deg(8, 8 + LOCKOUT_KNEE_FLEX_DEG, 0)


def clean_and_jerk_via_schedule(
    dyn: LagrangianDynamics, q_start: NDArray, bar_mass: float
) -> ViaSchedule:
    """Via poses from the first pull to the overhead recovery.

    The setup *q_start* is held until ``first_pull``, when the bar leaves the
    floor.  The front-squat catch is held from ``rack_catch`` until
    ``rack_recovery`` and the overhead catch from ``overhead_catch`` until
    ``overhead_recovery``.  The bar-knee clearance ends at ``turnover``, once
    the bar is above the hips.
    """

    def balanced(raw: NDArray, joint: int, convention: str) -> NDArray:
        return balance_config_pose(dyn, raw, convention, bar_mass, adjust_joint=joint)

    def held(knee_deg: float, trunk_deg: float, convention: str) -> NDArray:
        return balanced_shank_pose(
            dyn, knee_deg, trunk_deg, convention, bar_mass, _ANKLE_BOUNDS_DEG
        )

    rack_catch = held(RACK_CATCH_KNEE_FLEX_DEG, RACK_CATCH_TRUNK_DEG, _RACK)
    overhead_catch = held(OVERHEAD_CATCH_KNEE_FLEX_DEG, OVERHEAD_CATCH_TRUNK_DEG, _OVERHEAD)
    poses = {
        "first_pull": np.asarray(q_start, dtype=float),
        "second_pull": balanced(_knee_pass_angles(), 0, _HANG),
        "turnover": balanced(_extension_angles(), 2, _HANG),
        "rack_catch": rack_catch,
        "rack_recovery": rack_catch,
        "jerk_dip_drive": balanced(_standing_angles(), 2, _RACK),
        "dip_bottom": held(OVERHEAD_CATCH_KNEE_FLEX_DEG, _DIP_TRUNK_DEG, _RACK),
        "drive_extension": balanced(_extension_angles(), 2, _RACK),
        "overhead_catch": overhead_catch,
        "overhead_recovery": overhead_catch,
    }
    times = _times()
    points = tuple(ViaPoint(times[k], q, k) for k, q in poses.items())
    return ViaSchedule(points, clearance_end_frac=CLEAN_AND_JERK_EVENTS["turnover"])


def clean_and_jerk_bar_weights(t_frac: NDArray) -> tuple[NDArray, NDArray]:
    """Blend weights ``(to_rack, to_overhead)`` at each time fraction.

    ``to_rack`` is 0 through the pulls, 1 from the rack catch on, with a
    smoothstep across the turnover; ``to_overhead`` is 0 until the jerk
    drive's full extension, 1 from the overhead catch on, with a smoothstep
    while the arms drive the bar up off the shoulders.
    """
    events = _times()
    to_rack = smoothstep(t_frac, events["turnover"], events["rack_catch"])
    to_overhead = smoothstep(t_frac, events["drive_extension"], events["overhead_catch"])
    return to_rack, to_overhead


def clean_and_jerk_bar_path(dyn: LagrangianDynamics, q: NDArray, t_frac: NDArray) -> NDArray:
    """Phase-correct bar positions along a clean-and-jerk trajectory.

    The bar hangs one arm length below the shoulder (the ``deadlift`` bar
    model) through the pulls, sits on the shoulder (the front-rack
    ``clean_and_jerk`` model) from the rack catch to the jerk drive and is one
    arm length above it (the overhead ``jerk`` model) from the overhead catch
    on, moving between them across the turnover and the drive.

    Args:
        dyn: Dynamics providing :meth:`bar_position`.
        q: Joint-angle trajectory, shape ``(n, 3)``.
        t_frac: Time of each sample as a fraction of the lift, shape ``(n,)``.

    Returns:
        Bar positions, shape ``(n, 2)``.

    Raises:
        ValueError: If *t_frac* does not have one entry per sample of *q*.
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    t_frac = np.asarray(t_frac, dtype=float)
    if t_frac.shape != (q.shape[0],):
        raise ValueError("t_frac must have one entry per trajectory sample")
    hang, rack, overhead = (
        np.array([dyn.bar_position(row, convention) for row in q])
        for convention in (_HANG, _RACK, _OVERHEAD)
    )
    to_rack, to_overhead = (w[:, None] for w in clean_and_jerk_bar_weights(t_frac))
    shelf = (1.0 - to_rack) * hang + to_rack * rack
    path: NDArray = (1.0 - to_overhead) * shelf + to_overhead * overhead
    return path


def make_clean_and_jerk_config(
    body: BodyModel, bar_mass: float
) -> tuple[LagrangianDynamics, NDArray, NDArray, NDArray, ViaSchedule]:
    """Create dynamics and trajectory config for the full clean and jerk.

    The deadlift-style dynamics (arms hanging, load ``m_arms + bar_mass`` at
    the shoulder) are used for the whole lift.  In this planar model the bar
    sits directly below, at or above the shoulder, so the hanging, front-rack
    and overhead bar models share one horizontal position and the COM balance
    is the same for all three; only the reported bar height
    (:func:`clean_and_jerk_bar_path`) changes with the phase.

    Returns:
        ``(dynamics, q_start, q_end, q_bounds, q_via)`` where ``q_via`` is a
        :class:`~movement_optimizer.trajectory.ViaSchedule` accepted by
        :class:`~movement_optimizer.trajectory.TrajectoryOptimizer` with the
        ``"clean_and_jerk"`` exercise type.

    Raises:
        ValueError: If *bar_mass* is negative or not finite.
    """
    if not (np.isfinite(bar_mass) and bar_mass >= 0.0):
        raise ValueError(f"bar_mass must be finite and non-negative: {bar_mass!r}")
    load = body.m_arms + bar_mass
    dyn = LagrangianDynamics(body, body.m_deadlift.copy(), body.I_deadlift.copy(), load)
    q_start_raw = pull_start_angles(body, q2_deg=52)
    q_start = balance_config_pose(dyn, q_start_raw, _HANG, bar_mass, adjust_joint=0)
    q_end = balance_config_pose(dyn, _standing_angles(), _OVERHEAD, bar_mass, adjust_joint=2)
    schedule = clean_and_jerk_via_schedule(dyn, q_start, bar_mass)
    q_bounds = default_bounds_deg(_ANKLE_BOUNDS_DEG, (-110, 10), (-10, 80))
    return dyn, q_start, q_end, q_bounds, schedule
