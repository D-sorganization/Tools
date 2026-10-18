# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Snatch exercise configuration.

The snatch lifts the bar from the floor to overhead in one continuous
motion: a first pull (floor to knees), a second pull (knees to full
extension), the turnover (the lifter pulls under the bar and turns the arms
over), the catch (bar received overhead in a squat) and the recovery
(standing with the bar locked out).

Each phase starts at the time fraction of the biomechanical parity
standard (UpstreamDrift ``biomech_parity_standard.json``,
``exercises.snatch.phases``), recorded here as :data:`SNATCH_PHASES`.  The
configuration pins one via pose at each phase start through a
:class:`~movement_optimizer.trajectory.ViaSchedule`, so the shoulder rises
through both pulls before the lifter drops under the bar.

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

#: Phase start times as fractions of the lift (parity standard, snatch).
SNATCH_PHASES: Mapping[str, float] = MappingProxyType(
    {
        "setup": 0.0,
        "first_pull": 0.1,
        "second_pull": 0.3,
        "turnover": 0.45,
        "catch": 0.55,
        "recovery": 0.7,
    }
)

#: Catch trunk lean from vertical: the standard's catch pelvis pitch
#: (``via_poses.catch.root.orientation_rpy_rad[1]`` = 0.3264 rad).
SNATCH_CATCH_TRUNK_DEG: float = float(np.degrees(0.3264))

#: Catch knee flexion: the standard's ``knee_flex`` at the catch (-1.5332 rad).
SNATCH_CATCH_KNEE_FLEX_DEG: float = float(np.degrees(-1.5332))

#: Lockout knee flexion: the standard's ``knee_flex`` at recovery (-0.05 rad).
SNATCH_LOCKOUT_KNEE_FLEX_DEG: float = float(np.degrees(-0.05))

#: Shank-angle search range for balancing the catch (snatch ankle bounds).
_ANKLE_BOUNDS_DEG: tuple[float, float] = (-5.0, 45.0)


def _knee_pass_angles() -> NDArray:
    """Start of the second pull: bar just above the knees.

    Knees moved back out of the bar path (shank near vertical) while the
    trunk keeps roughly its setup inclination.
    """
    return pose_deg(5, -35, 45)


def _power_position_angles() -> NDArray:
    """End of the second pull: full extension, the highest shoulder.

    Knees extended to the standard's lockout flexion with the shank
    inclined 5 deg; the trunk is then balanced, inclining it over the bar.
    The bar hangs at the hips.
    """
    return pose_deg(5, 5 + SNATCH_LOCKOUT_KNEE_FLEX_DEG, 0)


def _lockout_angles() -> NDArray:
    """Standing with the bar locked out overhead (the standard's knee)."""
    return pose_deg(8, 8 + SNATCH_LOCKOUT_KNEE_FLEX_DEG, 0)


def _balanced_catch(dyn: LagrangianDynamics, bar_mass: float) -> NDArray:
    """Overhead-squat catch with the standard's trunk lean and knee flexion.

    The shank angle is solved so the whole-body COM sits over the centre of
    the inner base of support; the knee flexion and trunk lean are held at
    the standard's catch values.

    Raises:
        ValueError: If no shank angle within the ankle bounds balances it.
    """
    return balanced_shank_pose(
        dyn,
        SNATCH_CATCH_KNEE_FLEX_DEG,
        SNATCH_CATCH_TRUNK_DEG,
        "snatch",
        bar_mass,
        _ANKLE_BOUNDS_DEG,
    )


def snatch_via_schedule(dyn: LagrangianDynamics, q_start: NDArray, bar_mass: float) -> ViaSchedule:
    """Via poses at the starts of the first pull through the recovery.

    The bar leaves the floor at ``first_pull`` (the setup is held until
    then), the catch is held from ``catch`` until ``recovery``, and the
    bar-knee clearance ends at ``turnover``, once the bar is above the hips.
    """
    catch = _balanced_catch(dyn, bar_mass)

    def balanced(raw: NDArray, joint: int) -> NDArray:
        return balance_config_pose(dyn, raw, "snatch", bar_mass, adjust_joint=joint)

    poses = {
        "second_pull": balanced(_knee_pass_angles(), 0),
        "turnover": balanced(_power_position_angles(), 2),
        "catch": catch,
        "recovery": catch,
    }
    points = tuple(ViaPoint(SNATCH_PHASES[k], q, k) for k, q in poses.items())
    return ViaSchedule(points, clearance_end_frac=SNATCH_PHASES["turnover"])


def snatch_bar_weight(t_frac: NDArray) -> NDArray:
    """Fraction of the way from hanging to overhead at each time fraction.

    0 through the pulls, 1 from the catch on, and a smoothstep across the
    turnover, when the arms turn the bar over.
    """
    return smoothstep(t_frac, SNATCH_PHASES["turnover"], SNATCH_PHASES["catch"])


def snatch_bar_path(dyn: LagrangianDynamics, q: NDArray, t_frac: NDArray) -> NDArray:
    """Phase-correct bar positions along a snatch trajectory.

    The bar hangs one arm length below the shoulder (the ``deadlift`` bar
    model) through both pulls, is one arm length above it (the ``snatch``
    overhead model) from the catch on, and moves between the two across the
    turnover.

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
    overhead = np.array([dyn.bar_position(row, "snatch") for row in q])
    w = snatch_bar_weight(t_frac)[:, None]
    path: NDArray = (1.0 - w) * hang + w * overhead
    return path


def make_snatch_config(
    body: BodyModel, bar_mass: float
) -> tuple[LagrangianDynamics, NDArray, NDArray, NDArray, ViaSchedule]:
    """Create dynamics and trajectory config for the snatch.

    The deadlift-style dynamics (arms hanging, load ``m_arms + bar_mass`` at
    the shoulder) are used for the whole lift.  In this planar model the bar
    sits directly above or below the shoulder, so the hanging and overhead
    bar models share one horizontal position and the COM balance is the same
    for both; only the reported bar height (:func:`snatch_bar_path`) changes
    with the phase.

    Returns:
        ``(dynamics, q_start, q_end, q_bounds, q_via)`` where ``q_via`` is a
        :class:`~movement_optimizer.trajectory.ViaSchedule` accepted by
        :class:`~movement_optimizer.trajectory.TrajectoryOptimizer`.
    """
    load = body.m_arms + bar_mass
    dyn = LagrangianDynamics(body, body.m_deadlift.copy(), body.I_deadlift.copy(), load)

    q_start_raw = pull_start_angles(body, q2_deg=48)
    q_start = balance_config_pose(dyn, q_start_raw, "deadlift", bar_mass, adjust_joint=0)
    q_end = balance_config_pose(dyn, _lockout_angles(), "snatch", bar_mass, adjust_joint=2)
    schedule = snatch_via_schedule(dyn, q_start, bar_mass)
    q_bounds = default_bounds_deg(_ANKLE_BOUNDS_DEG, (-110, 10), (-10, 80))
    return dyn, q_start, q_end, q_bounds, schedule
