# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Ordered, timed via points for multi-phase lifts.

The legacy ``q_via`` argument of :class:`TrajectoryOptimizer` is a single
configuration pinned at the control-grid midpoint.  That is enough for a
lift with one intermediate pose (the clean's power position) but not for a
lift whose phases have their own timing, such as the snatch (pull, turnover,
catch, recovery).  A :class:`ViaSchedule` passed as ``q_via`` instead pins
each :class:`ViaPoint` at its own time fraction; the free spline waypoints
are spread over the segments between the pinned times in proportion to the
segment durations.

A schedule may also end the bar-knee clearance constraint at a time
fraction: the clearance keeps the bar in front of the knees while it is
below them, and has no meaning once the bar is overhead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

__all__ = [
    "ViaLayout",
    "ViaPoint",
    "ViaSchedule",
    "assemble_controls",
    "clearance_times",
    "schedule_splines",
    "via_initial_guess",
    "via_layout",
]


@dataclass(frozen=True)
class ViaPoint:
    """One configuration pinned at a fraction of the movement duration.

    Attributes:
        t_frac: Time as a fraction of the duration, strictly inside (0, 1).
        q: Joint-angle configuration in rad, shape ``(n_dof,)``.
        name: Phase or event the pose marks (informational).
    """

    t_frac: float
    q: NDArray
    name: str = ""

    def __post_init__(self) -> None:
        if not (0.0 < float(self.t_frac) < 1.0):
            raise ValueError(f"t_frac must lie in (0, 1): {self.t_frac!r}")
        q = np.asarray(self.q, dtype=float)
        if q.ndim != 1 or not np.all(np.isfinite(q)):
            raise ValueError("q must be a finite 1-D configuration")
        object.__setattr__(self, "q", q)


@dataclass(frozen=True)
class ViaSchedule:
    """Ordered via points plus the end of the bar-knee clearance window.

    Attributes:
        points: Via points with strictly increasing ``t_frac``.
        clearance_end_frac: Time fraction after which the bar-knee clearance
            is no longer enforced, or ``None`` to enforce it throughout.
    """

    points: tuple[ViaPoint, ...]
    clearance_end_frac: float | None = None

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if not points:
            raise ValueError("a ViaSchedule needs at least one via point")
        fracs = np.array([p.t_frac for p in points])
        if np.any(np.diff(fracs) <= 0.0):
            raise ValueError("via point t_frac values must be strictly increasing")
        if len({p.q.shape for p in points}) != 1:
            raise ValueError("all via points must have the same dimension")
        end = self.clearance_end_frac
        if end is not None and not (0.0 < float(end) <= 1.0):
            raise ValueError(f"clearance_end_frac must lie in (0, 1]: {end!r}")
        object.__setattr__(self, "points", points)

    @property
    def t_fracs(self) -> NDArray:
        """Via time fractions, shape ``(k,)``."""
        return np.array([p.t_frac for p in self.points], dtype=float)

    @property
    def configurations(self) -> NDArray:
        """Via configurations, shape ``(k, n_dof)``."""
        return np.vstack([p.q for p in self.points])


@dataclass(frozen=True)
class ViaLayout:
    """Control-point time grid of a schedule.

    Attributes:
        t_ctrl: Control times (start, free waypoints, vias, end), increasing.
        via_index: Positions of the via points within ``t_ctrl``.
        free_index: Positions of the free waypoints within ``t_ctrl``.
    """

    t_ctrl: NDArray
    via_index: NDArray
    free_index: NDArray


def _free_counts(segment_fracs: NDArray, n_waypoints: int) -> NDArray:
    """Split *n_waypoints* over segments in proportion to their lengths.

    Largest-remainder apportionment; the counts sum to *n_waypoints*.
    """
    ideal = segment_fracs * n_waypoints
    counts = np.floor(ideal).astype(int)
    order = np.argsort(-(ideal - counts), kind="stable")
    counts[order[: n_waypoints - int(counts.sum())]] += 1
    return counts


def via_layout(schedule: ViaSchedule, n_waypoints: int, duration: float) -> ViaLayout:
    """Control grid with the vias at their times and free points in between.

    Preconditions:
        n_waypoints >= 1 and duration > 0.

    Postconditions:
        ``len(t_ctrl) == n_waypoints + 2 + len(schedule.points)``; the vias
        sit exactly at ``t_frac * duration``; each segment's free points are
        evenly spaced strictly inside it.
    """
    if n_waypoints < 1:
        raise ValueError("n_waypoints must be >= 1")
    if not duration > 0.0:
        raise ValueError("duration must be positive")
    anchors = np.concatenate([[0.0], schedule.t_fracs, [1.0]])
    counts = _free_counts(np.diff(anchors), n_waypoints)
    times: list[float] = [0.0]
    via_index: list[int] = []
    for seg, count in enumerate(counts):
        lo, hi = anchors[seg], anchors[seg + 1]
        times.extend(np.linspace(lo, hi, count + 2)[1:-1])
        if seg < len(schedule.points):
            via_index.append(len(times))
        times.append(hi)
    t_ctrl = np.asarray(times) * duration
    fixed = {0, len(times) - 1, *via_index}
    free_index = np.array([i for i in range(len(times)) if i not in fixed])
    return ViaLayout(t_ctrl, np.asarray(via_index), free_index)


def assemble_controls(
    wp: NDArray,
    q_start: NDArray,
    q_end: NDArray,
    schedule: ViaSchedule,
    layout: ViaLayout,
) -> NDArray:
    """Stack start, free waypoints, vias and end in control-time order.

    Preconditions:
        ``wp.shape == (len(layout.free_index), n_dof)``.
    """
    q_all = np.empty((layout.t_ctrl.size, wp.shape[1]), dtype=float)
    q_all[0], q_all[-1] = q_start, q_end
    q_all[layout.via_index] = schedule.configurations
    q_all[layout.free_index] = wp
    return q_all


def via_initial_guess(
    q_start: NDArray, q_end: NDArray, schedule: ViaSchedule, layout: ViaLayout
) -> NDArray:
    """Free waypoints on the piecewise-linear path start -> vias -> end.

    Returns:
        Waypoint array of shape ``(len(layout.free_index), n_dof)``.
    """
    knot_t = layout.t_ctrl[np.concatenate([[0], layout.via_index, [-1]])]
    knot_q = np.vstack([q_start, schedule.configurations, q_end])
    t_free = layout.t_ctrl[layout.free_index]
    return np.column_stack(
        [np.interp(t_free, knot_t, knot_q[:, j]) for j in range(knot_q.shape[1])]
    )


def clearance_times(t_eval: NDArray, q_via: object, duration: float) -> NDArray:
    """Evaluation times at which the bar-knee clearance is enforced.

    All of *t_eval* unless *q_via* is a :class:`ViaSchedule` with a
    ``clearance_end_frac``, in which case only the samples up to that time.
    """
    if not isinstance(q_via, ViaSchedule) or q_via.clearance_end_frac is None:
        return t_eval
    return t_eval[t_eval <= q_via.clearance_end_frac * duration + 1e-12]


def schedule_splines(
    x: NDArray,
    q_start: NDArray,
    q_end: NDArray,
    schedule: ViaSchedule,
    layout: ViaLayout,
) -> CubicSpline:
    """Clamped cubic spline through start, free waypoints, vias and end.

    Preconditions:
        ``x.size == len(layout.free_index) * q_start.size``.
    """
    wp = np.asarray(x, dtype=float).reshape(layout.free_index.size, q_start.size)
    q_all = assemble_controls(wp, q_start, q_end, schedule, layout)
    return CubicSpline(layout.t_ctrl, q_all, bc_type="clamped")
