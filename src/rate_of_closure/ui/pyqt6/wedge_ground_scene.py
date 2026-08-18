"""Matplotlib 3-D annotations for one swept wedge-clearance snapshot."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from rate_of_closure.simulation import RunGroundClearanceSnapshot

__all__ = ["draw_wedge_ground_overlay_3d"]


def _nearest_index(times: np.ndarray, time_s: float) -> int:
    return int(np.argmin(np.abs(times - time_s)))


def draw_wedge_ground_overlay_3d(
    axes: Any,
    snapshot: RunGroundClearanceSnapshot,
    time_s: float,
    display: Callable[[np.ndarray], np.ndarray],
    chart_color: Callable[[int], str],
) -> None:
    """Draw the sole envelope, current clearance, and auditable event markers."""
    analysis = snapshot.analysis
    if not analysis.envelope:
        return
    times = np.asarray([sample.time_s for sample in analysis.envelope])
    points = np.asarray([sample.world_point_m for sample in analysis.envelope])
    shown = display(points)
    axes.plot(
        shown[:, 0],
        shown[:, 1],
        shown[:, 2],
        color=chart_color(2),
        alpha=0.55,
        lw=1.4,
        label="Wedge Sole Envelope",
    )
    current_index = _nearest_index(times, time_s)
    current = points[current_index]
    current_clearance_mm = analysis.envelope[current_index].minimum_clearance_m * 1000.0
    ground = current.copy()
    ground[1] = 0.0
    clearance_line = display(np.vstack([current, ground]))
    axes.plot(
        clearance_line[:, 0],
        clearance_line[:, 1],
        clearance_line[:, 2],
        color=chart_color(9),
        ls="--",
        lw=1.6,
        label=f"Live Clearance {current_clearance_mm:.1f} mm",
    )
    if analysis.ball_contact_time_s is not None:
        ball_point = shown[_nearest_index(times, analysis.ball_contact_time_s)]
        axes.scatter(
            *ball_point, color=chart_color(9), s=55, zorder=8, label="Ball Contact"
        )
    if analysis.first_ground_contact is not None:
        contact = display(np.asarray(analysis.first_ground_contact.world_point_m))
        axes.scatter(
            *contact,
            color=chart_color(3),
            s=60,
            marker="X",
            zorder=9,
            label="Ground Contact",
        )
    low = display(np.asarray(analysis.low_point_world_m))
    axes.scatter(
        *low, color=chart_color(4), s=45, marker="D", zorder=7, label="Swept Low Point"
    )
