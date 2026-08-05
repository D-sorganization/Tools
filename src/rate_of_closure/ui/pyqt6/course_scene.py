"""Matplotlib course-scene painters (epic #4125, H7a).

Shared by the swing 3D scene and the flight viewers: the ground reads
as grass, with a lighter fairway strip along the target line and a
distinct putting green + hole/flag at the layout's configurable
distance, a tee marker at the origin. All tones come from
:func:`rate_of_closure.ui.course.course_colors` (theme-palette derived —
nothing hard-coded here); geometry comes from
:class:`rate_of_closure.ui.course.CourseLayout`.

App frame: x downrange along the target line, y up, z right. The 3D
axes use display order (z, x, y) like the swing view.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.patches import Circle

from rate_of_closure.ui.course import CourseColors, CourseLayout, course_colors

__all__ = [
    "draw_course_ground_3d",
    "draw_course_side",
    "draw_course_top",
]

_FLAG_HEIGHT_M = 2.5  # regulation-ish flagstick, tall enough to read


def _surface(axes: Any, xs: Any, zs: Any, y: float, color: str, alpha: float) -> None:
    """A flat y-level rectangle in display axes (X=z right, Y=x downrange)."""
    gx, gz = np.meshgrid(np.asarray(xs, float), np.asarray(zs, float))
    gy = np.full_like(gx, y)
    axes.plot_surface(gz, gx, gy, color=color, alpha=alpha, linewidth=0.0, shade=False)


def draw_course_ground_3d(
    axes: Any,
    extent: float,
    *,
    layout: CourseLayout | None = None,
    elements: bool = True,
    colors: CourseColors | None = None,
) -> None:
    """Grass ground plane (+ optional course elements) for a 3D scene.

    The rough covers the whole extent; when ``elements`` is on, the
    fairway strip runs downrange from the tee, the green disc + flag
    appear once the extent reaches them, and the tee marker sits at
    the origin.
    """
    layout = layout or CourseLayout()
    tones = colors or course_colors()
    axes.computed_zorder = False  # keep paths/markers above the grass
    _surface(axes, [-extent, extent], [-extent, extent], 0.0, tones.rough, 0.30)
    if not elements:
        return
    hw = min(layout.fairway_half_width_m, extent)
    _surface(axes, [0.0, extent], [-hw, hw], 0.0, tones.fairway, 0.45)
    d, r = layout.green_distance_m, layout.green_radius_m
    if d - r <= extent:
        theta = np.linspace(0.0, 2.0 * np.pi, 40)
        gx = d + r * np.cos(theta)
        gz = r * np.sin(theta)
        # Display axes: X=z right, Y=x downrange.
        axes.plot(gz, gx, np.zeros_like(gx), color=tones.green, lw=1.2)
        axes.plot_trisurf(gz, gx, np.zeros_like(gx), color=tones.green, alpha=0.55)
        axes.plot([0.0, 0.0], [d, d], [0.0, _FLAG_HEIGHT_M], color=tones.flag, lw=1.4)
        axes.scatter([0.0], [d], [_FLAG_HEIGHT_M], color=tones.flag, marker=">", s=40)
        axes.scatter([0.0], [d], [0.0], color=tones.hole, s=18, zorder=4)
    axes.scatter([0.0], [0.0], [0.0], color=tones.tee, marker="s", s=26, zorder=4)


def draw_course_side(
    axes: Any,
    carry_extent: float,
    *,
    layout: CourseLayout | None = None,
    elements: bool = True,
    colors: CourseColors | None = None,
) -> None:
    """Course styling for a side profile (height vs carry) panel."""
    layout = layout or CourseLayout()
    tones = colors or course_colors()
    axes.axvspan(0.0, carry_extent, ymax=0.025, color=tones.rough, alpha=0.75)
    if not elements:
        return
    d, r = layout.green_distance_m, layout.green_radius_m
    if d - r <= carry_extent:
        axes.axvspan(
            max(d - r, 0.0),
            min(d + r, carry_extent),
            ymax=0.025,
            color=tones.green,
            alpha=0.95,
        )
        axes.axvline(d, ymax=0.14, color=tones.flag, lw=1.2)
        axes.plot(
            [d], [0.0], marker="^", ms=5, color=tones.flag, clip_on=False, zorder=4
        )
    axes.plot([0.0], [0.0], marker="s", ms=4, color=tones.tee, zorder=4)


def draw_course_top(
    axes: Any,
    carry_extent: float,
    lateral_extent: float,
    *,
    layout: CourseLayout | None = None,
    elements: bool = True,
    colors: CourseColors | None = None,
) -> None:
    """Course styling for a top-down (lateral vs carry) panel."""
    layout = layout or CourseLayout()
    tones = colors or course_colors()
    axes.axhspan(-lateral_extent, lateral_extent, color=tones.rough, alpha=0.30)
    if not elements:
        return
    hw = min(layout.fairway_half_width_m, lateral_extent)
    axes.axhspan(-hw, hw, color=tones.fairway, alpha=0.40)
    d, r = layout.green_distance_m, layout.green_radius_m
    if d - r <= carry_extent:
        axes.add_patch(
            Circle((d, 0.0), r, facecolor=tones.green, edgecolor=tones.green, alpha=0.6)
        )
        axes.plot([d], [0.0], marker="o", ms=3, color=tones.hole, zorder=4)
        axes.plot([d], [0.0], marker=">", ms=6, color=tones.flag, zorder=5, alpha=0.9)
    axes.plot([0.0], [0.0], marker="s", ms=4, color=tones.tee, zorder=4)
