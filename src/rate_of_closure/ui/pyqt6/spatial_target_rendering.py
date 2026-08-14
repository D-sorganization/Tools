"""Matplotlib projection painters for canonical spatial targets."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
from matplotlib.patches import Circle, Rectangle

from rate_of_closure.ui.course import course_colors
from shared.python.swing_sim.solver import (
    BoxTolerance,
    SpatialTarget,
    SphereTolerance,
    SurfaceCircleTolerance,
)

_LINE_WIDTH = 1.6
_POINT_SIZE = 38


def draw_spatial_target_side(axes: Any, target: SpatialTarget) -> None:
    """Draw the x/elevation projection with an explicit target label."""
    x_m, elevation_m, _right_m = target.point.app_coordinates_m
    color = course_colors().flag
    tolerance = target.tolerance
    patch: Circle | Rectangle
    if isinstance(tolerance, SphereTolerance):
        patch = Circle((x_m, elevation_m), tolerance.radius_m)
    elif isinstance(tolerance, BoxTolerance):
        half_x, half_elevation, _half_right = tolerance.half_extents_m
        patch = _rectangle(x_m, elevation_m, half_x, half_elevation)
    elif isinstance(tolerance, SurfaceCircleTolerance):
        patch = _surface_band(x_m, elevation_m, tolerance.radius_m)
    else:
        patch = _surface_band(x_m, elevation_m, tolerance.half_length_m)
    _style_patch(patch, color)
    axes.add_patch(patch)
    _label_2d(axes, x_m, elevation_m, target.label, color)


def draw_spatial_target_top(axes: Any, target: SpatialTarget) -> None:
    """Draw the x/right projection with an explicit target label."""
    x_m, _elevation_m, right_m = target.point.app_coordinates_m
    color = course_colors().flag
    tolerance = target.tolerance
    patch: Circle | Rectangle
    if isinstance(tolerance, (SphereTolerance, SurfaceCircleTolerance)):
        patch = Circle((x_m, right_m), tolerance.radius_m)
    elif isinstance(tolerance, BoxTolerance):
        half_x, _half_elevation, half_right = tolerance.half_extents_m
        patch = _rectangle(x_m, right_m, half_x, half_right)
    else:
        patch = _rectangle(
            x_m, right_m, tolerance.half_length_m, tolerance.half_width_m
        )
    _style_patch(patch, color)
    axes.add_patch(patch)
    _label_2d(axes, x_m, right_m, target.label, color)


def _rectangle(x_m: float, y_m: float, half_x: float, half_y: float) -> Rectangle:
    return Rectangle((x_m - half_x, y_m - half_y), 2.0 * half_x, 2.0 * half_y)


def _surface_band(x_m: float, elevation_m: float, half_x: float) -> Rectangle:
    visual_half_height = max(0.15, half_x * 0.015)
    return _rectangle(x_m, elevation_m, half_x, visual_half_height)


def _style_patch(patch: Any, color: str) -> None:
    patch.set_fill(False)
    patch.set_edgecolor(color)
    patch.set_linestyle("--")
    patch.set_linewidth(_LINE_WIDTH)
    patch.set_zorder(6)


def _label_2d(axes: Any, x_m: float, y_m: float, label: str, color: str) -> None:
    axes.scatter([x_m], [y_m], s=_POINT_SIZE, color=color, marker="x", zorder=7)
    axes.annotate(
        label,
        xy=(x_m, y_m),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=7,
        color=color,
        zorder=7,
    )


def draw_spatial_target_3d(axes: Any, target: SpatialTarget) -> None:
    """Draw the tolerance in display order (right, downrange, elevation)."""
    x_m, elevation_m, right_m = target.point.app_coordinates_m
    color = course_colors().flag
    axes.scatter(
        [right_m], [x_m], [elevation_m], s=_POINT_SIZE, color=color, marker="x"
    )
    tolerance = target.tolerance
    if isinstance(tolerance, SphereTolerance):
        _sphere(axes, x_m, elevation_m, right_m, tolerance.radius_m, color)
    elif isinstance(tolerance, BoxTolerance):
        _box(axes, x_m, elevation_m, right_m, tolerance.half_extents_m, color)
    elif isinstance(tolerance, SurfaceCircleTolerance):
        _surface_circle(axes, x_m, elevation_m, right_m, tolerance.radius_m, color)
    else:
        _surface_box(
            axes,
            x_m,
            elevation_m,
            right_m,
            tolerance.half_length_m,
            tolerance.half_width_m,
            color,
        )
    axes.text(right_m, x_m, elevation_m, target.label, color=color, fontsize=7)


def spatial_target_extents(target: SpatialTarget) -> tuple[float, float, float]:
    """Return positive carry, height, and absolute-lateral target bounds."""
    x_m, elevation_m, right_m = target.point.app_coordinates_m
    tolerance = target.tolerance
    if isinstance(tolerance, SphereTolerance):
        half_x = half_elevation = half_right = tolerance.radius_m
    elif isinstance(tolerance, SurfaceCircleTolerance):
        half_x = half_right = tolerance.radius_m
        half_elevation = 0.0
    elif isinstance(tolerance, BoxTolerance):
        half_x, half_elevation, half_right = tolerance.half_extents_m
    else:
        half_x = tolerance.half_length_m
        half_elevation = 0.0
        half_right = tolerance.half_width_m
    return (
        max(0.0, x_m + half_x),
        max(0.0, elevation_m + half_elevation),
        abs(right_m) + half_right,
    )


def _sphere(
    axes: Any, x_m: float, elevation_m: float, right_m: float, radius: float, color: str
) -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 48)
    for elevation_angle in (-np.pi / 4.0, 0.0, np.pi / 4.0):
        horizontal = radius * np.cos(elevation_angle)
        axes.plot(
            right_m + horizontal * np.sin(angle),
            x_m + horizontal * np.cos(angle),
            np.full_like(angle, elevation_m + radius * np.sin(elevation_angle)),
            color=color,
            linestyle="--",
            linewidth=0.8,
        )


def _box(
    axes: Any,
    x_m: float,
    elevation_m: float,
    right_m: float,
    half_extents: tuple[float, float, float],
    color: str,
) -> None:
    half_x, half_elevation, half_right = half_extents
    vertices = [
        (
            right_m + sr * half_right,
            x_m + sx * half_x,
            elevation_m + se * half_elevation,
        )
        for sr, sx, se in product((-1.0, 1.0), repeat=3)
    ]
    for first, second in _box_edges():
        start, end = vertices[first], vertices[second]
        axes.plot(
            *zip(start, end, strict=True), color=color, linestyle="--", linewidth=0.8
        )


def _box_edges() -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    for first in range(8):
        for bit in (1, 2, 4):
            second = first ^ bit
            if first < second:
                edges.append((first, second))
    return tuple(edges)


def _surface_circle(
    axes: Any, x_m: float, elevation_m: float, right_m: float, radius: float, color: str
) -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 64)
    axes.plot(
        right_m + radius * np.sin(angle),
        x_m + radius * np.cos(angle),
        np.full_like(angle, elevation_m),
        color=color,
        linestyle="--",
        linewidth=_LINE_WIDTH,
    )


def _surface_box(
    axes: Any,
    x_m: float,
    elevation_m: float,
    right_m: float,
    half_x: float,
    half_right: float,
    color: str,
) -> None:
    rights = right_m + np.array(
        [-half_right, half_right, half_right, -half_right, -half_right]
    )
    downrange = x_m + np.array([-half_x, -half_x, half_x, half_x, -half_x])
    axes.plot(
        rights,
        downrange,
        np.full(5, elevation_m),
        color=color,
        linestyle="--",
        linewidth=_LINE_WIDTH,
    )


__all__ = [
    "draw_spatial_target_3d",
    "draw_spatial_target_side",
    "draw_spatial_target_top",
    "spatial_target_extents",
]
