"""Text, labels, and scalar validation for the spatial-target editor."""

from __future__ import annotations

import math

from PyQt6.QtWidgets import QLineEdit

from shared.python.swing_sim.solver import (
    BoxTolerance,
    SpatialTarget,
    SphereTolerance,
    SurfaceCircleTolerance,
    TargetMiss,
)

DEFAULT_GROUND_SOURCE = "course.surface/default"
KIND_ITEMS = (
    ("Landing area (course surface)", "landing_area"),
    ("Aerial waypoint (3D volume)", "aerial_waypoint"),
)
FRAME_ITEMS = (
    ("App — x downrange, y up, z right", "app"),
    ("Flight — x forward, y left, z up", "flight"),
)
TOLERANCE_ITEMS = {
    "landing_area": (
        ("Surface circle", "surface_circle"),
        ("Surface corridor", "surface_corridor"),
    ),
    "aerial_waypoint": (("Sphere", "sphere"), ("Axis-aligned box", "box")),
}
COORDINATE_LABELS = {
    "app": ("Downrange x [m]", "Elevation y [m]", "Right z [m]"),
    "flight": ("Forward x [m]", "Left y [m]", "Up z [m]"),
}


def finite_number(edit: QLineEdit, label: str, *, positive: bool = False) -> float:
    """Parse one exact text field without spin-box clamping or coercion."""
    try:
        value = float(edit.text())
    except ValueError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if positive and value <= 0.0:
        raise ValueError(f"{label} must be greater than zero")
    return value


def target_summary(target: SpatialTarget) -> str:
    """Describe the active target's axes, units, provenance, and center."""
    x_m, elevation_m, right_m = target.point.app_coordinates_m
    kind = "landing area" if target.kind == "landing_area" else "aerial waypoint"
    source = target.ground_source or "absolute elevation"
    tolerance = target.tolerance
    if isinstance(tolerance, (SphereTolerance, SurfaceCircleTolerance)):
        shape = "sphere" if isinstance(tolerance, SphereTolerance) else "surface circle"
        geometry = f"{shape} radius {tolerance.radius_m:.2f} m"
    elif isinstance(tolerance, BoxTolerance):
        extents = ", ".join(f"{value:.2f}" for value in tolerance.half_extents_m)
        geometry = f"box half extents ({extents}) m"
    else:
        geometry = (
            f"surface corridor half-length {tolerance.half_length_m:.2f} m, "
            f"half-width {tolerance.half_width_m:.2f} m"
        )
    return (
        f'Current target: "{target.label}" · {kind} · canonical app frame '
        f"(x downrange, y up, z right), metres · center "
        f"({x_m:.2f}, {elevation_m:.2f}, {right_m:.2f}) · {geometry} · {source} · "
        f"authored in {target.point.source_frame} frame."
    )


def miss_summary(miss: TargetMiss, *, landing: bool) -> str:
    """Format a signed canonical closest-point residual."""
    status = "accepted" if miss.accepted else f"outside by {miss.distance_m:.2f} m"
    prefix = "Landing miss" if landing else "Closest trajectory miss"
    return (
        f"{prefix}: {status}; {_signed(miss.downrange_m, 'long', 'short')}, "
        f"{_signed(miss.elevation_m, 'high', 'low')}, "
        f"{_signed(miss.right_m, 'right', 'left')}."
    )


def _signed(value: float, positive: str, negative: str) -> str:
    direction = positive if value >= 0.0 else negative
    return f"{abs(value):.2f} m {direction}"


__all__ = [
    "COORDINATE_LABELS",
    "DEFAULT_GROUND_SOURCE",
    "FRAME_ITEMS",
    "KIND_ITEMS",
    "TOLERANCE_ITEMS",
    "finite_number",
    "miss_summary",
    "target_summary",
]
