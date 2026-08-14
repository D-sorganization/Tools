"""UI-facing target residual selection for a completed flight trajectory."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np

from shared.python.swing_sim.solver import (
    BoxTolerance,
    SpatialTarget,
    SphereTolerance,
    TargetMiss,
)

_RESOLVED_FLAT_GROUND_SOURCES = frozenset(
    {"course.surface/default", "legacy.course_surface/default"}
)


class LandingSurfaceResolutionError(ValueError):
    """Typed fail-closed diagnostic for unavailable course terrain."""

    def __init__(
        self,
        code: Literal["UNSUPPORTED_SURFACE_ELEVATION", "UNRESOLVED_GROUND_SOURCE"],
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


def validate_landing_surface(target: SpatialTarget) -> None:
    """Require the only surface currently resolved by the flat course model."""
    if target.kind != "landing_area":
        return
    if target.ground_source not in _RESOLVED_FLAT_GROUND_SOURCES:
        raise LandingSurfaceResolutionError(
            "UNRESOLVED_GROUND_SOURCE",
            f"ground source {target.ground_source!r} is not resolved by this course",
        )
    elevation_m = target.point.app_coordinates_m[1]
    if abs(elevation_m) > 1e-12:
        raise LandingSurfaceResolutionError(
            "UNSUPPORTED_SURFACE_ELEVATION",
            "landing elevation must be 0 m for the resolved flat course surface; "
            "terrain elevation is not available",
        )


def trajectory_target_miss(
    target: SpatialTarget, positions_m: np.ndarray
) -> TargetMiss:
    """Return landing or closest-passage miss using canonical app coordinates.

    Landing targets use the final sample's horizontal coordinates projected to
    the target's course-surface elevation. This intentionally removes ball
    radius and tee/support height from a surface-contact assessment. Aerial
    waypoints retain full 3D coordinates and use exact segment passage.
    """
    if not isinstance(target, SpatialTarget):
        raise TypeError("target must be a SpatialTarget")
    positions = np.asarray(positions_m, dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions):
        raise ValueError("positions_m must have shape (N, 3) with N > 0")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must be finite")
    if target.kind == "landing_area":
        validate_landing_surface(target)
        landing_point = positions[-1].copy()
        landing_point[1] = 0.0
        return target.miss(landing_point)
    misses = tuple(
        _segment_miss(target, start, end)
        for start, end in zip(positions[:-1], positions[1:], strict=True)
    )
    if not misses:
        return target.miss(positions[0])
    return min(misses, key=lambda miss: miss.distance_m)


def _segment_miss(
    target: SpatialTarget, start: np.ndarray, end: np.ndarray
) -> TargetMiss:
    tolerance = target.tolerance
    if isinstance(tolerance, SphereTolerance):
        point = _closest_point_to_center(target, start, end)
        return target.miss(point)
    if isinstance(tolerance, BoxTolerance):
        return _closest_box_segment_miss(target, tolerance, start, end)
    raise ValueError("aerial waypoint requires sphere or box tolerance")


def _closest_point_to_center(
    target: SpatialTarget, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    center = np.asarray(target.point.app_coordinates_m, dtype=float)
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared == 0.0:
        return start
    fraction = float(np.dot(center - start, direction) / length_squared)
    bounded_fraction = min(max(fraction, 0.0), 1.0)
    return cast(np.ndarray, start + bounded_fraction * direction)


def _closest_box_segment_miss(
    target: SpatialTarget,
    tolerance: BoxTolerance,
    start: np.ndarray,
    end: np.ndarray,
) -> TargetMiss:
    center = np.asarray(target.point.app_coordinates_m, dtype=float)
    extents = np.asarray(tolerance.half_extents_m, dtype=float)
    lower, upper = center - extents, center + extents
    direction = end - start
    breakpoints = _box_breakpoints(start, direction, lower, upper)
    candidates = set(breakpoints)
    for low_t, high_t in zip(breakpoints[:-1], breakpoints[1:], strict=True):
        stationary = _interval_stationary_t(
            start, direction, lower, upper, low_t, high_t
        )
        if stationary is not None:
            candidates.add(stationary)
    misses = tuple(target.miss(start + time * direction) for time in candidates)
    return min(misses, key=lambda miss: miss.distance_m)


def _box_breakpoints(
    start: np.ndarray,
    direction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[float, ...]:
    values = {0.0, 1.0}
    for axis in range(3):
        if direction[axis] == 0.0:
            continue
        for boundary in (lower[axis], upper[axis]):
            fraction = float((boundary - start[axis]) / direction[axis])
            if 0.0 <= fraction <= 1.0:
                values.add(fraction)
    return tuple(sorted(values))


def _interval_stationary_t(
    start: np.ndarray,
    direction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    low_t: float,
    high_t: float,
) -> float | None:
    midpoint = start + (low_t + high_t) * 0.5 * direction
    boundary = np.where(
        midpoint < lower, lower, np.where(midpoint > upper, upper, midpoint)
    )
    active = (midpoint < lower) | (midpoint > upper)
    denominator = float(np.dot(direction[active], direction[active]))
    if denominator == 0.0:
        return None
    numerator = float(np.dot(direction[active], start[active] - boundary[active]))
    fraction = -numerator / denominator
    if low_t <= fraction <= high_t:
        return fraction
    return None


__all__ = [
    "LandingSurfaceResolutionError",
    "trajectory_target_miss",
    "validate_landing_surface",
]
