"""UI-neutral three-dimensional target contracts in the Tools app frame.

The canonical coordinate order is ``(x downrange, y elevation, z right)`` in
metres.  A miss vector is ``actual - closest accepted point`` in that order;
therefore positive components mean long, high, and right respectively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from shared.python.swing_sim.flight.frames import from_flight_frame, to_flight_frame

from ._target_validation import (
    ElevationSource,
    TargetFrame,
    TargetKind,
    Vector3,
    finite_float,
    nonempty_text,
    positive_float,
    target_frame,
    vector3,
)
from .targets import TargetRegion

_ACCEPTANCE_ABS_TOL_M = 1e-12


@dataclass(frozen=True)
class TargetPoint:
    """Canonical app-frame target point with source-frame provenance."""

    x_m: float
    elevation_m: float
    right_m: float
    source_frame: TargetFrame = "app"

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_m", finite_float(self.x_m, "x_m"))
        object.__setattr__(
            self, "elevation_m", finite_float(self.elevation_m, "elevation_m")
        )
        object.__setattr__(self, "right_m", finite_float(self.right_m, "right_m"))
        object.__setattr__(
            self, "source_frame", target_frame(self.source_frame, "source_frame")
        )

    @classmethod
    def from_frame(
        cls, coordinates_m: object, source_frame: TargetFrame
    ) -> TargetPoint:
        """Create a canonical point from app- or flight-frame coordinates."""
        coordinates = vector3(coordinates_m, "coordinates_m")
        validated_frame = target_frame(source_frame, "source_frame")
        if validated_frame == "flight":
            converted = from_flight_frame(np.asarray(coordinates, dtype=float))
            coordinates = tuple(float(value) for value in converted)  # type: ignore[assignment]
        return cls(*coordinates, source_frame=validated_frame)

    @property
    def app_coordinates_m(self) -> Vector3:
        """Return ``(downrange, elevation, right)`` in metres."""
        return (self.x_m, self.elevation_m, self.right_m)

    def coordinates_in(self, frame: TargetFrame) -> Vector3:
        """Return this point in the requested existing app/flight frame."""
        validated_frame = target_frame(frame)
        if validated_frame == "app":
            return self.app_coordinates_m
        converted = to_flight_frame(np.asarray(self.app_coordinates_m, dtype=float))
        return tuple(float(value) for value in converted)  # type: ignore[return-value]


@dataclass(frozen=True)
class SphereTolerance:
    """Spherical three-dimensional acceptance volume."""

    radius_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius_m", positive_float(self.radius_m, "radius_m"))


@dataclass(frozen=True)
class BoxTolerance:
    """Axis-aligned 3D half extents in app-frame coordinate order."""

    half_extents_m: Vector3

    def __post_init__(self) -> None:
        extents = vector3(self.half_extents_m, "half_extents_m")
        if any(value <= 0.0 for value in extents):
            raise ValueError("half_extents_m must be finite and > 0")
        object.__setattr__(self, "half_extents_m", extents)


@dataclass(frozen=True)
class SurfaceCircleTolerance:
    """Circular landing acceptance region embedded in a course surface."""

    radius_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius_m", positive_float(self.radius_m, "radius_m"))


@dataclass(frozen=True)
class SurfaceCorridorTolerance:
    """Rectangular landing corridor embedded in a course surface."""

    half_length_m: float
    half_width_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "half_length_m",
            positive_float(self.half_length_m, "half_length_m"),
        )
        object.__setattr__(
            self, "half_width_m", positive_float(self.half_width_m, "half_width_m")
        )


AcceptanceGeometry: TypeAlias = (
    SphereTolerance | BoxTolerance | SurfaceCircleTolerance | SurfaceCorridorTolerance
)


@dataclass(frozen=True)
class TargetMiss:
    """Closest-point miss result in the canonical app frame."""

    closest_point_m: Vector3
    vector_m: Vector3
    distance_m: float
    accepted: bool

    def __post_init__(self) -> None:
        closest = vector3(self.closest_point_m, "closest_point_m")
        vector = vector3(self.vector_m, "vector_m")
        distance = finite_float(self.distance_m, "distance_m")
        if distance < 0.0:
            raise ValueError("distance_m must be >= 0")
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a bool")
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isclose(distance, norm, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("distance_m must equal the norm of vector_m")
        if self.accepted != (distance <= _ACCEPTANCE_ABS_TOL_M):
            raise ValueError("accepted must match the zero-distance condition")
        object.__setattr__(self, "closest_point_m", closest)
        object.__setattr__(self, "vector_m", vector)
        object.__setattr__(self, "distance_m", distance)

    @property
    def downrange_m(self) -> float:
        """Signed downrange miss: positive is long."""
        return self.vector_m[0]

    @property
    def elevation_m(self) -> float:
        """Signed elevation miss: positive is high."""
        return self.vector_m[1]

    @property
    def right_m(self) -> float:
        """Signed lateral miss: positive is right."""
        return self.vector_m[2]


@dataclass(frozen=True)
class SpatialTarget:
    """Version-ready target independent of UI, solver, and trajectory model."""

    label: str
    kind: TargetKind
    point: TargetPoint
    tolerance: AcceptanceGeometry
    elevation_source: ElevationSource
    ground_source: str | None = None
    units: str = "m"
    frame: str = "app"

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", nonempty_text(self.label, "label"))
        if not isinstance(self.kind, str):
            raise TypeError("kind must be a string")
        if self.kind not in ("landing_area", "aerial_waypoint"):
            raise ValueError(f"unknown target kind {self.kind!r}")
        if not isinstance(self.point, TargetPoint):
            raise TypeError("point must be a TargetPoint")
        geometry_types = (
            SphereTolerance,
            BoxTolerance,
            SurfaceCircleTolerance,
            SurfaceCorridorTolerance,
        )
        if not isinstance(self.tolerance, geometry_types):
            raise TypeError("tolerance must be an acceptance geometry")
        if not isinstance(self.elevation_source, str):
            raise TypeError("elevation_source must be a string")
        if not isinstance(self.units, str):
            raise TypeError("units must be a string")
        if self.units != "m":
            raise ValueError("units must be 'm'")
        if not isinstance(self.frame, str):
            raise TypeError("frame must be a string")
        if self.frame != "app":
            raise ValueError("frame must be 'app'")
        self._validate_kind_contract()

    def _validate_kind_contract(self) -> None:
        surface_types = (SurfaceCircleTolerance, SurfaceCorridorTolerance)
        volume_types = (SphereTolerance, BoxTolerance)
        if self.kind == "landing_area":
            if not isinstance(self.tolerance, surface_types):
                raise ValueError("landing_area requires a surface tolerance")
            if self.elevation_source != "course_surface":
                raise ValueError("landing_area requires course_surface elevation")
            if self.ground_source is None:
                raise ValueError("landing_area requires ground_source")
            nonempty_text(self.ground_source, "ground_source")
            return
        if not isinstance(self.tolerance, volume_types):
            raise ValueError("aerial_waypoint requires a 3D volume tolerance")
        if self.elevation_source != "absolute":
            raise ValueError("aerial_waypoint requires absolute elevation")
        if self.ground_source is not None:
            raise ValueError("aerial_waypoint ground_source must be None")

    def miss(self, actual_app_m: object) -> TargetMiss:
        """Return closest-point signed miss from an app-frame point.

        Postcondition: ``distance_m`` is the Euclidean norm of ``vector_m``;
        accepted points have a zero vector and distance.
        """
        actual = vector3(actual_app_m, "actual_app_m")
        closest = self._closest_point(actual)
        vector: Vector3 = (
            actual[0] - closest[0],
            actual[1] - closest[1],
            actual[2] - closest[2],
        )
        distance = math.sqrt(sum(value * value for value in vector))
        accepted = distance <= _ACCEPTANCE_ABS_TOL_M
        if accepted:
            vector = (0.0, 0.0, 0.0)
            distance = 0.0
        return TargetMiss(closest, vector, distance, accepted)

    def miss_from_frame(self, actual_m: object, frame: TargetFrame) -> TargetMiss:
        """Return a miss after converting an app- or flight-frame point."""
        actual = TargetPoint.from_frame(actual_m, source_frame=frame)
        return self.miss(actual.app_coordinates_m)

    def _closest_point(self, actual: Vector3) -> Vector3:
        center = self.point.app_coordinates_m
        tolerance = self.tolerance
        if isinstance(tolerance, SphereTolerance):
            return _closest_sphere(actual, center, tolerance.radius_m)
        if isinstance(tolerance, BoxTolerance):
            return _closest_box(actual, center, tolerance.half_extents_m)
        if isinstance(tolerance, SurfaceCircleTolerance):
            return _closest_surface_circle(actual, center, tolerance.radius_m)
        return _closest_surface_corridor(actual, center, tolerance)

    @classmethod
    def from_target_region(
        cls,
        region: TargetRegion,
        surface_elevation_m: float = 0.0,
        ground_source: str = "course.surface/default",
        label: str | None = None,
    ) -> SpatialTarget:
        """Lift the legacy 2D region into an explicit course-surface target."""
        if not isinstance(region, TargetRegion):
            raise TypeError("region must be a TargetRegion")
        elevation = finite_float(surface_elevation_m, "surface_elevation_m")
        lateral = region.lateral_m if region.kind == "green" else 0.0
        point = TargetPoint(region.distance_m, elevation, lateral)
        if region.kind == "green":
            tolerance: AcceptanceGeometry = SurfaceCircleTolerance(region.radius_m)
        else:
            tolerance = SurfaceCorridorTolerance(
                region.band_half_length_m, region.half_width_m
            )
        return cls(
            label=label or f"{region.kind.title()} Target",
            kind="landing_area",
            point=point,
            tolerance=tolerance,
            elevation_source="course_surface",
            ground_source=ground_source,
        )

    def to_target_region(self) -> TargetRegion:
        """Project a surface target back to the unchanged legacy 2D API."""
        tolerance = self.tolerance
        if isinstance(tolerance, SurfaceCircleTolerance):
            return TargetRegion(
                kind="green",
                distance_m=self.point.x_m,
                radius_m=tolerance.radius_m,
                lateral_m=self.point.right_m,
            )
        if isinstance(tolerance, SurfaceCorridorTolerance):
            return TargetRegion(
                kind="fairway",
                distance_m=self.point.x_m,
                band_half_length_m=tolerance.half_length_m,
                half_width_m=tolerance.half_width_m,
            )
        raise ValueError("only landing_area surface targets have a 2D projection")


def _closest_sphere(actual: Vector3, center: Vector3, radius_m: float) -> Vector3:
    delta = tuple(value - origin for value, origin in zip(actual, center, strict=True))
    distance = math.sqrt(sum(value * value for value in delta))
    if distance <= radius_m:
        return actual
    scale = radius_m / distance
    return tuple(
        origin + scale * value for origin, value in zip(center, delta, strict=True)
    )  # type: ignore[return-value]


def _closest_box(actual: Vector3, center: Vector3, half_extents: Vector3) -> Vector3:
    if all(
        abs(value - origin) <= extent
        for value, origin, extent in zip(actual, center, half_extents, strict=True)
    ):
        return actual
    return tuple(
        min(max(value, origin - extent), origin + extent)
        for value, origin, extent in zip(actual, center, half_extents, strict=True)
    )  # type: ignore[return-value]


def _closest_surface_circle(
    actual: Vector3, center: Vector3, radius_m: float
) -> Vector3:
    delta_x = actual[0] - center[0]
    delta_right = actual[2] - center[2]
    radial_distance = math.hypot(delta_x, delta_right)
    if radial_distance <= radius_m:
        return (actual[0], center[1], actual[2])
    scale = radius_m / radial_distance
    return (
        center[0] + scale * delta_x,
        center[1],
        center[2] + scale * delta_right,
    )


def _closest_surface_corridor(
    actual: Vector3,
    center: Vector3,
    tolerance: SurfaceCorridorTolerance,
) -> Vector3:
    return (
        min(
            max(actual[0], center[0] - tolerance.half_length_m),
            center[0] + tolerance.half_length_m,
        ),
        center[1],
        min(
            max(actual[2], center[2] - tolerance.half_width_m),
            center[2] + tolerance.half_width_m,
        ),
    )


__all__ = [
    "AcceptanceGeometry",
    "BoxTolerance",
    "ElevationSource",
    "SpatialTarget",
    "SphereTolerance",
    "SurfaceCircleTolerance",
    "SurfaceCorridorTolerance",
    "TargetFrame",
    "TargetKind",
    "TargetMiss",
    "TargetPoint",
]
