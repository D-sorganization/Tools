"""Fail-closed resolver for coplanar material regions and finite edges."""

from __future__ import annotations

import math

from ._vector_math import dot
from .contract_records import GroundSimulationRequest
from .contract_types import GroundSurfaceProfile
from .regional_surface_types import (
    PlanarSurfaceRegion,
    SurfaceRegionTransitionCrossing,
)
from .surface_motion_types import (
    PlanarSurfaceDomain,
    SurfaceBoundaryCrossing,
    SurfaceKinematicSegment,
)

_ROOT_TOLERANCE_S = 1e-12
_GEOMETRY_TOLERANCE = 1e-10
_MAX_REGIONS = 4_096


class SurfaceResolver:
    """Resolve one static plane with finite, precedence-ordered materials."""

    def __init__(
        self,
        domain: PlanarSurfaceDomain,
        regions: tuple[PlanarSurfaceRegion, ...] = (),
    ) -> None:
        if type(domain) is not PlanarSurfaceDomain:
            raise ValueError("resolver requires an exact planar surface domain")
        selected_regions = tuple(regions)
        if len(selected_regions) > _MAX_REGIONS:
            raise ValueError(f"resolver supports at most {_MAX_REGIONS} regions")
        if any(type(region) is not PlanarSurfaceRegion for region in selected_regions):
            raise ValueError("resolver regions must be exact planar surface regions")
        self._domain = domain
        self._regions = selected_regions
        self._validate_regions()

    @property
    def domain(self) -> PlanarSurfaceDomain:
        """Return the immutable qualified domain."""
        return self._domain

    @property
    def surface(self) -> GroundSurfaceProfile:
        """Return the request-bound base surface profile."""
        return self._domain.surface

    @property
    def regions(self) -> tuple[PlanarSurfaceRegion, ...]:
        """Return immutable regional material overlays."""
        return self._regions

    def _validate_regions(self) -> None:
        identities = tuple(region.region_id for region in self._regions)
        precedences = tuple(region.precedence for region in self._regions)
        if len(set(identities)) != len(identities):
            raise ValueError("region_id values must be unique")
        if len(set(precedences)) != len(precedences):
            raise ValueError("region precedence values must be unique")
        for region in self._regions:
            self._validate_region_geometry(region)

    def _validate_region_geometry(self, region: PlanarSurfaceRegion) -> None:
        candidate = region.domain
        same_axis = (
            candidate.axis_origin_m == self._domain.axis_origin_m
            and candidate.axis_unit == self._domain.axis_unit
        )
        surface = candidate.surface
        same_geometry = (
            surface.frame is self.surface.frame
            and surface.height_m == self.surface.height_m
            and surface.normal_unit == self.surface.normal_unit
            and surface.surface_velocity_m_s == self.surface.surface_velocity_m_s
        )
        if not same_axis or not same_geometry:
            raise ValueError(
                "regional profiles must share the base coplanar geometry, axis, "
                "and surface velocity"
            )
        lower = candidate.lower_coordinate_m
        upper = candidate.upper_coordinate_m
        if lower is None or upper is None:
            raise RuntimeError("validated regional bounds must be finite")
        base_lower = self._domain.lower_coordinate_m
        base_upper = self._domain.upper_coordinate_m
        if base_lower is not None and lower < base_lower - _GEOMETRY_TOLERANCE:
            raise ValueError("regional lower bound must lie inside the base domain")
        if base_upper is not None and upper > base_upper + _GEOMETRY_TOLERANCE:
            raise ValueError("regional upper bound must lie inside the base domain")

    def region_at(
        self, position_m: tuple[float, float, float]
    ) -> PlanarSurfaceRegion | None:
        """Return the highest-precedence material overlay at one exact point."""
        candidates = tuple(
            region for region in self._regions if region.domain.contains(position_m)
        )
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.precedence)

    def surface_for_region(self, region_id: str | None) -> GroundSurfaceProfile:
        """Return the material profile for a validated active region identity."""
        if region_id is None:
            return self.surface
        for region in self._regions:
            if region.region_id == region_id:
                return region.domain.surface
        raise ValueError(f"unknown active region: {region_id}")

    def validate_handoff(self, position_m: tuple[float, float, float]) -> None:
        """Require the impact-bound base profile to own the initial contact."""
        if self.region_at(position_m) is not None:
            raise ValueError("skid/roll handoff must begin on the request base region")

    def validate_request(self, request: GroundSimulationRequest) -> None:
        """Require exact surface identity, geometry, motion, and material data."""
        if type(request) is not GroundSimulationRequest:
            raise ValueError("resolver requires an exact ground request")
        expected = request.surface
        actual = self.surface
        provider = (actual.surface_id, actual.provider_id, actual.provider_version)
        expected_provider = (
            expected.surface_id,
            expected.provider_id,
            expected.provider_version,
        )
        if provider != expected_provider:
            raise ValueError("resolver provider identity must match the request")
        if actual.normal_unit != expected.normal_unit:
            raise ValueError("resolver normal must match the request")
        geometry = (actual.frame, actual.height_m, actual.surface_velocity_m_s)
        expected_geometry = (
            expected.frame,
            expected.height_m,
            expected.surface_velocity_m_s,
        )
        if geometry != expected_geometry:
            raise ValueError(
                "resolver plane and surface velocity must match the request"
            )
        if actual != expected:
            raise ValueError("resolver material must match the request")

    def first_crossing(
        self,
        segment: SurfaceKinematicSegment,
    ) -> SurfaceBoundaryCrossing | None:
        """Return the exact first outward finite-edge crossing, if any."""
        if type(segment) is not SurfaceKinematicSegment:
            raise ValueError("crossing requires an exact kinematic segment")
        if not self._domain.contains(segment.start_position_m):
            raise ValueError("motion segment must start inside the surface domain")
        candidates: list[SurfaceBoundaryCrossing] = []
        bounds = (
            (self._domain.lower_coordinate_m, -1.0),
            (self._domain.upper_coordinate_m, 1.0),
        )
        for bound, direction in bounds:
            if bound is None:
                continue
            crossing = self._crossing_at(segment, bound, direction)
            if crossing is not None:
                candidates.append(crossing)
        if not candidates:
            return None
        return min(candidates, key=lambda item: item.time_offset_s)

    def first_transition(
        self,
        segment: SurfaceKinematicSegment,
        active_region_id: str | None,
    ) -> SurfaceRegionTransitionCrossing | None:
        """Return the first exact coplanar material change within a segment."""
        if type(segment) is not SurfaceKinematicSegment:
            raise ValueError("transition requires an exact kinematic segment")
        self.surface_for_region(active_region_id)
        if not self._domain.contains(segment.start_position_m):
            raise ValueError("motion segment must start inside the surface domain")
        crossings = tuple(
            crossing
            for boundary in self._region_boundaries()
            for crossing in (self._transition_at(segment, boundary, active_region_id),)
            if crossing is not None
        )
        if not crossings:
            return None
        return min(crossings, key=lambda item: item.time_offset_s)

    def _region_boundaries(self) -> tuple[float, ...]:
        values = {
            bound
            for region in self._regions
            for bound in (
                region.domain.lower_coordinate_m,
                region.domain.upper_coordinate_m,
            )
            if bound is not None
        }
        return tuple(sorted(values))

    def _transition_at(
        self,
        segment: SurfaceKinematicSegment,
        boundary_m: float,
        active_region_id: str | None,
    ) -> SurfaceRegionTransitionCrossing | None:
        origin = self._domain.coordinate(segment.start_position_m) - boundary_m
        speed = dot(segment.start_velocity_m_s, self._domain.axis_unit)
        acceleration = dot(segment.acceleration_m_s2, self._domain.axis_unit)
        for time_s in _quadratic_roots(0.5 * acceleration, speed, origin):
            if not (
                -_ROOT_TOLERANCE_S <= time_s <= segment.duration_s + _ROOT_TOLERANCE_S
            ):
                continue
            bounded_time = max(0.0, min(segment.duration_s, time_s))
            direction = speed + acceleration * bounded_time
            if abs(direction) <= _ROOT_TOLERANCE_S:
                direction = acceleration
            if abs(direction) <= _ROOT_TOLERANCE_S:
                continue
            outgoing = math.nextafter(
                boundary_m,
                math.inf if direction > 0.0 else -math.inf,
            )
            next_region = self._region_at_coordinate(outgoing)
            next_id = None if next_region is None else next_region.region_id
            if next_id == active_region_id:
                continue
            return SurfaceRegionTransitionCrossing(
                bounded_time,
                segment.position_at(bounded_time),
                boundary_m,
                active_region_id,
                next_id,
                self.surface_for_region(next_id),
            )
        return None

    def _region_at_coordinate(self, coordinate_m: float) -> PlanarSurfaceRegion | None:
        candidates = tuple(
            region
            for region in self._regions
            if _coordinate_in_domain(coordinate_m, region.domain)
        )
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.precedence)

    def _crossing_at(
        self,
        segment: SurfaceKinematicSegment,
        boundary_m: float,
        outward_sign: float,
    ) -> SurfaceBoundaryCrossing | None:
        origin = self._domain.coordinate(segment.start_position_m) - boundary_m
        speed = dot(segment.start_velocity_m_s, self._domain.axis_unit)
        half_acceleration = 0.5 * dot(
            segment.acceleration_m_s2,
            self._domain.axis_unit,
        )
        for time_s in _quadratic_roots(half_acceleration, speed, origin):
            if (
                not -_ROOT_TOLERANCE_S
                <= time_s
                <= segment.duration_s + _ROOT_TOLERANCE_S
            ):
                continue
            time_s = max(0.0, min(segment.duration_s, time_s))
            derivative = speed + 2.0 * half_acceleration * time_s
            starts_accelerating_outward = (
                time_s <= _ROOT_TOLERANCE_S
                and abs(derivative) <= _ROOT_TOLERANCE_S
                and outward_sign * 2.0 * half_acceleration > 0.0
            )
            if outward_sign * derivative <= 0.0 and not starts_accelerating_outward:
                continue
            return SurfaceBoundaryCrossing(
                time_s,
                segment.position_at(time_s),
                boundary_m,
            )
        return None


def _quadratic_roots(a: float, b: float, c: float) -> tuple[float, ...]:
    if abs(a) <= 1e-15:
        return () if abs(b) <= 1e-15 else (-c / b,)
    discriminant = b * b - 4.0 * a * c
    if discriminant < -1e-14:
        return ()
    root = math.sqrt(max(0.0, discriminant))
    roots = ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))
    return tuple(sorted(set(roots)))


def _coordinate_in_domain(coordinate_m: float, domain: PlanarSurfaceDomain) -> bool:
    lower = domain.lower_coordinate_m
    upper = domain.upper_coordinate_m
    return (lower is None or coordinate_m >= lower) and (
        upper is None or coordinate_m <= upper
    )


__all__ = ["SurfaceResolver"]
