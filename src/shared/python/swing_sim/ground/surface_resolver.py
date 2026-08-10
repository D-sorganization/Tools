"""Fail-closed resolver for one immutable planar surface and finite edges."""

from __future__ import annotations

import math

from ._vector_math import dot
from .contract_records import GroundSimulationRequest
from .contract_types import GroundSurfaceProfile
from .surface_motion_types import (
    PlanarSurfaceDomain,
    SurfaceBoundaryCrossing,
    SurfaceKinematicSegment,
)

_ROOT_TOLERANCE_S = 1e-12


class SurfaceResolver:
    """Resolve a request's one static plane and optional finite axis bounds."""

    def __init__(self, domain: PlanarSurfaceDomain) -> None:
        if type(domain) is not PlanarSurfaceDomain:
            raise ValueError("resolver requires an exact planar surface domain")
        self._domain = domain

    @property
    def domain(self) -> PlanarSurfaceDomain:
        """Return the immutable qualified domain."""
        return self._domain

    @property
    def surface(self) -> GroundSurfaceProfile:
        """Return the one immutable surface profile."""
        return self._domain.surface

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


__all__ = ["SurfaceResolver"]
