"""Typed non-wire contracts for coplanar regional material transitions."""

from __future__ import annotations

from dataclasses import dataclass

from .contract_types import GroundSurfaceProfile, Vector3
from .surface_motion_types import PlanarSurfaceDomain, _finite, _vector


@dataclass(frozen=True)
class PlanarSurfaceRegion:
    """One finite coplanar material overlay with deterministic precedence."""

    region_id: str
    domain: PlanarSurfaceDomain
    precedence: int

    def __post_init__(self) -> None:
        if not isinstance(self.region_id, str) or not self.region_id.strip():
            raise ValueError("region_id must be nonempty")
        if self.region_id != self.region_id.strip():
            raise ValueError("region_id must not have edge whitespace")
        if type(self.domain) is not PlanarSurfaceDomain:
            raise ValueError("region domain must be an exact planar domain")
        if (
            self.domain.lower_coordinate_m is None
            or self.domain.upper_coordinate_m is None
        ):
            raise ValueError("regional material overlays require two finite bounds")
        if isinstance(self.precedence, bool) or not isinstance(self.precedence, int):
            raise ValueError("region precedence must be a nonnegative integer")
        if self.precedence < 0:
            raise ValueError("region precedence must be a nonnegative integer")


@dataclass(frozen=True)
class SurfaceRegionTransitionCrossing:
    """Exact coplanar material-region transition without a state impulse."""

    time_offset_s: float
    position_m: Vector3
    boundary_coordinate_m: float
    from_region_id: str | None
    to_region_id: str | None
    to_surface: GroundSurfaceProfile

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "time_offset_s", _finite(self.time_offset_s, "time_offset_s")
        )
        object.__setattr__(self, "position_m", _vector(self.position_m, "position_m"))
        object.__setattr__(
            self,
            "boundary_coordinate_m",
            _finite(self.boundary_coordinate_m, "boundary_coordinate_m"),
        )
        if self.time_offset_s < 0.0:
            raise ValueError("transition crossing time must be nonnegative")
        for name in ("from_region_id", "to_region_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be nonempty when provided")
        if self.from_region_id == self.to_region_id:
            raise ValueError("surface transition must change active regions")
        if type(self.to_surface) is not GroundSurfaceProfile:
            raise ValueError("transition surface must be an exact surface profile")


@dataclass(frozen=True)
class SurfaceRegionTransition:
    """Internal identity ledger for one emitted regional-boundary event."""

    event_sequence: int
    time_s: float
    position_m: Vector3
    from_region_id: str | None
    to_region_id: str | None
    from_surface_id: str
    to_surface_id: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.event_sequence, bool)
            or not isinstance(self.event_sequence, int)
            or self.event_sequence < 0
        ):
            raise ValueError("transition event_sequence must be nonnegative")
        object.__setattr__(self, "time_s", _finite(self.time_s, "time_s"))
        object.__setattr__(self, "position_m", _vector(self.position_m, "position_m"))
        if self.time_s < 0.0:
            raise ValueError("transition time must be nonnegative")
        for name in ("from_surface_id", "to_surface_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be nonempty")
        if self.from_region_id == self.to_region_id:
            raise ValueError("transition evidence must change active regions")


__all__ = [
    "PlanarSurfaceRegion",
    "SurfaceRegionTransition",
    "SurfaceRegionTransitionCrossing",
]
