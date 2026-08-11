"""Validated records for the coplanar regional-material plan boundary."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .contract_types import (
    UNIT_SYSTEM_SI,
    GroundProvenance,
    GroundSurfaceProfile,
    Vector3,
    _finite,
    _integer,
    _text,
    _vector,
    _WireRecord,
)
from .regional_surface_types import PlanarSurfaceRegion
from .surface_motion_types import PlanarSurfaceDomain
from .surface_resolver import SurfaceResolver

REGIONAL_PLAN_REQUEST_SCHEMA_VERSION = "ground-regional-material-plan-request/v1"
REGIONAL_PLAN_RESULT_SCHEMA_VERSION = "ground-regional-material-plan-result/v1"
REGIONAL_PLAN_GEOMETRY_MODEL = "coplanar_static_material_overlays"
REGIONAL_PLAN_LIMITATIONS = (
    "coplanar_static_surfaces_only",
    "material_changes_only_no_geometry_or_velocity_discontinuities",
)
MAX_REGIONAL_PLAN_REGIONS = 4_096
MAX_REGIONAL_PLAN_WIRE_BYTES = 1_048_576


def _fixed_value(value: object, expected: str, name: str) -> str:
    if value != expected:
        raise ValueError(f"unsupported {name}: {value}")
    return expected


def _fixed_limitations(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("limitations must be an array")
    limitations = tuple(value)
    if limitations != REGIONAL_PLAN_LIMITATIONS:
        raise ValueError("limitations must declare the complete v1 qualification")
    return limitations


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    digest = _text(value, name).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return digest


@dataclass(frozen=True)
class GroundRegionalMaterialRegion(_WireRecord):
    """One finite regional surface record bound to a stable identity."""

    region_id: str
    precedence: int
    lower_coordinate_m: float
    upper_coordinate_m: float
    surface: GroundSurfaceProfile

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", _text(self.region_id, "region_id"))
        object.__setattr__(self, "precedence", _integer(self.precedence, "precedence"))
        lower = _finite(self.lower_coordinate_m, "lower_coordinate_m")
        upper = _finite(self.upper_coordinate_m, "upper_coordinate_m")
        if lower >= upper:
            raise ValueError("lower_coordinate_m must be below upper_coordinate_m")
        if type(self.surface) is not GroundSurfaceProfile:
            raise ValueError("region surface must be an exact GroundSurfaceProfile")
        object.__setattr__(self, "lower_coordinate_m", lower)
        object.__setattr__(self, "upper_coordinate_m", upper)

    @classmethod
    def from_dict(cls, payload: object) -> GroundRegionalMaterialRegion:
        """Parse one exact regional material record."""
        from .regional_plan_wire import regional_material_region_from_dict

        return regional_material_region_from_dict(payload)

    def to_runtime(
        self, request: GroundRegionalMaterialPlanRequest
    ) -> PlanarSurfaceRegion:
        """Bind the wire material record to the qualified runtime axis."""
        domain = PlanarSurfaceDomain(
            self.surface,
            request.axis_origin_m,
            request.axis_unit,
            self.lower_coordinate_m,
            self.upper_coordinate_m,
        )
        return PlanarSurfaceRegion(self.region_id, domain, self.precedence)


@dataclass(frozen=True)
class GroundRegionalMaterialPlanRequest(_WireRecord):
    """Versioned finite coplanar regional-material execution request."""

    request_id: str
    base_surface: GroundSurfaceProfile
    axis_origin_m: Vector3
    axis_unit: Vector3
    lower_coordinate_m: float
    upper_coordinate_m: float
    regions: tuple[GroundRegionalMaterialRegion, ...]
    provenance: GroundProvenance
    geometry_model: str = REGIONAL_PLAN_GEOMETRY_MODEL
    limitations: tuple[str, ...] = REGIONAL_PLAN_LIMITATIONS
    unit_system: str = UNIT_SYSTEM_SI
    schema_version: str = REGIONAL_PLAN_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id"))
        self._validate_exact_types()
        object.__setattr__(
            self, "axis_origin_m", _vector(self.axis_origin_m, "axis_origin_m")
        )
        object.__setattr__(self, "axis_unit", _vector(self.axis_unit, "axis_unit"))
        object.__setattr__(
            self,
            "lower_coordinate_m",
            _finite(self.lower_coordinate_m, "lower_coordinate_m"),
        )
        object.__setattr__(
            self,
            "upper_coordinate_m",
            _finite(self.upper_coordinate_m, "upper_coordinate_m"),
        )
        regions = tuple(self.regions)
        self._validate_region_count(regions)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(
            self,
            "geometry_model",
            _fixed_value(
                self.geometry_model, REGIONAL_PLAN_GEOMETRY_MODEL, "geometry_model"
            ),
        )
        object.__setattr__(self, "limitations", _fixed_limitations(self.limitations))
        object.__setattr__(
            self,
            "unit_system",
            _fixed_value(self.unit_system, UNIT_SYSTEM_SI, "unit_system"),
        )
        object.__setattr__(
            self,
            "schema_version",
            _fixed_value(
                self.schema_version,
                REGIONAL_PLAN_REQUEST_SCHEMA_VERSION,
                "schema_version",
            ),
        )
        self._validate_material_plan()

    def _validate_exact_types(self) -> None:
        if type(self.base_surface) is not GroundSurfaceProfile:
            raise ValueError("base_surface must be an exact GroundSurfaceProfile")
        if type(self.provenance) is not GroundProvenance:
            raise ValueError("provenance must be an exact GroundProvenance")

    @staticmethod
    def _validate_region_count(
        regions: tuple[GroundRegionalMaterialRegion, ...],
    ) -> None:
        if not regions:
            raise ValueError("regional material plan requires at least one region")
        if len(regions) > MAX_REGIONAL_PLAN_REGIONS:
            message = "regional material plan supports at most "
            raise ValueError(f"{message}{MAX_REGIONAL_PLAN_REGIONS} regions")
        if any(type(region) is not GroundRegionalMaterialRegion for region in regions):
            raise ValueError("regions must contain exact regional material records")

    def _validate_material_plan(self) -> None:
        if self.base_surface.surface_velocity_m_s != (0.0, 0.0, 0.0):
            raise ValueError("regional plan v1 supports static surfaces only")
        self._validate_unique_identifiers()
        if any(
            region.surface.surface_velocity_m_s != (0.0, 0.0, 0.0)
            for region in self.regions
        ):
            raise ValueError("regional plan v1 supports static surfaces only")
        base_geometry = (
            self.base_surface.frame,
            self.base_surface.height_m,
            self.base_surface.normal_unit,
        )
        if any(
            self._surface_geometry(region) != base_geometry for region in self.regions
        ):
            raise ValueError(
                "regional profiles must share the coplanar static geometry"
            )
        resolver = self.to_surface_resolver()
        if resolver.surface is not self.base_surface:
            raise RuntimeError("regional resolver must retain the request base surface")

    def _validate_unique_identifiers(self) -> None:
        identities = tuple(region.region_id for region in self.regions)
        precedences = tuple(region.precedence for region in self.regions)
        surface_ids = (self.base_surface.surface_id,) + tuple(
            region.surface.surface_id for region in self.regions
        )
        if len(identities) != len(set(identities)):
            raise ValueError("region_id values must be unique")
        if len(precedences) != len(set(precedences)):
            raise ValueError("precedence values must be unique")
        if len(surface_ids) != len(set(surface_ids)):
            raise ValueError("surface_id values must be unique")

    @staticmethod
    def _surface_geometry(
        region: GroundRegionalMaterialRegion,
    ) -> tuple[object, float, Vector3]:
        return (
            region.surface.frame,
            region.surface.height_m,
            region.surface.normal_unit,
        )

    @classmethod
    def from_dict(cls, payload: object) -> GroundRegionalMaterialPlanRequest:
        """Parse one exact regional plan request mapping."""
        from .regional_plan_wire import regional_material_plan_request_from_dict

        return regional_material_plan_request_from_dict(payload)

    def to_surface_resolver(self) -> SurfaceResolver:
        """Construct the exact runtime resolver represented by this plan."""
        base = PlanarSurfaceDomain(
            self.base_surface,
            self.axis_origin_m,
            self.axis_unit,
            self.lower_coordinate_m,
            self.upper_coordinate_m,
        )
        regions = tuple(region.to_runtime(self) for region in self.regions)
        return SurfaceResolver(base, regions)


@dataclass(frozen=True)
class GroundRegionalMaterialPlanResult(_WireRecord):
    """Validated material-plan result without synthesized physics output."""

    request: GroundRegionalMaterialPlanRequest
    request_sha256: str
    ordered_regions: tuple[GroundRegionalMaterialRegion, ...]
    provenance: GroundProvenance
    limitations: tuple[str, ...] = REGIONAL_PLAN_LIMITATIONS
    unit_system: str = UNIT_SYSTEM_SI
    schema_version: str = REGIONAL_PLAN_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.request) is not GroundRegionalMaterialPlanRequest:
            raise ValueError("request must be an exact regional material plan request")
        if type(self.provenance) is not GroundProvenance:
            raise ValueError("provenance must be an exact GroundProvenance")
        digest = _digest(self.request_sha256, "request_sha256")
        if digest != _sha256(self.request.to_json()):
            raise ValueError("request_sha256 does not match the embedded request")
        if self.provenance.input_sha256 != digest:
            raise ValueError("result provenance input_sha256 must match request_sha256")
        ordered = tuple(self.ordered_regions)
        if any(type(region) is not GroundRegionalMaterialRegion for region in ordered):
            raise ValueError("ordered_regions must contain exact regional records")
        expected = _canonical_regions(self.request)
        if _region_keys(ordered) != _region_keys(expected):
            raise ValueError("ordered_regions must use canonical precedence order")
        if ordered != expected:
            raise ValueError(
                "ordered region surface identity must match request material evidence"
            )
        object.__setattr__(self, "request_sha256", digest)
        object.__setattr__(self, "ordered_regions", ordered)
        object.__setattr__(self, "limitations", _fixed_limitations(self.limitations))
        object.__setattr__(
            self,
            "unit_system",
            _fixed_value(self.unit_system, UNIT_SYSTEM_SI, "unit_system"),
        )
        object.__setattr__(
            self,
            "schema_version",
            _fixed_value(
                self.schema_version,
                REGIONAL_PLAN_RESULT_SCHEMA_VERSION,
                "schema_version",
            ),
        )

    @property
    def base_surface(self) -> GroundSurfaceProfile:
        """Return the request-bound base surface without duplicating wire data."""
        return self.request.base_surface

    @classmethod
    def from_dict(cls, payload: object) -> GroundRegionalMaterialPlanResult:
        """Parse one exact regional plan result mapping."""
        from .regional_plan_wire import regional_material_plan_result_from_dict

        return regional_material_plan_result_from_dict(payload)


def _canonical_regions(
    request: GroundRegionalMaterialPlanRequest,
) -> tuple[GroundRegionalMaterialRegion, ...]:
    return tuple(
        sorted(request.regions, key=lambda item: (-item.precedence, item.region_id))
    )


def _region_keys(
    regions: tuple[GroundRegionalMaterialRegion, ...],
) -> tuple[tuple[int, str], ...]:
    return tuple((region.precedence, region.region_id) for region in regions)


def build_regional_material_plan_result(
    request: GroundRegionalMaterialPlanRequest,
    provenance: GroundProvenance,
) -> GroundRegionalMaterialPlanResult:
    """Validate and canonically order one request without running physics."""
    if type(request) is not GroundRegionalMaterialPlanRequest:
        raise ValueError("request must be an exact regional material plan request")
    return GroundRegionalMaterialPlanResult(
        request,
        _sha256(request.to_json()),
        _canonical_regions(request),
        provenance,
    )


def regional_plan_to_surface_resolver(
    request: GroundRegionalMaterialPlanRequest,
) -> SurfaceResolver:
    """Bind a validated wire request to the existing Python reference resolver."""
    if type(request) is not GroundRegionalMaterialPlanRequest:
        raise ValueError("request must be an exact regional material plan request")
    return request.to_surface_resolver()


__all__ = [
    "MAX_REGIONAL_PLAN_REGIONS",
    "MAX_REGIONAL_PLAN_WIRE_BYTES",
    "REGIONAL_PLAN_GEOMETRY_MODEL",
    "REGIONAL_PLAN_LIMITATIONS",
    "REGIONAL_PLAN_REQUEST_SCHEMA_VERSION",
    "REGIONAL_PLAN_RESULT_SCHEMA_VERSION",
    "GroundRegionalMaterialPlanRequest",
    "GroundRegionalMaterialPlanResult",
    "GroundRegionalMaterialRegion",
    "build_regional_material_plan_result",
    "regional_plan_to_surface_resolver",
]
