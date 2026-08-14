"""Fail-closed parsers for regional material plan v1 documents."""

from __future__ import annotations

from typing import Any, cast

from .contract_types import GroundProvenance, GroundSurfaceProfile, Vector3
from .regional_plan_records import (
    MAX_REGIONAL_PLAN_WIRE_BYTES,
    GroundRegionalMaterialPlanRequest,
    GroundRegionalMaterialPlanResult,
    GroundRegionalMaterialRegion,
)
from .strict_json import strict_json_object

_REGION_FIELDS = {
    "lower_coordinate_m",
    "precedence",
    "region_id",
    "surface",
    "upper_coordinate_m",
}
_REQUEST_FIELDS = {
    "axis_origin_m",
    "axis_unit",
    "base_surface",
    "geometry_model",
    "limitations",
    "lower_coordinate_m",
    "provenance",
    "regions",
    "request_id",
    "schema_version",
    "unit_system",
    "upper_coordinate_m",
}
_RESULT_FIELDS = {
    "limitations",
    "ordered_regions",
    "provenance",
    "request",
    "request_sha256",
    "schema_version",
    "unit_system",
}


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _exact(payload: dict[str, Any], fields: set[str], name: str) -> None:
    if set(payload) != fields:
        raise ValueError(f"{name} fields do not match v1 schema")


def _surface(value: object) -> GroundSurfaceProfile:
    payload = _mapping(value, "surface")
    return cast(GroundSurfaceProfile, GroundSurfaceProfile.from_dict(payload))


def _provenance(value: object) -> GroundProvenance:
    payload = _mapping(value, "provenance")
    return cast(GroundProvenance, GroundProvenance.from_dict(payload))


def _vector(value: object, name: str) -> Vector3:
    return cast(Vector3, tuple(_sequence(value, name)))


def regional_material_region_from_dict(
    payload: object,
) -> GroundRegionalMaterialRegion:
    """Parse one exact regional material mapping."""
    data = _mapping(payload, "regional material")
    _exact(data, _REGION_FIELDS, "regional material")
    return GroundRegionalMaterialRegion(
        data["region_id"],
        data["precedence"],
        data["lower_coordinate_m"],
        data["upper_coordinate_m"],
        _surface(data["surface"]),
    )


def _regions(value: object, name: str) -> tuple[GroundRegionalMaterialRegion, ...]:
    return tuple(
        regional_material_region_from_dict(item) for item in _sequence(value, name)
    )


def regional_material_plan_request_from_dict(
    payload: object,
) -> GroundRegionalMaterialPlanRequest:
    """Parse one exact regional material plan request mapping."""
    data = _mapping(payload, "regional material plan request")
    _exact(data, _REQUEST_FIELDS, "regional material plan request")
    return GroundRegionalMaterialPlanRequest(
        data["request_id"],
        _surface(data["base_surface"]),
        _vector(data["axis_origin_m"], "axis_origin_m"),
        _vector(data["axis_unit"], "axis_unit"),
        data["lower_coordinate_m"],
        data["upper_coordinate_m"],
        _regions(data["regions"], "regions"),
        _provenance(data["provenance"]),
        data["geometry_model"],
        tuple(_sequence(data["limitations"], "limitations")),
        data["unit_system"],
        data["schema_version"],
    )


def regional_material_plan_result_from_dict(
    payload: object,
) -> GroundRegionalMaterialPlanResult:
    """Parse one exact regional material plan result mapping."""
    data = _mapping(payload, "regional material plan result")
    _exact(data, _RESULT_FIELDS, "regional material plan result")
    return GroundRegionalMaterialPlanResult(
        regional_material_plan_request_from_dict(data["request"]),
        data["request_sha256"],
        _regions(data["ordered_regions"], "ordered_regions"),
        _provenance(data["provenance"]),
        tuple(_sequence(data["limitations"], "limitations")),
        data["unit_system"],
        data["schema_version"],
    )


def _strict_document(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError("regional material plan JSON must be text")
    if len(text.encode("utf-8")) > MAX_REGIONAL_PLAN_WIRE_BYTES:
        raise ValueError("regional material plan exceeds maximum wire size")
    document: dict[str, Any] = strict_json_object(text)
    return document


def regional_material_plan_request_from_json(
    text: str,
) -> GroundRegionalMaterialPlanRequest:
    """Parse one bounded strict request JSON document."""
    return regional_material_plan_request_from_dict(_strict_document(text))


def regional_material_plan_result_from_json(
    text: str,
) -> GroundRegionalMaterialPlanResult:
    """Parse one bounded strict result JSON document."""
    return regional_material_plan_result_from_dict(_strict_document(text))


__all__ = [
    "regional_material_plan_request_from_dict",
    "regional_material_plan_request_from_json",
    "regional_material_plan_result_from_dict",
    "regional_material_plan_result_from_json",
    "regional_material_region_from_dict",
]
