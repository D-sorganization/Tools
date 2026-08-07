"""Deterministic, versioned serialization for :mod:`spatial_targets`.

Version 1 stores canonical app-frame coordinates while retaining the frame in
which the target was authored.  Unversioned ``TargetRegion`` mappings are the
only accepted legacy inputs and migrate to explicit course-surface targets.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from ._target_validation import finite_float
from .spatial_targets import (
    AcceptanceGeometry,
    BoxTolerance,
    SpatialTarget,
    SphereTolerance,
    SurfaceCircleTolerance,
    SurfaceCorridorTolerance,
    TargetPoint,
)
from .targets import TargetRegion

SPATIAL_TARGET_SCHEMA = "swing_sim.spatial_target"
SPATIAL_TARGET_SCHEMA_VERSION = 1
LEGACY_GROUND_SOURCE = "legacy.course_surface/default"

_CURRENT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "units",
        "frame",
        "source_frame",
        "label",
        "kind",
        "position_m",
        "elevation_source",
        "ground_source",
        "tolerance",
    }
)
_LEGACY_SNAKE_FIELDS = frozenset(
    {
        "kind",
        "distance_m",
        "radius_m",
        "lateral_m",
        "band_half_length_m",
        "half_width_m",
    }
)
_LEGACY_CAMEL_FIELDS = frozenset(
    {
        "kind",
        "distanceM",
        "radiusM",
        "lateralM",
        "bandHalfLengthM",
        "halfWidthM",
    }
)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _exact_fields(
    mapping: Mapping[str, object], allowed: frozenset[str], name: str
) -> None:
    unknown = set(mapping).difference(allowed)
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")
    missing = allowed.difference(mapping)
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")


def _required(value: Mapping[str, object], name: str) -> object:
    if name not in value:
        raise ValueError(f"missing field {name!r}")
    return value[name]


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _position_to_dict(point: TargetPoint) -> dict[str, float]:
    return {"x": point.x_m, "elevation": point.elevation_m, "right": point.right_m}


def _tolerance_to_dict(tolerance: AcceptanceGeometry) -> dict[str, object]:
    if isinstance(tolerance, SphereTolerance):
        return {"kind": "sphere", "radius_m": tolerance.radius_m}
    if isinstance(tolerance, BoxTolerance):
        x_m, elevation_m, right_m = tolerance.half_extents_m
        return {
            "kind": "box",
            "half_extents_m": {"x": x_m, "elevation": elevation_m, "right": right_m},
        }
    if isinstance(tolerance, SurfaceCircleTolerance):
        return {"kind": "surface_circle", "radius_m": tolerance.radius_m}
    return {
        "kind": "surface_corridor",
        "half_length_m": tolerance.half_length_m,
        "half_width_m": tolerance.half_width_m,
    }


def spatial_target_to_json_dict(target: SpatialTarget) -> dict[str, object]:
    """Return the complete version-1 persistence mapping.

    Postcondition: all coordinates are app-frame SI values and the source frame
    is retained as provenance.
    """
    if not isinstance(target, SpatialTarget):
        raise TypeError("target must be a SpatialTarget")
    return {
        "schema": SPATIAL_TARGET_SCHEMA,
        "schema_version": SPATIAL_TARGET_SCHEMA_VERSION,
        "units": target.units,
        "frame": target.frame,
        "source_frame": target.point.source_frame,
        "label": target.label,
        "kind": target.kind,
        "position_m": _position_to_dict(target.point),
        "elevation_source": target.elevation_source,
        "ground_source": target.ground_source,
        "tolerance": _tolerance_to_dict(target.tolerance),
    }


def spatial_target_to_json(target: SpatialTarget) -> str:
    """Encode a target as deterministic compact JSON."""
    return json.dumps(
        spatial_target_to_json_dict(target),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _parse_position(data: object, source_frame: str) -> TargetPoint:
    mapping = _mapping(data, "position_m")
    _exact_fields(mapping, frozenset(("x", "elevation", "right")), "position_m")
    return TargetPoint(
        _required(mapping, "x"),  # type: ignore[arg-type]
        _required(mapping, "elevation"),  # type: ignore[arg-type]
        _required(mapping, "right"),  # type: ignore[arg-type]
        source_frame=source_frame,  # type: ignore[arg-type]
    )


def _parse_half_extents(data: object) -> BoxTolerance:
    mapping = _mapping(data, "half_extents_m")
    _exact_fields(mapping, frozenset(("x", "elevation", "right")), "half_extents_m")
    return BoxTolerance(
        (
            _required(mapping, "x"),
            _required(mapping, "elevation"),
            _required(mapping, "right"),
        )  # type: ignore[arg-type]
    )


def _parse_tolerance(data: object) -> AcceptanceGeometry:
    mapping = _mapping(data, "tolerance")
    kind = _string(_required(mapping, "kind"), "tolerance.kind")
    if kind == "sphere":
        _exact_fields(mapping, frozenset(("kind", "radius_m")), "tolerance")
        return SphereTolerance(_required(mapping, "radius_m"))  # type: ignore[arg-type]
    if kind == "box":
        _exact_fields(mapping, frozenset(("kind", "half_extents_m")), "tolerance")
        return _parse_half_extents(_required(mapping, "half_extents_m"))
    if kind == "surface_circle":
        _exact_fields(mapping, frozenset(("kind", "radius_m")), "tolerance")
        return SurfaceCircleTolerance(_required(mapping, "radius_m"))  # type: ignore[arg-type]
    if kind == "surface_corridor":
        fields = frozenset(("kind", "half_length_m", "half_width_m"))
        _exact_fields(mapping, fields, "tolerance")
        return SurfaceCorridorTolerance(
            _required(mapping, "half_length_m"),  # type: ignore[arg-type]
            _required(mapping, "half_width_m"),  # type: ignore[arg-type]
        )
    raise ValueError(f"unknown tolerance kind {kind!r}")


def _parse_current(data: Mapping[str, object]) -> SpatialTarget:
    _exact_fields(data, _CURRENT_FIELDS, "spatial target")
    schema = _string(data["schema"], "schema")
    if schema != SPATIAL_TARGET_SCHEMA:
        raise ValueError(f"schema must be {SPATIAL_TARGET_SCHEMA!r}")
    version = data["schema_version"]
    if type(version) is not int or version != SPATIAL_TARGET_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version!r}")
    units = _string(data["units"], "units")
    if units != "m":
        raise ValueError("units must be 'm'")
    frame = _string(data["frame"], "frame")
    if frame != "app":
        raise ValueError("frame must be 'app'")
    source_frame = _string(data["source_frame"], "source_frame")
    point = _parse_position(data["position_m"], source_frame)
    ground_source = data["ground_source"]
    if ground_source is not None:
        ground_source = _string(ground_source, "ground_source")
    return SpatialTarget(
        label=_string(data["label"], "label"),
        kind=_string(data["kind"], "kind"),  # type: ignore[arg-type]
        point=point,
        tolerance=_parse_tolerance(data["tolerance"]),
        elevation_source=_string(  # type: ignore[arg-type]
            data["elevation_source"], "elevation_source"
        ),
        ground_source=ground_source,
        units="m",
        frame="app",
    )


def _legacy_number(
    data: Mapping[str, object], snake_name: str, camel_name: str, default: float
) -> float:
    if snake_name in data:
        value = data[snake_name]
    else:
        value = data.get(camel_name, default)
    return finite_float(value, snake_name)


def _parse_legacy(data: Mapping[str, object]) -> SpatialTarget:
    allowed = (
        _LEGACY_CAMEL_FIELDS
        if any("M" in key for key in data)
        else _LEGACY_SNAKE_FIELDS
    )
    unknown = set(data).difference(allowed)
    if unknown:
        raise ValueError(f"legacy target has unknown fields: {sorted(unknown)}")
    kind_value = _string(_required(data, "kind"), "kind")
    kind: Literal["green", "fairway"]
    if kind_value == "green":
        kind = "green"
    elif kind_value == "fairway":
        kind = "fairway"
    else:
        raise ValueError("legacy target kind must be 'green' or 'fairway'")
    region = TargetRegion(
        kind=kind,
        distance_m=_legacy_number(data, "distance_m", "distanceM", 230.0),
        radius_m=_legacy_number(data, "radius_m", "radiusM", 10.0),
        lateral_m=_legacy_number(data, "lateral_m", "lateralM", 0.0),
        band_half_length_m=_legacy_number(
            data, "band_half_length_m", "bandHalfLengthM", 15.0
        ),
        half_width_m=_legacy_number(data, "half_width_m", "halfWidthM", 16.0),
    )
    return SpatialTarget.from_target_region(
        region,
        surface_elevation_m=0.0,
        ground_source=LEGACY_GROUND_SOURCE,
        label=f"Migrated {kind.title()} Target",
    )


def spatial_target_from_json_dict(data: Mapping[str, object]) -> SpatialTarget:
    """Decode version 1 or migrate an explicit unversioned 2D target mapping."""
    mapping = _mapping(data, "spatial target")
    if "schema_version" not in mapping:
        return _parse_legacy(mapping)
    return _parse_current(mapping)


def spatial_target_from_json(text: str) -> SpatialTarget:
    """Decode target JSON with stable boundary errors."""
    if not isinstance(text, str):
        raise TypeError("target JSON must be text")
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("target must contain valid JSON") from exc
    return spatial_target_from_json_dict(data)


__all__ = [
    "LEGACY_GROUND_SOURCE",
    "SPATIAL_TARGET_SCHEMA",
    "SPATIAL_TARGET_SCHEMA_VERSION",
    "spatial_target_from_json",
    "spatial_target_from_json_dict",
    "spatial_target_to_json",
    "spatial_target_to_json_dict",
]
