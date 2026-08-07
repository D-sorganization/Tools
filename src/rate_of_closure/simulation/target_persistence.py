"""Canonical spatial-target metadata for run/project persistence."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from shared.python.swing_sim.solver import (
    SpatialTarget,
    TargetRegion,
    spatial_target_from_json_dict,
    spatial_target_to_json_dict,
)

TARGET_CSV_COLUMNS: tuple[str, ...] = (
    "target_schema",
    "target_schema_version",
    "target_label",
    "target_kind",
    "target_x_downrange_m",
    "target_y_up_m",
    "target_z_right_m",
    "target_frame",
    "target_source_frame",
    "target_units",
    "target_elevation_source",
    "target_ground_source",
    "target_tolerance_json",
)

_FORMAT_PATTERN = re.compile(r"^rate_of_closure\.simulation_run(?:\.web)?/(\d+)$")


def target_csv_values(target: SpatialTarget) -> tuple[object, ...]:
    """Flatten only fields from the canonical serializer for CSV rows."""
    record = spatial_target_to_json_dict(target)
    position = _mapping(record["position_m"], "position_m")
    return (
        record["schema"],
        record["schema_version"],
        record["label"],
        record["kind"],
        position["x"],
        position["elevation"],
        position["right"],
        record["frame"],
        record["source_frame"],
        record["units"],
        record["elevation_source"],
        record["ground_source"],
        json.dumps(
            record["tolerance"], allow_nan=False, separators=(",", ":"), sort_keys=True
        ),
    )


def target_document_fields(target: SpatialTarget) -> dict[str, object]:
    """Return canonical target plus consumer manifests without a shadow schema."""
    record = spatial_target_to_json_dict(target)
    return {
        "spatial_target": record,
        "solver_manifest": {
            "schema": "swing_sim.solver_manifest",
            "schema_version": 1,
            "target": record,
        },
        "variation_manifest": {
            "schema": "swing_sim.variation_manifest",
            "schema_version": 1,
            "target": record,
        },
    }


def spatial_target_from_simulation_document(value: object) -> SpatialTarget:
    """Load a canonical target or migrate an older 2D run/project target."""
    document = _mapping(value, "simulation document")
    version, is_web = _simulation_format(document)
    parameters = _optional_mapping(document.get("parameters"), "parameters")
    manifest = _optional_mapping(document.get("solver_manifest"), "solver_manifest")
    if manifest is not None:
        _validate_solver_manifest(manifest)
    raw_target = document.get("spatial_target")
    if raw_target is None and version == 5:
        dialect = "web " if is_web else ""
        raise ValueError(
            f"{dialect}simulation schema version 5 requires spatial_target"
        )
    if raw_target is None and parameters is not None:
        raw_target = parameters.get("spatial_target", parameters.get("target"))
    if raw_target is None:
        raw_target = document.get("target")
    if raw_target is None and manifest is not None:
        raw_target = manifest.get("target")
    if raw_target is None:
        if is_web and version >= 4:
            raise ValueError(
                f"web simulation schema version {version} requires spatial_target"
            )
        return default_spatial_target()
    return spatial_target_from_json_dict(_mapping(raw_target, "spatial_target"))


def simulation_document_format(value: object) -> tuple[int, bool]:
    """Return ``(version, is_web)`` after validating a simulation format tag."""
    return _simulation_format(_mapping(value, "simulation document"))


def _simulation_format(document: Mapping[str, object]) -> tuple[int, bool]:
    value = document.get("format")
    if value is None:
        return (0, False)
    if not isinstance(value, str):
        raise TypeError("simulation format must be a string")
    match = _FORMAT_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported simulation format {value!r}")
    version = int(match.group(1))
    is_web = ".web/" in value
    maximum = 5
    if version < 1 or version > maximum:
        raise ValueError(f"unsupported simulation schema version {version}")
    return version, is_web


def _validate_solver_manifest(manifest: Mapping[str, object]) -> None:
    if manifest.get("schema") != "swing_sim.solver_manifest":
        raise ValueError("solver_manifest schema must be 'swing_sim.solver_manifest'")
    if manifest.get("schema_version") != 1:
        raise ValueError("solver_manifest schema_version must be 1")


def default_spatial_target() -> SpatialTarget:
    """Return the explicit canonical target used by target-less legacy runs."""
    return SpatialTarget.from_target_region(
        TargetRegion(kind="green", distance_m=230.0),
        ground_source="course.surface/default",
        label="Landing target",
    )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _optional_mapping(value: object, name: str) -> Mapping[str, Any] | None:
    return None if value is None else _mapping(value, name)


__all__ = [
    "TARGET_CSV_COLUMNS",
    "default_spatial_target",
    "simulation_document_format",
    "spatial_target_from_simulation_document",
    "target_csv_values",
    "target_document_fields",
]
