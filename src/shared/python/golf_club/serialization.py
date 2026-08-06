"""Deterministic versioned JSON persistence for club assemblies."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ._validation import reject_unknown_fields, require_mapping
from .assembly import ClubAssembly
from .types import (
    ClubComponent,
    ClubLengthConvention,
    ClubLengthMeasurement,
    ComponentMassProperties,
    ComponentRole,
    RigidTransform,
)

CURRENT_FORMAT = "golf_club.assembly/1"
LEGACY_FORMAT = "golf_club.assembly/0"

_ASSEMBLY_FIELDS = frozenset(
    {"format", "assembly_id", "frame_id", "components", "club_length"}
)
_LEGACY_FIELDS = frozenset(
    {"format", "assembly_id", "frame_id", "components", "club_length_m"}
)
_COMPONENT_FIELDS = frozenset({"mass_properties", "transform_to_club"})
_MASS_FIELDS = frozenset(
    {
        "component_id",
        "role",
        "frame_id",
        "mass_kg",
        "center_of_mass_m",
        "inertia_at_com_kg_m2",
    }
)
_TRANSFORM_FIELDS = frozenset(
    {"from_frame_id", "to_frame_id", "rotation", "translation_m"}
)
_LENGTH_FIELDS = frozenset(
    {
        "length_m",
        "convention",
        "measurement_frame_id",
        "lower_reference_id",
        "upper_reference_id",
    }
)


def assembly_to_json_dict(assembly: ClubAssembly) -> dict[str, Any]:
    """Return the canonical version-one assembly mapping."""
    if not isinstance(assembly, ClubAssembly):
        raise TypeError("assembly must be a ClubAssembly")
    return {
        "format": CURRENT_FORMAT,
        "assembly_id": assembly.assembly_id,
        "frame_id": assembly.frame_id,
        "components": [_component_to_dict(item) for item in assembly.components],
        "club_length": _length_to_dict(assembly.club_length),
    }


def assembly_to_json(assembly: ClubAssembly) -> str:
    """Serialize with deterministic key ordering and no non-finite extensions."""
    return json.dumps(
        assembly_to_json_dict(assembly),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def assembly_from_json(text: str) -> ClubAssembly:
    """Parse one assembly JSON document with useful corruption errors."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        data = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ValueError("text must contain valid JSON") from error
    return assembly_from_json_dict(require_mapping(data, "assembly JSON"))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting ambiguous duplicate fields."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate field {key!r}")
        result[key] = value
    return result


def assembly_from_json_dict(data: Mapping[str, Any]) -> ClubAssembly:
    """Validate and migrate a supported assembly mapping."""
    source = require_mapping(data, "assembly JSON")
    format_name = source.get("format")
    if not isinstance(format_name, str):
        raise TypeError("format must be a string")
    if format_name == CURRENT_FORMAT:
        return _assembly_from_current(source)
    if format_name == LEGACY_FORMAT:
        return _assembly_from_legacy(source)
    raise ValueError(f"unsupported golf-club assembly format {format_name!r}")


def _assembly_from_current(data: Mapping[str, Any]) -> ClubAssembly:
    """Load the current schema after strict field validation."""
    reject_unknown_fields(data, _ASSEMBLY_FIELDS, "assembly JSON")
    components = _components_from_json(data.get("components"))
    length = _length_from_dict(data.get("club_length"))
    return ClubAssembly(
        assembly_id=data.get("assembly_id"),  # type: ignore[arg-type]
        frame_id=data.get("frame_id"),  # type: ignore[arg-type]
        components=components,
        club_length=length,
    )


def _assembly_from_legacy(data: Mapping[str, Any]) -> ClubAssembly:
    """Migrate version zero's scalar length into an explicit datum record."""
    reject_unknown_fields(data, _LEGACY_FIELDS, "legacy assembly JSON")
    frame_id = data.get("frame_id")
    length = ClubLengthMeasurement(
        length_m=data.get("club_length_m"),  # type: ignore[arg-type]
        convention=ClubLengthConvention.DECLARED_DATUMS,
        measurement_frame_id=frame_id,  # type: ignore[arg-type]
        lower_reference_id="unspecified legacy lower datum",
        upper_reference_id="unspecified legacy upper datum",
    )
    return ClubAssembly(
        assembly_id=data.get("assembly_id"),  # type: ignore[arg-type]
        frame_id=frame_id,  # type: ignore[arg-type]
        components=_components_from_json(data.get("components")),
        club_length=length,
    )


def _components_from_json(value: object) -> tuple[ClubComponent, ...]:
    """Load a component list without accepting strings or arbitrary iterables."""
    if not isinstance(value, list):
        raise TypeError("components must be a JSON array")
    return tuple(_component_from_dict(item) for item in value)


def _component_from_dict(value: object) -> ClubComponent:
    """Load one placed component mapping."""
    data = require_mapping(value, "component")
    reject_unknown_fields(data, _COMPONENT_FIELDS, "component")
    return ClubComponent(
        mass_properties=_mass_from_dict(data.get("mass_properties")),
        transform_to_club=_transform_from_dict(data.get("transform_to_club")),
    )


def _mass_from_dict(value: object) -> ComponentMassProperties:
    """Load one component mass-properties mapping."""
    data = require_mapping(value, "mass_properties")
    reject_unknown_fields(data, _MASS_FIELDS, "mass_properties")
    role_value = data.get("role")
    if not isinstance(role_value, str):
        raise TypeError("role must be a string")
    try:
        role = ComponentRole(role_value)
    except ValueError as error:
        raise ValueError(f"unknown component role {role_value!r}") from error
    return ComponentMassProperties(
        component_id=data.get("component_id"),  # type: ignore[arg-type]
        role=role,
        frame_id=data.get("frame_id"),  # type: ignore[arg-type]
        mass_kg=data.get("mass_kg"),  # type: ignore[arg-type]
        center_of_mass_m=data.get("center_of_mass_m"),  # type: ignore[arg-type]
        inertia_at_com_kg_m2=data.get("inertia_at_com_kg_m2"),  # type: ignore[arg-type]
    )


def _transform_from_dict(value: object) -> RigidTransform:
    """Load one rigid-transform mapping."""
    data = require_mapping(value, "transform_to_club")
    reject_unknown_fields(data, _TRANSFORM_FIELDS, "transform_to_club")
    return RigidTransform(
        from_frame_id=data.get("from_frame_id"),  # type: ignore[arg-type]
        to_frame_id=data.get("to_frame_id"),  # type: ignore[arg-type]
        rotation=data.get("rotation"),  # type: ignore[arg-type]
        translation_m=data.get("translation_m"),  # type: ignore[arg-type]
    )


def _length_from_dict(value: object) -> ClubLengthMeasurement:
    """Load one declared club-length record."""
    data = require_mapping(value, "club_length")
    reject_unknown_fields(data, _LENGTH_FIELDS, "club_length")
    convention_value = data.get("convention")
    if not isinstance(convention_value, str):
        raise TypeError("convention must be a string")
    try:
        convention = ClubLengthConvention(convention_value)
    except ValueError as error:
        raise ValueError(
            f"unknown club length convention {convention_value!r}"
        ) from error
    return ClubLengthMeasurement(
        length_m=data.get("length_m"),  # type: ignore[arg-type]
        convention=convention,
        measurement_frame_id=data.get("measurement_frame_id"),  # type: ignore[arg-type]
        lower_reference_id=data.get("lower_reference_id"),  # type: ignore[arg-type]
        upper_reference_id=data.get("upper_reference_id"),  # type: ignore[arg-type]
    )


def _component_to_dict(component: ClubComponent) -> dict[str, Any]:
    """Serialize one placed component using explicit SI field names."""
    mass = component.mass_properties
    transform = component.transform_to_club
    return {
        "mass_properties": {
            "component_id": mass.component_id,
            "role": mass.role.value,
            "frame_id": mass.frame_id,
            "mass_kg": mass.mass_kg,
            "center_of_mass_m": list(mass.center_of_mass_m),
            "inertia_at_com_kg_m2": [list(row) for row in mass.inertia_at_com_kg_m2],
        },
        "transform_to_club": {
            "from_frame_id": transform.from_frame_id,
            "to_frame_id": transform.to_frame_id,
            "rotation": [list(row) for row in transform.rotation],
            "translation_m": list(transform.translation_m),
        },
    }


def _length_to_dict(measurement: ClubLengthMeasurement) -> dict[str, Any]:
    """Serialize the declared length and all reference provenance."""
    return {
        "length_m": measurement.length_m,
        "convention": measurement.convention.value,
        "measurement_frame_id": measurement.measurement_frame_id,
        "lower_reference_id": measurement.lower_reference_id,
        "upper_reference_id": measurement.upper_reference_id,
    }


__all__ = [
    "CURRENT_FORMAT",
    "LEGACY_FORMAT",
    "assembly_from_json",
    "assembly_from_json_dict",
    "assembly_to_json",
    "assembly_to_json_dict",
]
