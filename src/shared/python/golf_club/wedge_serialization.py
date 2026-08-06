"""Strict versioned JSON persistence for wedge-family parameters."""

from __future__ import annotations

import json
from typing import Any

from ._validation import reject_unknown_fields, require_mapping
from .wedge_parameters import (
    Handedness,
    WedgeGeometryProvenance,
    WedgeHeadParameters,
)

WEDGE_PARAMETERS_FORMAT = "golf_club.wedge_parameters/1"
_DOCUMENT_FIELDS = frozenset({"format", "parameters"})
_PARAMETER_FIELDS = frozenset(
    {
        "head_id",
        "handedness",
        "loft_deg",
        "lie_deg",
        "bounce_deg",
        "face_length_m",
        "face_height_m",
        "sole_width_m",
        "topline_thickness_m",
        "leading_edge_radius_m",
        "rear_curve_depth_fraction",
        "face_progression_m",
        "hosel_outer_diameter_m",
        "hosel_bore_diameter_m",
        "hosel_length_m",
        "material_density_kg_m3",
        "target_mass_kg",
        "provenance",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "source_name",
        "geometry_basis",
        "uncertainty_note",
        "source_uri",
        "data_license",
    }
)


def wedge_parameters_to_json(parameters: WedgeHeadParameters) -> str:
    """Serialize one wedge parameter set deterministically."""
    if not isinstance(parameters, WedgeHeadParameters):
        raise TypeError("parameters must be WedgeHeadParameters")
    return json.dumps(
        {
            "format": WEDGE_PARAMETERS_FORMAT,
            "parameters": _parameters_to_dict(parameters),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def wedge_parameters_from_json(text: str) -> WedgeHeadParameters:
    """Parse a strict current-version wedge parameter document."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ValueError("text must contain valid JSON") from error
    document = require_mapping(value, "wedge parameter JSON")
    reject_unknown_fields(document, _DOCUMENT_FIELDS, "wedge parameter JSON")
    format_name = document.get("format")
    if not isinstance(format_name, str):
        raise TypeError("format must be a string")
    if format_name != WEDGE_PARAMETERS_FORMAT:
        raise ValueError(f"unsupported wedge-parameter format {format_name!r}")
    return _parameters_from_dict(document.get("parameters"))


def _parameters_to_dict(parameters: WedgeHeadParameters) -> dict[str, Any]:
    provenance = parameters.provenance
    return {
        "head_id": parameters.head_id,
        "handedness": parameters.handedness.value,
        "loft_deg": parameters.loft_deg,
        "lie_deg": parameters.lie_deg,
        "bounce_deg": parameters.bounce_deg,
        "face_length_m": parameters.face_length_m,
        "face_height_m": parameters.face_height_m,
        "sole_width_m": parameters.sole_width_m,
        "topline_thickness_m": parameters.topline_thickness_m,
        "leading_edge_radius_m": parameters.leading_edge_radius_m,
        "rear_curve_depth_fraction": parameters.rear_curve_depth_fraction,
        "face_progression_m": parameters.face_progression_m,
        "hosel_outer_diameter_m": parameters.hosel_outer_diameter_m,
        "hosel_bore_diameter_m": parameters.hosel_bore_diameter_m,
        "hosel_length_m": parameters.hosel_length_m,
        "material_density_kg_m3": parameters.material_density_kg_m3,
        "target_mass_kg": parameters.target_mass_kg,
        "provenance": {
            "source_name": provenance.source_name,
            "geometry_basis": provenance.geometry_basis,
            "uncertainty_note": provenance.uncertainty_note,
            "source_uri": provenance.source_uri,
            "data_license": provenance.data_license,
        },
    }


def _parameters_from_dict(value: object) -> WedgeHeadParameters:
    data = require_mapping(value, "wedge parameters")
    reject_unknown_fields(data, _PARAMETER_FIELDS, "wedge parameters")
    handedness_value = data.get("handedness")
    if not isinstance(handedness_value, str):
        raise TypeError("handedness must be a string")
    try:
        handedness = Handedness(handedness_value)
    except ValueError as error:
        raise ValueError(f"unknown handedness {handedness_value!r}") from error
    return WedgeHeadParameters(
        head_id=data.get("head_id"),  # type: ignore[arg-type]
        handedness=handedness,
        loft_deg=data.get("loft_deg"),  # type: ignore[arg-type]
        lie_deg=data.get("lie_deg"),  # type: ignore[arg-type]
        bounce_deg=data.get("bounce_deg"),  # type: ignore[arg-type]
        face_length_m=data.get("face_length_m"),  # type: ignore[arg-type]
        face_height_m=data.get("face_height_m"),  # type: ignore[arg-type]
        sole_width_m=data.get("sole_width_m"),  # type: ignore[arg-type]
        topline_thickness_m=data.get("topline_thickness_m"),  # type: ignore[arg-type]
        leading_edge_radius_m=data.get("leading_edge_radius_m"),  # type: ignore[arg-type]
        rear_curve_depth_fraction=data.get("rear_curve_depth_fraction"),  # type: ignore[arg-type]
        face_progression_m=data.get("face_progression_m"),  # type: ignore[arg-type]
        hosel_outer_diameter_m=data.get("hosel_outer_diameter_m"),  # type: ignore[arg-type]
        hosel_bore_diameter_m=data.get("hosel_bore_diameter_m"),  # type: ignore[arg-type]
        hosel_length_m=data.get("hosel_length_m"),  # type: ignore[arg-type]
        material_density_kg_m3=data.get("material_density_kg_m3"),  # type: ignore[arg-type]
        target_mass_kg=data.get("target_mass_kg"),  # type: ignore[arg-type]
        provenance=_provenance_from_dict(data.get("provenance")),
    )


def _provenance_from_dict(value: object) -> WedgeGeometryProvenance:
    data = require_mapping(value, "wedge provenance")
    reject_unknown_fields(data, _PROVENANCE_FIELDS, "wedge provenance")
    return WedgeGeometryProvenance(
        source_name=data.get("source_name"),  # type: ignore[arg-type]
        geometry_basis=data.get("geometry_basis"),  # type: ignore[arg-type]
        uncertainty_note=data.get("uncertainty_note"),  # type: ignore[arg-type]
        source_uri=data.get("source_uri"),
        data_license=data.get("data_license"),
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate field {key!r}")
        result[key] = value
    return result


__all__ = [
    "WEDGE_PARAMETERS_FORMAT",
    "wedge_parameters_from_json",
    "wedge_parameters_to_json",
]
