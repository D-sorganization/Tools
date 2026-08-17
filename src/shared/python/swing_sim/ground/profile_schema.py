"""Draft 2020-12 JSON Schemas for ground material profile v1 documents."""

from __future__ import annotations

from typing import Any, cast

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json

from .profile_types import (
    CANONICAL_GROUND_PARAMETER_IDS,
    GROUND_MATERIAL_PROFILE_SCHEMA_VERSION,
    GROUND_PROFILE_LIBRARY_SCHEMA_VERSION,
    GroundEvidenceKind,
    GroundMaterialProfile,
    GroundModelUseStatus,
    GroundProfileLibrary,
    GroundQualificationGateId,
    GroundQualificationStatus,
)

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_MIN_POSITIVE = 0.00000000001
_UNITS = ("1", "1", "1", "1", "Pa", "1", "m", "1", "1", "kg/m^3", "1")


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "additionalProperties": False,
        "properties": properties,
        "required": sorted(properties),
        "type": "object",
    }


def _text() -> dict[str, Any]:
    return {
        "allOf": [
            {"pattern": r"^[^\u0000-\u001F\u007F]+$"},
            {"pattern": r"^[^\uD800-\uDFFF]+$"},
            {"pattern": r"^[^ \t\r\n\f\v](?:[\s\S]*[^ \t\r\n\f\v])?$"},
        ],
        "minLength": 1,
        "type": "string",
    }


def _number(
    minimum: float | None = None, maximum: float | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "number"}
    if minimum is not None:
        result["minimum"] = minimum
    if maximum is not None:
        result["maximum"] = maximum
    return result


def _enum(values: tuple[str, ...]) -> dict[str, Any]:
    return {"enum": list(values), "type": "string"}


def _ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/$defs/{name}"}


def _fixed_array(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "items": False,
        "maxItems": len(items),
        "minItems": len(items),
        "prefixItems": items,
        "type": "array",
    }


def _parameter_value_schema(parameter_id: str) -> dict[str, Any]:
    if parameter_id in {
        "normal_restitution",
        "rolling_resistance",
        "hardness_fraction",
        "compressibility_fraction",
        "compression_damping_fraction",
        "moisture_fraction",
    }:
        return _number(0.0, 1.0)
    if parameter_id in {"static_friction", "kinetic_friction"}:
        return _number(0.0, 5.0)
    if parameter_id == "firmness_pa":
        return _number(_MIN_POSITIVE)
    return _number(0.0)


def _parameter_definition(parameter_id: str, unit: str) -> dict[str, Any]:
    return _object(
        {
            "confidence_level": {
                "exclusiveMinimum": 0.0,
                "maximum": 1.0,
                "type": "number",
            },
            "coverage_factor": _number(1.0),
            "parameter_id": {"const": parameter_id},
            "standard_uncertainty_si": _number(0.0),
            "unit_si": {"const": unit},
            "validity_lower_evidence_ids": _evidence_id_array(),
            "validity_lower_si": _parameter_value_schema(parameter_id),
            "validity_upper_evidence_ids": _evidence_id_array(),
            "validity_upper_si": _parameter_value_schema(parameter_id),
            "value_si": _parameter_value_schema(parameter_id),
        }
    )


def _parameter_array() -> dict[str, Any]:
    return _fixed_array(
        [
            _parameter_definition(str(parameter_id), unit)
            for parameter_id, unit in zip(
                CANONICAL_GROUND_PARAMETER_IDS, _UNITS, strict=True
            )
        ]
    )


def _parameter_id_array() -> dict[str, Any]:
    return {
        "items": _enum(tuple(str(item) for item in CANONICAL_GROUND_PARAMETER_IDS)),
        "minItems": 1,
        "type": "array",
        "uniqueItems": True,
    }


def _evidence_id_array() -> dict[str, Any]:
    return {
        "items": _text(),
        "minItems": 1,
        "type": "array",
        "uniqueItems": True,
    }


def _definitions() -> dict[str, Any]:
    definitions = {
        "applicability": _applicability_definition(),
        "calibration": _calibration_definition(),
        "evidence": _evidence_definition(),
        "provenance": _provenance_definition(),
        "qualification": _qualification_definition(),
        "rights": _rights_definition(),
    }
    definitions["profile"] = _object(_profile_properties())
    return definitions


def _evidence_definition() -> dict[str, Any]:
    return _object(
        {
            "citation": _text(),
            "evidence_id": _text(),
            "kind": _enum(tuple(str(item) for item in GroundEvidenceKind)),
            "parameter_ids": _parameter_id_array(),
            "source_sha256": {"pattern": "^[0-9a-f]{64}$", "type": "string"},
            "source_uri": _text(),
        }
    )


def _rights_definition() -> dict[str, Any]:
    return _object(
        {
            "derivative_use_allowed": {"type": "boolean"},
            "license_id": _text(),
            "redistribution_allowed": {"type": "boolean"},
            "rights_holder": _text(),
        }
    )


def _applicability_definition() -> dict[str, Any]:
    return _object(
        {
            "moisture_max_fraction": _number(0.0, 1.0),
            "moisture_min_fraction": _number(0.0, 1.0),
            "surface_classes": {
                "items": _text(),
                "minItems": 1,
                "type": "array",
                "uniqueItems": True,
            },
            "temperature_max_k": _number(_MIN_POSITIVE),
            "temperature_min_k": _number(_MIN_POSITIVE),
        }
    )


def _calibration_definition() -> dict[str, Any]:
    return _object(
        {
            "calibration_id": _text(),
            "evidence_ids": {
                "items": _text(),
                "minItems": 1,
                "type": "array",
                "uniqueItems": True,
            },
            "method": _text(),
            "parameter_ids": _parameter_id_array(),
        }
    )


def _provenance_definition() -> dict[str, Any]:
    return _object(
        {
            "producer": _text(),
            "producer_version": _text(),
            "source_revision": _text(),
            "source_sha256": {"pattern": "^[0-9a-f]{64}$", "type": "string"},
        }
    )


def _qualification_definition() -> dict[str, Any]:
    gates = [
        _object({"gate_id": {"const": str(gate_id)}, "passed": {"type": "boolean"}})
        for gate_id in GroundQualificationGateId
    ]
    return _object(
        {
            "gates": _fixed_array(gates),
            "status": _enum(tuple(str(item) for item in GroundQualificationStatus)),
        }
    )


def _profile_properties() -> dict[str, Any]:
    return {
        "applicability": _ref("applicability"),
        "calibration": _ref("calibration"),
        "display_name": _text(),
        "evidence": {"items": _ref("evidence"), "minItems": 1, "type": "array"},
        "model_use_status": _enum(tuple(str(item) for item in GroundModelUseStatus)),
        "parameters": _parameter_array(),
        "profile_id": _text(),
        "provenance": _ref("provenance"),
        "qualification": _ref("qualification"),
        "revision": _text(),
        "rights": _ref("rights"),
        "schema_version": {"const": GROUND_MATERIAL_PROFILE_SCHEMA_VERSION},
    }


def _document(
    schema_id: str, properties: dict[str, Any], semantic_validator: str
) -> dict[str, Any]:
    document = _object(properties)
    document.update(
        {
            "$defs": _definitions(),
            "$id": f"https://d-sorganization.github.io/schemas/{schema_id}.json",
            "$schema": JSON_SCHEMA_DIALECT,
            "$comment": (
                "Structural validation is necessary but not sufficient; the "
                "named semantic validator is the authoritative v1 boundary."
            ),
            "x-semantic-validator": semantic_validator,
        }
    )
    return document


def profile_json_schema() -> dict[str, Any]:
    """Return the mandatory structural profile schema."""
    return _document(
        "ground-material-profile-v1",
        _profile_properties(),
        "shared.python.swing_sim.ground.profile_schema.validate_profile_payload",
    )


def library_json_schema() -> dict[str, Any]:
    """Return an independent strict ground-profile-library/v1 schema."""
    properties = {
        "library_id": _text(),
        "profiles": {"items": _ref("profile"), "minItems": 1, "type": "array"},
        "provenance": _ref("provenance"),
        "revision": _text(),
        "schema_version": {"const": GROUND_PROFILE_LIBRARY_SCHEMA_VERSION},
    }
    return _document(
        "ground-profile-library-v1",
        properties,
        "shared.python.swing_sim.ground.profile_schema.validate_library_payload",
    )


def validate_profile_payload(payload: dict[str, Any]) -> GroundMaterialProfile:
    """Apply the authoritative structural and cross-record profile contract."""
    return cast(GroundMaterialProfile, GroundMaterialProfile.from_dict(payload))


def validate_library_payload(payload: dict[str, Any]) -> GroundProfileLibrary:
    """Apply the authoritative structural and cross-record library contract."""
    return cast(GroundProfileLibrary, GroundProfileLibrary.from_dict(payload))


def schema_json(schema: dict[str, Any]) -> str:
    """Serialize one schema deterministically as canonical numeric JSON."""
    return str(canonical_numeric_json(schema))


__all__ = [
    "JSON_SCHEMA_DIALECT",
    "library_json_schema",
    "profile_json_schema",
    "schema_json",
    "validate_library_payload",
    "validate_profile_payload",
]
