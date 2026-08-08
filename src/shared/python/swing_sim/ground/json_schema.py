"""Machine-readable JSON Schema documents for flight-to-ground v1."""

from __future__ import annotations

from typing import Any

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json

from .contract_types import (
    REQUEST_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    UNIT_SYSTEM_SI,
    CalibrationKind,
    GroundEventType,
    GroundFrame,
    GroundPhase,
    GroundResultStatus,
    GroundTerminationReason,
    GroundWarningSeverity,
)
from .unavailable_types import GroundUnavailableFieldId, GroundUnavailableReason

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
MAX_SAFE_INTEGER = 9_007_199_254_740_991
MIN_CANONICAL_POSITIVE = 0.00000000001


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "additionalProperties": False,
        "properties": properties,
        "required": sorted(properties),
        "type": "object",
    }


def _number(
    minimum: float | None = None, maximum: float | None = None
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "number"}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def _text() -> dict[str, Any]:
    return {
        "allOf": [
            {"pattern": r"^[^\uD800-\uDFFF]+$"},
            {"pattern": r"^[^ \t\r\n\f\v](?:[\s\S]*[^ \t\r\n\f\v])?$"},
        ],
        "minLength": 1,
        "type": "string",
    }


def _enum(values: list[str]) -> dict[str, Any]:
    return {"enum": values, "type": "string"}


def _ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/$defs/{name}"}


def _array(item: dict[str, Any], minimum: int = 0) -> dict[str, Any]:
    return {"items": item, "minItems": minimum, "type": "array"}


def _vector() -> dict[str, Any]:
    return {
        "items": _number(),
        "maxItems": 3,
        "minItems": 3,
        "type": "array",
    }


def _identity_definitions() -> dict[str, Any]:
    return {
        "calibration": _object(
            {
                "calibration_id": _text(),
                "confidence": _number(0.0, 1.0),
                "kind": _enum([item.value for item in CalibrationKind]),
                "source": _text(),
            }
        ),
        "provenance": _object(
            {
                "input_sha256": {
                    "pattern": "^[0-9a-f]{64}$",
                    "type": "string",
                },
                "producer": _text(),
                "producer_version": _text(),
                "source_revision": _text(),
            }
        ),
    }


def _surface_definition() -> dict[str, Any]:
    properties = {
        "compressibility_fraction": _number(0.0, 1.0),
        "compression_damping_fraction": _number(0.0, 1.0),
        "firmness_pa": {"minimum": MIN_CANONICAL_POSITIVE, "type": "number"},
        "frame": {"const": GroundFrame.TARGET.value},
        "grass_height_m": _number(0.0),
        "hardness_fraction": _number(0.0, 1.0),
        "height_m": _number(),
        "kinetic_friction": _number(0.0, 5.0),
        "moisture_fraction": _number(0.0, 1.0),
        "normal_restitution": _number(0.0, 1.0),
        "normal_unit": _vector(),
        "provider_id": _text(),
        "provider_version": _text(),
        "rolling_resistance": _number(0.0, 1.0),
        "static_friction": _number(0.0, 5.0),
        "surface_id": _text(),
        "surface_velocity_m_s": _vector(),
        "turf_density_kg_m3": _number(0.0),
    }
    return _object(properties)


def _contact_definition() -> dict[str, Any]:
    return _object(
        {
            "angular_velocity_rad_s": _vector(),
            "frame": {"const": GroundFrame.TARGET.value},
            "position_m": _vector(),
            "time_s": _number(0.0),
            "velocity_m_s": _vector(),
        }
    )


def _result_definitions() -> dict[str, Any]:
    definitions = _identity_definitions()
    definitions.update(
        {
            "event": _object(
                {
                    "angular_velocity_after_rad_s": _vector(),
                    "angular_velocity_before_rad_s": _vector(),
                    "event_type": _enum([item.value for item in GroundEventType]),
                    "frame": {"const": GroundFrame.TARGET.value},
                    "position_m": _vector(),
                    "sequence": {
                        "maximum": MAX_SAFE_INTEGER,
                        "minimum": 0,
                        "type": "integer",
                    },
                    "time_s": _number(0.0),
                    "velocity_after_m_s": _vector(),
                    "velocity_before_m_s": _vector(),
                }
            ),
            "point": _object(
                {
                    "angular_velocity_rad_s": _vector(),
                    "frame": {"const": GroundFrame.TARGET.value},
                    "phase": _enum([item.value for item in GroundPhase]),
                    "position_m": _vector(),
                    "time_s": _number(0.0),
                    "velocity_m_s": _vector(),
                }
            ),
            "summary": _summary_definition(),
            "termination": _termination_definition(),
            "unavailable_field": _object(
                {
                    "field_id": _enum(
                        [item.value for item in GroundUnavailableFieldId]
                    ),
                    "provenance": _text(),
                    "reason": _enum([item.value for item in GroundUnavailableReason]),
                }
            ),
            "warning": _warning_definition(),
        }
    )
    return definitions


def _summary_definition() -> dict[str, Any]:
    return _object(
        {
            "bounce_air_distance_m": _number(0.0),
            "bounce_count": {
                "maximum": MAX_SAFE_INTEGER,
                "minimum": 0,
                "type": "integer",
            },
            "carry_distance_m": _number(0.0),
            "final_downrange_m": _number(),
            "final_offline_m": _number(),
            "roll_distance_m": _number(0.0),
            "skid_distance_m": _number(0.0),
            "surface_path_distance_m": _number(0.0),
            "total_distance_m": _number(0.0),
        }
    )


def _termination_definition() -> dict[str, Any]:
    return _object(
        {
            "completed": {"type": "boolean"},
            "reason": _enum([item.value for item in GroundTerminationReason]),
            "time_s": _number(0.0),
        }
    )


def _warning_definition() -> dict[str, Any]:
    return _object(
        {
            "code": _text(),
            "message": _text(),
            "severity": _enum([item.value for item in GroundWarningSeverity]),
        }
    )


def request_json_schema() -> dict[str, Any]:
    """Return the strict request schema as an independent mapping."""
    properties = {
        "ball_mass_kg": {"minimum": MIN_CANONICAL_POSITIVE, "type": "number"},
        "ball_radius_m": {"minimum": MIN_CANONICAL_POSITIVE, "type": "number"},
        "calibration": _ref("calibration"),
        "first_penetrating_state": _ref("contact"),
        "last_separated_state": _ref("contact"),
        "max_events": {
            "maximum": MAX_SAFE_INTEGER,
            "minimum": 1,
            "type": "integer",
        },
        "max_time_s": {"minimum": MIN_CANONICAL_POSITIVE, "type": "number"},
        "output_interval_s": {
            "minimum": MIN_CANONICAL_POSITIVE,
            "type": "number",
        },
        "provenance": _ref("provenance"),
        "request_id": _text(),
        "rotational_inertia_factor": {
            "minimum": MIN_CANONICAL_POSITIVE,
            "maximum": 1.0,
            "type": "number",
        },
        "schema_version": {"const": REQUEST_SCHEMA_VERSION},
        "surface": _ref("surface"),
        "unit_system": {"const": UNIT_SYSTEM_SI},
    }
    definitions = _identity_definitions()
    definitions.update(
        {"contact": _contact_definition(), "surface": _surface_definition()}
    )
    return _document("flight-to-ground-request-v1", properties, definitions)


def result_json_schema() -> dict[str, Any]:
    """Return the strict result schema as an independent mapping."""
    properties = {
        "calibration": _ref("calibration"),
        "events": _array(_ref("event")),
        "frame": {"const": GroundFrame.TARGET.value},
        "model_id": _text(),
        "model_version": _text(),
        "provenance": _ref("provenance"),
        "request_id": _text(),
        "schema_version": {"const": RESULT_SCHEMA_VERSION},
        "status": _enum([item.value for item in GroundResultStatus]),
        "summary": {"anyOf": [_ref("summary"), {"type": "null"}]},
        "surface_id": _text(),
        "termination": _ref("termination"),
        "trajectory": _array(_ref("point")),
        "unit_system": {"const": UNIT_SYSTEM_SI},
        "unavailable_fields": _array(_ref("unavailable_field")),
        "warnings": _array(_ref("warning")),
    }
    return _document("flight-to-ground-result-v1", properties, _result_definitions())


def _document(
    schema_id: str,
    properties: dict[str, Any],
    definitions: dict[str, Any],
) -> dict[str, Any]:
    document = _object(properties)
    document.update(
        {
            "$defs": definitions,
            "$id": f"https://d-sorganization.github.io/schemas/{schema_id}.json",
            "$schema": JSON_SCHEMA_DIALECT,
        }
    )
    return document


def schema_json(schema: dict[str, Any]) -> str:
    """Serialize one schema deterministically."""
    return str(canonical_numeric_json(schema))


__all__ = [
    "JSON_SCHEMA_DIALECT",
    "request_json_schema",
    "result_json_schema",
    "schema_json",
]
