"""Strict deterministic wire parsers for flight-to-ground v1 records."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, cast

from shared.python.compatibility import StrEnum

from .contract_records import GroundSimulationRequest, GroundSimulationResult
from .contract_types import (
    GroundCalibration,
    GroundContactState,
    GroundEvent,
    GroundProvenance,
    GroundSurfaceProfile,
    GroundTrajectoryPoint,
)
from .result_types import GroundSummary, GroundTermination, GroundWarning
from .strict_json import strict_json_object
from .unavailable_types import GroundUnavailableField


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


def _vector(value: object, name: str) -> tuple[float, float, float]:
    items = _sequence(value, name)
    if len(items) != 3:
        raise ValueError(f"{name} must contain three components")
    return (items[0], items[1], items[2])


def record_to_dict(record: object) -> dict[str, Any]:
    """Convert a recognized immutable record to JSON-compatible values."""
    if not is_dataclass(record):
        raise TypeError("ground contract record must be a dataclass")
    raw = asdict(cast(Any, record))
    return cast(dict[str, Any], _wire_value(raw))


def _wire_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {key: _wire_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire_value(item) for item in value]
    return value


def _provenance(payload: object) -> GroundProvenance:
    data = _mapping(payload, "provenance")
    _exact(
        data,
        {"input_sha256", "producer", "producer_version", "source_revision"},
        "provenance",
    )
    return GroundProvenance(
        data["producer"],
        data["producer_version"],
        data["source_revision"],
        data["input_sha256"],
    )


def _calibration(payload: object) -> GroundCalibration:
    data = _mapping(payload, "calibration")
    _exact(data, {"calibration_id", "confidence", "kind", "source"}, "calibration")
    return GroundCalibration(
        data["calibration_id"], data["kind"], data["source"], data["confidence"]
    )


def _surface(payload: object) -> GroundSurfaceProfile:
    data = _mapping(payload, "surface")
    _exact(
        data,
        {
            "firmness_pa",
            "frame",
            "grass_height_m",
            "hardness_fraction",
            "height_m",
            "kinetic_friction",
            "compressibility_fraction",
            "compression_damping_fraction",
            "moisture_fraction",
            "normal_restitution",
            "normal_unit",
            "provider_id",
            "provider_version",
            "rolling_resistance",
            "static_friction",
            "surface_id",
            "surface_velocity_m_s",
            "turf_density_kg_m3",
        },
        "surface",
    )
    return GroundSurfaceProfile(
        data["surface_id"],
        data["provider_id"],
        data["provider_version"],
        data["frame"],
        data["height_m"],
        _vector(data["normal_unit"], "normal_unit"),
        _vector(data["surface_velocity_m_s"], "surface_velocity_m_s"),
        data["normal_restitution"],
        data["static_friction"],
        data["kinetic_friction"],
        data["rolling_resistance"],
        data["firmness_pa"],
        data["hardness_fraction"],
        data["grass_height_m"],
        data["compressibility_fraction"],
        data["compression_damping_fraction"],
        data["turf_density_kg_m3"],
        data["moisture_fraction"],
    )


def _contact(payload: object) -> GroundContactState:
    data = _mapping(payload, "contact state")
    _exact(
        data,
        {"angular_velocity_rad_s", "frame", "position_m", "time_s", "velocity_m_s"},
        "contact state",
    )
    return GroundContactState(
        data["time_s"],
        data["frame"],
        _vector(data["position_m"], "position_m"),
        _vector(data["velocity_m_s"], "velocity_m_s"),
        _vector(data["angular_velocity_rad_s"], "angular_velocity_rad_s"),
    )


def _point(payload: object) -> GroundTrajectoryPoint:
    data = _mapping(payload, "trajectory point")
    _exact(
        data,
        {
            "angular_velocity_rad_s",
            "frame",
            "phase",
            "position_m",
            "time_s",
            "velocity_m_s",
        },
        "trajectory point",
    )
    return GroundTrajectoryPoint(
        data["time_s"],
        data["frame"],
        _vector(data["position_m"], "position_m"),
        _vector(data["velocity_m_s"], "velocity_m_s"),
        _vector(data["angular_velocity_rad_s"], "angular_velocity_rad_s"),
        data["phase"],
    )


def _event(payload: object) -> GroundEvent:
    data = _mapping(payload, "ground event")
    _exact(
        data,
        {
            "angular_velocity_after_rad_s",
            "angular_velocity_before_rad_s",
            "event_type",
            "frame",
            "position_m",
            "sequence",
            "time_s",
            "velocity_after_m_s",
            "velocity_before_m_s",
        },
        "ground event",
    )
    return GroundEvent(
        data["sequence"],
        data["event_type"],
        data["time_s"],
        data["frame"],
        _vector(data["position_m"], "position_m"),
        _vector(data["velocity_before_m_s"], "velocity_before_m_s"),
        _vector(data["velocity_after_m_s"], "velocity_after_m_s"),
        _vector(
            data["angular_velocity_before_rad_s"],
            "angular_velocity_before_rad_s",
        ),
        _vector(
            data["angular_velocity_after_rad_s"],
            "angular_velocity_after_rad_s",
        ),
    )


def _summary(payload: object) -> GroundSummary:
    data = _mapping(payload, "ground summary")
    _exact(
        data,
        {
            "bounce_air_distance_m",
            "bounce_count",
            "carry_distance_m",
            "final_downrange_m",
            "final_offline_m",
            "roll_distance_m",
            "skid_distance_m",
            "surface_path_distance_m",
            "total_distance_m",
        },
        "ground summary",
    )
    return GroundSummary(
        data["carry_distance_m"],
        data["bounce_air_distance_m"],
        data["skid_distance_m"],
        data["roll_distance_m"],
        data["surface_path_distance_m"],
        data["total_distance_m"],
        data["final_downrange_m"],
        data["final_offline_m"],
        data["bounce_count"],
    )


def _termination(payload: object) -> GroundTermination:
    data = _mapping(payload, "termination")
    _exact(data, {"completed", "reason", "time_s"}, "termination")
    return GroundTermination(data["reason"], data["time_s"], data["completed"])


def _warning(payload: object) -> GroundWarning:
    data = _mapping(payload, "warning")
    _exact(data, {"code", "message", "severity"}, "warning")
    return GroundWarning(data["code"], data["severity"], data["message"])


def _unavailable_field(payload: object) -> GroundUnavailableField:
    data = _mapping(payload, "unavailable field")
    _exact(data, {"field_id", "provenance", "reason"}, "unavailable field")
    return GroundUnavailableField(data["field_id"], data["reason"], data["provenance"])


def _request(payload: dict[str, Any]) -> GroundSimulationRequest:
    _exact(
        payload,
        {
            "ball_mass_kg",
            "ball_radius_m",
            "calibration",
            "first_penetrating_state",
            "last_separated_state",
            "max_events",
            "max_time_s",
            "output_interval_s",
            "provenance",
            "request_id",
            "rotational_inertia_factor",
            "schema_version",
            "surface",
            "unit_system",
        },
        "ground simulation request",
    )
    return GroundSimulationRequest(
        payload["request_id"],
        _surface(payload["surface"]),
        _contact(payload["last_separated_state"]),
        _contact(payload["first_penetrating_state"]),
        payload["ball_radius_m"],
        payload["ball_mass_kg"],
        payload["rotational_inertia_factor"],
        payload["max_time_s"],
        payload["output_interval_s"],
        payload["max_events"],
        _calibration(payload["calibration"]),
        _provenance(payload["provenance"]),
        payload["unit_system"],
        payload["schema_version"],
    )


def _result(payload: dict[str, Any]) -> GroundSimulationResult:
    _exact(
        payload,
        {
            "calibration",
            "events",
            "frame",
            "model_id",
            "model_version",
            "provenance",
            "request_id",
            "schema_version",
            "summary",
            "surface_id",
            "termination",
            "trajectory",
            "unit_system",
            "unavailable_fields",
            "warnings",
            "status",
        },
        "ground simulation result",
    )
    return GroundSimulationResult(
        payload["request_id"],
        payload["surface_id"],
        payload["frame"],
        payload["model_id"],
        payload["model_version"],
        payload["status"],
        tuple(_point(item) for item in _sequence(payload["trajectory"], "trajectory")),
        tuple(_event(item) for item in _sequence(payload["events"], "events")),
        None if payload["summary"] is None else _summary(payload["summary"]),
        _termination(payload["termination"]),
        _calibration(payload["calibration"]),
        tuple(_warning(item) for item in _sequence(payload["warnings"], "warnings")),
        tuple(
            _unavailable_field(item)
            for item in _sequence(payload["unavailable_fields"], "unavailable_fields")
        ),
        _provenance(payload["provenance"]),
        payload["unit_system"],
        payload["schema_version"],
    )


_PARSERS = {
    GroundCalibration: _calibration,
    GroundContactState: _contact,
    GroundEvent: _event,
    GroundProvenance: _provenance,
    GroundSimulationRequest: _request,
    GroundSimulationResult: _result,
    GroundSummary: _summary,
    GroundSurfaceProfile: _surface,
    GroundTermination: _termination,
    GroundTrajectoryPoint: _point,
    GroundUnavailableField: _unavailable_field,
    GroundWarning: _warning,
}


def record_from_dict(record_type: type[Any], payload: dict[str, Any]) -> Any:
    """Parse one recognized record type with exact-field validation."""
    try:
        parser = _PARSERS[record_type]
    except KeyError as exc:
        raise TypeError(f"unsupported ground contract record: {record_type!r}") from exc
    return parser(_mapping(payload, record_type.__name__))


def request_from_json(text: str) -> GroundSimulationRequest:
    """Parse one strict request JSON document."""
    return cast(
        GroundSimulationRequest,
        GroundSimulationRequest.from_dict(strict_json_object(text)),
    )


def result_from_json(text: str) -> GroundSimulationResult:
    """Parse one strict result JSON document."""
    return cast(
        GroundSimulationResult,
        GroundSimulationResult.from_dict(strict_json_object(text)),
    )


__all__ = [
    "record_from_dict",
    "record_to_dict",
    "request_from_json",
    "result_from_json",
]
