"""Strict deterministic wire boundary for repeated-bounce prefix evidence."""

from __future__ import annotations

from typing import Any, cast

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json

from .bounce_types import (
    BounceAirSegment,
    BounceTermination,
    RepeatedBounceResult,
)
from .contract_types import (
    UNIT_SYSTEM_SI,
    GroundContactState,
    GroundEvent,
    GroundFrame,
    GroundTrajectoryPoint,
    _bounded,
    _finite,
    _nonnegative,
    _positive,
    _text,
    _vector,
)
from .contract_wire import record_to_dict
from .impact_types import ImpactEnergyLedger, ImpactImpulseResult
from .strict_json import strict_json_object

REPEATED_BOUNCE_SCHEMA_VERSION = "ground-repeated-bounce-result/v1"
MAX_REPEATED_BOUNCE_WIRE_BYTES = 1_048_576

_RESULT_FIELDS = {
    "airborne_segments",
    "events",
    "frame",
    "handoff_state",
    "impacts",
    "model_id",
    "model_version",
    "request_fingerprint_sha256",
    "request_id",
    "schema_version",
    "surface_id",
    "termination",
    "trajectory",
    "unit_system",
    "warnings",
}
_IMPACT_FIELDS = {
    "contact_velocity_after_m_s",
    "contact_velocity_before_m_s",
    "effective_restitution",
    "energy",
    "friction_utilization",
    "normal_impulse_n_s",
    "regime",
    "state_after",
    "state_before",
    "tangential_impulse_n_s",
    "total_impulse_n_s",
}
_ENERGY_FIELDS = {
    "boundary_work_j",
    "dissipation_j",
    "kinetic_after_j",
    "kinetic_before_j",
}
_AIR_SEGMENT_FIELDS = {
    "completed_at_contact",
    "end_position_m",
    "end_time_s",
    "horizontal_distance_m",
    "start_position_m",
    "start_time_s",
}
_TERMINATION_FIELDS = {"elapsed_time_s", "reason", "time_s"}


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return value


def _exact(data: dict[str, Any], fields: set[str], name: str) -> None:
    if set(data) != fields:
        raise ValueError(f"{name} fields do not match v1 schema")


def _sequence(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _contact(value: object) -> GroundContactState:
    return cast(
        GroundContactState,
        GroundContactState.from_dict(_mapping(value, "contact state")),
    )


def _point(value: object) -> GroundTrajectoryPoint:
    return cast(
        GroundTrajectoryPoint,
        GroundTrajectoryPoint.from_dict(_mapping(value, "trajectory point")),
    )


def _event(value: object) -> GroundEvent:
    return cast(GroundEvent, GroundEvent.from_dict(_mapping(value, "ground event")))


def _energy(value: object) -> ImpactEnergyLedger:
    data = _mapping(value, "impact energy")
    _exact(data, _ENERGY_FIELDS, "impact energy")
    return ImpactEnergyLedger(
        _nonnegative(data["kinetic_before_j"], "kinetic_before_j"),
        _nonnegative(data["kinetic_after_j"], "kinetic_after_j"),
        _finite(data["boundary_work_j"], "boundary_work_j"),
        _nonnegative(data["dissipation_j"], "dissipation_j"),
    )


def _impact(value: object) -> ImpactImpulseResult:
    data = _mapping(value, "impact result")
    _exact(data, _IMPACT_FIELDS, "impact result")
    return ImpactImpulseResult(
        _contact(data["state_before"]),
        _contact(data["state_after"]),
        data["regime"],
        _positive(data["normal_impulse_n_s"], "normal_impulse_n_s"),
        _vector(data["tangential_impulse_n_s"], "tangential_impulse_n_s"),
        _vector(data["total_impulse_n_s"], "total_impulse_n_s"),
        _vector(data["contact_velocity_before_m_s"], "contact_velocity_before_m_s"),
        _vector(data["contact_velocity_after_m_s"], "contact_velocity_after_m_s"),
        _bounded(data["effective_restitution"], "effective_restitution"),
        _bounded(data["friction_utilization"], "friction_utilization"),
        _energy(data["energy"]),
    )


def _air_segment(value: object) -> BounceAirSegment:
    data = _mapping(value, "airborne segment")
    _exact(data, _AIR_SEGMENT_FIELDS, "airborne segment")
    completed = data["completed_at_contact"]
    if not isinstance(completed, bool):
        raise ValueError("completed_at_contact must be a boolean")
    return BounceAirSegment(
        _nonnegative(data["start_time_s"], "start_time_s"),
        _nonnegative(data["end_time_s"], "end_time_s"),
        _vector(data["start_position_m"], "start_position_m"),
        _vector(data["end_position_m"], "end_position_m"),
        _nonnegative(data["horizontal_distance_m"], "horizontal_distance_m"),
        completed,
    )


def _termination(value: object) -> BounceTermination:
    data = _mapping(value, "bounce termination")
    _exact(data, _TERMINATION_FIELDS, "bounce termination")
    return BounceTermination(
        data["reason"],
        _nonnegative(data["time_s"], "termination time_s"),
        _nonnegative(data["elapsed_time_s"], "elapsed_time_s"),
    )


def repeated_bounce_result_from_dict(payload: object) -> RepeatedBounceResult:
    """Parse one exact v1 repeated-bounce evidence mapping without executing physics."""
    data = _mapping(payload, "repeated bounce result")
    _exact(data, _RESULT_FIELDS, "repeated bounce result")
    if data["schema_version"] != REPEATED_BOUNCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {data['schema_version']}")
    if data["unit_system"] != UNIT_SYSTEM_SI:
        raise ValueError(f"unsupported unit_system: {data['unit_system']}")
    frame = GroundFrame(data["frame"])
    warnings = tuple(
        _text(item, "warning") for item in _sequence(data["warnings"], "warnings")
    )
    handoff = data["handoff_state"]
    return RepeatedBounceResult(
        _text(data["request_id"], "request_id"),
        _text(data["surface_id"], "surface_id"),
        frame,
        _text(data["model_id"], "model_id"),
        _text(data["model_version"], "model_version"),
        _text(data["request_fingerprint_sha256"], "request fingerprint"),
        tuple(_point(item) for item in _sequence(data["trajectory"], "trajectory")),
        tuple(_event(item) for item in _sequence(data["events"], "events")),
        tuple(_impact(item) for item in _sequence(data["impacts"], "impacts")),
        tuple(
            _air_segment(item)
            for item in _sequence(data["airborne_segments"], "airborne_segments")
        ),
        None if handoff is None else _contact(handoff),
        _termination(data["termination"]),
        warnings,
    )


def repeated_bounce_result_to_dict(result: RepeatedBounceResult) -> dict[str, Any]:
    """Return a validated exact v1 JSON-compatible evidence mapping."""
    if type(result) is not RepeatedBounceResult:
        raise TypeError("repeated bounce evidence must be an exact result record")
    payload = record_to_dict(result)
    payload.update(
        schema_version=REPEATED_BOUNCE_SCHEMA_VERSION,
        unit_system=UNIT_SYSTEM_SI,
    )
    try:
        validated = repeated_bounce_result_from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"repeated bounce canonical evidence is invalid: {exc}"
        ) from exc
    normalized = cast(dict[str, Any], record_to_dict(validated))
    normalized.update(
        schema_version=REPEATED_BOUNCE_SCHEMA_VERSION,
        unit_system=UNIT_SYSTEM_SI,
    )
    return normalized


def repeated_bounce_result_to_json(result: RepeatedBounceResult) -> str:
    """Serialize validated evidence with deterministic canonical numeric JSON."""
    text = str(canonical_numeric_json(repeated_bounce_result_to_dict(result)))
    if len(text.encode("utf-8")) > MAX_REPEATED_BOUNCE_WIRE_BYTES:
        raise ValueError("repeated bounce evidence exceeds maximum wire size")
    return text


def repeated_bounce_result_from_json(text: str) -> RepeatedBounceResult:
    """Parse bounded UTF-8 JSON with duplicate-key rejection at every depth."""
    if not isinstance(text, str):
        raise TypeError("repeated bounce JSON must be text")
    if len(text.encode("utf-8")) > MAX_REPEATED_BOUNCE_WIRE_BYTES:
        raise ValueError("repeated bounce evidence exceeds maximum wire size")
    return repeated_bounce_result_from_dict(strict_json_object(text))


__all__ = [
    "MAX_REPEATED_BOUNCE_WIRE_BYTES",
    "REPEATED_BOUNCE_SCHEMA_VERSION",
    "repeated_bounce_result_from_dict",
    "repeated_bounce_result_from_json",
    "repeated_bounce_result_to_dict",
    "repeated_bounce_result_to_json",
]
