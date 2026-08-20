"""Fail-closed parser for regional ground execution result v1."""

from __future__ import annotations

from typing import Any, cast

from .contract_records import GroundSimulationResult
from .contract_types import (
    GroundProvenance,
    _integer,
    _nonnegative,
    _text,
    _vector,
)
from .regional_execution_records import (
    MAX_REGIONAL_GROUND_EXECUTION_WIRE_BYTES,
    RegionalGroundExecutionResult,
)
from .regional_plan_wire import regional_material_plan_request_from_dict
from .regional_surface_types import SurfaceRegionTransition
from .strict_json import strict_json_object

_RESULT_FIELDS = {
    "executor_provenance",
    "failure_reason",
    "ground_request_sha256",
    "ground_result",
    "limitations",
    "model_id",
    "model_version",
    "plan_id",
    "plan_provenance",
    "regional_plan_sha256",
    "regional_plan",
    "request_id",
    "schema_version",
    "status",
    "surface_id",
    "transitions",
    "unit_system",
}
_TRANSITION_FIELDS = {
    "event_sequence",
    "from_region_id",
    "from_surface_id",
    "position_m",
    "time_s",
    "to_region_id",
    "to_surface_id",
}


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return value


def _exact(payload: dict[str, Any], fields: set[str], name: str) -> None:
    if set(payload) != fields:
        raise ValueError(f"{name} fields do not match v1 schema")


def _sequence(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _provenance(value: object) -> GroundProvenance:
    return cast(
        GroundProvenance, GroundProvenance.from_dict(_mapping(value, "provenance"))
    )


def _transition(value: object) -> SurfaceRegionTransition:
    data = _mapping(value, "regional transition")
    _exact(data, _TRANSITION_FIELDS, "regional transition")
    position = tuple(_sequence(data["position_m"], "position_m"))
    from_region = _optional_text(data["from_region_id"], "from_region_id")
    to_region = _optional_text(data["to_region_id"], "to_region_id")
    from_surface = _text(data["from_surface_id"], "from_surface_id")
    to_surface = _text(data["to_surface_id"], "to_surface_id")
    if from_surface == to_surface:
        raise ValueError("transition surface identities must differ")
    return SurfaceRegionTransition(
        _integer(data["event_sequence"], "event_sequence"),
        _nonnegative(data["time_s"], "time_s"),
        _vector(cast(tuple[float, float, float], position), "position_m"),
        from_region,
        to_region,
        from_surface,
        to_surface,
    )


def _optional_text(value: object, name: str) -> str | None:
    return None if value is None else _text(cast(str, value), name)


def regional_execution_result_from_dict(
    payload: object,
) -> RegionalGroundExecutionResult:
    """Parse one exact regional execution result mapping."""
    data = _mapping(payload, "regional ground execution result")
    _exact(data, _RESULT_FIELDS, "regional ground execution result")
    ground_payload = data["ground_result"]
    ground_result = (
        None
        if ground_payload is None
        else GroundSimulationResult.from_dict(_mapping(ground_payload, "ground_result"))
    )
    return RegionalGroundExecutionResult(
        data["request_id"],
        data["surface_id"],
        data["plan_id"],
        data["ground_request_sha256"],
        data["regional_plan_sha256"],
        regional_material_plan_request_from_dict(data["regional_plan"]),
        data["status"],
        data["failure_reason"],
        ground_result,
        _provenance(data["plan_provenance"]),
        _provenance(data["executor_provenance"]),
        data["model_id"],
        data["model_version"],
        tuple(
            _transition(item) for item in _sequence(data["transitions"], "transitions")
        ),
        tuple(_sequence(data["limitations"], "limitations")),
        data["unit_system"],
        data["schema_version"],
    )


def regional_ground_execution_result_from_json(
    text: str,
) -> RegionalGroundExecutionResult:
    """Parse one bounded JSON document with duplicate-key rejection."""
    if not isinstance(text, str):
        raise TypeError("regional ground execution JSON must be text")
    if len(text.encode("utf-8")) > MAX_REGIONAL_GROUND_EXECUTION_WIRE_BYTES:
        raise ValueError("regional ground execution exceeds maximum wire size")
    return regional_execution_result_from_dict(strict_json_object(text))


__all__ = [
    "regional_execution_result_from_dict",
    "regional_ground_execution_result_from_json",
]
