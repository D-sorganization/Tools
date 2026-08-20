"""Strict JSON-wire type validation for capability workflow documents."""

from __future__ import annotations

import math
from typing import Any

_PARAMETER_FIELDS = {
    "baseline",
    "bias",
    "evidence_lower_bound",
    "evidence_upper_bound",
    "lower_bound",
    "parameter_id",
    "standard_deviation",
    "unit",
    "upper_bound",
}
_PARAMETER_NUMBERS = _PARAMETER_FIELDS - {"parameter_id", "unit"}
_TARGET_FIELDS = {
    "band_half_length_m",
    "distance_m",
    "half_width_m",
    "kind",
    "lateral_m",
    "radius_m",
}
_REQUEST_FIELDS = {
    "alternatives_count",
    "candidate_budget",
    "club_ids",
    "cvar_alpha",
    "ensemble_size",
    "minimum_success_fraction",
    "objective",
    "problem_id",
    "schema_version",
    "seed",
    "target",
}
MAX_CAPABILITY_WIRE_MAGNITUDE = 1e300


def _record(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields do not match v1 schema")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    if abs(parsed) > MAX_CAPABILITY_WIRE_MAGNITUDE:
        raise ValueError(
            f"{name} magnitude must not exceed {MAX_CAPABILITY_WIRE_MAGNITUDE:g}"
        )
    return parsed


def _integer(value: object, name: str) -> int:
    parsed = _number(value, name)
    if not parsed.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(parsed)


def _validate_parameter(value: object) -> None:
    parameter = _record(value, _PARAMETER_FIELDS, "capability parameter")
    _text(parameter["parameter_id"], "parameter_id")
    _text(parameter["unit"], "parameter unit")
    for field in _PARAMETER_NUMBERS:
        _number(parameter[field], field)


def _validate_club(value: object) -> None:
    fields = {
        "club_id",
        "confidence",
        "matrix",
        "matrix_kind",
        "parameters",
        "provenance",
    }
    club = _record(value, fields, "club capability")
    for field in ("club_id", "matrix_kind", "provenance"):
        _text(club[field], field)
    _number(club["confidence"], "club confidence")
    for parameter in _array(club["parameters"], "club parameters"):
        _validate_parameter(parameter)
    for row in _array(club["matrix"], "capability matrix"):
        for entry in _array(row, "capability matrix row"):
            _number(entry, "capability matrix entry")


def _validate_profile(value: object) -> None:
    fields = {"clubs", "confidence", "profile_id", "provenance", "schema_version"}
    profile = _record(value, fields, "player capability profile")
    for field in ("profile_id", "provenance", "schema_version"):
        _text(profile[field], field)
    _number(profile["confidence"], "profile confidence")
    for club in _array(profile["clubs"], "profile clubs"):
        _validate_club(club)


def _validate_target(value: object) -> None:
    target = _record(value, _TARGET_FIELDS, "target definition")
    _text(target["kind"], "target kind")
    for field in _TARGET_FIELDS - {"kind"}:
        _number(target[field], field)


def _validate_request(value: object) -> None:
    request = _record(value, _REQUEST_FIELDS, "optimization request")
    for field in ("problem_id", "objective", "schema_version"):
        _text(request[field], field)
    for club_id in _array(request["club_ids"], "request club_ids"):
        _text(club_id, "request club_id")
    for field in ("candidate_budget", "ensemble_size", "alternatives_count", "seed"):
        request[field] = _integer(request[field], field)
    for field in ("cvar_alpha", "minimum_success_fraction"):
        _number(request[field], field)
    _validate_target(request["target"])


def _validate_config(value: object) -> None:
    fields = {"max_time_s", "spin_defaults", "trajectory_sample_interval_s"}
    config = _record(value, fields, "evaluator_config")
    _number(config["max_time_s"], "max_time_s")
    _number(config["trajectory_sample_interval_s"], "trajectory_sample_interval_s")
    spin_fields = {"club_id", "provenance", "spin_axis_tilt_deg", "total_spin_rpm"}
    for value in _array(config["spin_defaults"], "spin_defaults"):
        spin = _record(value, spin_fields, "spin default")
        _text(spin["club_id"], "spin default club_id")
        _text(spin["provenance"], "spin default provenance")
        _number(spin["total_spin_rpm"], "total_spin_rpm")
        _number(spin["spin_axis_tilt_deg"], "spin_axis_tilt_deg")


def validate_capability_workflow_wire(payload: object) -> dict[str, Any]:
    """Validate exact structure and primitive JSON types before model parsing."""
    fields = {"evaluator_config", "profile", "request", "schema_version"}
    document = _record(payload, fields, "capability workflow")
    _text(document["schema_version"], "schema_version")
    _validate_profile(document["profile"])
    _validate_request(document["request"])
    _validate_config(document["evaluator_config"])
    return document


__all__ = ["MAX_CAPABILITY_WIRE_MAGNITUDE", "validate_capability_workflow_wire"]
