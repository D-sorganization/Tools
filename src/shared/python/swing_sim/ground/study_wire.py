"""Strict deterministic wire format for ground-study projections."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from shared.python.swing_sim.solver.spatial_targets import TargetMiss
from shared.python.swing_sim.solver.target_serialization import (
    spatial_target_from_json_dict,
    spatial_target_to_json_dict,
)

from .contract_types import (
    GroundResultStatus,
    GroundSurfaceProfile,
    GroundTerminationReason,
)
from .profile_binding import ProfileOperatingCondition
from .profile_types import GroundMaterialProfile
from .result_types import GroundSummary, GroundWarning
from .strict_json import strict_json_object
from .study_record import GroundStudyProjection
from .study_target import canonical_ground_target
from .study_types import (
    GROUND_STUDY_SCHEMA_VERSION,
    GroundEndpointKind,
    GroundSolverEligibility,
    GroundSolverEligibilityReason,
    GroundStudyMetrics,
    GroundStudyProfile,
    GroundStudyStatus,
    GroundTargetEvaluation,
    GroundTargetUnavailableReason,
)
from .unavailable_types import GroundUnavailableField

_ROOT_FIELDS = {
    "final_target",
    "final_target_unavailable_reason",
    "first_contact_target",
    "ball_radius_m",
    "metrics",
    "model_id",
    "model_version",
    "profile",
    "request_id",
    "request_sha256",
    "result_sha256",
    "result_status",
    "schema_version",
    "solver_eligibility",
    "status",
    "surface",
    "target",
    "termination_reason",
    "unavailable_fields",
    "warnings",
}


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    return value


def _exact(data: Mapping[str, object], fields: set[str], name: str) -> None:
    if set(data) != fields:
        raise ValueError(f"{name} fields do not match the v1 contract")


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _vector_dict(value: tuple[float, float, float]) -> list[float]:
    return list(value)


def _vector(value: object, name: str) -> tuple[float, float, float]:
    items = _sequence(value, name)
    if len(items) != 3:
        raise ValueError(f"{name} must contain three components")
    return cast(tuple[float, float, float], tuple(items))


def _miss_to_dict(miss: TargetMiss) -> dict[str, object]:
    return {
        "accepted": miss.accepted,
        "closest_point_m": _vector_dict(miss.closest_point_m),
        "distance_m": miss.distance_m,
        "vector_m": _vector_dict(miss.vector_m),
    }


def _miss(value: object) -> TargetMiss:
    data = _mapping(value, "target miss")
    _exact(
        data, {"accepted", "closest_point_m", "distance_m", "vector_m"}, "target miss"
    )
    return TargetMiss(
        _vector(data["closest_point_m"], "closest_point_m"),
        _vector(data["vector_m"], "vector_m"),
        _number(data["distance_m"], "distance_m"),
        _boolean(data["accepted"], "accepted"),
    )


def _evaluation_to_dict(value: GroundTargetEvaluation) -> dict[str, object]:
    return {
        "ball_center_m": _vector_dict(value.ball_center_m),
        "center_distance_m": value.center_distance_m,
        "center_residual_m": _vector_dict(value.center_residual_m),
        "contact_point_m": _vector_dict(value.contact_point_m),
        "endpoint_kind": value.endpoint_kind.value,
        "miss": _miss_to_dict(value.miss),
        "target_center_m": _vector_dict(value.target_center_m),
        "target_label": value.target_label,
    }


def _evaluation(value: object) -> GroundTargetEvaluation:
    data = _mapping(value, "target evaluation")
    fields = {
        "ball_center_m",
        "center_distance_m",
        "center_residual_m",
        "contact_point_m",
        "endpoint_kind",
        "miss",
        "target_center_m",
        "target_label",
    }
    _exact(data, fields, "target evaluation")
    miss = data["miss"]
    return GroundTargetEvaluation(
        _string(data["target_label"], "target_label"),
        GroundEndpointKind(_string(data["endpoint_kind"], "endpoint_kind")),
        _vector(data["ball_center_m"], "ball_center_m"),
        _vector(data["contact_point_m"], "contact_point_m"),
        _vector(data["target_center_m"], "target_center_m"),
        _vector(data["center_residual_m"], "center_residual_m"),
        _number(data["center_distance_m"], "center_distance_m"),
        _miss(miss),
    )


def _metrics_to_dict(value: GroundStudyMetrics) -> dict[str, object]:
    return {
        "final_observed_position_m": _vector_dict(value.final_observed_position_m),
        "first_contact_position_m": _vector_dict(value.first_contact_position_m),
        "ground_elapsed_s": value.ground_elapsed_s,
        "summary": value.summary.to_dict(),
    }


def _metrics(value: object) -> GroundStudyMetrics:
    data = _mapping(value, "study metrics")
    fields = {
        "final_observed_position_m",
        "first_contact_position_m",
        "ground_elapsed_s",
        "summary",
    }
    _exact(data, fields, "study metrics")
    summary = GroundSummary.from_dict(dict(_mapping(data["summary"], "summary")))
    return GroundStudyMetrics(
        summary,
        _vector(data["first_contact_position_m"], "first_contact_position_m"),
        _vector(data["final_observed_position_m"], "final_observed_position_m"),
        _number(data["ground_elapsed_s"], "ground_elapsed_s"),
    )


def _profile_to_dict(value: GroundStudyProfile) -> dict[str, object]:
    return {
        "material_profile": value.material_profile.to_dict(),
        "operating_condition": {
            "moisture_fraction": value.operating_condition.moisture_fraction,
            "surface_class": value.operating_condition.surface_class,
            "temperature_k": value.operating_condition.temperature_k,
        },
        "warnings": list(value.warnings),
    }


def _profile(value: object) -> GroundStudyProfile:
    data = _mapping(value, "study profile")
    fields = {"material_profile", "operating_condition", "warnings"}
    _exact(data, fields, "study profile")
    condition = _mapping(data["operating_condition"], "operating condition")
    _exact(
        condition,
        {"moisture_fraction", "surface_class", "temperature_k"},
        "operating condition",
    )
    return GroundStudyProfile(
        GroundMaterialProfile.from_dict(
            dict(_mapping(data["material_profile"], "material profile"))
        ),
        ProfileOperatingCondition(
            _string(condition["surface_class"], "surface_class"),
            _number(condition["temperature_k"], "temperature_k"),
            _number(condition["moisture_fraction"], "moisture_fraction"),
        ),
        tuple(
            _string(item, "profile warning")
            for item in _sequence(data["warnings"], "profile warnings")
        ),
    )


def _eligibility_to_dict(value: GroundSolverEligibility) -> dict[str, object]:
    return {
        "eligible": value.eligible,
        "reasons": [item.value for item in value.reasons],
    }


def _eligibility(value: object) -> GroundSolverEligibility:
    data = _mapping(value, "solver eligibility")
    _exact(data, {"eligible", "reasons"}, "solver eligibility")
    return GroundSolverEligibility(
        _boolean(data["eligible"], "eligible"),
        tuple(
            GroundSolverEligibilityReason(_string(item, "eligibility reason"))
            for item in _sequence(data["reasons"], "eligibility reasons")
        ),
    )


def study_to_dict(value: GroundStudyProjection) -> dict[str, Any]:
    """Return one complete exact-field projection mapping."""
    if type(value) is not GroundStudyProjection:
        raise TypeError("value must use the exact GroundStudyProjection type")
    return {
        "final_target": None
        if value.final_target is None
        else _evaluation_to_dict(value.final_target),
        "final_target_unavailable_reason": None
        if value.final_target_unavailable_reason is None
        else value.final_target_unavailable_reason.value,
        "first_contact_target": None
        if value.first_contact_target is None
        else _evaluation_to_dict(value.first_contact_target),
        "ball_radius_m": value.ball_radius_m,
        "metrics": None if value.metrics is None else _metrics_to_dict(value.metrics),
        "model_id": value.model_id,
        "model_version": value.model_version,
        "profile": None if value.profile is None else _profile_to_dict(value.profile),
        "request_id": value.request_id,
        "request_sha256": value.request_sha256,
        "result_sha256": value.result_sha256,
        "result_status": value.result_status.value,
        "schema_version": value.schema_version,
        "solver_eligibility": _eligibility_to_dict(value.solver_eligibility),
        "status": value.status.value,
        "surface": value.surface.to_dict(),
        "target": None
        if value.target is None
        else spatial_target_to_json_dict(value.target),
        "termination_reason": value.termination_reason.value,
        "unavailable_fields": [item.to_dict() for item in value.unavailable_fields],
        "warnings": [item.to_dict() for item in value.warnings],
    }


def study_to_json(value: GroundStudyProjection) -> str:
    """Return canonical numeric JSON without nonfinite values."""
    return str(canonical_numeric_json(study_to_dict(value)))


def _optional(value: object, parser: Any) -> Any:
    return None if value is None else parser(value)


def _unavailable_fields(value: object) -> tuple[GroundUnavailableField, ...]:
    return tuple(
        GroundUnavailableField.from_dict(dict(_mapping(item, "unavailable field")))
        for item in _sequence(value, "unavailable_fields")
    )


def _warnings(value: object) -> tuple[GroundWarning, ...]:
    return tuple(
        GroundWarning.from_dict(dict(_mapping(item, "warning")))
        for item in _sequence(value, "warnings")
    )


def study_from_dict(payload: dict[str, Any]) -> GroundStudyProjection:
    """Parse one strict exact-field projection mapping."""
    data = _mapping(payload, "ground study projection")
    _exact(data, _ROOT_FIELDS, "ground study projection")
    if data["schema_version"] != GROUND_STUDY_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {data['schema_version']}")
    target = _optional(
        data["target"],
        lambda item: canonical_ground_target(
            spatial_target_from_json_dict(_mapping(item, "target"))
        ),
    )
    return GroundStudyProjection(
        _string(data["request_id"], "request_id"),
        _string(data["request_sha256"], "request_sha256"),
        _string(data["result_sha256"], "result_sha256"),
        GroundSurfaceProfile.from_dict(
            dict(_mapping(data["surface"], "ground surface"))
        ),
        _number(data["ball_radius_m"], "ball_radius_m"),
        _string(data["model_id"], "model_id"),
        _string(data["model_version"], "model_version"),
        GroundResultStatus(_string(data["result_status"], "result_status")),
        GroundStudyStatus(_string(data["status"], "status")),
        GroundTerminationReason(
            _string(data["termination_reason"], "termination_reason")
        ),
        _optional(data["metrics"], _metrics),
        target,
        _optional(data["first_contact_target"], _evaluation),
        _optional(data["final_target"], _evaluation),
        _optional(
            data["final_target_unavailable_reason"],
            lambda item: GroundTargetUnavailableReason(
                _string(item, "final target unavailable reason")
            ),
        ),
        _eligibility(data["solver_eligibility"]),
        _optional(data["profile"], _profile),
        _warnings(data["warnings"]),
        _unavailable_fields(data["unavailable_fields"]),
        _string(data["schema_version"], "schema_version"),
    )


def study_from_json(text: str) -> GroundStudyProjection:
    """Parse strict JSON while rejecting duplicate keys at every level."""
    return study_from_dict(strict_json_object(text))


__all__ = ["study_from_dict", "study_from_json", "study_to_dict", "study_to_json"]
