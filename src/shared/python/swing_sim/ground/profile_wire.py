"""Strict canonical JSON wire boundary for ground material profile v1."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, cast

from shared.python.compatibility import StrEnum
from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json

from .profile_types import (
    GroundApplicability,
    GroundCalibrationRecord,
    GroundMaterialParameter,
    GroundMaterialProfile,
    GroundModelUseStatus,
    GroundProfileEvidence,
    GroundProfileLibrary,
    GroundProfileProvenance,
    GroundProfileQualification,
    GroundProfileRights,
    GroundQualificationGate,
)
from .profile_validation import array_value, exact_fields, object_mapping


def _wire_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {key: _wire_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_wire_value(item) for item in value]
    return value


def document_to_dict(document: object) -> dict[str, Any]:
    """Convert a recognized immutable document to JSON-compatible values."""
    if type(document) not in {GroundMaterialProfile, GroundProfileLibrary}:
        raise TypeError("ground profile document must use an exact document type")
    if not is_dataclass(document):  # defensive against proxy subclasses
        raise TypeError("ground profile document must be a dataclass")
    return cast(dict[str, Any], _wire_value(asdict(cast(Any, document))))


def document_to_json(document: object) -> str:
    """Serialize one profile or library as canonical numeric JSON."""
    return str(canonical_numeric_json(document_to_dict(document)))


def _parameter(payload: object) -> GroundMaterialParameter:
    data = object_mapping(payload, "material parameter")
    exact_fields(
        data,
        {
            "confidence_level",
            "coverage_factor",
            "parameter_id",
            "standard_uncertainty_si",
            "unit_si",
            "validity_lower_evidence_ids",
            "validity_lower_si",
            "validity_upper_evidence_ids",
            "validity_upper_si",
            "value_si",
        },
        "material parameter",
    )
    return GroundMaterialParameter(
        data["parameter_id"],
        data["value_si"],
        data["standard_uncertainty_si"],
        data["coverage_factor"],
        data["confidence_level"],
        data["validity_lower_si"],
        data["validity_upper_si"],
        tuple(
            array_value(
                data["validity_lower_evidence_ids"], "validity lower evidence_ids"
            )
        ),
        tuple(
            array_value(
                data["validity_upper_evidence_ids"], "validity upper evidence_ids"
            )
        ),
        data["unit_si"],
    )


def _evidence(payload: object) -> GroundProfileEvidence:
    data = object_mapping(payload, "profile evidence")
    exact_fields(
        data,
        {
            "citation",
            "evidence_id",
            "kind",
            "parameter_ids",
            "source_sha256",
            "source_uri",
        },
        "profile evidence",
    )
    return GroundProfileEvidence(
        data["evidence_id"],
        data["kind"],
        data["citation"],
        data["source_uri"],
        data["source_sha256"],
        tuple(array_value(data["parameter_ids"], "evidence parameter_ids")),
    )


def _rights(payload: object) -> GroundProfileRights:
    data = object_mapping(payload, "profile rights")
    exact_fields(
        data,
        {
            "derivative_use_allowed",
            "license_id",
            "redistribution_allowed",
            "rights_holder",
        },
        "profile rights",
    )
    return GroundProfileRights(
        data["license_id"],
        data["rights_holder"],
        data["redistribution_allowed"],
        data["derivative_use_allowed"],
    )


def _applicability(payload: object) -> GroundApplicability:
    data = object_mapping(payload, "profile applicability")
    exact_fields(
        data,
        {
            "moisture_max_fraction",
            "moisture_min_fraction",
            "surface_classes",
            "temperature_max_k",
            "temperature_min_k",
        },
        "profile applicability",
    )
    return GroundApplicability(
        tuple(array_value(data["surface_classes"], "surface_classes")),
        data["temperature_min_k"],
        data["temperature_max_k"],
        data["moisture_min_fraction"],
        data["moisture_max_fraction"],
    )


def _calibration(payload: object) -> GroundCalibrationRecord:
    data = object_mapping(payload, "profile calibration")
    exact_fields(
        data,
        {"calibration_id", "evidence_ids", "method", "parameter_ids"},
        "profile calibration",
    )
    return GroundCalibrationRecord(
        data["calibration_id"],
        data["method"],
        tuple(array_value(data["evidence_ids"], "calibration evidence_ids")),
        tuple(array_value(data["parameter_ids"], "calibration parameter_ids")),
    )


def _provenance(payload: object) -> GroundProfileProvenance:
    data = object_mapping(payload, "profile provenance")
    exact_fields(
        data,
        {"producer", "producer_version", "source_revision", "source_sha256"},
        "profile provenance",
    )
    return GroundProfileProvenance(
        data["producer"],
        data["producer_version"],
        data["source_revision"],
        data["source_sha256"],
    )


def _qualification(payload: object) -> GroundProfileQualification:
    data = object_mapping(payload, "profile qualification")
    exact_fields(data, {"gates", "status"}, "profile qualification")
    gates = tuple(
        _qualification_gate(item)
        for item in array_value(data["gates"], "qualification gates")
    )
    return GroundProfileQualification(data["status"], gates)


def _qualification_gate(payload: object) -> GroundQualificationGate:
    data = object_mapping(payload, "qualification gate")
    exact_fields(data, {"gate_id", "passed"}, "qualification gate")
    return GroundQualificationGate(data["gate_id"], data["passed"])


def _profile(payload: object) -> GroundMaterialProfile:
    data = object_mapping(payload, "ground material profile")
    exact_fields(
        data,
        {
            "applicability",
            "calibration",
            "display_name",
            "evidence",
            "model_use_status",
            "parameters",
            "profile_id",
            "provenance",
            "qualification",
            "revision",
            "rights",
            "schema_version",
        },
        "ground material profile",
    )
    profile = GroundMaterialProfile(
        data["profile_id"],
        data["display_name"],
        data["revision"],
        tuple(
            _parameter(item) for item in array_value(data["parameters"], "parameters")
        ),
        tuple(_evidence(item) for item in array_value(data["evidence"], "evidence")),
        _rights(data["rights"]),
        _applicability(data["applicability"]),
        _calibration(data["calibration"]),
        _provenance(data["provenance"]),
        data["schema_version"],
    )
    if _qualification(data["qualification"]) != profile.qualification:
        raise ValueError("qualification must equal the derived gate results")
    if GroundModelUseStatus(data["model_use_status"]) is not profile.model_use_status:
        raise ValueError("model_use_status must equal the derived qualification")
    return profile


def _library(payload: object) -> GroundProfileLibrary:
    data = object_mapping(payload, "ground profile library")
    exact_fields(
        data,
        {"library_id", "profiles", "provenance", "revision", "schema_version"},
        "ground profile library",
    )
    return GroundProfileLibrary(
        data["library_id"],
        data["revision"],
        tuple(_profile(item) for item in array_value(data["profiles"], "profiles")),
        _provenance(data["provenance"]),
        data["schema_version"],
    )


def document_from_dict(document_type: type[Any], payload: dict[str, Any]) -> Any:
    """Parse one profile or library with exact-field nested validation."""
    if document_type is GroundMaterialProfile:
        return _profile(payload)
    if document_type is GroundProfileLibrary:
        return _library(payload)
    raise TypeError(f"unsupported ground profile document: {document_type!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _invalid_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _strict_canonical_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError("ground profile JSON must be text")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("ground profile JSON is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("ground profile JSON must be an object")
    return value


def profile_from_json(text: str) -> GroundMaterialProfile:
    """Parse canonical profile JSON, rejecting alternate encodings."""
    profile = _profile(_strict_canonical_object(text))
    if profile.to_json() != text:
        raise ValueError("ground material profile JSON must be canonical")
    return profile


def library_from_json(text: str) -> GroundProfileLibrary:
    """Parse canonical library JSON, rejecting alternate encodings."""
    library = _library(_strict_canonical_object(text))
    if library.to_json() != text:
        raise ValueError("ground profile library JSON must be canonical")
    return library


__all__ = [
    "document_from_dict",
    "document_to_dict",
    "document_to_json",
    "library_from_json",
    "profile_from_json",
]
