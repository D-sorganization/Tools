"""Strict deterministic JSON persistence for reduced turf-contact profiles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ._validation import reject_unknown_fields, require_mapping
from .turf_contact import (
    TurfCalibrationStatus,
    TurfContactProfile,
    TurfProfileProvenance,
)

TURF_PROFILE_FORMAT = "golf-club.turf-profile/v1"

_ROOT_FIELDS = frozenset({"format", "profile"})
_PROFILE_FIELDS = frozenset(
    {
        "profile_id",
        "normal_stiffness_n_m",
        "normal_damping_n_s_m",
        "friction_coefficient",
        "friction_regularization_mps",
        "max_penetration_m",
        "calibration_status",
        "provenance",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {"source_name", "parameter_basis", "uncertainty_note", "source_uri"}
)


def turf_profile_to_json_dict(profile: TurfContactProfile) -> dict[str, Any]:
    """Return a complete versioned JSON boundary record."""
    if not isinstance(profile, TurfContactProfile):
        raise TypeError("profile must be TurfContactProfile")
    provenance = profile.provenance
    return {
        "format": TURF_PROFILE_FORMAT,
        "profile": {
            "profile_id": profile.profile_id,
            "normal_stiffness_n_m": profile.normal_stiffness_n_m,
            "normal_damping_n_s_m": profile.normal_damping_n_s_m,
            "friction_coefficient": profile.friction_coefficient,
            "friction_regularization_mps": profile.friction_regularization_mps,
            "max_penetration_m": profile.max_penetration_m,
            "calibration_status": profile.calibration_status.value,
            "provenance": {
                "source_name": provenance.source_name,
                "parameter_basis": provenance.parameter_basis,
                "uncertainty_note": provenance.uncertainty_note,
                "source_uri": provenance.source_uri,
            },
        },
    }


def turf_profile_to_json(profile: TurfContactProfile) -> str:
    """Serialize deterministically for storage, hashing, and replay."""
    return json.dumps(
        turf_profile_to_json_dict(profile),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )


def _required(data: Mapping[str, Any], field: str, boundary: str) -> Any:
    if field not in data:
        raise ValueError(f"{boundary} is missing required field {field!r}")
    return data[field]


def turf_profile_from_json_dict(source: object) -> TurfContactProfile:
    """Validate and load one exact v1 profile record."""
    root = require_mapping(source, "turf profile payload")
    reject_unknown_fields(root, _ROOT_FIELDS, "turf profile payload")
    if _required(root, "format", "turf profile payload") != TURF_PROFILE_FORMAT:
        raise ValueError("unsupported turf profile format")
    data = require_mapping(
        _required(root, "profile", "turf profile payload"), "profile"
    )
    reject_unknown_fields(data, _PROFILE_FIELDS, "profile")
    provenance_data = require_mapping(
        _required(data, "provenance", "profile"), "provenance"
    )
    reject_unknown_fields(provenance_data, _PROVENANCE_FIELDS, "provenance")
    source_uri = provenance_data.get("source_uri")
    if source_uri is not None and not isinstance(source_uri, str):
        raise TypeError("source_uri must be a string or null")
    return TurfContactProfile(
        profile_id=_required(data, "profile_id", "profile"),
        normal_stiffness_n_m=_required(data, "normal_stiffness_n_m", "profile"),
        normal_damping_n_s_m=_required(data, "normal_damping_n_s_m", "profile"),
        friction_coefficient=_required(data, "friction_coefficient", "profile"),
        friction_regularization_mps=_required(
            data, "friction_regularization_mps", "profile"
        ),
        max_penetration_m=_required(data, "max_penetration_m", "profile"),
        calibration_status=TurfCalibrationStatus(
            _required(data, "calibration_status", "profile")
        ),
        provenance=TurfProfileProvenance(
            source_name=_required(provenance_data, "source_name", "provenance"),
            parameter_basis=_required(provenance_data, "parameter_basis", "provenance"),
            uncertainty_note=_required(
                provenance_data, "uncertainty_note", "provenance"
            ),
            source_uri=source_uri,
        ),
    )


def turf_profile_from_json(payload: str) -> TurfContactProfile:
    """Load a profile from strict JSON text."""
    if not isinstance(payload, str):
        raise TypeError("payload must be a string")
    try:
        source = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("payload must be valid JSON") from error
    return turf_profile_from_json_dict(source)


__all__ = [
    "TURF_PROFILE_FORMAT",
    "turf_profile_from_json",
    "turf_profile_from_json_dict",
    "turf_profile_to_json",
    "turf_profile_to_json_dict",
]
