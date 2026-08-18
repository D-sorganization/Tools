"""Schema and migration parity tests for ground profile contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

from shared.python.swing_sim.ground.profile_migration import (
    migrate_library_to_current,
    migrate_profile_to_current,
)
from shared.python.swing_sim.ground.profile_schema import (
    JSON_SCHEMA_DIALECT,
    library_json_schema,
    profile_json_schema,
    schema_json,
    validate_library_payload,
    validate_profile_payload,
)

from .test_profile_contract import _library, _profile


def _assert_strict_object(schema: dict[str, Any]) -> None:
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(cast(list[str], schema["required"])) == set(
        cast(dict[str, Any], schema["properties"])
    )


def test_profile_and_library_schemas_are_deterministic_and_strict() -> None:
    for schema in (profile_json_schema(), library_json_schema()):
        assert schema["$schema"] == JSON_SCHEMA_DIALECT
        assert schema["x-semantic-validator"].endswith("_payload")
        assert "necessary but not sufficient" in schema["$comment"]
        _assert_strict_object(schema)
        assert schema_json(schema) == schema_json(deepcopy(schema))
        Draft202012Validator.check_schema(schema)

    profile_defs = cast(dict[str, dict[str, Any]], profile_json_schema()["$defs"])
    for definition in profile_defs.values():
        if definition.get("type") == "object":
            _assert_strict_object(definition)


def test_runtime_payloads_validate_against_machine_readable_schemas() -> None:
    profile_payload = _profile().to_dict()
    library_payload = _library().to_dict()

    assert Draft202012Validator(profile_json_schema()).is_valid(profile_payload)
    assert Draft202012Validator(library_json_schema()).is_valid(library_payload)
    parameter_properties = profile_json_schema()["$defs"]["profile"]["properties"][
        "parameters"
    ]["prefixItems"][0]["properties"]
    assert set(parameter_properties) >= {
        "validity_lower_si",
        "validity_upper_si",
        "validity_lower_evidence_ids",
        "validity_upper_evidence_ids",
    }
    assert profile_json_schema()["$defs"]["profile"]["properties"]["model_use_status"][
        "enum"
    ] == ["illustrative", "calibrated"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("parameters", 0, "value_si"), True),
        (("parameters", 0, "parameter_id"), "unknown"),
        (("rights", "redistribution_allowed"), 1),
    ],
)
def test_schema_and_runtime_reject_the_same_invalid_profile_shapes(
    path: tuple[str | int, ...], value: object
) -> None:
    payload = _profile().to_dict()
    target: Any = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    assert not Draft202012Validator(profile_json_schema()).is_valid(payload)
    with pytest.raises((TypeError, ValueError)):
        migrate_profile_to_current(payload)


def test_non_json_nan_is_rejected_by_runtime_and_canonical_serialization() -> None:
    payload = _profile().to_dict()
    payload["parameters"][0]["value_si"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        migrate_profile_to_current(payload)
    with pytest.raises(ValueError, match="finite"):
        schema_json({"invalid_non_json_number": float("nan")})


def test_current_migration_is_v1_only_canonical_and_nonmutating() -> None:
    profile = _profile().to_dict()
    library = _library().to_dict()
    original_profile = deepcopy(profile)
    original_library = deepcopy(library)

    assert migrate_profile_to_current(profile) == original_profile
    assert migrate_library_to_current(library) == original_library
    assert profile == original_profile
    assert library == original_library

    profile["schema_version"] = "ground-material-profile/v0"
    with pytest.raises(ValueError, match="schema_version"):
        migrate_profile_to_current(profile)
    library["schema_version"] = "ground-profile-library/v2"
    with pytest.raises(ValueError, match="schema_version"):
        migrate_library_to_current(library)


def test_migration_rejects_unknown_nested_fields() -> None:
    profile = _profile().to_dict()
    profile["evidence"][0]["unreviewed_note"] = "must not pass silently"

    assert not Draft202012Validator(profile_json_schema()).is_valid(profile)
    with pytest.raises(ValueError, match="fields"):
        migrate_profile_to_current(profile)


@pytest.mark.parametrize(
    "mutation",
    [
        "reversed_evidence_parameters",
        "missing_calibration_reference",
        "incoherent_qualification",
        "duplicate_evidence_identity",
        "kinetic_above_static",
        "inverted_temperature_range",
        "unsorted_surface_classes",
        "validity_bounds_do_not_enclose_value",
        "unknown_validity_source",
        "incoherent_model_use_status",
    ],
)
def test_structural_schema_requires_authoritative_semantic_validation(
    mutation: str,
) -> None:
    payload = _profile().to_dict()
    if mutation == "reversed_evidence_parameters":
        payload["evidence"][0]["parameter_ids"].reverse()
    elif mutation == "missing_calibration_reference":
        payload["calibration"]["evidence_ids"] = ["missing"]
    elif mutation == "incoherent_qualification":
        payload["qualification"]["status"] = "unqualified"
    elif mutation == "duplicate_evidence_identity":
        payload["evidence"].append(deepcopy(payload["evidence"][0]))
    elif mutation == "kinetic_above_static":
        payload["parameters"][2]["value_si"] = 0.9
    elif mutation == "inverted_temperature_range":
        payload["applicability"]["temperature_min_k"] = 400.0
    elif mutation == "unsorted_surface_classes":
        payload["applicability"]["surface_classes"] = ["rough", "fairway"]
    elif mutation == "validity_bounds_do_not_enclose_value":
        payload["parameters"][0]["validity_lower_si"] = 0.5
    elif mutation == "unknown_validity_source":
        payload["parameters"][0]["validity_lower_evidence_ids"] = ["missing"]
        payload["qualification"]["gates"][1]["passed"] = False
        payload["qualification"]["status"] = "unqualified"
        payload["model_use_status"] = "illustrative"
    elif mutation == "incoherent_model_use_status":
        payload["model_use_status"] = "illustrative"

    assert Draft202012Validator(profile_json_schema()).is_valid(payload)
    with pytest.raises((TypeError, ValueError)):
        validate_profile_payload(payload)


def test_library_semantics_reject_duplicate_profile_identity_after_schema() -> None:
    payload = _library().to_dict()
    payload["profiles"].append(deepcopy(payload["profiles"][0]))

    assert Draft202012Validator(library_json_schema()).is_valid(payload)
    with pytest.raises(ValueError, match="unique"):
        validate_library_payload(payload)
