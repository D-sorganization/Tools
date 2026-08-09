"""Machine-readable schema and fail-closed migration gateway tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

from shared.python.swing_sim.ground import (
    JSON_SCHEMA_DIALECT,
    migrate_request_to_current,
    migrate_result_to_current,
    request_json_schema,
    result_json_schema,
    schema_json,
)

from ._support import _request, _result


def _assert_strict_object(schema: dict[str, Any]) -> None:
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(cast(list[str], schema["required"])) == set(
        cast(dict[str, Any], schema["properties"])
    )


def test_machine_readable_schemas_are_deterministic_and_strict() -> None:
    request_schema = request_json_schema()
    result_schema = result_json_schema()

    for schema in (request_schema, result_schema):
        assert schema["$schema"] == JSON_SCHEMA_DIALECT
        _assert_strict_object(schema)
        for definition in cast(dict[str, dict[str, Any]], schema["$defs"]).values():
            _assert_strict_object(definition)
        assert schema_json(schema) == schema_json(deepcopy(schema))

    assert (
        cast(dict[str, dict[str, Any]], request_schema["properties"])["schema_version"][
            "const"
        ]
        == _request().schema_version
    )
    assert (
        cast(dict[str, dict[str, Any]], result_schema["properties"])["schema_version"][
            "const"
        ]
        == _result().schema_version
    )


def test_current_payload_migration_is_canonical_and_nonmutating() -> None:
    request = _request().to_dict()
    result = _result().to_dict()

    assert migrate_request_to_current(deepcopy(request)) == request
    assert migrate_result_to_current(deepcopy(result)) == result


def test_schema_integer_numbers_normalize_consistently_at_runtime() -> None:
    request = _request().to_dict()
    request["max_events"] = 64.0
    assert migrate_request_to_current(request)["max_events"] == 64

    result = _result().to_dict()
    result["events"][0]["sequence"] = 0.0
    result["summary"]["bounce_count"] = 1.0
    migrated = migrate_result_to_current(result)
    assert migrated["events"][0]["sequence"] == 0
    assert migrated["summary"]["bounce_count"] == 1


def test_schema_and_runtime_reject_noncanonical_positive_and_text_edges() -> None:
    schema = Draft202012Validator(request_json_schema())
    tiny = _request().to_dict()
    tiny["output_interval_s"] = 1e-12
    assert not schema.is_valid(tiny)
    with pytest.raises(ValueError, match="at least"):
        migrate_request_to_current(tiny)

    for request_id in ("   ", " ground-run-001 "):
        padded = _request().to_dict()
        padded["request_id"] = request_id
        assert not schema.is_valid(padded)
        with pytest.raises(ValueError, match="whitespace"):
            migrate_request_to_current(padded)


def test_migration_rejects_versions_that_need_an_undefined_transform() -> None:
    request = _request().to_dict()
    request["schema_version"] = "flight-to-ground-request/v0"
    with pytest.raises(ValueError, match="schema_version"):
        migrate_request_to_current(request)

    result = _result().to_dict()
    result["schema_version"] = "flight-to-ground-result/v2"
    with pytest.raises(ValueError, match="schema_version"):
        migrate_result_to_current(result)
