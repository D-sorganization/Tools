from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[6]
SCHEMA = ROOT / "schemas" / "mocap" / "mocap-session-v1.schema.json"
FIXTURE = ROOT / "tests" / "fixtures" / "mocap" / "session-v1.golden.json"


def test_golden_session_validates_against_strict_schema() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(fixture)
    assert schema["additionalProperties"] is False


def test_schema_rejects_unknown_root_field() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixture["unknown"] = True
    errors = list(Draft202012Validator(schema).iter_errors(fixture))
    assert any("Additional properties" in error.message for error in errors)


def test_schema_rejects_mutable_or_ambiguous_transform_direction() -> None:
    schema_text = SCHEMA.read_text(encoding="utf-8")
    assert "T_world_from_camera" in schema_text
    assert '"extrinsics"' not in schema_text
    assert '"additionalProperties": false' in schema_text


def test_fixture_uses_no_private_or_local_uris() -> None:
    text = FIXTURE.read_text(encoding="utf-8").lower()
    forbidden = (
        "file://",
        "c:\\users\\",
        "tools_private",
        "github.com/d-sorganization",
    )
    assert not [value for value in forbidden if value in text]


@pytest.mark.parametrize("path", [SCHEMA, FIXTURE])
def test_contract_files_end_with_newline(path: Path) -> None:
    assert path.read_bytes().endswith(b"\n")
