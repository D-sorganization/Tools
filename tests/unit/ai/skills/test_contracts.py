"""Tests for ai.skills.contracts (Tools #2737)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.python.ai.skills.contracts import (
    SkillDescriptor,
    SkillInvocation,
    SkillResult,
)

pytestmark = pytest.mark.unit


def _make_descriptor(**overrides: object) -> SkillDescriptor:
    payload: dict[str, object] = {
        "id": "demo.echo",
        "name": "Demo Echo",
        "version": "1.0.0",
        "description": "Echoes the supplied payload.",
        "inputs": {"message": "string"},
        "outputs": {"echoed": "string"},
        "preconditions": ["message_is_non_empty"],
        "postconditions": ["echoed_equals_message"],
    }
    payload.update(overrides)
    return SkillDescriptor(**payload)  # type: ignore[arg-type]


class TestSkillDescriptor:
    def test_round_trip_via_model_dump(self) -> None:
        descriptor = _make_descriptor()
        clone = SkillDescriptor(**descriptor.model_dump())
        assert clone == descriptor

    def test_rejects_blank_id(self) -> None:
        with pytest.raises(ValidationError):
            _make_descriptor(id="")

    def test_strips_required_string_fields(self) -> None:
        descriptor = _make_descriptor(id=" demo.echo ", name=" Demo Echo ")
        assert descriptor.id == "demo.echo"
        assert descriptor.name == "Demo Echo"

    def test_rejects_blank_version(self) -> None:
        with pytest.raises(ValidationError):
            _make_descriptor(version="")

    def test_preconditions_default_to_empty_list(self) -> None:
        descriptor = SkillDescriptor(
            id="demo.bare",
            name="Bare",
            version="0.1.0",
            description="bare",
            inputs={},
            outputs={},
        )
        assert descriptor.preconditions == []
        assert descriptor.postconditions == []


class TestSkillInvocation:
    def test_request_id_defaults_to_uuid_string(self) -> None:
        inv = SkillInvocation(skill_id="demo.echo", args={"message": "hi"})
        assert isinstance(inv.request_id, str)
        assert len(inv.request_id) >= 8

    def test_explicit_request_id_preserved(self) -> None:
        inv = SkillInvocation(
            skill_id="demo.echo",
            args={"message": "hi"},
            request_id="req-1",
        )
        assert inv.request_id == "req-1"

    def test_negative_timeout_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SkillInvocation(skill_id="demo.echo", args={}, timeout_s=-1.0)


class TestSkillResult:
    def test_audit_trail_defaults_to_empty(self) -> None:
        result = SkillResult(
            request_id="req-1",
            success=True,
            value={"echoed": "x"},
            error=None,
            elapsed_ms=1.0,
        )
        assert result.audit_trail == []

    def test_error_field_serialises_to_string(self) -> None:
        result = SkillResult(
            request_id="req-2",
            success=False,
            value=None,
            error="boom",
            elapsed_ms=0.0,
        )
        payload = result.model_dump()
        assert payload["error"] == "boom"
        assert payload["success"] is False
