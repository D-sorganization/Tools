"""Tests for the built-in echo skill (Tools #2737)."""

from __future__ import annotations

import pytest

from shared.python.ai.skills import SkillRegistry, SkillRunner
from shared.python.ai.skills.builtin.echo import EchoSkill
from shared.python.ai.skills.contracts import SkillInvocation


pytestmark = pytest.mark.unit


async def test_echo_skill_round_trip() -> None:
    registry = SkillRegistry()
    registry.register(EchoSkill)
    runner = SkillRunner(registry=registry)

    result = await runner.run(
        SkillInvocation(
            skill_id=EchoSkill.descriptor.id,
            args={"message": "hello"},
            request_id="echo-1",
        )
    )

    assert result.success is True
    assert result.value == {"echoed": "hello"}
    assert result.error is None
    # Audit trail records start + completion.
    kinds = [event["kind"] for event in result.audit_trail]
    assert "started" in kinds
    assert "completed" in kinds


async def test_echo_skill_rejects_empty_message() -> None:
    registry = SkillRegistry()
    registry.register(EchoSkill)
    runner = SkillRunner(registry=registry)

    result = await runner.run(
        SkillInvocation(
            skill_id=EchoSkill.descriptor.id,
            args={"message": ""},
            request_id="echo-2",
        )
    )

    assert result.success is False
    assert result.error is not None


def test_echo_descriptor_declares_pre_and_post() -> None:
    assert EchoSkill.descriptor.preconditions, "must declare preconditions"
    assert EchoSkill.descriptor.postconditions, "must declare postconditions"
