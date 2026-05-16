"""Tests for the built-in summarize skill (Tools #2737)."""

from __future__ import annotations

import pytest

from shared.python.ai.skills import SkillRegistry, SkillRunner
from shared.python.ai.skills.builtin.summarize import (
    StubLLMClient,
    SummarizeSkill,
)
from shared.python.ai.skills.contracts import SkillInvocation


pytestmark = pytest.mark.unit


async def test_summarize_skill_with_stub_llm() -> None:
    stub = StubLLMClient(canned_response="SUMMARY: hello world")
    skill = SummarizeSkill(llm_client=stub)

    registry = SkillRegistry()
    registry.register_instance(skill)
    runner = SkillRunner(registry=registry)

    result = await runner.run(
        SkillInvocation(
            skill_id=SummarizeSkill.descriptor.id,
            args={"text": "Hello world. This is a test passage that needs summarising."},
            request_id="sum-1",
        )
    )

    assert result.success is True
    assert result.value is not None
    assert "summary" in result.value
    assert result.value["summary"] == "SUMMARY: hello world"
    # Audit trail should record the LLM call.
    kinds = [event["kind"] for event in result.audit_trail]
    assert "llm_call" in kinds
    assert stub.call_count == 1


async def test_summarize_skill_rejects_blank_input() -> None:
    stub = StubLLMClient(canned_response="never used")
    skill = SummarizeSkill(llm_client=stub)

    registry = SkillRegistry()
    registry.register_instance(skill)
    runner = SkillRunner(registry=registry)

    result = await runner.run(
        SkillInvocation(
            skill_id=SummarizeSkill.descriptor.id,
            args={"text": "   "},
            request_id="sum-2",
        )
    )

    assert result.success is False
    assert stub.call_count == 0, "LLM must not be called when preconditions fail"


def test_summarize_descriptor_declares_pre_and_post() -> None:
    assert SummarizeSkill.descriptor.preconditions
    assert SummarizeSkill.descriptor.postconditions
