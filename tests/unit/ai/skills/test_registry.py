"""Tests for ai.skills.registry (Tools #2737)."""

from __future__ import annotations

import pytest

from shared.python.ai.skills import (
    Skill,
    SkillRegistry,
    register_skill,
)
from shared.python.ai.skills.contracts import (
    SkillDescriptor,
    SkillInvocation,
    SkillResult,
)
from shared.python.ai.skills.errors import SkillNotFoundError

pytestmark = pytest.mark.unit


class _DummySkill(Skill):
    descriptor = SkillDescriptor(
        id="dummy.one",
        name="Dummy One",
        version="0.0.1",
        description="dummy",
        inputs={},
        outputs={},
        preconditions=[],
        postconditions=[],
    )

    async def run(self, invocation: SkillInvocation) -> SkillResult:
        return SkillResult(
            request_id=invocation.request_id,
            success=True,
            value={},
            error=None,
            elapsed_ms=0.0,
        )


class _DummySkillTwo(Skill):
    descriptor = SkillDescriptor(
        id="dummy.two",
        name="Dummy Two",
        version="0.0.1",
        description="dummy two",
        inputs={},
        outputs={},
        preconditions=[],
        postconditions=[],
    )

    async def run(self, invocation: SkillInvocation) -> SkillResult:
        return SkillResult(
            request_id=invocation.request_id,
            success=True,
            value={},
            error=None,
            elapsed_ms=0.0,
        )


class TestSkillRegistry:
    def test_concrete_skill_without_descriptor_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="must define a 'descriptor"):

            class _MissingDescriptor(Skill):
                async def run(self, invocation: SkillInvocation) -> SkillResult:
                    return SkillResult(
                        request_id=invocation.request_id,
                        success=True,
                        value={},
                        error=None,
                        elapsed_ms=0.0,
                    )

    def test_register_and_get(self) -> None:
        registry = SkillRegistry()
        registry.register(_DummySkill)
        skill = registry.get("dummy.one")
        assert isinstance(skill, _DummySkill)

    def test_get_unknown_raises(self) -> None:
        registry = SkillRegistry()
        with pytest.raises(SkillNotFoundError):
            registry.get("does.not.exist")

    def test_list_returns_descriptors(self) -> None:
        registry = SkillRegistry()
        registry.register(_DummySkill)
        registry.register(_DummySkillTwo)
        descriptors = list(registry.list())
        ids = {d.id for d in descriptors}
        assert ids == {"dummy.one", "dummy.two"}

    def test_duplicate_register_raises(self) -> None:
        registry = SkillRegistry()
        registry.register(_DummySkill)
        with pytest.raises(ValueError):
            registry.register(_DummySkill)

    def test_duplicate_register_instance_raises(self) -> None:
        registry = SkillRegistry()
        registry.register_instance(_DummySkill())
        with pytest.raises(ValueError):
            registry.register_instance(_DummySkill())

    def test_register_decorator(self) -> None:
        registry = SkillRegistry()

        @register_skill(registry=registry)
        class _Decorated(Skill):
            descriptor = SkillDescriptor(
                id="dummy.decorated",
                name="Decorated",
                version="0.0.1",
                description="decorated",
                inputs={},
                outputs={},
                preconditions=[],
                postconditions=[],
            )

            async def run(self, invocation: SkillInvocation) -> SkillResult:
                return SkillResult(
                    request_id=invocation.request_id,
                    success=True,
                    value={},
                    error=None,
                    elapsed_ms=0.0,
                )

        assert isinstance(registry.get("dummy.decorated"), _Decorated)


class TestDefaultRegistry:
    def test_default_registry_is_singleton(self) -> None:
        from shared.python.ai.skills.registry import default_registry

        a = default_registry()
        b = default_registry()
        assert a is b
