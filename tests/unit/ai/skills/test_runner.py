"""Tests for ai.skills.runner (Tools #2737)."""

from __future__ import annotations

import asyncio

import pytest

from shared.python.ai.skills import (
    Skill,
    SkillRegistry,
    SkillRunner,
)
from shared.python.ai.skills.contracts import (
    SkillDescriptor,
    SkillInvocation,
    SkillResult,
)
from shared.python.ai.skills.errors import (
    SkillExecutionError,
    SkillNotFoundError,
    SkillPostconditionError,
    SkillPreconditionError,
    SkillTimeoutError,
)

pytestmark = pytest.mark.unit


class _OkSkill(Skill):
    descriptor = SkillDescriptor(
        id="t.ok",
        name="Ok",
        version="0.0.1",
        description="ok",
        inputs={"x": "int"},
        outputs={"y": "int"},
        preconditions=["x_is_int"],
        postconditions=["y_equals_x_plus_one"],
    )

    def validate_preconditions(self, args: dict[str, object]) -> None:
        if not isinstance(args.get("x"), int):
            raise ValueError("x_is_int")

    def validate_postconditions(self, result: dict[str, object]) -> None:
        assert isinstance(result["y"], int)

    async def run(self, invocation: SkillInvocation) -> SkillResult:
        x = int(invocation.args["x"])
        return SkillResult(
            request_id=invocation.request_id,
            success=True,
            value={"y": x + 1},
            error=None,
            elapsed_ms=0.0,
        )


class _BadPostSkill(Skill):
    descriptor = SkillDescriptor(
        id="t.badpost",
        name="BadPost",
        version="0.0.1",
        description="bad post",
        inputs={},
        outputs={"y": "int"},
        preconditions=[],
        postconditions=["y_is_int"],
    )

    def validate_postconditions(self, result: dict[str, object]) -> None:
        if not isinstance(result.get("y"), int):
            raise ValueError("y_is_int")

    async def run(self, invocation: SkillInvocation) -> SkillResult:
        return SkillResult(
            request_id=invocation.request_id,
            success=True,
            value={"y": "not-an-int"},
            error=None,
            elapsed_ms=0.0,
        )


class _SlowSkill(Skill):
    descriptor = SkillDescriptor(
        id="t.slow",
        name="Slow",
        version="0.0.1",
        description="slow",
        inputs={},
        outputs={},
        preconditions=[],
        postconditions=[],
    )

    async def run(self, invocation: SkillInvocation) -> SkillResult:
        await asyncio.sleep(1.0)
        return SkillResult(
            request_id=invocation.request_id,
            success=True,
            value={},
            error=None,
            elapsed_ms=0.0,
        )


class _RaisesSkill(Skill):
    descriptor = SkillDescriptor(
        id="t.raises",
        name="Raises",
        version="0.0.1",
        description="raises from body",
        inputs={},
        outputs={},
        preconditions=[],
        postconditions=[],
    )

    async def run(self, invocation: SkillInvocation) -> SkillResult:
        raise RuntimeError("body exploded")


def _build_runner(*skills: type[Skill]) -> SkillRunner:
    registry = SkillRegistry()
    for skill_cls in skills:
        registry.register(skill_cls)
    return SkillRunner(registry=registry)


class TestSkillRunner:
    async def test_happy_path_succeeds(self) -> None:
        runner = _build_runner(_OkSkill)
        result = await runner.run(
            SkillInvocation(skill_id="t.ok", args={"x": 41}, request_id="r1")
        )
        assert result.success is True
        assert result.value == {"y": 42}
        assert result.error is None
        assert result.audit_trail, "expected audit trail emitted"

    async def test_unknown_skill_raises(self) -> None:
        runner = _build_runner()
        with pytest.raises(SkillNotFoundError):
            await runner.run(
                SkillInvocation(skill_id="t.missing", args={}, request_id="r2")
            )

    async def test_precondition_failure_returns_structured_result(self) -> None:
        runner = _build_runner(_OkSkill)
        result = await runner.run(
            SkillInvocation(skill_id="t.ok", args={"x": "not-int"}, request_id="r3")
        )
        assert result.success is False
        assert result.error is not None
        assert "x_is_int" in result.error
        # Runner should classify this as a precondition error in the audit trail.
        kinds = [event["kind"] for event in result.audit_trail]
        assert "precondition_failed" in kinds

    async def test_postcondition_failure_returns_structured_result(self) -> None:
        runner = _build_runner(_BadPostSkill)
        result = await runner.run(
            SkillInvocation(skill_id="t.badpost", args={}, request_id="r4")
        )
        assert result.success is False
        assert result.error is not None
        kinds = [event["kind"] for event in result.audit_trail]
        assert "postcondition_failed" in kinds

    async def test_timeout_returns_structured_result(self) -> None:
        runner = _build_runner(_SlowSkill)
        result = await runner.run(
            SkillInvocation(
                skill_id="t.slow",
                args={},
                request_id="r5",
                timeout_s=0.05,
            )
        )
        assert result.success is False
        assert result.error is not None
        kinds = [event["kind"] for event in result.audit_trail]
        assert "timeout" in kinds

    async def test_body_exception_returns_structured_execution_error(self) -> None:
        runner = _build_runner(_RaisesSkill)
        result = await runner.run(
            SkillInvocation(skill_id="t.raises", args={}, request_id="r-exec")
        )

        assert result.success is False
        assert result.error == "body exploded"
        kinds = [event["kind"] for event in result.audit_trail]
        assert "execution_error" in kinds
        assert result.audit_trail[-1]["extra"]["failure_kind"] == "execution_error"

    async def test_raises_directly_when_skill_id_unknown(self) -> None:
        runner = _build_runner(_OkSkill)
        with pytest.raises(SkillNotFoundError):
            await runner.run(SkillInvocation(skill_id="nope", args={}, request_id="r6"))

    async def test_precondition_error_type_can_be_raised(self) -> None:
        # SkillPreconditionError exists and is the runner-internal raise type.
        err = SkillPreconditionError("nope")
        assert str(err) == "nope"

    async def test_postcondition_error_type_can_be_raised(self) -> None:
        err = SkillPostconditionError("nope")
        assert str(err) == "nope"

    async def test_timeout_error_type_can_be_raised(self) -> None:
        err = SkillTimeoutError("nope")
        assert str(err) == "nope"

    async def test_execution_error_type_can_be_raised(self) -> None:
        err = SkillExecutionError("nope")
        assert str(err) == "nope"
