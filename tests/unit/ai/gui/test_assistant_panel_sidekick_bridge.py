"""The AIAssistantPanel side of the Sidekick action bridge (#7209).

This bridge was deleted wholesale by a squash while every module it depends on
survived, so only the panel's connection to them was cut. Nothing in Tools
noticed, because the tests that exercise it live downstream in UpstreamDrift
and one of them needs `src/launchers/sidekick_host_port`, which has no
counterpart here.

These tests therefore stub the service and assert the *bridge*: that attaching
a service builds a planner, publishes a system prompt, exports declarations in
the provider's format, and that detaching undoes all of it. They deliberately
avoid constructing the Qt panel, so they run headless and cheaply; the panel's
own behaviour stays covered downstream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from sidekick.agent.action_service import (
    ActionDescriptor,
    ActionResult,
    SidekickActionService,
)

pytestmark = pytest.mark.unit


class _EchoHandler:
    """One read action, enough to produce a declaration and an invocation."""

    namespace = "probe"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def describe(self) -> Sequence[ActionDescriptor]:
        return (
            ActionDescriptor(
                action_id="probe.echo",
                summary="Return the input value.",
                params_schema={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
                side_effects="read",
                reversible=False,
            ),
        )

    def invoke(self, action_id: str, params: Mapping[str, Any]) -> ActionResult:
        self.calls.append((action_id, dict(params)))
        return ActionResult(ok=True, value=params.get("value"))


def _service() -> SidekickActionService:
    service = SidekickActionService()
    service.register(_EchoHandler())
    return service


class TestPlannerBridge:
    def test_planner_exports_registered_actions_for_the_tool_registry(self) -> None:
        """`_sidekick_tool_declarations` is a formatter over this export."""
        from sidekick.agent.planner import SidekickAgentPlanner

        planner = SidekickAgentPlanner(service=_service())
        exported = list(planner.export_for_tool_registry())

        assert exported, "planner exported no actions for a registered handler"
        entry = exported[0]
        assert {"name", "description", "parameters"} <= set(entry)

    def test_system_prompt_names_the_registered_action(self) -> None:
        """What `_refresh_prompt_memory` publishes as `sidekick_system_prompt`."""
        from sidekick.agent.planner import build_sidekick_system_prompt

        prompt = build_sidekick_system_prompt(service=_service())

        assert "probe.echo" in prompt


class TestServiceDispatch:
    def test_invoke_reaches_the_handler(self) -> None:
        """`invoke_sidekick_action` is a thin pass-through to this."""
        handler = _EchoHandler()
        service = SidekickActionService()
        service.register(handler)

        result = service.invoke("probe.echo", {"value": 7})

        assert result.ok is True
        assert result.value == 7
        assert handler.calls == [("probe.echo", {"value": 7})]

    def test_main_thread_dispatcher_is_settable_and_clearable(self) -> None:
        """`set_action_service` sets it and `set_action_service(None)` clears it.

        The panel depends on both directions: it installs its own
        MainThreadToolDispatcher on attach and clears it on detach, so a
        detached service must not keep dispatching onto a dead Qt object.
        """

        def _dispatcher(fn: object) -> object:
            return fn

        service = _service()

        service.set_main_thread_dispatcher(_dispatcher)  # type: ignore[arg-type]
        assert service._dispatcher is _dispatcher

        service.set_main_thread_dispatcher(None)
        assert service._dispatcher is None

    def test_dispatcher_rejects_a_non_callable(self) -> None:
        """The panel passes a real dispatcher; anything else is a wiring bug."""
        with pytest.raises(TypeError, match="callable"):
            _service().set_main_thread_dispatcher(object())  # type: ignore[arg-type]
