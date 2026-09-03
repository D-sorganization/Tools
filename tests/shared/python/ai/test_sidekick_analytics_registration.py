"""The Sidekick analytics tool must actually be reachable from chat.

``register_sidekick_analytics_tools()`` shipped with #5464 but was called from
nowhere in ``src/``, so ``summarize_simulation_run`` was dead code: the chat
assistant could not invoke it and the system prompt never mentioned it.

Both halves are asserted here -- registration *and* the prompt -- because the
failure mode being fixed is precisely the two drifting apart.

Part of D-sorganization/UpstreamDrift#9474.
"""

from __future__ import annotations

import pytest

from src.shared.python.ai.sample_tools import register_golf_suite_tools
from src.shared.python.ai.system_prompts import build_system_prompt
from src.shared.python.ai.tool_registry import ToolRegistry
from src.shared.python.ai.tools.sidekick_analytics import (
    SIDEKICK_ANALYTICS_TOOL_NAME,
    register_sidekick_analytics_tools,
)

pytestmark = [pytest.mark.unit]


def test_golf_suite_registration_includes_sidekick_analytics() -> None:
    """The suite entry point must reach the analytics registrar."""
    registry = ToolRegistry()
    register_golf_suite_tools(registry)

    assert registry.get_tool(SIDEKICK_ANALYTICS_TOOL_NAME) is not None, (
        f"{SIDEKICK_ANALYTICS_TOOL_NAME} is registered nowhere reachable from "
        "register_golf_suite_tools(); the chat assistant cannot invoke it."
    )


def test_direct_registration_still_works() -> None:
    """The standalone registrar remains usable on its own registry."""
    registry = ToolRegistry()
    register_sidekick_analytics_tools(registry)

    assert registry.get_tool(SIDEKICK_ANALYTICS_TOOL_NAME) is not None


def test_system_prompt_advertises_the_tool() -> None:
    """A tool the prompt advertises must exist, and vice versa."""
    prompt = build_system_prompt("upstream_drift")

    assert SIDEKICK_ANALYTICS_TOOL_NAME in prompt, (
        "build_system_prompt('upstream_drift') does not mention "
        f"{SIDEKICK_ANALYTICS_TOOL_NAME}, so the assistant will not know it "
        "can call it."
    )


def test_registration_failure_is_not_swallowed() -> None:
    """A missing analytics module must raise, not log-and-continue.

    The system prompt advertises the tool unconditionally, so silently
    skipping registration would leave the assistant offering a capability it
    cannot call -- the exact defect this suite pins.
    """
    import src.shared.python.ai.sample_tools as sample_tools

    def _boom(_registry: ToolRegistry) -> None:
        raise ImportError("simulated missing analytics module")

    original = sample_tools._register_sidekick_analytics
    sample_tools._register_sidekick_analytics = _boom
    try:
        with pytest.raises(ImportError):
            register_golf_suite_tools(ToolRegistry())
    finally:
        sample_tools._register_sidekick_analytics = original
