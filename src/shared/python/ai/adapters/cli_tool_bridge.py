"""Shared bridge glue connecting CLI adapters to ``sidekick.agent.cli_bridge``."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from shared.python.ai.types import ConversationContext, ToolCall
from shared.python.sidekick.agent.cli_bridge import (
    DEFAULT_MAX_GOAL_STEPS,
    MAX_GOAL_REPAIRS,
    CliBridgeGoalOutcome,
    CliBridgeLoopStatus,
    CliTurnResponse,
    ToolStubSpec,
    build_nonce_tool_prompt,
    parse_nonce_tool_envelopes,
    probe_cli_tool_capability,
    run_bounded_cli_goal_loop,
    validate_cli_goal_loop_bounds,
)

_ENVELOPE_RE = re.compile(
    r'<sidekick_tool_call\s+nonce="([^"]+)"\s*>(.*?)</sidekick_tool_call>',
    re.DOTALL,
)


def tool_declaration_to_stub_spec(
    tool: ToolDeclaration | ToolStubSpec,
) -> ToolStubSpec:
    """Convert a ``ToolDeclaration`` into a sidekick ``ToolStubSpec``."""
    if isinstance(tool, ToolStubSpec):
        return tool
    params = dict(tool.parameters or {})
    if "properties" in params and isinstance(params["properties"], Mapping):
        schema: dict[str, Any] = {
            "type": str(params.get("type", "object")),
            "properties": dict(params["properties"]),
            "required": list(tool.required or params.get("required") or ()),
        }
    else:
        schema = {
            "type": "object",
            "properties": params,
            "required": list(tool.required or ()),
        }
    return ToolStubSpec(
        name=tool.name,
        description=tool.description or "",
        parameters_schema=schema,
    )


def build_cli_tool_prompt_section(
    tools: Sequence[ToolDeclaration | ToolStubSpec],
    nonce: str,
    *,
    fallback_full_schema: bool = False,
) -> str:
    """Delegate compact TypeScript tool-stub generation to ``cli_bridge``."""
    stubs = [tool_declaration_to_stub_spec(tool) for tool in tools]
    rendered: str = build_nonce_tool_prompt(
        stubs, nonce=nonce, fallback_full_schema=fallback_full_schema
    )
    return rendered


def parse_cli_tool_envelopes_with_diagnostics(
    raw_text: str,
    *,
    expected_nonce: str,
    tools: Sequence[ToolDeclaration | ToolStubSpec],
) -> tuple[str, list[ToolCall], int, tuple[str, ...]]:
    """Extract nonce-verified ``ToolCall`` requests plus diagnostic errors."""
    if not expected_nonce:
        raise ValueError("expected_nonce must be non-empty")
    allowed = {
        spec.name: spec for spec in (tool_declaration_to_stub_spec(t) for t in tools)
    }
    calls: list[ToolCall] = []
    errors: list[str] = []
    rejected = 0
    spans: list[tuple[int, int]] = []
    for match in _ENVELOPE_RE.finditer(raw_text or ""):
        batch = parse_nonce_tool_envelopes(
            match.group(0),
            expected_nonce=expected_nonce,
            channel="assistant",
            allowed_tools=allowed,
        )
        rejected += batch.rejected_nonce_count
        errors.extend(batch.errors)
        if batch.calls:
            calls.extend(
                ToolCall.create(name=call.action_id, arguments=dict(call.params))
                for call in batch.calls
            )
            spans.append(match.span())
    cleaned = raw_text or ""
    for start, end in reversed(spans):
        cleaned = (cleaned[:start] + cleaned[end:]).strip()
    return cleaned.strip(), calls, rejected, tuple(errors)


def parse_cli_tool_envelopes(
    raw_text: str, *, expected_nonce: str, tools: Sequence[ToolDeclaration]
) -> tuple[str, list[ToolCall], int]:
    """Extract nonce-verified ``ToolCall`` requests from CLI assistant output."""
    cleaned, calls, rejected, _ = parse_cli_tool_envelopes_with_diagnostics(
        raw_text, expected_nonce=expected_nonce, tools=tools
    )
    return cleaned, calls, rejected


class _AdapterTurnTransport:
    """Adapt a synchronous ``BaseAgentAdapter`` to ``cli_bridge.CliTurnTransport``."""

    def __init__(self, adapter: BaseAgentAdapter) -> None:
        self._adapter = adapter

    def send_turn(self, prompt: str, *, nonce: str) -> CliTurnResponse:  # noqa: ARG002
        response = self._adapter.send_message(prompt, ConversationContext(), [])
        return CliTurnResponse(assistant_text=response.content, channel="assistant")


def run_adapter_cli_goal_loop(
    adapter: BaseAgentAdapter,
    goal_prompt: str,
    *,
    tools: Sequence[ToolDeclaration | ToolStubSpec],
    executor: Callable[[Any], str],
    max_steps: int = DEFAULT_MAX_GOAL_STEPS,
    max_repairs: int = MAX_GOAL_REPAIRS,
    probe_version: bool = False,
) -> CliBridgeGoalOutcome:
    """Run ``cli_bridge.run_bounded_cli_goal_loop`` backed by a CLI adapter."""
    validate_cli_goal_loop_bounds(max_steps, max_repairs)
    stubs = tuple(tool_declaration_to_stub_spec(tool) for tool in tools)
    if probe_version:
        ok, msg = adapter.validate_connection()
        version_text = msg.split(":", 1)[-1].strip() if ":" in msg else msg
        report = probe_cli_tool_capability(
            getattr(adapter, "PROVIDER_NAME", ""), version_text
        )
        if not ok or not report.tools_supported:
            return CliBridgeGoalOutcome(
                CliBridgeLoopStatus.CAPABILITY_UNAVAILABLE,
                "",
                (),
                0,
                0,
                report.reason or msg,
            )
    return run_bounded_cli_goal_loop(
        goal_prompt,
        transport=_AdapterTurnTransport(adapter),
        tools=stubs,
        executor=executor,
        max_steps=max_steps,
        max_repairs=max_repairs,
    )
