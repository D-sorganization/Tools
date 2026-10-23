"""Authorization-free CLI tool-call protocol & loop orchestrator (#5728, Epic #5727).

Reviewed by 4-Seat Expert Panel ``th_8820201930a3`` (Astra, Argon, Fable, Barb):
per-turn CSPRNG nonce envelope ``<sidekick_tool_call nonce="<hex>">{...}``, assistant-
only channel check, reflected tag neutralization + 2KB truncation, minified TS stubs,
bounded loop (``max_steps <= 5``) with 1 repair turn, capability probe, & secret scrub.
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from .planner import ToolCall

DEFAULT_MAX_GOAL_STEPS = 5
MAX_TOOL_OUTPUT_BYTES = 2048
_NONCE_BYTES = 8
_LOOKALIKE_TAG_RE = re.compile(r"<\s*(/?)\s*sidekick_tool_call\b", re.IGNORECASE)
_VAULT_HANDLE_RE = re.compile(r"\$vault:([A-Za-z0-9_.-]{1,64})")
_SENSITIVE_ENV_SUFFIXES = ("_API_KEY", "_SECRET", "_TOKEN", "_PASSWORD", "_PRIVATE_KEY")
_PINNED_CLI_VERSION_PREFIXES: dict[str, tuple[str, ...]] = {
    "claude": ("1.", "2.", "claude-code/"),
    "codex": ("0.", "1.", "codex-cli/"),
    "agy": ("0.", "1.", "2.", "antigravity/"),
}


class CliBridgeLoopStatus(StrEnum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REPAIR_EXHAUSTED = "repair_exhausted"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"


@dataclass(frozen=True, slots=True)
class ToolStubSpec:
    """Compact, authorization-free tool declaration for CLI prompt injection."""

    name: str
    description: str
    parameters_schema: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ToolStubSpec.name must be a non-empty string")
        if not isinstance(self.description, str):
            raise TypeError("ToolStubSpec.description must be a string")


@dataclass(frozen=True, slots=True)
class ParsedEnvelopeBatch:
    calls: tuple[ToolCall, ...]
    errors: tuple[str, ...]
    rejected_nonce_count: int = 0


@dataclass(frozen=True, slots=True)
class NeutralizedToolOutput:
    content: str
    truncated_bytes: int
    neutralized_tag_count: int


@dataclass(frozen=True, slots=True)
class CliCapabilityReport:
    provider: str
    version_string: str
    tools_supported: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class CliTurnResponse:
    assistant_text: str
    channel: str = "assistant"


@dataclass(frozen=True, slots=True)
class CliBridgeGoalOutcome:
    status: CliBridgeLoopStatus
    final_text: str
    executed_calls: tuple[ToolCall, ...]
    steps_used: int
    repairs_used: int
    error_detail: str = ""


class CliTurnTransport(Protocol):
    def send_turn(self, prompt: str, *, nonce: str) -> CliTurnResponse: ...


def generate_turn_nonce() -> str:
    return secrets.token_hex(_NONCE_BYTES)


def _format_arg_type(key: str, spec: Any, required: set[str]) -> str:
    kind = spec.get("type", "any") if isinstance(spec, Mapping) else "any"
    return f"{key}{'' if key in required else '?'}:{kind}"


def format_minified_tool_stubs(
    tools: Sequence[ToolStubSpec], *, fallback_full_schema: bool = False
) -> str:
    """Serialize tool schemas into compact TypeScript stubs or full JSON fallback."""
    if not tools:
        return "// No tools registered."
    if fallback_full_schema:
        payload = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": dict(t.parameters_schema),
            }
            for t in tools
        ]
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)
    lines: list[str] = []
    for tool in tools:
        props = tool.parameters_schema.get("properties")
        req = set(tool.parameters_schema.get("required") or ())
        sig = (
            ", ".join(_format_arg_type(k, v, req) for k, v in props.items())
            if isinstance(props, Mapping) and props
            else ""
        )
        desc = " ".join(tool.description.split())
        lines.append(f"type {tool.name} = ({sig}) =>Result; // {desc}")
    return "\n".join(lines)


def build_nonce_tool_prompt(
    tools: Sequence[ToolStubSpec], *, nonce: str, fallback_full_schema: bool = False
) -> str:
    """Build the compact per-turn instruction block binding tool calls to ``nonce``."""
    if not nonce or not isinstance(nonce, str):
        raise ValueError("nonce must be a non-empty string")
    stubs = format_minified_tool_stubs(tools, fallback_full_schema=fallback_full_schema)
    return (
        f"Available host tools:\n{stubs}\nTo invoke a tool, emit ONLY:\n"
        f'<sidekick_tool_call nonce="{nonce}">'
        f'{{"action_id":"<tool_name>","params":{{...}},"rationale":"..."}}'
        f"</sidekick_tool_call>"
    )


def neutralize_and_truncate_tool_output(
    raw_output: str, *, max_bytes: int = MAX_TOOL_OUTPUT_BYTES
) -> NeutralizedToolOutput:
    """Neutralize lookalike ``<sidekick_tool_call`` tags and cap at ``max_bytes``."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    neutralized_count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal neutralized_count
        neutralized_count += 1
        return f"[{match.group(1)}sidekick_tool_call_neutralized"

    sanitized = _LOOKALIKE_TAG_RE.sub(_replace, raw_output or "")
    encoded = sanitized.encode("utf-8")
    if len(encoded) <= max_bytes:
        return NeutralizedToolOutput(sanitized, 0, neutralized_count)
    truncated_bytes = len(encoded) - max_bytes
    clipped = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return NeutralizedToolOutput(
        f"{clipped} [truncated {truncated_bytes} bytes]",
        truncated_bytes,
        neutralized_count,
    )


def _validate_envelope_payload(
    payload: Any, allowed_tools: Mapping[str, ToolStubSpec] | None
) -> tuple[ToolCall | None, str | None]:
    if not isinstance(payload, Mapping):
        return None, "tool envelope payload must be a JSON object"
    action_id = payload.get("action_id") or payload.get("name")
    params = (
        payload.get("params") if "params" in payload else payload.get("arguments", {})
    )
    if not isinstance(action_id, str) or not action_id.strip():
        return None, "missing non-empty 'action_id' in tool envelope"
    if not isinstance(params, Mapping):
        return None, f"params for '{action_id}' must be a JSON object"
    if allowed_tools is not None:
        if action_id not in allowed_tools:
            return None, f"unknown tool '{action_id}'"
        req = set(allowed_tools[action_id].parameters_schema.get("required") or ())
        missing = sorted(req - set(params.keys()))
        if missing:
            return (
                None,
                f"tool '{action_id}' missing required params: {', '.join(missing)}",
            )
    return (
        ToolCall(action_id, dict(params), str(payload.get("rationale") or "")),
        None,
    )


def parse_nonce_tool_envelopes(
    text: str,
    *,
    expected_nonce: str,
    channel: str = "assistant",
    allowed_tools: Mapping[str, ToolStubSpec] | None = None,
) -> ParsedEnvelopeBatch:
    """Parse ``<sidekick_tool_call nonce="...">`` strictly from ``assistant``."""
    if not expected_nonce:
        raise ValueError("expected_nonce must be non-empty")
    if channel != "assistant":
        return ParsedEnvelopeBatch(calls=(), errors=(), rejected_nonce_count=0)
    pattern = re.compile(
        r'<sidekick_tool_call\s+nonce="([^"]+)"\s*>(.*?)</sidekick_tool_call>',
        re.DOTALL,
    )
    calls: list[ToolCall] = []
    errors: list[str] = []
    rejected_nonce = 0
    for match in pattern.finditer(text or ""):
        found_nonce, raw_json = match.group(1), match.group(2).strip()
        if not secrets.compare_digest(found_nonce, expected_nonce):
            rejected_nonce += 1
            errors.append("nonce mismatch on <sidekick_tool_call> envelope")
            continue
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON in tool envelope: {exc.msg}")
            continue
        call, err = _validate_envelope_payload(payload, allowed_tools)
        if err:
            errors.append(err)
        elif call is not None:
            calls.append(call)
    return ParsedEnvelopeBatch(tuple(calls), tuple(errors), rejected_nonce)


def probe_cli_tool_capability(
    provider: str, version_output: str
) -> CliCapabilityReport:
    """Verify whether a CLI version string matches a known fixture-tested prefix."""
    norm, clean = provider.strip().lower(), (version_output or "").strip()
    prefixes = _PINNED_CLI_VERSION_PREFIXES.get(norm)
    if not prefixes:
        return CliCapabilityReport(norm, clean, False, f"unknown CLI provider '{norm}'")
    if not clean:
        return CliCapabilityReport(norm, "", False, "empty CLI version string")
    low = clean.lower()
    if any(t.startswith(prefixes) for t in low.split()) or low.startswith(prefixes):
        return CliCapabilityReport(norm, clean, True)
    return CliCapabilityReport(
        norm, clean, False, f"unrecognized {norm} version '{clean}'"
    )


def scrub_cli_environment(env: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of ``env`` stripped of raw API keys, secrets, and vault values."""
    return {
        k: v
        for k, v in env.items()
        if not any(k.upper().endswith(s) for s in _SENSITIVE_ENV_SUFFIXES)
        and not k.upper().startswith("VAULT_SECRET_")
    }


def extract_vault_key_names(text: str) -> tuple[str, ...]:
    """Return deduplicated ``$vault:<KEY>`` key names for audit logs (never values)."""
    return tuple(
        dict.fromkeys(m.group(1) for m in _VAULT_HANDLE_RE.finditer(text or ""))
    )


class FakeCliAdapter:
    """Deterministic fake CLI transport for contract and failure-mode testing."""

    def __init__(
        self,
        scripted_builders: Sequence[Callable[[str, str], CliTurnResponse]],
        *,
        provider: str = "claude",
        version_string: str = "claude-code/1.0.0",
    ) -> None:
        self._builders = list(scripted_builders)
        self.provider = provider
        self.version_string = version_string
        self.prompts_received: list[str] = []

    def send_turn(self, prompt: str, *, nonce: str) -> CliTurnResponse:
        self.prompts_received.append(prompt)
        if not self._builders:
            return CliTurnResponse(assistant_text="Done.", channel="assistant")
        return self._builders.pop(0)(prompt, nonce)


def run_bounded_cli_goal_loop(
    goal_prompt: str,
    *,
    transport: CliTurnTransport,
    tools: Sequence[ToolStubSpec],
    executor: Callable[[ToolCall], str],
    max_steps: int = DEFAULT_MAX_GOAL_STEPS,
    max_repairs: int = 1,
) -> CliBridgeGoalOutcome:
    """Execute a host-orchestrated CLI goal loop with ``max_steps`` & 1 repair turn."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    tool_map = {t.name: t for t in tools}
    executed: list[ToolCall] = []
    repairs_used, rolling = 0, goal_prompt
    for step in range(1, max_steps + 1):
        nonce = generate_turn_nonce()
        hdr = build_nonce_tool_prompt(
            tools, nonce=nonce, fallback_full_schema=repairs_used > 0
        )
        turn = transport.send_turn(f"{hdr}\n\n{rolling}", nonce=nonce)
        batch = parse_nonce_tool_envelopes(
            turn.assistant_text,
            expected_nonce=nonce,
            channel=turn.channel,
            allowed_tools=tool_map,
        )
        if batch.errors:
            if repairs_used < max_repairs:
                repairs_used += 1
                err_str = "; ".join(batch.errors)
                rolling = f"{rolling}\n[Validation error: {err_str}. Repair once.]"
                continue
            return CliBridgeGoalOutcome(
                CliBridgeLoopStatus.REPAIR_EXHAUSTED,
                turn.assistant_text,
                tuple(executed),
                step,
                repairs_used,
                "; ".join(batch.errors),
            )
        if not batch.calls:
            return CliBridgeGoalOutcome(
                CliBridgeLoopStatus.COMPLETED,
                turn.assistant_text,
                tuple(executed),
                step,
                repairs_used,
            )
        obs = [
            f"Tool {c.action_id} result: "
            f"{neutralize_and_truncate_tool_output(executor(c)).content}"
            for c in batch.calls
        ]
        executed.extend(batch.calls)
        rolling = f"{rolling}\n" + "\n".join(obs)
    return CliBridgeGoalOutcome(
        CliBridgeLoopStatus.BUDGET_EXHAUSTED,
        "Goal step budget exhausted.",
        tuple(executed),
        max_steps,
        repairs_used,
        f"Exceeded max_steps={max_steps}",
    )
