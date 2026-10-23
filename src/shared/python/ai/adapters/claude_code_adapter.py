"""Claude Code CLI adapter.

Wraps the ``claude`` CLI binary (Claude Code agent) as a chat provider. The
CLI handles its own authentication via OAuth keychain or ``ANTHROPIC_API_KEY``;
this adapter only spawns it and parses output.

Distinct from the ``AnthropicAdapter`` which talks directly to the Anthropic
REST API and requires an API key in the process environment. The CLI adapter
is the right choice when the user has ``claude`` installed and logged in but
does not want to expose an API key to the application.

Requirements:
    - ``claude`` CLI on ``PATH`` (or a known install location).
    - User logged in (``claude login`` or ``ANTHROPIC_API_KEY`` set).

Invocation pattern::

    claude -p [--model MODEL] < prompt.txt

Prompt is fed via stdin; response comes on stdout. Exit code 0 on success.

Example::

    >>> from shared.python.ai.adapters.claude_code_adapter import ClaudeCodeAdapter
    >>> adapter = ClaudeCodeAdapter()
    >>> ok, msg = adapter.validate_connection()
    >>> if ok:
    ...     response = adapter.send_message("hello", ctx, tools=[])
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import signal
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from shared.python.ai.exceptions import (
    AIConnectionError,
    AIProviderError,
    AITimeoutError,
)
from shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
    ProviderCapability,
    ToolCall,
)
from shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Default invocation timeout. Claude Code's first response on a cold prompt
# typically returns in 2-5s; subsequent calls are faster. Keep generous so a
# slow first call doesn't surface as a misleading "CLI broken" error.
DEFAULT_CLAUDE_CODE_TIMEOUT = 120.0  # [s]

# Known install locations probed when ``claude`` is not on PATH. Order matters:
# Windows install location first because the launcher runs on Windows.
_FALLBACK_PATHS = (
    r"%LOCALAPPDATA%\Programs\Claude Code\claude.exe",
    r"~\.local\bin\claude.exe",
    "~/.local/bin/claude",
    "/usr/local/bin/claude",
)

# Static model catalogue. Claude Code does not expose a `list models` command,
# so this is the documented model dropdown — kept in sync with anthropic_adapter.
_STATIC_MODELS: tuple[str, ...] = (
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "sonnet",
    "opus",
    "haiku",
)


_SENSITIVE_ENV_SUFFIXES = ("_SECRET", "_PASSWORD", "_PRIVATE_KEY", "_TOKEN")
_ENVELOPE_RE = re.compile(
    r'<sidekick_tool_call\s+nonce="([^"]+)"\s*>(.*?)</sidekick_tool_call>',
    re.DOTALL,
)


def scrub_cli_subprocess_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a copy of ``env`` stripped of vault secrets and raw credential tokens."""
    source = os.environ if env is None else env
    return {
        k: v
        for k, v in source.items()
        if not k.upper().startswith("VAULT_SECRET_")
        and not any(k.upper().endswith(s) for s in _SENSITIVE_ENV_SUFFIXES)
    }


def terminate_cli_process_group(pid: int | None) -> bool:
    """Best-effort SIGKILL of the CLI process group on timeout or cancellation."""
    if pid is None or pid <= 0 or not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
        return True
    except OSError:
        return False


def build_cli_tool_prompt_section(tools: Sequence[ToolDeclaration], nonce: str) -> str:
    """Format compact TypeScript tool stubs bound to per-turn CSPRNG ``nonce``."""
    if not nonce or not isinstance(nonce, str):
        raise ValueError("nonce must be a non-empty string")
    lines: list[str] = []
    for tool in tools:
        req = set(tool.required or ())
        sig = ", ".join(
            f"{k}{'' if k in req else '?'}:{v.get('type', 'any') if isinstance(v, Mapping) else 'any'}"  # noqa: E501
            for k, v in (tool.parameters or {}).items()
        )
        desc = " ".join((tool.description or "").split())
        lines.append(f"type {tool.name} = ({sig}) => Result; // {desc}")
    stubs = "\n".join(lines) if lines else "// No tools registered."
    return (
        f"Available host tools:\n{stubs}\nTo invoke a tool, emit ONLY:\n"
        f'<sidekick_tool_call nonce="{nonce}">'
        f'{{"action_id":"<tool_name>","params":{{...}},"rationale":"..."}}'
        f"</sidekick_tool_call>"
    )


def parse_cli_tool_envelopes(
    raw_text: str, *, expected_nonce: str, tools: Sequence[ToolDeclaration]
) -> tuple[str, list[ToolCall], int]:
    """Extract nonce-verified ``ToolCall`` requests from CLI assistant output."""
    if not expected_nonce:
        raise ValueError("expected_nonce must be non-empty")
    allowed = {t.name: t for t in tools}
    calls: list[ToolCall] = []
    rejected = 0
    spans: list[tuple[int, int]] = []
    for match in _ENVELOPE_RE.finditer(raw_text or ""):
        found_nonce, raw_json = match.group(1), match.group(2).strip()
        if not secrets.compare_digest(found_nonce, expected_nonce):
            rejected += 1
            continue
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        action_id = payload.get("action_id") or payload.get("name")
        params = (
            payload.get("params")
            if "params" in payload
            else payload.get("arguments", {})
        )
        if (
            not isinstance(action_id, str)
            or action_id not in allowed
            or not isinstance(params, Mapping)
        ):
            continue
        if set(allowed[action_id].required or ()) - set(params.keys()):
            continue
        calls.append(ToolCall.create(name=action_id, arguments=dict(params)))
        spans.append(match.span())
    cleaned = raw_text or ""
    for s, e in reversed(spans):
        cleaned = (cleaned[:s] + cleaned[e:]).strip()
    return cleaned.strip(), calls, rejected


def _resolve_binary(explicit: str | None = None) -> str | None:
    """Locate the ``claude`` binary.

    Returns the resolved absolute path, or ``None`` when the binary cannot be
    found. Callers must handle the ``None`` case gracefully — the chat UI
    should surface "Claude Code CLI not installed" rather than crashing.
    """
    if explicit:
        if Path(explicit).exists():
            return explicit
        # Caller passed something invalid; fall through to PATH lookup.
        logger.warning(
            "ClaudeCodeAdapter: explicit binary %s does not exist; "
            "falling back to PATH search",
            explicit,
        )
    found = shutil.which("claude")
    if found:
        return found
    import os

    for candidate in _FALLBACK_PATHS:
        expanded = os.path.expanduser(os.path.expandvars(candidate))
        if Path(expanded).exists():
            return expanded
    return None


class ClaudeCodeAdapter(BaseAgentAdapter):
    """Adapter that delegates chat to the Claude Code CLI.

    The CLI handles auth, retries, and model routing internally. This adapter
    is a thin subprocess wrapper that translates :class:`ConversationContext`
    into a single prompt string and parses the CLI's stdout back into an
    :class:`AgentResponse`.

    Attributes:
        binary: Absolute path to the resolved ``claude`` binary (or ``None``).
        model: Model identifier passed via ``--model``. ``None`` lets the CLI
            choose its own default.
        timeout: Per-invocation timeout in seconds.
    """

    PROVIDER_NAME = "claude_code"

    def __init__(
        self,
        binary: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            binary: Optional explicit path to the ``claude`` binary. Resolved
                lazily so construction never fails just because the CLI is
                missing — callers test availability via
                :meth:`validate_connection`.
            model: Model id (e.g. ``"sonnet"``, ``"claude-opus-4-5"``).
                ``None`` uses the CLI default.
            timeout: Subprocess timeout per call. Defaults to
                ``DEFAULT_CLAUDE_CODE_TIMEOUT``.
        """
        self.binary = _resolve_binary(binary)
        self.model = model
        self.timeout = (
            float(timeout) if timeout is not None else DEFAULT_CLAUDE_CODE_TIMEOUT
        )
        self._capabilities = ProviderCapabilities(
            supported=frozenset(
                {
                    ProviderCapability.FUNCTION_CALLING,
                    ProviderCapability.SYSTEM_MESSAGE,
                    ProviderCapability.STREAMING,
                }
            ),
            max_tokens=200_000,
            model_name=model or "default",
            provider_name=self.PROVIDER_NAME,
        )

    # ------------------------------------------------------------------ #
    # BaseAgentAdapter surface
    # ------------------------------------------------------------------ #

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def list_models(self) -> list[str]:
        """Return the documented model catalogue.

        The Claude Code CLI does not expose a model listing endpoint, so this
        returns the static, hand-maintained list. Always non-empty.
        """
        return list(_STATIC_MODELS)

    def thinking_capabilities(self) -> Any:
        """Claude Code CLI does not surface reasoning budgets to the host."""
        from shared.python.chat_contracts.models import make_none_only_capabilities

        return make_none_only_capabilities(provider=self.PROVIDER_NAME)

    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],  # noqa: ARG002 (CLI is its own tool host)
    ) -> AgentResponse:
        """Send a message and return the CLI's response.

        Tools declared by the host application are intentionally ignored: the
        Claude Code CLI manages its own tool inventory (file edits, bash, web
        fetch, etc.). Surfacing host tools here would duplicate or conflict
        with the CLI's built-ins. If you need host-tool integration, use
        :class:`AnthropicAdapter` instead.

        Raises:
            AIConnectionError: When the binary cannot be located.
            AITimeoutError: When the subprocess exceeds ``self.timeout``.
            AIProviderError: When the CLI returns a non-zero exit code.
        """
        if message is None or not message.strip():
            raise ValueError("message must be a non-empty string")
        if self.binary is None:
            raise AIConnectionError(
                "Claude Code CLI not found on PATH. Install from "
                "https://claude.com/claude-code or set the binary path explicitly.",
                provider=self.PROVIDER_NAME,
            )

        nonce = secrets.token_hex(8) if tools else None
        prompt = self._build_prompt(message, context, tools=tools, nonce=nonce)
        args = [self.binary, "-p"]
        if self.model:
            args += ["--model", self.model]

        try:
            result = subprocess.run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
                encoding="utf-8",
                env=scrub_cli_subprocess_env(),
                start_new_session=(os.name != "nt"),
            )
        except subprocess.TimeoutExpired as exc:
            terminate_cli_process_group(getattr(exc, "pid", None))
            raise AITimeoutError(
                f"Claude Code CLI timed out after {self.timeout}s",
                provider=self.PROVIDER_NAME,
            ) from exc
        except FileNotFoundError as exc:
            raise AIConnectionError(
                f"Claude Code binary at {self.binary} disappeared mid-invocation",
                provider=self.PROVIDER_NAME,
            ) from exc

        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-500:]
            raise AIProviderError(
                f"Claude Code CLI exited {result.returncode}: {stderr_tail}",
                provider=self.PROVIDER_NAME,
            )

        raw_stdout = (result.stdout or "").strip()
        tool_calls: list[ToolCall] = []
        meta: dict[str, Any] = {
            "provider": self.PROVIDER_NAME,
            "model": self.model or "default",
            "stderr": (result.stderr or "")[-500:],
        }
        if tools and nonce is not None:
            raw_stdout, tool_calls, rejected = parse_cli_tool_envelopes(
                raw_stdout, expected_nonce=nonce, tools=tools
            )
            meta["turn_nonce"] = nonce
            meta["rejected_tool_nonces"] = rejected
        return AgentResponse(
            content=raw_stdout,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage={},
            metadata=meta,
        )

    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        """Yield the response as a single chunk.

        True streaming would require the CLI's ``--output-format=stream-json``
        mode, which is more complex to parse and is left as a follow-up
        (Tools issue tracker). For now the chat dock will simply render the
        completed response as one bubble.
        """
        response = self.send_message(message, context, tools)
        for idx, tc in enumerate(response.tool_calls):
            yield AgentChunk(
                content="", tool_call_delta=tc.to_dict(), is_final=False, index=idx
            )
        yield AgentChunk(
            content=response.content,
            is_final=True,
            index=len(response.tool_calls),
        )

    def validate_connection(self) -> tuple[bool, str]:
        """Probe the CLI is installed and authenticated.

        Runs ``claude --version`` (cheap, <1s when warm). A non-zero exit or
        missing binary is reported with a user-friendly hint.
        """
        if self.binary is None:
            return False, (
                "Claude Code CLI not found. Install: https://claude.com/claude-code"
            )
        try:
            result = subprocess.run(
                [self.binary, "--version"],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
                encoding="utf-8",
            )
        except subprocess.TimeoutExpired:
            return False, "Claude Code CLI did not respond to --version within 10s"
        except OSError as exc:
            return False, f"Could not execute {self.binary}: {exc}"

        if result.returncode != 0:
            return False, (
                f"Claude Code --version exited {result.returncode}: "
                f"{(result.stderr or '').strip()[:200]}"
            )
        version = (
            (result.stdout or "").strip().splitlines()[0] if result.stdout else "?"
        )
        return True, f"Claude Code CLI available: {version}"

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_prompt(
        message: str,
        context: ConversationContext,
        *,
        tools: Sequence[ToolDeclaration] | None = None,
        nonce: str | None = None,
    ) -> str:
        """Concatenate conversation history into a single CLI prompt.

        The Claude Code CLI's non-interactive ``-p`` mode takes a single
        prompt; there is no native conversation-history flag. We render the
        history as alternating ``User:`` / ``Assistant:`` blocks followed by
        the current message — the same pattern AnthropicAdapter and friends
        use when collapsing context.
        """
        parts: list[str] = (
            [build_cli_tool_prompt_section(tools, nonce)] if (tools and nonce) else []
        )
        for msg in getattr(context, "messages", []) or []:
            role = getattr(msg, "role", "user").capitalize()
            content = getattr(msg, "content", "") or ""
            if content:
                parts.append(f"{role}: {content}")
        parts.append(f"User: {message}")
        return "\n\n".join(parts)
