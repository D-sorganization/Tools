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

import os
import secrets
import shutil
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from shared.python.ai.adapters.cli_subprocess_runner import (  # noqa: F401
    PromptDelivery,
    invoke_cli_subprocess_for_adapter,
    scrub_cli_subprocess_env,
    terminate_cli_process_group,
)
from shared.python.ai.adapters.cli_tool_bridge import (  # noqa: F401
    build_cli_tool_prompt_section,
    parse_cli_tool_envelopes,
    parse_cli_tool_envelopes_with_diagnostics,
    run_adapter_cli_goal_loop,
)
from shared.python.ai.adapters.cli_turn_lifecycle import (
    CliCancelToken,
    run_adapter_cli_turn,
)
from shared.python.ai.exceptions import (
    AIConnectionError,
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
from shared.python.lifecycle import LifecycleSnapshot
from shared.python.logging_pkg.logging_config import get_logger
from shared.python.sidekick.agent.cli_bridge import (
    DEFAULT_MAX_GOAL_STEPS,
    CliBridgeGoalOutcome,
    ToolStubSpec,
)

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
        idle_timeout: float | None = None,
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
            idle_timeout: Optional silence timeout between stdout/stderr reads.
        """
        self.binary = _resolve_binary(binary)
        self.model = model
        self.timeout = (
            float(timeout) if timeout is not None else DEFAULT_CLAUDE_CODE_TIMEOUT
        )
        self.idle_timeout = float(idle_timeout) if idle_timeout is not None else None
        self._active_cancel_token: CliCancelToken | None = None
        self.last_turn_lifecycle: LifecycleSnapshot | None = None
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

    def cancel(self) -> bool:
        """Cancel the currently active CLI turn, if one is running."""
        token = self._active_cancel_token
        return token.cancel() if token is not None else False

    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],  # noqa: ARG002 (CLI is its own tool host)
        *,
        cancel_token: CliCancelToken | None = None,
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

        nonce = secrets.token_hex(8) if tools else None
        prompt = self._build_prompt(message, context, tools=tools, nonce=nonce)
        args = [self.binary or "claude", "-p"]
        if self.model:
            args += ["--model", self.model]

        result = run_adapter_cli_turn(
            self,
            args,
            runner_fn=invoke_cli_subprocess_for_adapter,
            prompt=prompt,
            prompt_delivery=PromptDelivery.STDIN,
            cancel_token=cancel_token,
            missing_binary_message=(
                "Claude Code CLI not found on PATH. Install from "
                "https://claude.com/claude-code or set the binary path explicitly."
            ),
        )

        raw_stdout = (result.stdout or "").strip()
        tool_calls: list[ToolCall] = []
        meta: dict[str, Any] = {
            "provider": self.PROVIDER_NAME,
            "model": self.model or "default",
            "stderr": (result.stderr or "")[-500:],
        }
        if tools and nonce is not None:
            raw_stdout, tool_calls, rejected, envelope_errors = (
                parse_cli_tool_envelopes_with_diagnostics(
                    raw_stdout, expected_nonce=nonce, tools=tools
                )
            )
            meta["turn_nonce"] = nonce
            meta["rejected_tool_nonces"] = rejected
            meta["tool_envelope_errors"] = list(envelope_errors)
            meta["tool_envelope_error_count"] = len(envelope_errors)
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
            result = invoke_cli_subprocess_for_adapter(
                self.binary,
                [self.binary, "--version"],
                provider=self.PROVIDER_NAME,
                timeout=10.0,
            )
        except AITimeoutError:
            return False, "Claude Code CLI did not respond to --version within 10s"
        except AIConnectionError as exc:
            return False, str(exc)

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

    def run_goal_loop(
        self,
        goal_prompt: str,
        *,
        tools: Sequence[ToolDeclaration | ToolStubSpec],
        executor: Callable[[Any], str],
        max_steps: int = DEFAULT_MAX_GOAL_STEPS,
        max_repairs: int = 1,
        probe_version: bool = False,
    ) -> CliBridgeGoalOutcome:
        """Execute a bounded host-orchestrated tool loop via ``cli_bridge``."""
        return run_adapter_cli_goal_loop(
            self,
            goal_prompt,
            tools=tools,
            executor=executor,
            max_steps=max_steps,
            max_repairs=max_repairs,
            probe_version=probe_version,
        )
