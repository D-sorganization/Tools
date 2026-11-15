"""OpenAI Codex CLI adapter.

Wraps the ``codex`` CLI binary (the ``@openai/codex`` npm package) as a chat
provider. The CLI handles its own authentication and provider routing; this
adapter only spawns it and parses output.

Distinct from :class:`~shared.python.ai.adapters.openai_adapter.OpenAIAdapter`,
which talks directly to the OpenAI REST API and requires an API key in the
process environment. Use the CLI adapter when the user has ``codex`` installed
and logged in but does not want to expose an API key to the application.

Requirements:
    - ``codex`` CLI on ``PATH`` (or a known install location).
    - User logged in (``codex login`` or ``OPENAI_API_KEY`` set).

Invocation pattern::

    codex exec --skip-git-repo-check "prompt"

The ``--skip-git-repo-check`` flag is required because Codex's safety check
refuses to run outside a "trusted directory". The chat UI is not necessarily
launched from a git repo, so we always pass the flag.

Example::

    >>> from shared.python.ai.adapters.codex_cli_adapter import CodexCliAdapter
    >>> adapter = CodexCliAdapter()
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
from shared.python.ai.adapters.cli_subprocess_runner import (
    PromptDelivery,
    invoke_cli_subprocess_for_adapter,
)
from shared.python.ai.adapters.cli_tool_bridge import (
    build_cli_tool_prompt_section,
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

# Default invocation timeout. Codex's first response on a cold prompt can take
# 30-60s due to skill catalogue loading + model warm-up. Keep generous so the
# chat doesn't report a misleading timeout on the very first message.
DEFAULT_CODEX_CLI_TIMEOUT = 180.0  # [s]

# Known install locations probed when ``codex`` is not on PATH. Windows-npm
# global-bin first because that's the npm install target on the host that
# typically runs the launcher.
_FALLBACK_PATHS = (
    r"%APPDATA%\npm\codex.cmd",
    r"%APPDATA%\npm\codex",
    "~/.npm-global/bin/codex",
    "/usr/local/bin/codex",
)

# Static model catalogue. The Codex CLI does not expose a model listing
# command; this is the documented set of models that the CLI accepts via
# ``--model``. Update when OpenAI adds new entries.
_STATIC_MODELS: tuple[str, ...] = (
    "gpt-5.5",
    "gpt-5",
    "gpt-5-mini",
    "o4",
    "o4-mini",
    "default",
)


def _resolve_binary(explicit: str | None = None) -> str | None:
    """Locate the ``codex`` binary.

    Returns the resolved absolute path, or ``None`` when the binary cannot be
    found. Callers handle the ``None`` case via :meth:`validate_connection`.
    """
    if explicit:
        if Path(explicit).exists():
            return explicit
        logger.warning(
            "CodexCliAdapter: explicit binary %s does not exist; "
            "falling back to PATH search",
            explicit,
        )
    found = shutil.which("codex")
    if found:
        return found
    for candidate in _FALLBACK_PATHS:
        expanded = os.path.expanduser(os.path.expandvars(candidate))
        if Path(expanded).exists():
            return expanded
    return None


class CodexCliAdapter(BaseAgentAdapter):
    """Adapter that delegates chat to the OpenAI Codex CLI.

    The CLI handles auth, model selection, and tool inventory internally.
    This adapter is a thin subprocess wrapper that translates
    :class:`ConversationContext` into a single prompt and parses the CLI's
    stdout back into an :class:`AgentResponse`.

    Attributes:
        binary: Absolute path to the resolved ``codex`` binary (or ``None``).
        model: Optional model id passed via ``--model``. ``None`` lets the
            CLI choose its own default.
        timeout: Per-invocation timeout in seconds.
    """

    PROVIDER_NAME = "codex_cli"

    def __init__(
        self,
        binary: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        idle_timeout: float | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            binary: Optional explicit path to the ``codex`` binary. Resolved
                lazily so construction never fails when the CLI is missing.
            model: Optional model identifier.
            timeout: Subprocess timeout. Defaults to
                ``DEFAULT_CODEX_CLI_TIMEOUT``.
            idle_timeout: Optional silence timeout between stdout/stderr reads.
        """
        self.binary = _resolve_binary(binary)
        self.model = model
        self.timeout = (
            float(timeout) if timeout is not None else DEFAULT_CODEX_CLI_TIMEOUT
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
            max_tokens=128_000,
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
        """Return the documented model catalogue. Always non-empty."""
        return list(_STATIC_MODELS)

    def thinking_capabilities(self) -> Any:
        """Codex CLI does not surface reasoning budgets to the host."""
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
        Codex CLI manages its own tool inventory (file edits, shell, web
        browse). Surfacing host tools would duplicate or conflict with the
        CLI's built-ins. For host-tool integration use
        :class:`OpenAIAdapter` instead.

        Raises:
            AIConnectionError: When the binary cannot be located.
            AITimeoutError: When the subprocess exceeds ``self.timeout``.
            AIProviderError: When the CLI returns a non-zero exit code.
        """
        if message is None or not message.strip():
            raise ValueError("message must be a non-empty string")

        nonce = secrets.token_hex(8) if tools else None
        prompt = self._build_prompt(message, context, tools=tools, nonce=nonce)
        args = [self.binary or "codex", "exec", "--skip-git-repo-check"]
        if self.model:
            args += ["--model", self.model]
        result = run_adapter_cli_turn(
            self,
            args,
            runner_fn=invoke_cli_subprocess_for_adapter,
            prompt=prompt,
            prompt_delivery=PromptDelivery.AUTO,
            cancel_token=cancel_token,
            missing_binary_message=(
                "Codex CLI not found on PATH. Install with "
                "`npm install -g @openai/codex` and run `codex login`."
            ),
        )

        cleaned_stdout = self._strip_telemetry(result.stdout or "")
        tool_calls: list[ToolCall] = []
        meta: dict[str, Any] = {
            "provider": self.PROVIDER_NAME,
            "model": self.model or "default",
            "stderr": (result.stderr or "")[-500:],
        }
        if tools and nonce is not None:
            cleaned_stdout, tool_calls, rejected, envelope_errors = (
                parse_cli_tool_envelopes_with_diagnostics(
                    cleaned_stdout, expected_nonce=nonce, tools=tools
                )
            )
            meta["turn_nonce"] = nonce
            meta["rejected_tool_nonces"] = rejected
            meta["tool_envelope_errors"] = list(envelope_errors)
            meta["tool_envelope_error_count"] = len(envelope_errors)
        return AgentResponse(
            content=cleaned_stdout,
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

        Codex CLI does not yet have a stable streaming output format suitable
        for incremental rendering. We surface the complete response as one
        chunk; the chat dock handles this gracefully.
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

        Runs ``codex --version`` (returns in <2s). A non-zero exit or missing
        binary is reported with a user-friendly hint.
        """
        if self.binary is None:
            return False, ("Codex CLI not found. Install: npm install -g @openai/codex")
        try:
            result = invoke_cli_subprocess_for_adapter(
                self.binary,
                [self.binary, "--version"],
                provider=self.PROVIDER_NAME,
                timeout=10.0,
            )
        except AITimeoutError:
            return False, "Codex CLI did not respond to --version within 10s"
        except AIConnectionError as exc:
            return False, str(exc)

        if result.returncode != 0:
            return False, (
                f"Codex --version exited {result.returncode}: "
                f"{(result.stderr or '').strip()[:200]}"
            )
        stdout_lines = (result.stdout or "").strip().splitlines()
        version = stdout_lines[0] if stdout_lines else "?"
        return True, f"Codex CLI available: {version}"

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

        Codex's ``exec`` mode takes a single positional prompt with no native
        conversation-history flag. We render the history as alternating
        ``User:`` / ``Assistant:`` blocks followed by the current message —
        matching the convention used by the other CLI-shaped adapters.
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

    @staticmethod
    def _strip_telemetry(stdout: str) -> str:
        """Drop the noisy ``[year-month-day…]`` telemetry preamble.

        Codex prefixes responses with ANSI-coloured timing/runtime lines that
        look like ``[2026-05-17T23:13:03] thinking…``. Drop those leading
        lines so the chat bubble only shows the actual response text.
        """
        lines = stdout.splitlines()
        # Skip leading lines that look like timestamps in square brackets.
        first_real = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("[20"):
                first_real = i + 1
                continue
            break
        return "\n".join(lines[first_real:]).strip()

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
