"""Google Gemini CLI adapter.

Wraps the ``gemini`` CLI binary (Google's ``@google/gemini-cli`` npm package)
as a chat provider. The CLI handles its own authentication; this adapter
only spawns it and parses output.

Distinct from the ``GeminiAdapter`` which talks directly to Google's
Generative Language REST API and requires a ``GEMINI_API_KEY``. Use this
adapter when the user has ``gemini`` installed and logged in but does not
want to expose an API key to the application.

Requirements:
    - ``gemini`` CLI on ``PATH`` (or a known install location).
    - User logged in (``gemini`` prompts on first run, or set
      ``GEMINI_API_KEY``).

Invocation pattern::

    gemini -p "<prompt>" [--model MODEL] [--skip-trust]

Prompt is the argument to ``-p``; response comes on stdout.

Example::

    >>> from shared.python.ai.adapters.gemini_cli_adapter import GeminiCliAdapter
    >>> adapter = GeminiCliAdapter()
    >>> ok, msg = adapter.validate_connection()
    >>> if ok:
    ...     response = adapter.send_message("hello", ctx, tools=[])
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from shared.python.ai.adapters.claude_code_adapter import (
    build_cli_tool_prompt_section,
    parse_cli_tool_envelopes,
    scrub_cli_subprocess_env,
    terminate_cli_process_group,
)
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

# Default invocation timeout. Gemini's CLI is reasonably fast (<10s typical)
# but the first call after install can be slow because the runtime warms up.
DEFAULT_GEMINI_CLI_TIMEOUT = 120.0  # [s]

# Known install locations probed when ``gemini`` is not on PATH.
_FALLBACK_PATHS = (
    r"%APPDATA%\npm\gemini.cmd",
    r"%APPDATA%\npm\gemini",
    "~/.npm-global/bin/gemini",
    "/usr/local/bin/gemini",
)

# Static model catalogue. The CLI accepts these via --model; update as
# Google rolls out new versions.
_STATIC_MODELS: tuple[str, ...] = (
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "default",
)


def _resolve_binary(explicit: str | None = None) -> str | None:
    """Locate the ``gemini`` binary.

    Returns the resolved absolute path, or ``None`` when the binary cannot be
    found. Callers handle the ``None`` case via :meth:`validate_connection`.
    """
    if explicit:
        if Path(explicit).exists():
            return explicit
        logger.warning(
            "GeminiCliAdapter: explicit binary %s does not exist; "
            "falling back to PATH search",
            explicit,
        )
    found = shutil.which("gemini")
    if found:
        return found
    for candidate in _FALLBACK_PATHS:
        expanded = os.path.expanduser(os.path.expandvars(candidate))
        if Path(expanded).exists():
            return expanded
    return None


class GeminiCliAdapter(BaseAgentAdapter):
    """Adapter that delegates chat to Google's Gemini CLI.

    Attributes:
        binary: Absolute path to the resolved ``gemini`` binary (or ``None``).
        model: Optional model id passed via ``--model``.
        timeout: Per-invocation timeout in seconds.
    """

    PROVIDER_NAME = "gemini_cli"

    def __init__(
        self,
        binary: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            binary: Optional explicit path to the ``gemini`` binary. Resolved
                lazily so construction never fails when the CLI is missing.
            model: Optional model identifier (e.g. ``"gemini-2.5-pro"``).
            timeout: Subprocess timeout. Defaults to
                ``DEFAULT_GEMINI_CLI_TIMEOUT``.
        """
        self.binary = _resolve_binary(binary)
        self.model = model
        self.timeout = (
            float(timeout) if timeout is not None else DEFAULT_GEMINI_CLI_TIMEOUT
        )
        self._capabilities = ProviderCapabilities(
            supported=frozenset(
                {
                    ProviderCapability.FUNCTION_CALLING,
                    ProviderCapability.SYSTEM_MESSAGE,
                    ProviderCapability.STREAMING,
                }
            ),
            max_tokens=1_000_000,
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
        """Gemini CLI does not surface reasoning budgets to the host."""
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
        Gemini CLI manages its own tool inventory. For host-tool integration
        use :class:`GeminiAdapter` (REST API) instead.

        Raises:
            AIConnectionError: When the binary cannot be located.
            AITimeoutError: When the subprocess exceeds ``self.timeout``.
            AIProviderError: When the CLI returns a non-zero exit code.
        """
        if message is None or not message.strip():
            raise ValueError("message must be a non-empty string")
        if self.binary is None:
            raise AIConnectionError(
                "Gemini CLI not found on PATH. Install with "
                "`npm install -g @google/gemini-cli` and run `gemini` once "
                "to authenticate.",
                provider=self.PROVIDER_NAME,
            )

        nonce = secrets.token_hex(8) if tools else None
        prompt = self._build_prompt(message, context, tools=tools, nonce=nonce)
        # `--skip-trust` because the chat UI is not necessarily launched from
        # a workspace the user has explicitly trusted; the CLI otherwise
        # refuses to run in non-interactive mode against an untrusted dir.
        args = [self.binary, "--skip-trust", "-p", prompt]
        if self.model:
            args[1:1] = ["--model", self.model]

        try:
            result = subprocess.run(
                args,
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
                f"Gemini CLI timed out after {self.timeout}s",
                provider=self.PROVIDER_NAME,
            ) from exc
        except FileNotFoundError as exc:
            raise AIConnectionError(
                f"Gemini binary at {self.binary} disappeared mid-invocation",
                provider=self.PROVIDER_NAME,
            ) from exc

        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-500:]
            raise AIProviderError(
                f"Gemini CLI exited {result.returncode}: {stderr_tail}",
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
        """Yield tool-call deltas (if any) followed by the final text chunk."""
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

        Runs ``gemini --version`` (returns in <1s). A non-zero exit or
        missing binary is reported with a user-friendly hint.
        """
        if self.binary is None:
            return False, (
                "Gemini CLI not found. Install: npm install -g @google/gemini-cli"
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
            return False, "Gemini CLI did not respond to --version within 10s"
        except OSError as exc:
            return False, f"Could not execute {self.binary}: {exc}"

        if result.returncode != 0:
            return False, (
                f"Gemini --version exited {result.returncode}: "
                f"{(result.stderr or '').strip()[:200]}"
            )
        stdout_lines = (result.stdout or "").strip().splitlines()
        version = stdout_lines[0] if stdout_lines else "?"
        return True, f"Gemini CLI available: {version}"

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
        """Concatenate conversation history into a single CLI prompt."""
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
