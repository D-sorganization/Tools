"""Google Antigravity CLI (``agy``) adapter.

Wraps Google's ``agy`` CLI binary (Antigravity agent runtime) as a chat
provider. The CLI handles its own authentication; this adapter only spawns
it and parses output.

Distinct from :class:`GeminiAdapter` (REST API) and
:class:`GeminiCliAdapter` (the older ``gemini`` CLI). Antigravity is
Google's successor agentic CLI; this adapter is the migration path as the
standalone ``gemini`` CLI is phased out.

Requirements:
    - ``agy`` CLI on ``PATH`` (or a known install location).
    - User logged in (``agy`` prompts on first run, or set
      ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``).

Invocation pattern::

    agy -p "<prompt>" [--model MODEL]

Prompt is the argument to ``-p``; response comes on stdout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from src.shared.python.ai.exceptions import (
    AIConnectionError,
    AIProviderError,
    AITimeoutError,
)
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
    ProviderCapability,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Antigravity's first-call warmup is comparable to gemini-cli; reuse the
# same generous default so users don't hit spurious timeouts.
DEFAULT_AGY_CLI_TIMEOUT = 120.0  # [s]

# Known install locations probed when ``agy`` is not on PATH.
_FALLBACK_PATHS = (
    r"%APPDATA%\npm\agy.cmd",
    r"%APPDATA%\npm\agy",
    r"%LOCALAPPDATA%\Programs\Antigravity\agy.exe",
    "/home/dieterolson/.npm-global/bin/agy",
    "/usr/local/bin/agy",
    "/opt/homebrew/bin/agy",
)

# Static model catalogue. Antigravity routes to the underlying Gemini
# model family; keep this list in sync with what ``agy --model`` accepts.
_STATIC_MODELS: tuple[str, ...] = (
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "default",
)


def _resolve_binary(explicit: str | None = None) -> str | None:
    """Locate the ``agy`` binary.

    Returns the resolved absolute path, or ``None`` when the binary cannot
    be found. Callers handle the ``None`` case via
    :meth:`validate_connection`.
    """
    if explicit:
        if Path(explicit).exists():
            return explicit
        logger.warning(
            "AgyCliAdapter: explicit binary %s does not exist; "
            "falling back to PATH search",
            explicit,
        )
    found = shutil.which("agy")
    if found:
        return found
    for candidate in _FALLBACK_PATHS:
        expanded = os.path.expandvars(candidate)
        if Path(expanded).exists():
            return expanded
    return None


class AgyCliAdapter(BaseAgentAdapter):
    """Adapter that delegates chat to Google's Antigravity (``agy``) CLI.

    Attributes:
        binary: Absolute path to the resolved ``agy`` binary (or ``None``).
        model: Optional model id passed via ``--model``.
        timeout: Per-invocation timeout in seconds.
    """

    PROVIDER_NAME = "agy_cli"

    def __init__(
        self,
        binary: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.binary = _resolve_binary(binary)
        self.model = model
        self.timeout = (
            float(timeout) if timeout is not None else DEFAULT_AGY_CLI_TIMEOUT
        )
        self._capabilities = ProviderCapabilities(
            supported=frozenset(
                {
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
        return list(_STATIC_MODELS)

    def thinking_capabilities(self) -> Any:
        from src.shared.python.chat.models import make_none_only_capabilities

        return make_none_only_capabilities(provider=self.PROVIDER_NAME)

    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],  # noqa: ARG002 (CLI is its own tool host)
    ) -> AgentResponse:
        """Send a message and return the CLI's response.

        Tools declared by the host application are intentionally ignored:
        the Antigravity CLI manages its own tool inventory. For host-tool
        integration use :class:`GeminiAdapter` (REST API) instead.

        Raises:
            AIConnectionError: When the binary cannot be located.
            AITimeoutError: When the subprocess exceeds ``self.timeout``.
            AIProviderError: When the CLI returns a non-zero exit code.
        """
        if message is None or not message.strip():
            raise ValueError("message must be a non-empty string")
        if self.binary is None:
            raise AIConnectionError(
                "Antigravity CLI (agy) not found on PATH. Install from "
                "https://antigravity.google.com/ and run `agy` once to "
                "authenticate.",
                provider=self.PROVIDER_NAME,
            )

        prompt = self._build_prompt(message, context)
        args = [self.binary, "-p", prompt]
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
            )
        except subprocess.TimeoutExpired as exc:
            raise AITimeoutError(
                f"Antigravity CLI timed out after {self.timeout}s",
                provider=self.PROVIDER_NAME,
            ) from exc
        except FileNotFoundError as exc:
            raise AIConnectionError(
                f"Antigravity binary at {self.binary} disappeared mid-invocation",
                provider=self.PROVIDER_NAME,
            ) from exc

        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-500:]
            raise AIProviderError(
                f"Antigravity CLI exited {result.returncode}: {stderr_tail}",
                provider=self.PROVIDER_NAME,
            )

        return AgentResponse(
            content=(result.stdout or "").strip(),
            usage={},
            metadata={
                "provider": self.PROVIDER_NAME,
                "model": self.model or "default",
                "stderr": (result.stderr or "")[-500:],
            },
        )

    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        """Yield the response as a single chunk.

        Mirrors :class:`GeminiCliAdapter`: real streaming is possible via
        the CLI's JSON output mode but is not yet wired into the chat dock.
        """
        response = self.send_message(message, context, tools)
        yield AgentChunk(
            content=response.content,
            is_final=True,
        )

    def validate_connection(self) -> tuple[bool, str]:
        """Probe the CLI is installed and authenticated.

        Runs ``agy --version``. Non-zero exit or missing binary is
        reported with a user-friendly hint.
        """
        if self.binary is None:
            return False, (
                "Antigravity CLI (agy) not found. Install from "
                "https://antigravity.google.com/"
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
            return False, "Antigravity CLI did not respond to --version within 10s"
        except OSError as exc:
            return False, f"Could not execute {self.binary}: {exc}"

        if result.returncode != 0:
            return False, (
                f"agy --version exited {result.returncode}: "
                f"{(result.stderr or '').strip()[:200]}"
            )
        stdout_lines = (result.stdout or "").strip().splitlines()
        version = stdout_lines[0] if stdout_lines else "?"
        return True, f"Antigravity CLI available: {version}"

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_prompt(message: str, context: ConversationContext) -> str:
        """Concatenate conversation history into a single CLI prompt."""
        parts: list[str] = []
        for msg in getattr(context, "messages", []) or []:
            role = getattr(msg, "role", "user").capitalize()
            content = getattr(msg, "content", "") or ""
            if content:
                parts.append(f"{role}: {content}")
        parts.append(f"User: {message}")
        return "\n\n".join(parts)
