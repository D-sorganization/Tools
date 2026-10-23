"""Antigravity (`agy`) subscription CLI adapter and Runner Dashboard staff adapter."""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from shared.python.ai.adapters.gemini_cli_adapter import GeminiCliAdapter
from shared.python.ai.exceptions import AIConnectionError, AITimeoutError
from shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
    ProviderCapability,
)
from shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_AGY_MODEL = "gemini-3.8-flash-high"
DEFAULT_STAFF_BASE_URL = "http://127.0.0.1:8321"
SUBSCRIPTION_CLI_ENABLED_BY_DEFAULT = False


def normalize_cli_provider_id(provider: str) -> str:
    """Normalize CLI provider identifiers, mapping legacy `gemini_cli` to `agy`."""
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("provider must be a non-empty string")
    cleaned = provider.strip().lower()
    if cleaned in {"gemini_cli", "gemini-cli"}:
        warnings.warn(
            "Provider 'gemini_cli' is deprecated; use 'agy' (Antigravity "
            "subscription CLI with gemini-3.8-flash-high) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return "agy"
    return cleaned


def _resolve_binary(explicit: str | None = None) -> str | None:
    """Locate the `agy` binary on PATH or in known install locations."""
    if explicit:
        if Path(explicit).exists():
            return explicit
        logger.warning("Explicit agy binary %s missing; checking PATH", explicit)
    if found := shutil.which("agy"):
        return found
    for candidate in (
        "~/.npm-global/bin/agy",
        "~/.local/bin/agy",
        "/usr/local/bin/agy",
    ):
        expanded = os.path.expanduser(os.path.expandvars(candidate))
        if Path(expanded).exists():
            return expanded
    return None


class AntigravityCliAdapter(GeminiCliAdapter):
    """Adapter that delegates chat to the Antigravity (`agy`) subscription CLI."""

    PROVIDER_NAME = "agy"
    ENABLED_BY_DEFAULT = SUBSCRIPTION_CLI_ENABLED_BY_DEFAULT

    def __init__(
        self,
        binary: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        resolved = DEFAULT_AGY_MODEL if model in (None, "", "default") else str(model)
        super().__init__(binary=binary, model=resolved, timeout=timeout)
        self.binary = _resolve_binary(binary)
        self.model = resolved

    @classmethod
    def from_legacy_config(cls, config: Mapping[str, Any]) -> AntigravityCliAdapter:
        """Construct an `AntigravityCliAdapter` from a stored provider config."""
        if not isinstance(config, Mapping):
            raise ValueError("config must be a mapping")
        normalize_cli_provider_id(str(config.get("provider", "agy")))
        binary, model, timeout = (
            config.get("binary") or config.get("host"),
            config.get("model"),
            config.get("timeout"),
        )
        return cls(
            binary=str(binary) if binary else None,
            model=str(model) if model else None,
            timeout=float(timeout) if timeout is not None else None,
        )

    def list_models(self) -> list[str]:
        return [DEFAULT_AGY_MODEL, "gemini-2.5-pro", "gemini-2.5-flash", "default"]


def probe_runner_dashboard_staff(
    base_url: str = DEFAULT_STAFF_BASE_URL,
    timeout: float = 2.0,
) -> tuple[bool, str]:
    """Probe whether the local Runner Dashboard Staff endpoint (`:8321`) is online."""
    if not base_url or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    req = urllib.request.Request(f"{base_url.rstrip('/')}/api/staff/summary")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 # nosec B310
            if getattr(resp, "status", 200) == 200:
                return True, f"Runner Dashboard Staff available at {base_url}"
            return False, f"Runner Dashboard Staff returned HTTP {resp.status}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, f"Runner Dashboard Staff unreachable at {base_url}: {exc}"


class RunnerDashboardStaffAdapter(BaseAgentAdapter):
    """Optional federation adapter targeting local Runner Dashboard Staff (`:8321`)."""

    PROVIDER_NAME = "runner_dashboard_staff"
    ENABLED_BY_DEFAULT = False

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or DEFAULT_STAFF_BASE_URL).rstrip("/")
        self.model = model or "barb"
        self.timeout = float(timeout) if timeout is not None else 30.0
        caps = {ProviderCapability.SYSTEM_MESSAGE, ProviderCapability.STREAMING}
        self._capabilities = ProviderCapabilities(
            supported=frozenset(caps),
            max_tokens=128_000,
            model_name=self.model,
            provider_name=self.PROVIDER_NAME,
        )

    def thinking_capabilities(self) -> Any:
        from shared.python.chat_contracts.models import make_none_only_capabilities

        return make_none_only_capabilities()

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def list_models(self) -> list[str]:
        return ["barb", "panel", "astra", "argon", "fable"]

    def validate_connection(self) -> tuple[bool, str]:
        return probe_runner_dashboard_staff(self.base_url, min(self.timeout, 5.0))

    def send_message(
        self,
        message: str,
        context: ConversationContext,  # noqa: ARG002
        tools: list[ToolDeclaration],  # noqa: ARG002
    ) -> AgentResponse:
        if message is None or not message.strip():
            raise ValueError("message must be a non-empty string")
        payload = json.dumps({"message": message, "seat": self.model}).encode("utf-8")
        req = urllib.request.Request(  # noqa: S310
            f"{self.base_url}/api/v1/staff/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 # nosec B310
                raw = resp.read().decode("utf-8")
        except TimeoutError as exc:
            raise AITimeoutError(f"Staff timed out after {self.timeout}s") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise AIConnectionError(f"Staff unreachable: {exc}") from exc
        data = json.loads(raw) if raw.strip() else {}
        return AgentResponse(
            content=str(data.get("reply") or data.get("content") or ""),
            usage={},
            metadata={"provider": self.PROVIDER_NAME, "seat": self.model},
        )

    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        yield AgentChunk(
            content=self.send_message(message, context, tools).content, is_final=True
        )
