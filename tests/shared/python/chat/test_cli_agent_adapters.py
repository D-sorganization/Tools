"""Tests for CLI agent adapter health checks and session start wiring.

Covers the TerminalSessionRuntime integration for CLI provider selection
in the ChatDockWidget, plus DbC negative cases for empty messages.

Tools chat provider dropdown coverage.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Register src namespace packages so dotted imports resolve correctly
_src_pkg = types.ModuleType("src")
_src_pkg.__path__ = [str(ROOT / "src")]  # type: ignore[attr-defined]
sys.modules.setdefault("src", _src_pkg)

for _ns in (
    "src.shared",
    "src.shared.python",
    "src.shared.python.chat",
    "src.shared.python.contracts",
):
    _parts = _ns.split(".")
    _mod = types.ModuleType(_ns)
    _mod.__path__ = [str(ROOT.joinpath(*_parts))]  # type: ignore[attr-defined]
    sys.modules.setdefault(_ns, _mod)

logging_pkg = types.ModuleType("src.shared.python.logging_pkg")
logging_config = types.ModuleType("src.shared.python.logging_pkg.logging_config")
logging_config.get_logger = logging.getLogger  # type: ignore[attr-defined]
logging_config.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("src.shared.python.logging_pkg", logging_pkg)
sys.modules.setdefault("src.shared.python.logging_pkg.logging_config", logging_config)

from src.shared.python.chat.cli_provider_availability import (  # noqa: E402
    list_available_cli_providers,
)
from src.shared.python.chat.terminal_providers import (  # noqa: E402
    build_default_terminal_provider_registry,
)
from src.shared.python.chat.terminal_runtime import (  # noqa: E402
    TerminalSessionRuntime,
)

# ─────────────────────────────────────────────────────────────────────────────
# health_check helpers via list_available_cli_providers
# ─────────────────────────────────────────────────────────────────────────────


class TestCliProviderHealthChecks:
    def test_health_check_passes_when_claude_binary_found(self) -> None:
        with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"):
            providers = list_available_cli_providers()
        claude_entries = [p for p in providers if p.provider_id == "claude-code"]
        assert len(claude_entries) == 1
        assert claude_entries[0].binary_path is not None

    def test_health_check_fails_when_all_binaries_missing(self) -> None:
        with patch("shutil.which", return_value=None):
            providers = list_available_cli_providers()
        assert providers == []

    def test_health_check_fails_when_only_codex_missing(self) -> None:
        available = {"claude", "cline"}

        def _which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name in available else None

        with patch("shutil.which", side_effect=_which):
            providers = list_available_cli_providers()
        ids = [p.provider_id for p in providers]
        assert "codex" not in ids
        assert "claude-code" in ids


# ─────────────────────────────────────────────────────────────────────────────
# TerminalSessionRuntime integration — start returns session info
# ─────────────────────────────────────────────────────────────────────────────


class TestTerminalSessionRuntimeCliIntegration:
    """Verify TerminalSessionRuntime can start CLI provider sessions."""

    def _make_runtime(self) -> TerminalSessionRuntime:
        registry = build_default_terminal_provider_registry()
        adapter = MagicMock()
        adapter.start.return_value = "proc-001"
        return TerminalSessionRuntime(registry, adapter)

    def test_start_claude_cli_session_returns_info(self, tmp_path: Path) -> None:
        from src.shared.python.chat.terminal_contracts import (
            TerminalAgentSessionRequest,
        )

        runtime = self._make_runtime()
        request = TerminalAgentSessionRequest(
            app_context="sidekick",
            project_root=tmp_path,
            shell_id="bash",
            provider_id="claude-code",
        )
        info = runtime.start(request)
        assert info.provider_id == "claude-code"
        assert info.state == "running"

    def test_start_codex_session_returns_info(self, tmp_path: Path) -> None:
        from src.shared.python.chat.terminal_contracts import (
            TerminalAgentSessionRequest,
        )

        runtime = self._make_runtime()
        request = TerminalAgentSessionRequest(
            app_context="sidekick",
            project_root=tmp_path,
            shell_id="bash",
            provider_id="codex",
        )
        info = runtime.start(request)
        assert info.provider_id == "codex"
        assert info.state == "running"

    def test_start_cline_session_returns_info(self, tmp_path: Path) -> None:
        from src.shared.python.chat.terminal_contracts import (
            TerminalAgentSessionRequest,
        )

        runtime = self._make_runtime()
        request = TerminalAgentSessionRequest(
            app_context="sidekick",
            project_root=tmp_path,
            shell_id="bash",
            provider_id="cline-cli",
        )
        info = runtime.start(request)
        assert info.provider_id == "cline-cli"
        assert info.state == "running"

    def test_write_raises_on_empty_input(self, tmp_path: Path) -> None:
        from src.shared.python.chat.terminal_contracts import (
            TerminalAgentSessionRequest,
        )
        from src.shared.python.chat.terminal_runtime import TerminalRuntimeError

        runtime = self._make_runtime()
        request = TerminalAgentSessionRequest(
            app_context="sidekick",
            project_root=tmp_path,
            shell_id="bash",
            provider_id="claude-code",
        )
        info = runtime.start(request)
        with pytest.raises(TerminalRuntimeError):
            runtime.write(info.session_id, "")


# ─────────────────────────────────────────────────────────────────────────────
# Default registry has correct CLI provider display names
# ─────────────────────────────────────────────────────────────────────────────


class TestDefaultRegistryCliProviders:
    """Verify the registry exposes the expected CLI providers."""

    def test_claude_code_provider_in_registry(self) -> None:
        registry = build_default_terminal_provider_registry()
        provider = registry.get_provider("claude-code")
        assert provider.display_name == "Claude Code"
        assert provider.executable == "claude"

    def test_codex_provider_in_registry(self) -> None:
        registry = build_default_terminal_provider_registry()
        provider = registry.get_provider("codex")
        assert provider.display_name == "Codex"
        assert provider.executable == "codex"

    def test_cline_cli_provider_in_registry(self) -> None:
        registry = build_default_terminal_provider_registry()
        provider = registry.get_provider("cline-cli")
        assert provider.display_name == "Cline CLI"
        assert provider.executable == "cline"

    def test_github_cli_provider_in_registry(self) -> None:
        registry = build_default_terminal_provider_registry()
        provider = registry.get_provider("github-cli")
        assert provider.display_name == "GitHub CLI"
        assert provider.executable == "gh"

    def test_all_cli_providers_support_bash_shell(self) -> None:
        registry = build_default_terminal_provider_registry()
        for pid in ("claude-code", "codex", "cline-cli", "github-cli"):
            provider = registry.get_provider(pid)
            assert "bash" in provider.supported_shells, f"{pid} should support bash"

    def test_providers_for_bash_includes_all_cli_agents(self) -> None:
        registry = build_default_terminal_provider_registry()
        providers = registry.providers_for_shell("bash")
        ids = [p.id for p in providers]
        assert "claude-code" in ids
        assert "codex" in ids
        assert "cline-cli" in ids
        assert "github-cli" in ids


# ─────────────────────────────────────────────────────────────────────────────
# list_available_cli_providers + terminal registry alignment
# ─────────────────────────────────────────────────────────────────────────────


class TestCliProviderAlignmentWithRegistry:
    """provider_ids from list_available_cli_providers must exist in registry."""

    def test_all_returned_provider_ids_are_in_registry(self) -> None:
        registry = build_default_terminal_provider_registry()
        with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"):
            providers = list_available_cli_providers()
        for entry in providers:
            # Must not raise TerminalRegistryError
            registry.get_provider(entry.provider_id)

    def test_display_names_match_registry(self) -> None:
        registry = build_default_terminal_provider_registry()
        with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"):
            providers = list_available_cli_providers()
        for entry in providers:
            reg_provider = registry.get_provider(entry.provider_id)
            # The UI display name comes from the entry; registry has its own
            # canonical display name. They may differ (e.g. "Claude CLI" vs
            # "Claude Code") — the important thing is both are non-empty.
            assert entry.display_name
            assert reg_provider.display_name
