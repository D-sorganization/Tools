"""Contract tests for shared chat terminal-agent descriptors."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError


def test_terminal_contract_symbols_are_public() -> None:
    """Consumers can import terminal-agent contracts from the chat facade."""
    import chat

    expected = {
        "TerminalAgentEvent",
        "TerminalAgentProviderInfo",
        "TerminalAgentSessionInfo",
        "TerminalAgentSessionRequest",
        "TerminalProviderRegistry",
        "TerminalRegistryError",
        "TerminalShellInfo",
    }

    assert expected.issubset(set(chat.__all__))
    for name in expected:
        assert getattr(chat, name) is not None


def test_registry_rejects_duplicate_shell_ids() -> None:
    """Shell ids are unique registry keys."""
    from chat import TerminalProviderRegistry, TerminalRegistryError, TerminalShellInfo

    registry = TerminalProviderRegistry()
    shell = TerminalShellInfo(
        id="powershell",
        display_name="PowerShell",
        executable="pwsh",
    )

    registry.register_shell(shell)

    with pytest.raises(TerminalRegistryError, match="already registered"):
        registry.register_shell(shell)


def test_registry_rejects_duplicate_provider_ids() -> None:
    """Provider ids are unique registry keys."""
    from chat import (
        TerminalAgentProviderInfo,
        TerminalProviderRegistry,
        TerminalRegistryError,
        TerminalShellInfo,
    )

    registry = TerminalProviderRegistry()
    registry.register_shell(
        TerminalShellInfo(id="bash", display_name="Bash", executable="bash")
    )
    provider = TerminalAgentProviderInfo(
        id="codex",
        display_name="Codex",
        executable="codex",
        supported_shells=["bash"],
    )

    registry.register_provider(provider)

    with pytest.raises(TerminalRegistryError, match="already registered"):
        registry.register_provider(provider)


def test_provider_must_reference_registered_shell() -> None:
    """Provider descriptors cannot point at unknown shell ids."""
    from chat import (
        TerminalAgentProviderInfo,
        TerminalProviderRegistry,
        TerminalRegistryError,
    )

    registry = TerminalProviderRegistry()
    provider = TerminalAgentProviderInfo(
        id="gemini",
        display_name="Gemini CLI",
        executable="gemini",
        supported_shells=["bash"],
    )

    with pytest.raises(TerminalRegistryError, match="unknown shell"):
        registry.register_provider(provider)


def test_registry_validates_supported_shell_provider_pair() -> None:
    """A provider can be registered once and selected only on supported shells."""
    from chat import (
        TerminalAgentProviderInfo,
        TerminalProviderRegistry,
        TerminalRegistryError,
        TerminalShellInfo,
    )

    registry = TerminalProviderRegistry()
    registry.register_shell(
        TerminalShellInfo(id="powershell", display_name="PowerShell", executable="pwsh")
    )
    registry.register_shell(
        TerminalShellInfo(id="bash", display_name="Bash", executable="bash")
    )
    registry.register_provider(
        TerminalAgentProviderInfo(
            id="claude-code",
            display_name="Claude Code",
            executable="claude",
            supported_shells=["bash"],
        )
    )

    assert registry.validate_selection("bash", "claude-code").provider.id == (
        "claude-code"
    )

    with pytest.raises(TerminalRegistryError, match="does not support shell"):
        registry.validate_selection("powershell", "claude-code")


def test_registry_can_add_new_provider_without_ui_changes() -> None:
    """Dropdown-like consumers can enumerate provider data from the registry."""
    from chat import (
        TerminalAgentProviderInfo,
        TerminalProviderRegistry,
        TerminalShellInfo,
    )

    registry = TerminalProviderRegistry()
    registry.register_shell(
        TerminalShellInfo(id="wsl", display_name="WSL", executable="wsl")
    )
    registry.register_provider(
        TerminalAgentProviderInfo(
            id="future-agent",
            display_name="Future Agent",
            executable="future-agent",
            supported_shells=["wsl"],
        )
    )

    choices = [
        (provider.id, provider.display_name) for provider in registry.providers()
    ]

    assert choices == [("future-agent", "Future Agent")]


def test_session_request_resolves_project_root(tmp_path: Path) -> None:
    """Session requests normalize the project root before runtime launch."""
    from chat import TerminalAgentSessionRequest

    request = TerminalAgentSessionRequest(
        app_context="tools",
        project_root=tmp_path,
        shell_id="powershell",
        provider_id="codex",
    )

    assert request.project_root == tmp_path.resolve()


def test_session_request_rejects_missing_project_root(tmp_path: Path) -> None:
    """Terminal sessions cannot be requested against a missing directory."""
    from chat import TerminalAgentSessionRequest

    with pytest.raises(ValidationError):
        TerminalAgentSessionRequest(
            app_context="tools",
            project_root=tmp_path / "missing",
            shell_id="bash",
            provider_id="codex",
        )


def test_descriptor_ids_must_be_lowercase() -> None:
    """Stable registry ids stay lowercase and predictable."""
    from chat import TerminalShellInfo

    with pytest.raises(ValidationError):
        TerminalShellInfo(
            id="PowerShell",
            display_name="PowerShell",
            executable="pwsh",
        )
