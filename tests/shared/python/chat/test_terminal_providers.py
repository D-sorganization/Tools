"""Tests for data-driven terminal-agent provider descriptors."""

from __future__ import annotations


def test_default_provider_descriptors_cover_initial_agent_set() -> None:
    """The initial terminal-agent provider set is available without UI code."""
    from chat.terminal_providers import default_terminal_agent_providers

    providers = default_terminal_agent_providers()
    provider_ids = {provider.id for provider in providers}

    assert provider_ids == {
        "claude-code",
        "codex",
        "cline-cli",
        "gemini-cli",
        "github-cli",
    }
    assert {provider.executable for provider in providers} == {
        "claude",
        "codex",
        "cline",
        "gemini",
        "gh",
    }
    assert all(provider.install_probe_args == ["--version"] for provider in providers)


def test_default_registry_registers_shells_before_providers() -> None:
    """Consumers can enumerate shell-compatible providers from one registry."""
    from chat.terminal_providers import build_default_terminal_provider_registry

    registry = build_default_terminal_provider_registry()

    assert [shell.id for shell in registry.shells()] == [
        "powershell",
        "pwsh",
        "bash",
        "wsl",
    ]
    assert [provider.id for provider in registry.providers_for_shell("pwsh")] == [
        "claude-code",
        "codex",
        "cline-cli",
        "gemini-cli",
        "github-cli",
    ]
    assert registry.validate_selection("bash", "gemini-cli").provider.executable == (
        "gemini"
    )


def test_provider_probe_commands_are_metadata_only_and_redacted() -> None:
    """Probe command helpers expose safe diagnostics without running commands."""
    from chat.terminal_providers import (
        provider_probe_commands,
        redact_terminal_command,
    )

    probes = provider_probe_commands("claude-code")

    assert probes["install"] == ["claude", "--version"]
    assert probes["auth"] == ["claude", "auth", "status"]
    assert redact_terminal_command(
        ["agent", "--api-key", "secret-value", "--token=another-secret"]
    ) == ["agent", "--api-key", "***", "--token=***"]


def test_default_provider_helpers_are_public() -> None:
    """Provider helpers are part of the documented chat facade."""
    import chat

    expected = {
        "build_default_terminal_provider_registry",
        "default_terminal_agent_providers",
        "default_terminal_shells",
        "provider_probe_commands",
        "redact_terminal_command",
    }

    assert expected.issubset(set(chat.__all__))
    for name in expected:
        assert getattr(chat, name) is not None
