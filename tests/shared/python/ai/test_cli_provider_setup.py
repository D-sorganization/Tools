"""Tests for the CLI-provider install/auth guide module.

Pins:
- :func:`get_cli_setup_status` never raises (even on probe exception or
  unknown provider).
- :data:`CLI_PROVIDERS` exposes every CLI-shaped adapter so the chat UI
  never has to hardcode install instructions.
- Each spec has non-empty install_command + install_url + auth_instructions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.ai.adapters.cli_provider_setup import (  # noqa: E402
    CLI_PROVIDERS,
    CliSetupStatus,
    _safe_version,
    get_all_cli_setup_statuses,
    get_cli_setup_status,
)


class TestCatalogue:
    def test_all_cli_providers_covered(self) -> None:
        """Every CLI-shaped provider must have an install/auth card."""
        expected = {"claude_code", "codex_cli", "gemini_cli", "cline"}
        assert expected.issubset(CLI_PROVIDERS.keys()), (
            f"Missing CLI providers: {expected - set(CLI_PROVIDERS.keys())}"
        )

    @pytest.mark.parametrize(
        "provider", ["claude_code", "codex_cli", "gemini_cli", "cline"]
    )
    def test_each_spec_has_required_fields(self, provider: str) -> None:
        spec = CLI_PROVIDERS[provider]
        assert spec.display_name, f"{provider}: empty display_name"
        assert spec.install_command, f"{provider}: empty install_command"
        assert spec.install_url.startswith(("http://", "https://")), (
            f"{provider}: install_url not a URL: {spec.install_url!r}"
        )
        assert len(spec.auth_instructions) > 20, (
            f"{provider}: auth_instructions too short to be useful"
        )


class TestStatusProbe:
    def test_unknown_provider_returns_fallback(self) -> None:
        status = get_cli_setup_status("definitely-not-a-provider")
        assert isinstance(status, CliSetupStatus)
        assert status.installed is False
        assert status.binary_path is None
        # Caller can still render a card with the (empty) commands without crashing.
        assert isinstance(status.install_command, str)

    def test_probe_exception_treated_as_not_installed(self) -> None:
        """A buggy probe must not crash the chat UI."""
        with patch.dict(
            "src.shared.python.ai.adapters.cli_provider_setup.CLI_PROVIDERS",
            {
                "_broken": MagicMock(
                    provider="_broken",
                    display_name="Broken",
                    install_command="echo install",
                    install_url="https://example.com",
                    auth_instructions="auth instructions here.",
                    auth_command=None,
                    probe=MagicMock(side_effect=RuntimeError("probe blew up")),
                )
            },
            clear=False,
        ):
            status = get_cli_setup_status("_broken")
            assert status.installed is False
            assert status.binary_path is None

    def test_probe_returning_path_marks_installed(self) -> None:
        with patch.dict(
            "src.shared.python.ai.adapters.cli_provider_setup.CLI_PROVIDERS",
            {
                "_test": MagicMock(
                    provider="_test",
                    display_name="Test CLI",
                    install_command="echo install",
                    install_url="https://example.com",
                    auth_instructions="auth instructions here.",
                    auth_command="test login",
                    probe=MagicMock(return_value=("/usr/bin/test-cli", "1.2.3")),
                )
            },
            clear=False,
        ):
            status = get_cli_setup_status("_test")
            assert status.installed is True
            assert status.binary_path == "/usr/bin/test-cli"
            assert status.version == "1.2.3"


class TestGetAll:
    def test_returns_one_status_per_provider(self) -> None:
        statuses = get_all_cli_setup_statuses()
        assert set(statuses.keys()) == set(CLI_PROVIDERS.keys())
        for provider, status in statuses.items():
            assert isinstance(status, CliSetupStatus)
            assert status.provider == provider


class TestSafeVersion:
    def test_returns_first_line_on_success(self) -> None:
        fake = MagicMock(returncode=0, stdout="1.0.0\nmore noise\n", stderr="")
        with patch("subprocess.run", return_value=fake):
            assert _safe_version("/bin/x", ["--version"]) == "1.0.0"

    def test_returns_none_on_nonzero_exit(self) -> None:
        fake = MagicMock(returncode=1, stdout="", stderr="boom")
        with patch("subprocess.run", return_value=fake):
            assert _safe_version("/bin/x", ["--version"]) is None

    def test_returns_none_on_timeout(self) -> None:
        import subprocess

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1.0),
        ):
            assert _safe_version("/bin/x", ["--version"]) is None

    def test_returns_none_on_oserror(self) -> None:
        with patch("subprocess.run", side_effect=OSError("not found")):
            assert _safe_version("/bin/x", ["--version"]) is None
