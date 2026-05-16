"""Tests for the Sidekick shell discovery helpers (UpstreamDrift #5617).

The discovery module is the single source of truth for which interactive
shells the Sidekick OS Terminal can launch. The widget, settings UI, and
all tests must consume :func:`discover_shells` (DRY).
"""

from __future__ import annotations

import platform
import shutil
from collections.abc import Iterator
from unittest.mock import patch

import pytest


def _is_linux() -> bool:
    return platform.system() == "Linux"


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _is_posix() -> bool:
    return platform.system() in {"Linux", "Darwin"}


@pytest.fixture
def empty_path(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Clear PATH so :func:`shutil.which` finds nothing."""
    monkeypatch.setenv("PATH", "")

    def _missing(_name: str, *_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(shutil, "which", _missing)
    yield


def test_discover_shells_returns_shell_descriptors() -> None:
    """:func:`discover_shells` returns a list of ``ShellDescriptor`` items."""
    from upstream_drift_tools.ui.tools_sidebar.shell_discovery import (
        ShellDescriptor,
        discover_shells,
    )

    shells = discover_shells()
    assert isinstance(shells, list)
    for shell in shells:
        assert isinstance(shell, ShellDescriptor)
        assert isinstance(shell.identifier, str) and shell.identifier
        assert isinstance(shell.label, str) and shell.label
        assert isinstance(shell.command, tuple)
        assert all(isinstance(part, str) for part in shell.command)


def test_discover_shells_empty_environment_returns_empty(empty_path: None) -> None:
    """Empty PATH and no platform shells leaves an empty descriptor list."""
    from upstream_drift_tools.ui.tools_sidebar import shell_discovery

    # Also stub out platform-specific probes so the discovery has truly
    # nothing to find.
    with (
        patch.object(shell_discovery, "_discover_posix_shells", return_value=[]),
        patch.object(shell_discovery, "_discover_windows_shells", return_value=[]),
        patch.object(shell_discovery, "_discover_wsl_distros", return_value=[]),
    ):
        assert shell_discovery.discover_shells() == []


@pytest.mark.skipif(not _is_linux(), reason="POSIX-only check")
def test_discover_shells_finds_bash_on_linux() -> None:
    """``bash`` appears on every supported Linux runner."""
    from upstream_drift_tools.ui.tools_sidebar.shell_discovery import discover_shells

    identifiers = {shell.identifier for shell in discover_shells()}
    assert "bash" in identifiers


@pytest.mark.skipif(not _is_posix(), reason="POSIX-only check")
def test_posix_discovery_uses_shutil_which() -> None:
    """The POSIX probe consults :func:`shutil.which` to locate shells."""
    from upstream_drift_tools.ui.tools_sidebar.shell_discovery import (
        _discover_posix_shells,
    )

    fake_path = "/usr/bin/bash"

    def fake_which(name: str) -> str | None:
        return fake_path if name == "bash" else None

    with patch("shutil.which", side_effect=fake_which):
        shells = _discover_posix_shells()

    identifiers = {shell.identifier for shell in shells}
    assert "bash" in identifiers
    bash = next(shell for shell in shells if shell.identifier == "bash")
    assert bash.command[0] == fake_path


@pytest.mark.skipif(not _is_windows(), reason="Windows-only check")
def test_discover_shells_finds_pwsh_or_powershell_on_windows() -> None:
    """At least one Windows shell is found on a standard Windows install."""
    from upstream_drift_tools.ui.tools_sidebar.shell_discovery import discover_shells

    identifiers = {shell.identifier for shell in discover_shells()}
    assert identifiers & {"pwsh", "powershell", "cmd"}


def test_wsl_distros_enumerated_from_wsl_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_discover_wsl_distros`` parses ``wsl --list --quiet`` output."""
    from upstream_drift_tools.ui.tools_sidebar import shell_discovery

    # `wsl --list --quiet` on Windows emits UTF-16-LE with NUL padding; the
    # helper must already have decoded the bytes by the time we see them.
    fake_output = "Ubuntu-22.04\nDebian\n"

    def fake_run(args: list[str]) -> str | None:
        if args[:3] == ["wsl", "--list", "--quiet"]:
            return fake_output
        return None

    monkeypatch.setattr(shell_discovery, "_run_text_command", fake_run)
    monkeypatch.setattr(shell_discovery, "_wsl_executable", lambda: "wsl")

    descriptors = shell_discovery._discover_wsl_distros()
    identifiers = {shell.identifier for shell in descriptors}
    assert "wsl:Ubuntu-22.04" in identifiers
    assert "wsl:Debian" in identifiers


def test_wsl_unavailable_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the ``wsl`` executable is missing the helper returns ``[]``."""
    from upstream_drift_tools.ui.tools_sidebar import shell_discovery

    monkeypatch.setattr(shell_discovery, "_wsl_executable", lambda: None)
    assert shell_discovery._discover_wsl_distros() == []


def test_shell_descriptor_rejects_empty_identifier() -> None:
    """``ShellDescriptor`` validates inputs (DbC preconditions)."""
    from upstream_drift_tools.ui.tools_sidebar.shell_discovery import ShellDescriptor

    with pytest.raises(ValueError):
        ShellDescriptor(identifier="", label="x", command=("/bin/sh",))
    with pytest.raises(ValueError):
        ShellDescriptor(identifier="bash", label="", command=("/bin/bash",))
    with pytest.raises(ValueError):
        ShellDescriptor(identifier="bash", label="bash", command=())
