"""Tests for preset application + installation probes via config_loader."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.ai.mcp.config_loader import (
    apply_preset_to_config,
    is_preset_installed,
)


def test_apply_preset_to_config_writes_new_entry(tmp_path) -> None:
    target = tmp_path / "mcp_servers.json"
    cfg = apply_preset_to_config(
        "memory",
        target.resolve(),
        env={},
    )
    assert cfg.name == "memory"
    import json

    data = json.loads(target.read_text())
    assert "memory" in data["mcpServers"]


def test_apply_preset_to_config_with_env_overrides(tmp_path) -> None:
    target = tmp_path / "mcp_servers.json"
    cfg = apply_preset_to_config(
        "github",
        target.resolve(),
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_test"},
    )
    assert cfg.env["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_test"


def test_apply_preset_to_config_custom_name(tmp_path) -> None:
    target = tmp_path / "mcp_servers.json"
    cfg = apply_preset_to_config(
        "memory",
        target.resolve(),
        name="my_memory",
    )
    assert cfg.name == "my_memory"
    import json

    data = json.loads(target.read_text())
    assert "my_memory" in data["mcpServers"]


def test_apply_preset_unknown_raises(tmp_path) -> None:
    target = tmp_path / "mcp_servers.json"
    with pytest.raises(KeyError):
        apply_preset_to_config("nope", target.resolve())


def test_is_preset_installed_unknown_returns_false() -> None:
    assert is_preset_installed("not_a_real_preset_xyz") is False


def test_is_preset_installed_npm_success() -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="1.0.0\n")
        assert is_preset_installed("memory") is True


def test_is_preset_installed_npm_failure() -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert is_preset_installed("memory") is False


def test_is_preset_installed_npm_timeout() -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="npm", timeout=5)
        assert is_preset_installed("memory") is False


def test_is_preset_installed_notebooklm_local_python() -> None:
    # NotebookLM uses local python shim — should be importable in-tree.
    assert is_preset_installed("notebooklm") is True
