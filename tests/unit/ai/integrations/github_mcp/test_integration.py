# ruff: noqa: E501
"""Tests for ``register_github_mcp`` — pool-integration helper."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.ai.integrations.github_mcp.integration import (
    register_github_mcp,
)
from src.shared.python.ai.mcp.contracts import McpServerConfig


class _FakePool:
    """Test double for ``McpClientPool`` exposing only the LOD-approved surface."""

    def __init__(self, started: bool = True) -> None:
        self._started = started
        self.add_server = MagicMock()

    @property
    def is_started(self) -> bool:
        return self._started


def test_register_uses_explicit_token() -> None:
    pool = _FakePool(started=True)
    register_github_mcp(pool, token="ghp_explicit")  # type: ignore[arg-type]

    pool.add_server.assert_called_once()
    cfg = pool.add_server.call_args.args[0]
    assert isinstance(cfg, McpServerConfig)
    assert cfg.name == "github"
    assert cfg.env["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_explicit"


def test_register_falls_back_to_env_var() -> None:
    pool = _FakePool(started=True)
    with patch.dict(
        os.environ, {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_from_env"}, clear=False
    ):
        register_github_mcp(pool)  # type: ignore[arg-type]

    cfg = pool.add_server.call_args.args[0]
    assert cfg.env["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_from_env"


def test_register_raises_when_pool_not_started() -> None:
    """DbC precondition: ``pool.is_started`` must hold."""
    pool = _FakePool(started=False)
    with pytest.raises(RuntimeError, match="pool"):
        register_github_mcp(pool, token="ghp_x")  # type: ignore[arg-type]
    pool.add_server.assert_not_called()


def test_register_raises_when_no_token_anywhere() -> None:
    pool = _FakePool(started=True)
    # Strip the env var if it's set in the host shell
    env_copy = {
        k: v for k, v in os.environ.items() if k != "GITHUB_PERSONAL_ACCESS_TOKEN"
    }
    with patch.dict(os.environ, env_copy, clear=True):  # noqa: SIM117
        with pytest.raises(ValueError, match="GitHub token required"):
            register_github_mcp(pool)  # type: ignore[arg-type]
    pool.add_server.assert_not_called()


def test_register_custom_name_propagates() -> None:
    pool = _FakePool(started=True)
    register_github_mcp(pool, token="ghp_x", name="github-work")  # type: ignore[arg-type]
    cfg = pool.add_server.call_args.args[0]
    assert cfg.name == "github-work"


def test_register_is_idempotent_when_duplicate_rejected_by_pool() -> None:
    """If the pool already has a 'github' server, the helper surfaces the error."""
    pool = _FakePool(started=True)
    pool.add_server.side_effect = ValueError("server already registered: github")
    with pytest.raises(ValueError, match="already registered"):
        register_github_mcp(pool, token="ghp_x")  # type: ignore[arg-type]
