"""Tests for ``build_github_mcp_config`` — config builder + DbC."""

from __future__ import annotations

import pytest

from src.shared.python.ai.integrations.github_mcp.server_config import (
    GITHUB_MCP_SERVER_NAME,
    GITHUB_MCP_TOKEN_ENV_VAR,
    build_github_mcp_config,
)
from src.shared.python.ai.mcp.contracts import McpServerConfig, McpTransport


def test_build_config_returns_mcp_server_config() -> None:
    """Builder returns a fully-validated ``McpServerConfig`` instance."""
    cfg = build_github_mcp_config(token="ghp_test_token_123")
    assert isinstance(cfg, McpServerConfig)


def test_build_config_default_name_is_github() -> None:
    cfg = build_github_mcp_config(token="ghp_test_token_123")
    assert cfg.name == GITHUB_MCP_SERVER_NAME
    assert cfg.name == "github"


def test_build_config_uses_npx_command() -> None:
    """The default invocation is ``npx -y @modelcontextprotocol/server-github``."""
    cfg = build_github_mcp_config(token="ghp_test_token_123")
    assert cfg.command == "npx"
    assert "@modelcontextprotocol/server-github" in cfg.args
    # ``-y`` ensures non-interactive install on first run.
    assert "-y" in cfg.args


def test_build_config_uses_stdio_transport() -> None:
    cfg = build_github_mcp_config(token="ghp_test_token_123")
    assert cfg.transport is McpTransport.STDIO


def test_build_config_sets_token_env_var() -> None:
    """The token is plumbed via ``GITHUB_PERSONAL_ACCESS_TOKEN``."""
    cfg = build_github_mcp_config(token="ghp_test_token_xyz")
    assert GITHUB_MCP_TOKEN_ENV_VAR == "GITHUB_PERSONAL_ACCESS_TOKEN"
    assert cfg.env[GITHUB_MCP_TOKEN_ENV_VAR] == "ghp_test_token_xyz"


def test_build_config_none_token_raises() -> None:
    """DbC precondition: token is required."""
    with pytest.raises(ValueError, match="GitHub token required"):
        build_github_mcp_config(token=None)


def test_build_config_empty_token_raises() -> None:
    """DbC precondition: empty / whitespace token is rejected."""
    with pytest.raises(ValueError, match="GitHub token required"):
        build_github_mcp_config(token="")
    with pytest.raises(ValueError, match="GitHub token required"):
        build_github_mcp_config(token="   ")


def test_build_config_custom_name_supported() -> None:
    """Operators can override the server name for multi-account setups."""
    cfg = build_github_mcp_config(token="ghp_x", name="github-personal")
    assert cfg.name == "github-personal"


def test_build_config_extra_env_merged() -> None:
    """Extra env vars supplied by caller are merged with the token env."""
    cfg = build_github_mcp_config(
        token="ghp_x",
        extra_env={"GITHUB_API_URL": "https://ghe.example.com/api/v3"},
    )
    assert cfg.env["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_x"
    assert cfg.env["GITHUB_API_URL"] == "https://ghe.example.com/api/v3"


def test_build_config_extra_env_does_not_override_token() -> None:
    """The explicit ``token`` argument wins over a colliding ``extra_env`` entry."""
    cfg = build_github_mcp_config(
        token="ghp_real",
        extra_env={"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_fake"},
    )
    assert cfg.env["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_real"
