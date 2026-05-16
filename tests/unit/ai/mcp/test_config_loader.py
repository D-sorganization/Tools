"""Tests for ``config_loader`` — Claude-Desktop-compatible MCP server config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.ai.mcp.config_loader import (
    expand_env_vars,
    load_mcp_servers,
    parse_mcp_servers,
)


def test_parse_minimal_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = {
        "mcpServers": {
            "notebooklm": {"command": "python", "args": ["-m", "notebooklm_mcp"]}
        }
    }
    configs = parse_mcp_servers(raw)
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.name == "notebooklm"
    assert cfg.transport.value == "stdio"
    assert cfg.command == "python"
    assert cfg.args == ["-m", "notebooklm_mcp"]


def test_parse_http_transport() -> None:
    raw = {
        "mcpServers": {
            "remote": {
                "transport": "http",
                "url": "https://example.com/mcp",
            }
        }
    }
    configs = parse_mcp_servers(raw)
    assert configs[0].transport.value == "http"
    assert configs[0].url == "https://example.com/mcp"


def test_env_var_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_API_KEY", "secret123")
    raw = {
        "mcpServers": {
            "nb": {
                "command": "python",
                "env": {"API_KEY": "${MY_API_KEY}"},
            }
        }
    }
    configs = parse_mcp_servers(raw)
    assert configs[0].env == {"API_KEY": "secret123"}


def test_missing_env_var_left_as_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UNSET_VAR", raising=False)
    assert expand_env_vars("${UNSET_VAR}") == "${UNSET_VAR}"


def test_invalid_entries_skipped_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw = {
        "mcpServers": {
            "good": {"command": "python"},
            "bad": {"transport": "websocket"},  # unsupported
            "empty_name": {},
        }
    }
    import logging

    with caplog.at_level(logging.WARNING):
        configs = parse_mcp_servers(raw)
    names = [c.name for c in configs]
    assert names == ["good"]
    # At least one warning emitted for skipped entries.
    assert any("skipping" in r.message.lower() for r in caplog.records)


def test_load_mcp_servers_from_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp_servers.json"
    cfg_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "nb": {"command": "python", "args": ["-m", "notebooklm_mcp"]}
                }
            }
        )
    )
    configs = load_mcp_servers(cfg_path)
    assert len(configs) == 1
    assert configs[0].name == "nb"


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    configs = load_mcp_servers(tmp_path / "does_not_exist.json")
    assert configs == []


def test_load_malformed_json_returns_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg_path = tmp_path / "mcp_servers.json"
    cfg_path.write_text("{ not valid json")
    import logging

    with caplog.at_level(logging.WARNING):
        configs = load_mcp_servers(cfg_path)
    assert configs == []
