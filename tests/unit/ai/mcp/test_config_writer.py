from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.ai.mcp.config_writer import (
    McpServersFile,
    load,
    read,
    validate_env_placeholders,
    write,
)
from src.shared.python.ai.mcp.contracts import McpServerConfig


def test_write_serializes_claude_desktop_shape_and_creates_parent(
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / "mcp_servers.json"
    servers = [
        McpServerConfig(
            name="local",
            command="python",
            args=["-m", "local_server"],
            env={"TOKEN": "${LOCAL_TOKEN}"},
        ),
        {
            "name": "remote",
            "transport": "http",
            "url": "https://example.test/mcp",
            "timeout_seconds": 45,
        },
    ]

    returned = write(servers, path=target)

    assert returned == target
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == {
        "version": 1,
        "mcpServers": {
            "local": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "local_server"],
                "env": {"TOKEN": "${LOCAL_TOKEN}"},
            },
            "remote": {
                "transport": "http",
                "url": "https://example.test/mcp",
                "timeoutSeconds": 45.0,
            },
        },
    }


def test_write_rejects_duplicate_names_and_invalid_entries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate MCP server name"):
        write(
            [
                {"name": "dup", "command": "python"},
                {"name": "dup", "command": "node"},
            ],
            path=tmp_path / "servers.json",
        )

    with pytest.raises(ValueError, match="stdio transport requires 'command'"):
        write([{"name": "missing-command"}], path=tmp_path / "missing.json")

    with pytest.raises(TypeError, match="McpServerConfig or dict"):
        write([object()], path=tmp_path / "bad-type.json")  # type: ignore[list-item]


@pytest.mark.parametrize(
    "value",
    [
        "${}",
        "${1BAD}",
        "${UNTERMINATED",
        "prefix ${} suffix",
        "prefix ${-BAD} suffix",
    ],
)
def test_validate_env_placeholders_rejects_malformed_values(value: str) -> None:
    with pytest.raises(ValueError, match="Malformed environment-variable placeholder"):
        validate_env_placeholders(value)


def test_validate_env_placeholders_accepts_plain_and_valid_values() -> None:
    assert validate_env_placeholders("plain-token") == "plain-token"
    assert validate_env_placeholders("${TOKEN}_${TOKEN_2}") == "${TOKEN}_${TOKEN_2}"

    with pytest.raises(ValueError, match="not None"):
        validate_env_placeholders(None)  # type: ignore[arg-type]


def test_write_preflights_env_placeholders(tmp_path: Path) -> None:
    target = tmp_path / "servers.json"

    with pytest.raises(ValueError, match="Malformed environment-variable placeholder"):
        write(
            [{"name": "bad-env", "command": "python", "env": {"TOKEN": "${}"}}],
            path=target,
        )

    assert not target.exists()


def test_read_missing_and_non_object_shapes_return_empty_models(tmp_path: Path) -> None:
    assert read(path=tmp_path / "missing.json").servers == []

    scalar_path = tmp_path / "scalar.json"
    scalar_path.write_text("[]", encoding="utf-8")
    assert read(path=scalar_path).servers == []

    invalid_servers_path = tmp_path / "invalid_servers.json"
    invalid_servers_path.write_text('{"servers": {"not": "a list"}}', encoding="utf-8")
    assert read(path=invalid_servers_path).servers == []


def test_read_claude_desktop_shape_filters_invalid_entries(tmp_path: Path) -> None:
    target = tmp_path / "servers.json"
    target.write_text(
        json.dumps(
            {
                "version": 2,
                "mcpServers": {
                    "local": {
                        "command": "python",
                        "args": ["-m", "local"],
                        "env": {"A": "B"},
                    },
                    "remote": {
                        "transport": "http",
                        "url": "https://example.test/mcp",
                    },
                    "bad": {"transport": "http"},
                    "not-object": "ignored",
                },
            }
        ),
        encoding="utf-8",
    )

    model = read(path=target)

    assert model.version == 2
    assert [server.name for server in model.servers] == ["local", "remote"]
    assert model.servers[0].args == ["-m", "local"]
    assert model.servers[0].env == {"A": "B"}
    assert model.servers[1].transport.value == "http"


def test_read_flat_shape_filters_invalid_entries_and_duplicates(
    tmp_path: Path,
) -> None:
    target = tmp_path / "servers.json"
    target.write_text(
        json.dumps(
            {
                "servers": [
                    {"name": "good", "command": "python"},
                    "ignored",
                    {"name": "bad-http", "transport": "http"},
                    {"name": "good", "command": "node"},
                ]
            }
        ),
        encoding="utf-8",
    )

    model = read(path=target)

    assert model.servers == []


def test_read_malformed_json_raises_value_error(tmp_path: Path) -> None:
    target = tmp_path / "servers.json"
    target.write_text("{ not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to read MCP servers file"):
        read(path=target)


def test_load_alias_matches_read(tmp_path: Path) -> None:
    target = tmp_path / "servers.json"
    write([{"name": "alias", "command": "python"}], path=target)

    loaded = load(path=target)

    assert isinstance(loaded, McpServersFile)
    assert [server.name for server in loaded.servers] == ["alias"]
