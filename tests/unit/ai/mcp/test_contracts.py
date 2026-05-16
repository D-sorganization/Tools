"""Tests for MCP Pydantic contracts.

Covers:
- ``McpServerConfig`` validation for stdio and http transports.
- ``McpToolDescriptor`` and ``McpResourceDescriptor`` round-trip.
- DbC: invalid configurations raise ``pydantic.ValidationError`` / ``ValueError``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.shared.python.ai.mcp.contracts import (
    McpResourceDescriptor,
    McpServerConfig,
    McpToolDescriptor,
    McpTransport,
)


class TestMcpServerConfig:
    def test_stdio_minimal(self) -> None:
        cfg = McpServerConfig(name="notebooklm", transport="stdio", command="python")
        assert cfg.transport is McpTransport.STDIO
        assert cfg.command == "python"
        assert cfg.args == []
        assert cfg.env == {}

    def test_stdio_with_args_and_env(self) -> None:
        cfg = McpServerConfig(
            name="nb",
            transport="stdio",
            command="python",
            args=["-m", "mcp"],
            env={"API_KEY": "abc"},
        )
        assert cfg.args == ["-m", "mcp"]
        assert cfg.env == {"API_KEY": "abc"}

    def test_http_minimal(self) -> None:
        cfg = McpServerConfig(
            name="remote", transport="http", url="https://example.com/mcp"
        )
        assert cfg.transport is McpTransport.HTTP
        assert cfg.url == "https://example.com/mcp"

    def test_stdio_requires_command(self) -> None:
        with pytest.raises(ValidationError):
            McpServerConfig(name="nb", transport="stdio")

    def test_http_requires_url(self) -> None:
        with pytest.raises(ValidationError):
            McpServerConfig(name="nb", transport="http")

    def test_name_required_nonempty(self) -> None:
        with pytest.raises(ValidationError):
            McpServerConfig(name="", transport="stdio", command="python")

    def test_unknown_transport_rejected(self) -> None:
        with pytest.raises(ValidationError):
            McpServerConfig(name="nb", transport="websocket", command="python")


class TestMcpToolDescriptor:
    def test_round_trip(self) -> None:
        desc = McpToolDescriptor(
            name="search_notebook",
            description="Search the notebook",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        dumped = desc.model_dump()
        rebuilt = McpToolDescriptor(**dumped)
        assert rebuilt == desc

    def test_default_schema_is_object(self) -> None:
        desc = McpToolDescriptor(name="t", description="d")
        assert desc.input_schema == {"type": "object", "properties": {}}


class TestMcpResourceDescriptor:
    def test_round_trip(self) -> None:
        res = McpResourceDescriptor(
            uri="notebook://abc",
            name="Notebook ABC",
            mime_type="application/json",
        )
        rebuilt = McpResourceDescriptor(**res.model_dump())
        assert rebuilt == res
