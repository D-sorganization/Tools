"""Tests for ``McpClient`` — single-server JSON-RPC wrapper.

The transport layer is faked: a ``FakeStdioTransport`` records sent frames and
returns canned JSON-RPC responses. No real subprocesses or HTTP calls are made.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.ai.mcp.client import McpClient
from src.shared.python.ai.mcp.contracts import McpServerConfig


class FakeStdioTransport:
    """In-memory transport that mimics the wire-level MCP behavior."""

    def __init__(self, responses: dict[str, dict[str, Any]] | None = None) -> None:
        self.responses = responses or {}
        self.sent: list[dict[str, Any]] = []
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.sent.append({"method": method, "params": params})
        if method in self.responses:
            return self.responses[method]
        raise RuntimeError(f"no canned response for {method}")


def _stdio_cfg() -> McpServerConfig:
    return McpServerConfig(name="nb", transport="stdio", command="python")


class TestMcpClientLifecycle:
    @pytest.mark.asyncio
    async def test_connect_disconnect(self) -> None:
        transport = FakeStdioTransport(
            responses={
                "initialize": {"protocolVersion": "2024-11-05"},
                "tools/list": {"tools": []},
                "resources/list": {"resources": []},
            }
        )
        client = McpClient(_stdio_cfg(), transport=transport)
        assert client.is_connected is False
        await client.connect()
        assert client.is_connected is True
        assert transport.started is True
        await client.disconnect()
        assert client.is_connected is False
        assert transport.stopped is True

    @pytest.mark.asyncio
    async def test_reconnect_idempotent(self) -> None:
        transport = FakeStdioTransport(
            responses={
                "initialize": {"protocolVersion": "2024-11-05"},
                "tools/list": {"tools": []},
                "resources/list": {"resources": []},
            }
        )
        client = McpClient(_stdio_cfg(), transport=transport)
        await client.connect()
        await client.connect()  # idempotent
        assert client.is_connected is True
        await client.disconnect()
        await client.disconnect()  # idempotent
        assert client.is_connected is False


class TestMcpClientListTools:
    @pytest.mark.asyncio
    async def test_list_tools_returns_descriptors(self) -> None:
        transport = FakeStdioTransport(
            responses={
                "initialize": {"protocolVersion": "2024-11-05"},
                "tools/list": {
                    "tools": [
                        {
                            "name": "search_notebook",
                            "description": "search",
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                    ]
                },
                "resources/list": {"resources": []},
            }
        )
        client = McpClient(_stdio_cfg(), transport=transport)
        await client.connect()
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "search_notebook"
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_list_tools_precondition_requires_connection(self) -> None:
        client = McpClient(_stdio_cfg(), transport=FakeStdioTransport())
        with pytest.raises(RuntimeError, match="not connected"):
            await client.list_tools()


class TestMcpClientCallTool:
    @pytest.mark.asyncio
    async def test_call_tool_success(self) -> None:
        transport = FakeStdioTransport(
            responses={
                "initialize": {"protocolVersion": "2024-11-05"},
                "tools/list": {"tools": []},
                "resources/list": {"resources": []},
                "tools/call": {"content": [{"type": "text", "text": "ok"}]},
            }
        )
        client = McpClient(_stdio_cfg(), transport=transport)
        await client.connect()
        result = await client.call_tool("search_notebook", {"q": "hello"})
        assert result == {"content": [{"type": "text", "text": "ok"}]}
        # Verify the request was correctly serialized.
        last = transport.sent[-1]
        assert last["method"] == "tools/call"
        assert last["params"]["name"] == "search_notebook"
        assert last["params"]["arguments"] == {"q": "hello"}
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_call_tool_empty_name_raises(self) -> None:
        client = McpClient(_stdio_cfg(), transport=FakeStdioTransport())
        with pytest.raises(ValueError):
            await client.call_tool("", {})

    @pytest.mark.asyncio
    async def test_call_tool_precondition_requires_connection(self) -> None:
        client = McpClient(_stdio_cfg(), transport=FakeStdioTransport())
        with pytest.raises(RuntimeError, match="not connected"):
            await client.call_tool("search", {})


class TestMcpClientListResources:
    @pytest.mark.asyncio
    async def test_list_resources(self) -> None:
        transport = FakeStdioTransport(
            responses={
                "initialize": {"protocolVersion": "2024-11-05"},
                "tools/list": {"tools": []},
                "resources/list": {
                    "resources": [{"uri": "notebook://abc", "name": "Notebook ABC"}]
                },
            }
        )
        client = McpClient(_stdio_cfg(), transport=transport)
        await client.connect()
        resources = await client.list_resources()
        assert len(resources) == 1
        assert resources[0].uri == "notebook://abc"
        await client.disconnect()
