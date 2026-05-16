"""Tests for ``ToolRegistry.refresh`` — MCP-pool merge.

Verifies that:
- Calling ``refresh`` on an empty registry with a pool merges all MCP tools
  under their namespaced names with the ``mcp:<server>`` source tag.
- Re-running ``refresh`` is idempotent: previously merged MCP tools are
  dropped and re-added so stale tools cannot linger.
- ``refresh(None)`` clears previously merged MCP tools without contacting
  any pool.
- Locally registered (non-MCP) tools are left untouched by refresh.
- Attempting to dispatch an MCP-sourced tool through the local
  ``ToolRegistry`` path raises (these tools must go through the pool).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.ai.exceptions import ToolExecutionError
from src.shared.python.ai.tool_registry import Tool, ToolRegistry
from src.shared.python.ai.types import ToolResult


class FakePool:
    """In-memory ``McpClientPool`` stand-in for refresh tests."""

    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self._tools = tools
        self.calls = 0

    async def refresh_all(self) -> list[dict[str, Any]]:
        self.calls += 1
        return list(self._tools)


def _mcp_entry(server: str, tool: str, description: str = "") -> dict[str, Any]:
    return {
        "name": tool,
        "namespaced_name": f"{server}:{tool}",
        "description": description,
        "input_schema": {"type": "object", "properties": {}},
        "source": f"mcp:{server}",
        "server": server,
    }


def _local_handler(**_: Any) -> ToolResult:
    return ToolResult(tool_call_id="", success=True, result={"ok": True})


@pytest.mark.asyncio
async def test_refresh_merges_mcp_tools() -> None:
    registry = ToolRegistry()
    pool = FakePool([_mcp_entry("nb", "search", "search a notebook")])
    count = await registry.refresh(pool)
    assert count == 1
    assert "nb:search" in registry
    tool = registry.get_tool("nb:search")
    assert tool is not None
    assert getattr(tool, "_mcp_source", None) == "mcp:nb"
    assert pool.calls == 1


@pytest.mark.asyncio
async def test_refresh_is_idempotent() -> None:
    registry = ToolRegistry()
    pool = FakePool(
        [
            _mcp_entry("nb", "search"),
            _mcp_entry("nb", "summarize"),
        ]
    )
    await registry.refresh(pool)
    # Second call must drop & re-add — no duplicates, no stale ghosts.
    await registry.refresh(pool)
    assert "nb:search" in registry
    assert "nb:summarize" in registry
    assert pool.calls == 2
    # Only the two MCP tools should be present.
    assert len(registry) == 2


@pytest.mark.asyncio
async def test_refresh_none_clears_mcp_tools() -> None:
    registry = ToolRegistry()
    pool = FakePool([_mcp_entry("nb", "search")])
    await registry.refresh(pool)
    assert "nb:search" in registry
    cleared = await registry.refresh(None)
    assert cleared == 0
    assert "nb:search" not in registry


@pytest.mark.asyncio
async def test_refresh_preserves_local_tools() -> None:
    registry = ToolRegistry()
    registry.register_tool(
        Tool(name="local_tool", description="local", handler=_local_handler)
    )
    pool = FakePool([_mcp_entry("nb", "search")])
    await registry.refresh(pool)
    assert "local_tool" in registry
    assert "nb:search" in registry
    # And clearing MCP tools leaves the local one intact.
    await registry.refresh(None)
    assert "local_tool" in registry
    assert "nb:search" not in registry


@pytest.mark.asyncio
async def test_executing_mcp_tool_locally_raises() -> None:
    registry = ToolRegistry()
    pool = FakePool([_mcp_entry("nb", "search")])
    await registry.refresh(pool)
    tool = registry.get_tool("nb:search")
    assert tool is not None
    with pytest.raises(ToolExecutionError):
        tool.handler()
