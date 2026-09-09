"""Exercise the actual optional MCP protocol over a local stdio subprocess."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest


def test_real_mcp_lists_only_retrieval_tools_and_observes_edits(
    repository: Path,
) -> None:
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def exercise() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "agent_context.mcp_server", "--root", str(repository)],
            env=dict(os.environ),
        )
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                listed = await session.list_tools()
                assert {tool.name for tool in listed.tools} == {
                    "search_components",
                    "get_component_context",
                    "context_status",
                }
                response = await session.call_tool(
                    "get_component_context", {"component_id": "provider"}
                )
                assert not response.isError
                first = json.loads(response.content[0].text)
                (repository / "src/provider.py").write_text(
                    "def convert(value):\n    return value * 5\n", encoding="utf-8"
                )
                response = await session.call_tool(
                    "get_component_context", {"component_id": "provider"}
                )
                second = json.loads(response.content[0].text)
                assert first["provenance"]["digest"] != second["provenance"]["digest"]
                assert "value * 5" in second["sources"][0]["excerpt"]
                unknown = await session.call_tool(
                    "get_component_context", {"component_id": "missing"}
                )
                assert unknown.isError

    asyncio.run(exercise())
