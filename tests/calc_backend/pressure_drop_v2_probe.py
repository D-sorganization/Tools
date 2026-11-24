"""Real stdio MCP probe for the pressure-drop-v2 slice."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def exercise(root: Path) -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "shared.python.calc_backend.mcp_server"],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            listed = await session.call_tool(
                "list_calculators", {"query": "pressure-drop-v2"}
            )
            page = json.loads(listed.content[0].text)
            assert {item["id"] for item in page["calculators"]} == {
                "pressure-drop-v2",
                "pressure-drop-v2-validate",
            }
            described = await session.call_tool(
                "describe_calculator", {"id": "pressure-drop-v2"}
            )
            description = json.loads(described.content[0].text)
            inputs = description["worked_example"]["request"]
            for diameter, status in ((0.1, "SUCCEEDED"), (0.0, "FAILED")):
                response = await session.call_tool(
                    "run_calculation",
                    {
                        "id": "pressure-drop-v2",
                        "inputs": {**inputs, "pipe_diameter_m": diameter},
                    },
                )
                assert not response.isError
                result = json.loads(response.content[0].text)
                assert result["status"] == status, result
                if status == "SUCCEEDED":
                    assert result["outputs"]["data"]["pressure_drop_pa"] > 0
                    assert (
                        result["provenance"]["calculation_status"]
                        == "verified-unapproved"
                    )
                else:
                    assert result["errors"][0]["field"] == "pipe_diameter_m"


if __name__ == "__main__":
    asyncio.run(exercise(Path(sys.argv[1])))
