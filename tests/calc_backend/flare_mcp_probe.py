"""Stdio MCP transport probe for the registered ``flare`` calculator (issue #6019)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

FLARE_ID = "flare"

_FLARE_WORKED_REQUEST = {
    "total_flow_kg_hr": 5000.0,
    "gas_composition": {"H2": 30, "CO": 40, "CH4": 10, "N2": 20},
    "temperature_k": 400.0,
    "pressure_bar": 1.5,
}


async def exercise(repo_root: Path) -> None:
    src = repo_root / "src"
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "shared.python.calc_backend.mcp_server"],
        env={**os.environ, "PYTHONPATH": str(src)},
    )
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            assert {
                "list_calculators",
                "describe_calculator",
                "run_calculation",
            } <= names

            catalog = await session.call_tool("list_calculators", {})
            assert not catalog.isError
            catalog_payload = json.loads(catalog.content[0].text)
            calc_ids = {item["id"] for item in catalog_payload.get("calculators", [])}
            assert FLARE_ID in calc_ids, sorted(calc_ids)

            described = await session.call_tool("describe_calculator", {"id": FLARE_ID})
            assert not described.isError
            describe_payload = json.loads(described.content[0].text)
            assert describe_payload.get("id") == FLARE_ID
            assert describe_payload.get("request_schema")
            assert describe_payload.get("worked_example", {}).get("request")

            success = await session.call_tool(
                "run_calculation",
                {"id": FLARE_ID, "inputs": dict(_FLARE_WORKED_REQUEST)},
            )
            assert not success.isError
            ok_payload = json.loads(success.content[0].text)
            assert ok_payload.get("status") == "SUCCEEDED", ok_payload
            assert ok_payload.get("outputs", {}).get("design")

            invalid = await session.call_tool(
                "run_calculation",
                {
                    "id": FLARE_ID,
                    "inputs": {
                        **_FLARE_WORKED_REQUEST,
                        "pressure_bar": -1.0,
                    },
                },
            )
            assert not invalid.isError
            fail_payload = json.loads(invalid.content[0].text)
            assert fail_payload.get("status") == "FAILED", fail_payload
            assert fail_payload.get("errors")


if __name__ == "__main__":
    asyncio.run(exercise(Path(sys.argv[1])))
