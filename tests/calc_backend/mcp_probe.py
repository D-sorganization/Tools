"""Real SDK transport probe for ud-tools-mcp (issue #6021)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


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
                "convert_units",
            } <= names

            for calc_id, inputs in (
                (
                    "flow-rate",
                    {
                        "value": 10.0,
                        "from_unit": "kg/h",
                        "to_unit": "kg/s",
                        "category": "mass",
                    },
                ),
                (
                    "financial",
                    {"plant_capacity_tpd": 10.0, "product_price_per_ton": 50.0},
                ),
                (
                    "acid-gas-dewpoint",
                    {"temperature_c": 60.0, "pressure_bar": 1.0, "method": "antoine"},
                ),
            ):
                response = await session.call_tool(
                    "run_calculation", {"id": calc_id, "inputs": inputs}
                )
                assert not response.isError
                payload = json.loads(response.content[0].text)
                assert payload.get("status") == "SUCCEEDED", payload

            bad = await session.call_tool(
                "run_calculation",
                {
                    "id": "acid-gas-dewpoint",
                    "inputs": {"temperature_c": 60.0, "pressure_bar": -1.0},
                },
            )
            assert not bad.isError
            err_payload = json.loads(bad.content[0].text)
            assert err_payload.get("status") == "FAILED"
            assert err_payload.get("errors")


if __name__ == "__main__":
    asyncio.run(exercise(Path(sys.argv[1])))
