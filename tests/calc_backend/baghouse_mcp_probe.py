"""Stdio MCP acceptance probe for the registered baghouse calculator (#6019)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_CALC_ID = "baghouse"

_VALID_INPUTS: dict[str, Any] = {
    "gas_flow_kg_s": 5.0,
    "inlet_temp_k": 400.0,
    "pressure_pa": 101325.0,
    "composition": {"N2": 0.7, "CO": 0.2, "H2": 0.1},
    "ash_in_kg_hr": 10.0,
}

_INVALID_INPUTS: dict[str, Any] = {
    **_VALID_INPUTS,
    "gas_flow_kg_s": -2.0,
}


def _payload(response: Any) -> dict[str, Any]:
    assert not response.isError, response
    text = response.content[0].text
    data = json.loads(text)
    assert isinstance(data, dict), data
    return data


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

            catalog = _payload(await session.call_tool("list_calculators", {}))
            listed = {item["id"] for item in catalog["calculators"]}
            assert _CALC_ID in listed, sorted(listed)

            described = _payload(
                await session.call_tool("describe_calculator", {"id": _CALC_ID})
            )
            assert described.get("id") == _CALC_ID
            assert described.get("contract_version")
            assert described.get("request_schema")
            assert described.get("worked_example", {}).get("request")

            ok = _payload(
                await session.call_tool(
                    "run_calculation", {"id": _CALC_ID, "inputs": _VALID_INPUTS}
                )
            )
            assert ok.get("status") == "SUCCEEDED", ok
            provenance = ok.get("provenance") or {}
            assert provenance.get("calculator_id") == _CALC_ID
            assert provenance.get("contract_version")

            bad = _payload(
                await session.call_tool(
                    "run_calculation", {"id": _CALC_ID, "inputs": _INVALID_INPUTS}
                )
            )
            assert bad.get("status") == "FAILED", bad
            errors = bad.get("errors")
            assert isinstance(errors, list) and errors, bad
            first = errors[0]
            assert first.get("field")
            assert first.get("problem")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", type=Path)
    args = parser.parse_args()
    asyncio.run(exercise(args.repo_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
