"""Stdio MCP probe for the registered ``thermal-profile`` calculator (#6019)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

THERMAL_ID = "thermal-profile"

_VALID_INPUTS = {
    "thermal_mass_j_per_k": 1_000_000.0,
    "heat_loss_coeff_w_per_k": 50.0,
    "initial_temp_c": 25.0,
    "ambient_temp_c": 20.0,
    "power_w": 50_000.0,
    "t_end_s": 3600.0,
}

_INVALID_INPUTS = {
    **_VALID_INPUTS,
    "t_start_s": 100.0,
    "t_end_s": 50.0,
}


def _json_payload(response: object) -> dict[str, object]:
    assert getattr(response, "isError", True) is False, response
    text = response.content[0].text  # type: ignore[union-attr]
    payload = json.loads(text)
    assert isinstance(payload, dict)
    return payload


async def exercise(repo_root: Path) -> None:
    """list_calculators, describe_calculator, and a successful run_calculation."""
    src = repo_root / "src"
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "shared.python.calc_backend.mcp_server"],
        env={**os.environ, "PYTHONPATH": str(src)},
    )
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert {
                "list_calculators",
                "describe_calculator",
                "run_calculation",
            } <= names

            catalog = _json_payload(await session.call_tool("list_calculators", {}))
            calc_ids = {str(item["id"]) for item in catalog.get("calculators", [])}
            assert THERMAL_ID in calc_ids, sorted(calc_ids)

            described = _json_payload(
                await session.call_tool("describe_calculator", {"id": THERMAL_ID})
            )
            assert described.get("id") == THERMAL_ID
            assert described.get("contract_version")
            assert described.get("request_schema")
            assert described.get("worked_example", {}).get("request")

            ok = _json_payload(
                await session.call_tool(
                    "run_calculation", {"id": THERMAL_ID, "inputs": _VALID_INPUTS}
                )
            )
            assert ok.get("status") == "SUCCEEDED", ok
            outputs = ok.get("outputs") or {}
            assert outputs.get("final_temp_c") is not None


async def exercise_fail_closed(repo_root: Path) -> None:
    """Invalid time span must yield a structured FAILED envelope over stdio MCP."""
    src = repo_root / "src"
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "shared.python.calc_backend.mcp_server"],
        env={**os.environ, "PYTHONPATH": str(src)},
    )
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            response = await session.call_tool(
                "run_calculation",
                {"id": THERMAL_ID, "inputs": _INVALID_INPUTS},
            )
            if response.isError:
                detail = response.content[0].text if response.content else ""
                raise AssertionError(
                    "run_calculation must not surface a transport error: " + detail
                )
            fail = _json_payload(response)
            assert fail.get("status") == "FAILED", fail
            errors = fail.get("errors")
            assert isinstance(errors, list) and errors
            first = errors[0]
            assert first.get("field")
            assert first.get("problem")


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if not args:
        raise SystemExit(
            "usage: thermal_profile_mcp_probe.py <repo_root> [fail_closed]"
        )
    mode = args[1] if len(args) > 1 else "happy"
    repo_root = Path(args[0])
    if mode == "fail_closed":
        asyncio.run(exercise_fail_closed(repo_root))
    else:
        asyncio.run(exercise(repo_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
