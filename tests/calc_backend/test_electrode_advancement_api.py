"""Electrode consumption REST/MCP contract and independent worked values."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from calc_backend.app import app
from calc_backend.mcp_server import _run_calculator
from calc_backend.registry.store import describe_calculator
from fastapi.testclient import TestClient

CALCULATOR_ID = "electrode-advancement"
PATH = "/api/calc/electrode-advancement"
INPUTS = {"consumption_rate_in_per_kah": 0.5, "current_ka": 20.0, "time_hrs": 2.0}


def test_discovery_schema_units_bounds_and_worked_example() -> None:
    """One typed contract supplies REST and MCP metadata."""
    description = describe_calculator(CALCULATOR_ID)
    openapi = app.openapi()
    operation = openapi["paths"][PATH]["post"]
    for key, schema in (
        (
            "request_schema",
            operation["requestBody"]["content"]["application/json"]["schema"],
        ),
        (
            "response_schema",
            operation["responses"]["200"]["content"]["application/json"]["schema"],
        ),
    ):
        name = schema["$ref"].rsplit("/", 1)[-1]
        assert description[key] == openapi["components"]["schemas"][name]
    fields = {field["name"]: field for field in description["request_fields"]}
    assert fields["current_ka"]["units"] == "kA"
    assert fields["time_hrs"]["units"] == "h"
    assert fields["consumption_rate_in_per_kah"]["units"] == "in/(kA*h)"
    assert fields["current_ka"]["valid_range"] == {"minimum": 0.0}
    assert fields["time_hrs"]["valid_range"] == {"minimum": 0.0}
    assert fields["consumption_rate_in_per_kah"]["valid_range"] == {
        "exclusiveMinimum": 0.0
    }
    example = description["worked_example"]
    output = description["response_schema"]["properties"]["consumption_in"]
    assert output["description"].endswith("[in]")
    assert output["minimum"] == 0
    assert example["request"] == INPUTS
    assert example["response_summary"] == {"consumption_in": 20.0}
    assert (
        f"POST {PATH}"
        in TestClient(app).get("/api/calc/endpoints").json()["calculators"]
    )


@pytest.mark.parametrize(
    "current,time,expected", [(20, 2, 20), (0, 2, 0), (20, 0, 0), (3, 0.5, 0.75)]
)
def test_consumption_matches_independent_values(
    current: float, time: float, expected: float
) -> None:
    inputs = {**INPUTS, "current_ka": current, "time_hrs": time}
    response = TestClient(app).post(PATH, json=inputs)
    assert response.status_code == 200, response.text
    assert response.json() == {"consumption_in": expected}
    result = _run_calculator(CALCULATOR_ID, inputs)
    assert result["status"] == "SUCCEEDED", result
    assert result["outputs"] == response.json()
    assert result["units"]["current_ka"] == "kA"
    assert result["provenance"]["calculation_status"] == "verified-unapproved"


@pytest.mark.parametrize(
    "field,value",
    [
        ("current_ka", -1),
        ("time_hrs", -1),
        ("consumption_rate_in_per_kah", 0),
        ("current_ka", "NaN"),
        ("time_hrs", "Infinity"),
        ("consumption_rate_in_per_kah", "-Infinity"),
        ("unexpected", 1),
        ("consumption_rate_in_per_kah", 1e308),
    ],
)
def test_invalid_inputs_rejected_by_both_projections(field: str, value: object) -> None:
    inputs = {**INPUTS, field: value}
    assert TestClient(app).post(PATH, json=inputs).status_code == 422
    result = _run_calculator(CALCULATOR_ID, inputs)
    assert result["status"] == "FAILED"
    assert result["errors"]


def test_real_stdio_mcp_discovery_describe_and_run(repo_root: Path) -> None:
    """The real SDK must discover and execute the new registry entry."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def exercise() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "shared.python.calc_backend.mcp_server"],
            env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        )
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

                async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
                    response = await session.call_tool(name, arguments)
                    assert not response.isError
                    return json.loads(response.content[0].text)

                listed = await call("list_calculators", {"query": CALCULATOR_ID})
                assert [item["id"] for item in listed["calculators"]] == [CALCULATOR_ID]
                described = await call("describe_calculator", {"id": CALCULATOR_ID})
                assert (
                    described["request_schema"]
                    == describe_calculator(CALCULATOR_ID)["request_schema"]
                )
                result = await call(
                    "run_calculation", {"id": CALCULATOR_ID, "inputs": INPUTS}
                )
                assert result["status"] == "SUCCEEDED"
                assert result["outputs"] == {"consumption_in": 20.0}
                bad = await call(
                    "run_calculation",
                    {"id": CALCULATOR_ID, "inputs": {**INPUTS, "current_ka": -1}},
                )
                assert bad["status"] == "FAILED"

    asyncio.run(exercise())
