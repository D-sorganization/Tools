"""Pressure-drop-v2 REST/MCP calculator slice (Refs #6019)."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from shared.python.calc_backend.app import app
from shared.python.calc_backend.mcp_server import _list_page, _run_calculator
from shared.python.calc_backend.registry.store import describe_calculator

CALCULATOR_ID = "pressure-drop-v2"
PATH = "/api/calc/pressure-drop-v2"
# A prescribed viscosity and low mass flow give laminar flow, f = 64 / Re.
INPUTS = {
    "pipe_diameter_m": 0.1,
    "pipe_length_m": 10.0,
    "flow_rate_kg_s": 0.001,
    "temperature_k": 300.0,
    "pressure_pa": 101325.0,
    "molecular_weight_kg_mol": 0.029,
    "viscosity_pa_s": 1.8e-5,
}


def test_v2_is_discoverable_with_typed_rest_schema() -> None:
    page = _list_page(category="process", query=CALCULATOR_ID, limit=50, cursor=None)
    assert {item["id"] for item in page["calculators"]} == {
        CALCULATOR_ID,
        "pressure-drop-v2-validate",
    }
    description = describe_calculator(CALCULATOR_ID)
    assert description["http_path"] == PATH
    fields = {field["name"]: field for field in description["request_fields"]}
    assert fields["molecular_weight_kg_mol"]["units"] == "kg/mol"
    assert fields["pipe_diameter_m"]["valid_range"]["exclusiveMinimum"] == 0
    schema = app.openapi()
    post = schema["paths"][PATH]["post"]
    for key, location in (
        ("request_schema", post["requestBody"]),
        ("response_schema", post["responses"]["200"]),
    ):
        name = location["content"]["application/json"]["schema"]["$ref"].split("/")[-1]
        assert description[key] == schema["components"]["schemas"][name]


def test_v2_mcp_matches_rest_and_laminar_pressure_drop() -> None:
    response = TestClient(app).post(PATH, json=INPUTS)
    assert response.status_code == 200
    rest = response.json()
    result = _run_calculator(CALCULATOR_ID, INPUTS)
    assert result["status"] == "SUCCEEDED"
    assert result["outputs"]["status"] == "success"
    assert result["outputs"]["data"] == rest["data"]
    data = rest["data"]
    assert data["reynolds_number"] < 2300
    assert data["friction_factor"] == pytest.approx(64 / data["reynolds_number"])
    # Hagen-Poiseuille expressed in mass flow, using the returned gas density.
    expected = (
        128
        * INPUTS["viscosity_pa_s"]
        * INPUTS["pipe_length_m"]
        * INPUTS["flow_rate_kg_s"]
        / (math.pi * data["density_kg_m3"] * INPUTS["pipe_diameter_m"] ** 4)
    )
    assert data["pressure_drop_pa"] == pytest.approx(expected)
    assert result["units"]["molecular_weight_kg_mol"] == "kg/mol"
    assert result["provenance"]["calculation_status"] == "verified-unapproved"


@pytest.mark.parametrize(
    "field", ["pipe_diameter_m", "flow_rate_kg_s", "viscosity_pa_s"]
)
def test_v2_rejects_nonpositive_inputs(field: str) -> None:
    inputs = {**INPUTS, field: 0.0}
    assert TestClient(app).post(PATH, json=inputs).status_code == 422
    result = _run_calculator(CALCULATOR_ID, inputs)
    assert result["status"] == "FAILED"
    assert result["errors"][0]["field"] == field
    assert "outputs" not in result


def test_v2_validation_is_callable_without_running_calculation() -> None:
    result = _run_calculator("pressure-drop-v2-validate", INPUTS)
    assert result["status"] == "SUCCEEDED"
    assert result["outputs"]["data"]["valid"] is True


def test_v2_real_stdio_transport() -> None:
    pytest.importorskip("mcp")
    root = Path(__file__).resolve().parents[2]
    probe = Path(__file__).with_name("pressure_drop_v2_probe.py")
    result = subprocess.run(
        [sys.executable, str(probe), str(root)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
