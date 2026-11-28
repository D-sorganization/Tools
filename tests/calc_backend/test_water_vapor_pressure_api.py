"""Water vapor pressure REST/MCP contract with independent reference values."""

from __future__ import annotations

import math

import pytest
from calc_backend.app import app
from calc_backend.contracts.water_vapor_pressure import WaterVaporPressureRequest
from calc_backend.mcp_server import _run_calculator
from calc_backend.registry.store import describe_calculator
from fastapi.testclient import TestClient
from pydantic import ValidationError

from shared.python.sidekick.process_calculators.constants import (
    IAPWS_MAX_TEMPERATURE_C,
    IAPWS_MIN_TEMPERATURE_C,
)

CALCULATOR_ID = "water-vapor-pressure"
PATH = "/api/calc/water-vapor-pressure"
INPUTS = {"temperature_c": 20.0, "method": "iapws"}

# Published saturation-pressure values (IAPWS-IF97 steam tables), in Pa.
STEAM_TABLE_PA = {20.0: 2339.2, 100.0: 101418.0, 200.0: 1554900.0}


def test_discovery_schema_units_bounds_and_worked_example() -> None:
    """One typed contract supplies REST and MCP metadata."""
    description = describe_calculator(CALCULATOR_ID)
    operation = app.openapi()["paths"][PATH]["post"]
    request_ref = operation["requestBody"]["content"]["application/json"]["schema"]
    name = request_ref["$ref"].rsplit("/", 1)[-1]
    assert description["request_schema"] == app.openapi()["components"]["schemas"][name]
    fields = {field["name"]: field for field in description["request_fields"]}
    assert fields["temperature_c"]["units"] == "degC"
    assert fields["temperature_c"]["valid_range"] == {
        "minimum": IAPWS_MIN_TEMPERATURE_C,
        "maximum": IAPWS_MAX_TEMPERATURE_C,
    }
    output = description["response_schema"]["properties"]["pressure_pa"]
    assert output["description"].endswith("[Pa]")
    assert output["minimum"] == 0
    assert description["worked_example"]["request"] == INPUTS
    assert (
        f"POST {PATH}"
        in TestClient(app).get("/api/calc/endpoints").json()["calculators"]
    )


@pytest.mark.parametrize("temperature_c", sorted(STEAM_TABLE_PA))
def test_iapws_matches_published_steam_tables(temperature_c: float) -> None:
    inputs = {"temperature_c": temperature_c, "method": "iapws"}
    response = TestClient(app).post(PATH, json=inputs)
    assert response.status_code == 200, response.text
    assert response.json()["pressure_pa"] == pytest.approx(
        STEAM_TABLE_PA[temperature_c], rel=1e-3
    )
    result = _run_calculator(CALCULATOR_ID, inputs)
    assert result["status"] == "SUCCEEDED", result
    assert result["outputs"] == response.json()
    assert result["units"]["temperature_c"] == "degC"
    assert result["provenance"]["calculation_status"] == "verified-unapproved"


@pytest.mark.parametrize(
    "method,temperature_c,tolerance",
    [
        ("buck", 20.0, 1e-2),
        ("buck", 100.0, 1e-2),
        ("magnus", 20.0, 1e-2),
        # The Magnus fit drifts above ~60 degC (about 2.6 % at the boiling point).
        ("magnus", 100.0, 3e-2),
    ],
)
def test_empirical_methods_track_reference_within_stated_tolerance(
    method: str, temperature_c: float, tolerance: float
) -> None:
    response = TestClient(app).post(
        PATH, json={"temperature_c": temperature_c, "method": method}
    )
    assert response.status_code == 200, response.text
    assert response.json()["pressure_pa"] == pytest.approx(
        STEAM_TABLE_PA[temperature_c], rel=tolerance
    )


def test_published_bounds_match_iapws_if97_domain() -> None:
    """Schema envelope is the IAPWS saturation interval, not 0 C to 374 C.

    0.0 C is below the triple point and 374.0 C is above the critical point.
    Both must be schema-invalid so a client is not offered a temperature the
    solver then rejects.
    """
    description = describe_calculator(CALCULATOR_ID)
    fields = {field["name"]: field for field in description["request_fields"]}
    bounds = fields["temperature_c"]["valid_range"]
    assert bounds["minimum"] == pytest.approx(0.01)
    assert bounds["maximum"] == pytest.approx(373.946)
    assert bounds["minimum"] > 0.0
    assert bounds["maximum"] < 374.0


@pytest.mark.parametrize("temperature_c", [0.0, 374.0])
def test_schema_rejects_temperatures_outside_iapws(temperature_c: float) -> None:
    """The request model rejects the two endpoints the solver cannot evaluate."""
    with pytest.raises(ValidationError):
        WaterVaporPressureRequest(temperature_c=temperature_c, method="iapws")


@pytest.mark.parametrize("temperature_c", [0.01, 373.946])
def test_iapws_domain_endpoints_return_finite_pressure(temperature_c: float) -> None:
    """Triple-point and critical-point temperatures stay inside the contract."""
    response = TestClient(app).post(
        PATH, json={"temperature_c": temperature_c, "method": "iapws"}
    )
    assert response.status_code == 200, response.text
    pressure_pa = response.json()["pressure_pa"]
    assert math.isfinite(pressure_pa)
    assert pressure_pa > 0


@pytest.mark.parametrize(
    "inputs",
    [
        {"temperature_c": -1.0},
        {"temperature_c": 375.0},
        {"temperature_c": "NaN"},
        {"temperature_c": "Infinity"},
        {"temperature_c": 20.0, "method": "antoine"},
        {"temperature_c": 150.0, "method": "magnus"},
        {"temperature_c": 150.0, "method": "buck"},
        {"temperature_c": 20.0, "unexpected": 1},
        {},
    ],
)
def test_invalid_inputs_rejected_by_both_projections(inputs: dict[str, object]) -> None:
    assert TestClient(app).post(PATH, json=inputs).status_code == 422
    result = _run_calculator(CALCULATOR_ID, inputs)
    assert result["status"] == "FAILED"
    assert result["errors"]
