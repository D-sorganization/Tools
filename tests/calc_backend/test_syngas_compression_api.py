"""Single-stage syngas compression REST/MCP contract (epic #6019)."""

from __future__ import annotations

import math

import pytest
from calc_backend.app import app
from calc_backend.mcp_server import _run_calculator
from calc_backend.registry.store import describe_calculator
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit

CALCULATOR_ID = "syngas-compression"
PATH = "/api/calc/syngas-compression"
# CODATA 2018 gas constant and the calculator's stated horsepower conversion.
_R_J_PER_MOL_K = 8.314462618
_WATTS_PER_HP = 745.7
_MOL_PER_KMOL = 1000.0
_SECONDS_PER_HOUR = 3600.0
# Approximate pure-gas gamma values stated by SyngasCompressionEngine.
_GAMMA = {"H2": 1.41, "N2": 1.40}

H2_ISENTROPIC = {
    "composition": {"H2": 1.0},
    "inlet_pressure_bar": 1.0,
    "outlet_pressure_bar": 3.0,
    "inlet_temperature_k": 300.0,
    "isentropic_efficiency": 0.85,
    "flow_rate_kmol_h": 100.0,
    "compression_type": "isentropic",
}


def _power_hp(flow_kmol_h: float, work_j_per_mol: float) -> float:
    mol_per_s = flow_kmol_h * _MOL_PER_KMOL / _SECONDS_PER_HOUR
    return mol_per_s * work_j_per_mol / _WATTS_PER_HP


def _ideal_isentropic(
    gamma: float, temperature_k: float, pressure_ratio: float, efficiency: float
) -> tuple[float, float]:
    """Ideal-gas isentropic shaft work [J/mol] and actual discharge temperature."""
    exponent = (gamma - 1.0) / gamma
    temperature_ratio = pressure_ratio**exponent
    work_isentropic = (
        (gamma / (gamma - 1.0))
        * _R_J_PER_MOL_K
        * temperature_k
        * (temperature_ratio - 1.0)
    )
    work_actual = work_isentropic / efficiency
    heat_capacity = _R_J_PER_MOL_K * gamma / (gamma - 1.0)
    return work_actual, temperature_k + work_actual / heat_capacity


def _ideal_isothermal(
    temperature_k: float, pressure_ratio: float, efficiency: float
) -> float:
    """Ideal-gas isothermal shaft work [J/mol], W = RT ln(Pr) / efficiency."""
    return _R_J_PER_MOL_K * temperature_k * math.log(pressure_ratio) / efficiency


def _ideal_polytropic(
    gamma: float, temperature_k: float, pressure_ratio: float, efficiency: float
) -> tuple[float, float]:
    """Polytropic work using n-1/n = (gamma-1)/(gamma*efficiency), then /efficiency."""
    exponent = (gamma - 1.0) / (gamma * efficiency)
    temperature_ratio = pressure_ratio**exponent
    work_actual = (
        (_R_J_PER_MOL_K * temperature_k / exponent)
        * (temperature_ratio - 1.0)
        / efficiency
    )
    return work_actual, temperature_k * temperature_ratio


def _problem_text(result: dict[str, object]) -> str:
    errors = result["errors"]
    assert isinstance(errors, list)
    parts: list[str] = []
    for err in errors:
        assert isinstance(err, dict)
        parts.append(str(err.get("problem") or err.get("message") or ""))
    return " ".join(parts)


def test_discovery_schema_units_bounds_and_worked_example() -> None:
    """One typed contract supplies REST and MCP metadata."""
    description = describe_calculator(CALCULATOR_ID)
    operation = app.openapi()["paths"][PATH]["post"]
    request_ref = operation["requestBody"]["content"]["application/json"]["schema"]
    name = request_ref["$ref"].rsplit("/", 1)[-1]
    assert description["request_schema"] == app.openapi()["components"]["schemas"][name]
    fields = {field["name"]: field for field in description["request_fields"]}
    assert fields["inlet_pressure_bar"]["units"] == "bar"
    assert fields["inlet_pressure_bar"]["valid_range"]["exclusiveMinimum"] == 0.0
    assert fields["inlet_temperature_k"]["units"] == "K"
    assert fields["flow_rate_kmol_h"]["units"] == "kmol/h"
    assert fields["isentropic_efficiency"]["valid_range"]["maximum"] == 1.0
    output = description["response_schema"]["properties"]["power_hp"]
    assert output["description"].endswith("[hp]")
    assert description["worked_example"]["request"] == H2_ISENTROPIC
    listed = TestClient(app).get("/api/calc/endpoints").json()["calculators"]
    assert f"POST {PATH}" in listed


def test_pure_hydrogen_isentropic_matches_ideal_gas_relations() -> None:
    work, outlet_k = _ideal_isentropic(
        _GAMMA["H2"], 300.0, 3.0, H2_ISENTROPIC["isentropic_efficiency"]
    )
    response = TestClient(app).post(PATH, json=H2_ISENTROPIC)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pressure_ratio"] == pytest.approx(3.0)
    assert body["heat_capacity_ratio"] == pytest.approx(_GAMMA["H2"])
    assert body["work_actual_j_per_mol"] == pytest.approx(work)
    assert body["outlet_temperature_k"] == pytest.approx(outlet_k)
    assert body["heat_rise_k"] == pytest.approx(outlet_k - 300.0)
    assert body["power_hp"] == pytest.approx(_power_hp(100.0, work))
    result = _run_calculator(CALCULATOR_ID, H2_ISENTROPIC)
    assert result["status"] == "SUCCEEDED", result
    assert result["outputs"] == body
    assert result["units"]["flow_rate_kmol_h"] == "kmol/h"
    assert result["provenance"]["calculation_status"] == "verified-unapproved"


def test_percentage_composition_normalizes_to_the_same_result() -> None:
    percent = dict(H2_ISENTROPIC)
    percent["composition"] = {"H2": 100.0}
    mole = TestClient(app).post(PATH, json=H2_ISENTROPIC)
    scaled = TestClient(app).post(PATH, json=percent)
    assert mole.status_code == 200 and scaled.status_code == 200
    assert scaled.json()["power_hp"] == pytest.approx(mole.json()["power_hp"])


def test_isothermal_and_polytropic_match_stated_relations() -> None:
    isothermal = dict(H2_ISENTROPIC)
    isothermal["composition"] = {"N2": 1.0}
    isothermal["compression_type"] = "isothermal"
    work = _ideal_isothermal(300.0, 3.0, 0.85)
    response = TestClient(app).post(PATH, json=isothermal)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["work_actual_j_per_mol"] == pytest.approx(work)
    assert body["outlet_temperature_k"] == pytest.approx(300.0)
    assert body["heat_rise_k"] == pytest.approx(0.0)
    assert body["power_hp"] == pytest.approx(_power_hp(100.0, work))

    polytropic = dict(H2_ISENTROPIC)
    polytropic["compression_type"] = "polytropic"
    poly_work, poly_out = _ideal_polytropic(_GAMMA["H2"], 300.0, 3.0, 0.85)
    poly_response = TestClient(app).post(PATH, json=polytropic)
    assert poly_response.status_code == 200, poly_response.text
    poly_body = poly_response.json()
    assert poly_body["work_actual_j_per_mol"] == pytest.approx(poly_work)
    assert poly_body["outlet_temperature_k"] == pytest.approx(poly_out)
    assert poly_body["heat_capacity_ratio"] == pytest.approx(_GAMMA["H2"])


@pytest.mark.parametrize(
    "inputs",
    [
        {"inlet_pressure_bar": 0.0},
        {"outlet_pressure_bar": 1.0},
        {"inlet_temperature_k": 0.0},
        {"inlet_temperature_k": "NaN"},
        {"isentropic_efficiency": 0.0},
        {"isentropic_efficiency": 1.1},
        {"flow_rate_kmol_h": 0.0},
        {"compression_type": "adiabatic"},
        {"composition": {}},
        {"composition": {"H2": -0.1}},
        {"unexpected": 1},
        {},
    ],
)
def test_invalid_inputs_rejected_by_both_projections(
    inputs: dict[str, object],
) -> None:
    payload = dict(H2_ISENTROPIC)
    payload.update(inputs)
    if inputs == {}:
        payload = {}
    assert TestClient(app).post(PATH, json=payload).status_code == 422
    result = _run_calculator(CALCULATOR_ID, payload)
    assert result["status"] == "FAILED"
    assert result["errors"]


def test_unknown_species_fail_closed() -> None:
    payload = dict(H2_ISENTROPIC)
    payload["composition"] = {"Xe": 1.0}
    response = TestClient(app).post(PATH, json=payload)
    assert response.status_code == 422
    assert "unknown species" in response.text
    result = _run_calculator(CALCULATOR_ID, payload)
    assert result["status"] == "FAILED"
    assert "unknown species" in _problem_text(result)
