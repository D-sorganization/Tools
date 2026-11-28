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
# CODATA 2018 R and the engine's stated horsepower conversion.
_R_J_PER_MOL_K = 8.314462618
_WATTS_PER_HP = 745.7
_GAMMA = {"H2": 1.41, "N2": 1.40}
_BASE = {
    "composition": {"H2": 1.0},
    "inlet_pressure_bar": 1.0,
    "outlet_pressure_bar": 3.0,
    "inlet_temperature_k": 300.0,
    "isentropic_efficiency": 0.85,
    "flow_rate_kmol_h": 100.0,
    "compression_type": "isentropic",
}


def _power_hp(work_j_per_mol: float) -> float:
    mol_per_s = _BASE["flow_rate_kmol_h"] * 1000.0 / 3600.0
    return mol_per_s * work_j_per_mol / _WATTS_PER_HP


def _polytropic_efficiency(
    gamma: float, pressure_ratio: float, isentropic_efficiency: float
) -> float:
    """eta_p equivalent to a nameplate eta_s.

    The engine's polytropic and isothermal branches are graded on the machine's
    own small-stage efficiency, so this oracle converts instead of reusing
    eta_s (issue #6189).
    """
    exponent = (gamma - 1.0) / gamma
    return (
        exponent
        * math.log(pressure_ratio)
        / math.log(1.0 + (pressure_ratio**exponent - 1.0) / isentropic_efficiency)
    )


def _reference(kind: str, gamma: float) -> tuple[float, float]:
    """Textbook shaft work [J/mol] and discharge temperature [K]."""
    temperature_k = float(_BASE["inlet_temperature_k"])
    pressure_ratio = 3.0
    efficiency = float(_BASE["isentropic_efficiency"])
    if kind != "isentropic":
        efficiency = _polytropic_efficiency(gamma, pressure_ratio, efficiency)
    if kind == "isothermal":
        work = _R_J_PER_MOL_K * temperature_k * math.log(pressure_ratio) / efficiency
        return work, temperature_k
    if kind == "polytropic":
        exponent = (gamma - 1.0) / (gamma * efficiency)
    else:
        exponent = (gamma - 1.0) / gamma
    temperature_ratio = pressure_ratio**exponent
    if kind == "polytropic":
        work = (
            _R_J_PER_MOL_K * temperature_k / exponent * (temperature_ratio - 1.0)
        ) / efficiency
        return work, temperature_k * temperature_ratio
    work_s = (gamma / (gamma - 1.0)) * _R_J_PER_MOL_K * temperature_k
    work = work_s * (temperature_ratio - 1.0) / efficiency
    heat_capacity = _R_J_PER_MOL_K * gamma / (gamma - 1.0)
    return work, temperature_k + work / heat_capacity


def _post(payload: dict[str, object]) -> object:
    return TestClient(app).post(PATH, json=payload)


def test_discovery_schema_units_bounds_and_worked_example() -> None:
    """One typed contract supplies REST and MCP metadata."""
    description = describe_calculator(CALCULATOR_ID)
    operation = app.openapi()["paths"][PATH]["post"]
    request_ref = operation["requestBody"]["content"]["application/json"]["schema"]
    name = request_ref["$ref"].rsplit("/", 1)[-1]
    schemas = app.openapi()["components"]["schemas"]
    assert description["request_schema"] == schemas[name]
    fields = {field["name"]: field for field in description["request_fields"]}
    assert fields["inlet_pressure_bar"]["units"] == "bar"
    assert fields["inlet_pressure_bar"]["valid_range"]["exclusiveMinimum"] == 0.0
    assert fields["inlet_temperature_k"]["units"] == "K"
    assert fields["flow_rate_kmol_h"]["units"] == "kmol/h"
    assert fields["isentropic_efficiency"]["valid_range"]["maximum"] == 1.0
    power = description["response_schema"]["properties"]["power_hp"]
    assert power["description"].endswith("[hp]")
    assert description["worked_example"]["request"] == _BASE
    listed = TestClient(app).get("/api/calc/endpoints").json()["calculators"]
    assert f"POST {PATH}" in listed


@pytest.mark.parametrize(
    ("kind", "species"),
    [("isentropic", "H2"), ("isothermal", "N2"), ("polytropic", "H2")],
)
def test_stage_matches_ideal_gas_relations(kind: str, species: str) -> None:
    payload = dict(_BASE)
    payload["composition"] = {species: 1.0}
    payload["compression_type"] = kind
    work, outlet_k = _reference(kind, _GAMMA[species])
    response = _post(payload)
    assert response.status_code == 200, response.text
    body = response.json()
    inlet_k = float(_BASE["inlet_temperature_k"])
    assert body["work_actual_j_per_mol"] == pytest.approx(work)
    assert body["outlet_temperature_k"] == pytest.approx(outlet_k)
    assert body["heat_rise_k"] == pytest.approx(outlet_k - inlet_k)
    assert body["power_hp"] == pytest.approx(_power_hp(work))
    assert body["pressure_ratio"] == pytest.approx(3.0)
    if kind == "isentropic":
        assert body["heat_capacity_ratio"] == pytest.approx(_GAMMA[species])
        result = _run_calculator(CALCULATOR_ID, payload)
        assert result["status"] == "SUCCEEDED", result
        assert result["outputs"] == body
        assert result["units"]["flow_rate_kmol_h"] == "kmol/h"
        assert result["provenance"]["calculation_status"] == "verified-unapproved"


def test_percentage_composition_normalizes_to_the_same_power() -> None:
    percent = dict(_BASE)
    percent["composition"] = {"H2": 100.0}
    mole = _post(_BASE)
    scaled = _post(percent)
    assert mole.status_code == 200 and scaled.status_code == 200
    assert scaled.json()["power_hp"] == pytest.approx(mole.json()["power_hp"])


@pytest.mark.parametrize(
    "override",
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
    override: dict[str, object],
) -> None:
    payload = {} if override == {} else {**_BASE, **override}
    assert _post(payload).status_code == 422
    result = _run_calculator(CALCULATOR_ID, payload)
    assert result["status"] == "FAILED"
    assert result["errors"]


def test_unknown_species_fail_closed() -> None:
    payload = dict(_BASE)
    payload["composition"] = {"Xe": 1.0}
    response = _post(payload)
    assert response.status_code == 422
    assert "unknown species" in response.text
    result = _run_calculator(CALCULATOR_ID, payload)
    assert result["status"] == "FAILED"
    errors = result["errors"]
    assert isinstance(errors, list)
    text = " ".join(
        str(err.get("problem") or err.get("message") or "") for err in errors
    )
    assert "unknown species" in text
