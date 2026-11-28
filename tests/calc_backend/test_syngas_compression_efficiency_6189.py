"""Mode-specific efficiency conversion and the H2S gamma default (issue #6189).

``SyngasCompressionRequest`` documents ``isentropic_efficiency``, but the REST
adapter passed it straight into ``CompressionStage.efficiency``, which the
engine reads as the *mode-native* efficiency (the polytropic exponent
``(gamma - 1) / (gamma * eta_p)`` from F-TL-010, and the isothermal divide).
``H2S`` has no entry in the engine's gamma table, so pure H2S silently
inherited the diatomic default.

Oracle note: every expected value below is a textbook ideal-gas relation
evaluated in this file, not a second call into the engine.
"""

from __future__ import annotations

import math

import pytest
from calc_backend.app import app
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit

PATH = "/api/calc/syngas-compression"
_R_J_PER_MOL_K = 8.314462618  # CODATA 2018 universal gas constant [J/(mol*K)]

# Reference stage from the issue: pressure ratio 4 at a 75 % isentropic
# efficiency, the point where the panel measured the +7.37 % shaft-power error.
_PRESSURE_RATIO = 4.0
_INLET_TEMPERATURE_K = 300.0
_ISENTROPIC_EFFICIENCY = 0.75
_FLOW_RATE_KMOL_H = 100.0
_GAMMA_DIATOMIC = 1.40  # N2 in the engine's compressor-sizing table

# H2S ideal-gas Cp/Cv near 298 K. Engineering ToolBox "Gases - Ratios of
# Specific Heat" tabulates hydrogen sulfide at 1.32; the NIST WebBook Shomate
# ideal-gas Cp(298.15 K) gives k = Cp / (Cp - R) = 1.32, and Air Liquide
# tabulates 1.337 at 15 degC and 1.013 bar.
_GAMMA_H2S = 1.32


def _payload(
    compression_type: str, composition: dict[str, float] | None = None
) -> dict[str, object]:
    return {
        "composition": composition or {"N2": 1.0},
        "inlet_pressure_bar": 1.0,
        "outlet_pressure_bar": _PRESSURE_RATIO,
        "inlet_temperature_k": _INLET_TEMPERATURE_K,
        "isentropic_efficiency": _ISENTROPIC_EFFICIENCY,
        "flow_rate_kmol_h": _FLOW_RATE_KMOL_H,
        "compression_type": compression_type,
    }


def _post(payload: dict[str, object]) -> dict[str, float]:
    response = TestClient(app).post(PATH, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _isentropic_exponent(gamma: float) -> float:
    return (gamma - 1.0) / gamma


def _polytropic_efficiency(gamma: float) -> float:
    """eta_p implied by eta_s through the shared adiabatic end state.

    eta_s = (PR^z - 1) / (PR^(z/eta_p) - 1) with z = (gamma - 1) / gamma,
    solved for eta_p.
    """
    exponent = _isentropic_exponent(gamma)
    numerator = exponent * math.log(_PRESSURE_RATIO)
    denominator = math.log(
        1.0 + (_PRESSURE_RATIO**exponent - 1.0) / _ISENTROPIC_EFFICIENCY
    )
    return numerator / denominator


def _adiabatic_reference(gamma: float) -> tuple[float, float]:
    """Shaft work [J/mol] and discharge temperature [K] of the real machine.

    An adiabatic compressor's actual work is the isentropic work divided by
    eta_s however the path is parameterised, so this is the reference for both
    the isentropic and the polytropic modes.
    """
    exponent = _isentropic_exponent(gamma)
    temperature_ratio = _PRESSURE_RATIO**exponent
    work = (
        (gamma / (gamma - 1.0) * _R_J_PER_MOL_K * _INLET_TEMPERATURE_K)
        * (temperature_ratio - 1.0)
        / _ISENTROPIC_EFFICIENCY
    )
    heat_capacity = _R_J_PER_MOL_K / exponent
    return work, _INLET_TEMPERATURE_K + work / heat_capacity


def test_polytropic_mode_converts_to_a_polytropic_efficiency() -> None:
    """Polytropic shaft work must be graded on eta_p, not the raw eta_s."""
    work, outlet_k = _adiabatic_reference(_GAMMA_DIATOMIC)
    body = _post(_payload("polytropic"))
    assert body["work_actual_j_per_mol"] == pytest.approx(work)
    assert body["outlet_temperature_k"] == pytest.approx(outlet_k)


def test_polytropic_mode_no_longer_over_predicts_power_by_seven_percent() -> None:
    """Pins the #6189 magnitude: feeding eta_s raw over-states power by 7.37 %."""
    body = _post(_payload("polytropic"))
    exponent = _isentropic_exponent(_GAMMA_DIATOMIC) / _ISENTROPIC_EFFICIENCY
    raw_work = (
        (_R_J_PER_MOL_K * _INLET_TEMPERATURE_K / exponent)
        * (_PRESSURE_RATIO**exponent - 1.0)
        / _ISENTROPIC_EFFICIENCY
    )
    assert raw_work / body["work_actual_j_per_mol"] - 1.0 == pytest.approx(
        0.0737, abs=2e-4
    )


def test_isothermal_mode_converts_to_a_polytropic_efficiency() -> None:
    """An intercooled train is graded against the isothermal ideal on eta_p."""
    work = (
        _R_J_PER_MOL_K
        * _INLET_TEMPERATURE_K
        * math.log(_PRESSURE_RATIO)
        / _polytropic_efficiency(_GAMMA_DIATOMIC)
    )
    body = _post(_payload("isothermal"))
    assert body["work_actual_j_per_mol"] == pytest.approx(work)
    assert body["outlet_temperature_k"] == pytest.approx(_INLET_TEMPERATURE_K)
    raw_work = (
        _R_J_PER_MOL_K * _INLET_TEMPERATURE_K * math.log(_PRESSURE_RATIO)
    ) / _ISENTROPIC_EFFICIENCY
    assert raw_work / work - 1.0 == pytest.approx(0.0572, abs=1e-3)


def test_isentropic_mode_still_uses_the_supplied_efficiency_directly() -> None:
    """Guard against converting an efficiency that is already the right one."""
    work, outlet_k = _adiabatic_reference(_GAMMA_DIATOMIC)
    body = _post(_payload("isentropic"))
    assert body["work_actual_j_per_mol"] == pytest.approx(work)
    assert body["outlet_temperature_k"] == pytest.approx(outlet_k)
    assert body["heat_capacity_ratio"] == pytest.approx(_GAMMA_DIATOMIC)


def test_unit_isentropic_efficiency_leaves_every_mode_unchanged() -> None:
    """At eta_s = 1 the conversion is the identity, so the modes agree."""
    isentropic = _payload("isentropic")
    isentropic["isentropic_efficiency"] = 1.0
    baseline = _post(isentropic)
    polytropic = _payload("polytropic")
    polytropic["isentropic_efficiency"] = 1.0
    assert _post(polytropic)["work_actual_j_per_mol"] == pytest.approx(
        baseline["work_actual_j_per_mol"]
    )


def test_pure_h2s_reports_the_cited_gamma_instead_of_the_default() -> None:
    """Pure H2S is triatomic: k = 1.32, not the 1.40 diatomic fallback."""
    body = _post(_payload("isentropic", {"H2S": 1.0}))
    assert body["heat_capacity_ratio"] == pytest.approx(_GAMMA_H2S)
    assert abs(body["heat_capacity_ratio"] - _GAMMA_DIATOMIC) > 0.05


def test_h2s_blends_from_component_heat_capacities() -> None:
    """Mixtures combine Cp/Cv, so a tabulated H2S changes the blend."""
    body = _post(_payload("isentropic", {"H2S": 0.5, "N2": 0.5}))
    cp_h2s = _GAMMA_H2S / (_GAMMA_H2S - 1.0) * _R_J_PER_MOL_K
    cp_n2 = _GAMMA_DIATOMIC / (_GAMMA_DIATOMIC - 1.0) * _R_J_PER_MOL_K
    cp_mix = 0.5 * cp_h2s + 0.5 * cp_n2
    expected = cp_mix / (cp_mix - _R_J_PER_MOL_K)
    assert body["heat_capacity_ratio"] == pytest.approx(expected, rel=1e-5)
