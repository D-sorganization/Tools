"""Unit tests for the pressure-drop public calculation facade.

``pressure_drop_api`` is the Qt-free user-facing API: flexible-unit inputs,
pipe-geometry/gas-flow resolution, fitting-list construction, and the syngas/
custom-gas convenience wrappers. Tests run the full engine end-to-end (no
mocks) plus the resolution helpers and their guards.
"""

from __future__ import annotations

import pytest
from sidekick.process_calculators.pressure_drop_calculator import (
    pressure_drop_api as api,
)
from sidekick.process_calculators.pressure_drop_calculator.models.pressure_drop_data_models import (  # noqa: E501
    GasComposition,
    PipeFitting,
)

# ---------------------------------------------------------------------------
# build_fitting_list
# ---------------------------------------------------------------------------


def test_build_fitting_list_none_is_empty() -> None:
    assert api.build_fitting_list(None) == []


def test_build_fitting_list_empty_is_empty() -> None:
    assert api.build_fitting_list([]) == []


def test_build_fitting_list_converts_dicts() -> None:
    fittings = api.build_fitting_list(
        [{"type": "90_elbow_std", "quantity": 2}, {"type": "custom", "k_factor": 1.25}]
    )
    assert all(isinstance(f, PipeFitting) for f in fittings)
    # First fitting takes its K from the standard table; second from k_factor.
    assert fittings[0].fitting_type == "90_elbow_std"
    assert fittings[0].quantity == 2
    assert fittings[0].k_factor == pytest.approx(0.75)
    assert fittings[1].k_factor == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# resolve_pipe_geometry
# ---------------------------------------------------------------------------


def test_resolve_pipe_geometry_with_explicit_diameter() -> None:
    diameter, roughness = api.resolve_pipe_geometry(
        None, None, 0.1, "Commercial Steel", None
    )
    assert diameter == pytest.approx(0.1)
    assert roughness > 0


def test_resolve_pipe_geometry_explicit_roughness_wins() -> None:
    _, roughness = api.resolve_pipe_geometry(None, None, 0.1, "Commercial Steel", 1e-3)
    assert roughness == pytest.approx(1e-3)


def test_resolve_pipe_geometry_from_size_and_schedule() -> None:
    diameter, roughness = api.resolve_pipe_geometry(
        "4", "40", None, "Commercial Steel", None
    )
    assert diameter > 0
    assert roughness > 0


def test_resolve_pipe_geometry_missing_inputs_raises() -> None:
    with pytest.raises(ValueError, match="Either provide pipe_diameter"):
        api.resolve_pipe_geometry(None, None, None, "Commercial Steel", None)


# ---------------------------------------------------------------------------
# resolve_gas_and_flow
# ---------------------------------------------------------------------------


def test_resolve_gas_and_flow_defaults_to_air() -> None:
    composition, mass_flow = api.resolve_gas_and_flow(
        flow_rate=1000.0,
        flow_unit="kg/h",
        gas_composition=None,
        temp_k=300.0,
        pressure_pa=1.0e5,
        compressibility_correction=True,
        standard_condition="STP",
    )
    assert isinstance(composition, GasComposition)
    assert "Air" in composition.components
    # 1000 kg/h -> kg/s
    assert mass_flow == pytest.approx(1000.0 / 3600.0, rel=1e-6)


def test_resolve_gas_and_flow_custom_composition_normalized() -> None:
    composition, mass_flow = api.resolve_gas_and_flow(
        flow_rate=1.0,
        flow_unit="kg/s",
        gas_composition={"H2": 1.0, "CO": 1.0},  # sums to 2 -> normalized to 0.5/0.5
        temp_k=500.0,
        pressure_pa=5.0e5,
        compressibility_correction=True,
        standard_condition="STP",
    )
    assert sum(composition.components.values()) == pytest.approx(1.0)
    assert mass_flow == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# calculate_pressure_drop — full engine round-trip
# ---------------------------------------------------------------------------


def _result_is_well_formed(result: dict) -> None:
    assert result["pressure_drop_pa"] >= 0
    assert result["outlet_pressure_pa"] > 0
    assert result["reynolds_number"] > 0
    assert result["flow_regime"] in {"laminar", "transitional", "turbulent"}


def test_calculate_pressure_drop_with_diameter() -> None:
    result = api.calculate_pressure_drop(
        pipe_diameter=0.1,
        pipe_length=100.0,
        flow_rate=1000.0,
        flow_unit="kg/h",
        pressure=5.0,
        pressure_unit="bar",
        temperature=300.0,
        temperature_unit="K",
        gas_composition={"CH4": 1.0},
    )
    _result_is_well_formed(result)


def test_calculate_pressure_drop_with_size_and_fittings() -> None:
    result = api.calculate_pressure_drop(
        pipe_size="4",
        pipe_schedule="40",
        pipe_length=50.0,
        flow_rate=500.0,
        flow_unit="kg/h",
        pressure=3.0,
        pressure_unit="bar",
        temperature=350.0,
        gas_composition={"H2": 0.5, "CO": 0.5},
        fittings=[{"type": "90_elbow_std", "quantity": 3}],
    )
    _result_is_well_formed(result)


def test_calculate_pressure_drop_custom_gas_wrapper() -> None:
    result = api.calculate_pressure_drop_custom_gas(
        pipe_diameter=0.08,
        pipe_length=40.0,
        gas_composition={"N2": 1.0},
        flow_rate=0.2,
        flow_unit="kg/s",
        pressure=2.0,
        temperature=320.0,
    )
    _result_is_well_formed(result)


def test_calculate_pressure_drop_syngas_wrapper() -> None:
    result = api.calculate_pressure_drop_syngas(
        pipe_size="6",
        pipe_schedule="40",
        pipe_length=80.0,
        flow_rate=2000.0,
        flow_unit="kg/h",
        pressure=10.0,
        temperature=500.0,
    )
    _result_is_well_formed(result)
