"""Unit tests for the legacy ``PressureDropCalculator`` API.

``_legacy.py`` retains the original single-class calculator for backwards
compatibility. It is pure-logic (ideal-gas density, Swamee-Jain friction,
Darcy-Weisbach) with explicit type/range guards, so tests cover each flow
regime, the viscosity fallback, and every precondition branch.
"""

from __future__ import annotations

import pytest
from sidekick.process_calculators.pressure_drop_calculator._legacy import (
    PIPE_DIMENSIONS_SCH40,
    ROUGHNESS_VALUES,
    PressureDropCalculator,
    PressureDropResult,
)


@pytest.fixture
def calc() -> PressureDropCalculator:
    return PressureDropCalculator()


def _base_kwargs(**overrides):
    kwargs = {
        "pipe_diameter_m": 0.1,
        "pipe_length_m": 100.0,
        "roughness_m": 4.5e-5,
        "flow_rate_kg_s": 1.0,
        "temperature_k": 300.0,
        "pressure_pa": 5.0e5,
        "molecular_weight_kg_mol": 0.016,  # CH4
        "viscosity_pa_s": 1.1e-5,
    }
    kwargs.update(overrides)
    return kwargs


def test_turbulent_result_is_well_formed(calc: PressureDropCalculator) -> None:
    result = calc.calculate_pressure_drop(**_base_kwargs())
    assert isinstance(result, PressureDropResult)
    assert result.pressure_drop_pa > 0
    assert result.reynolds_number > 4000
    assert result.flow_regime == "Turbulent"
    assert result.density > 0


def test_laminar_regime(calc: PressureDropCalculator) -> None:
    # Very low flow + high viscosity drives Re < 2300.
    result = calc.calculate_pressure_drop(
        **_base_kwargs(flow_rate_kg_s=1e-3, viscosity_pa_s=1e-2)
    )
    assert result.flow_regime == "Laminar"
    assert result.friction_factor == pytest.approx(64.0 / result.reynolds_number)


def test_transitional_regime(calc: PressureDropCalculator) -> None:
    # Tune viscosity so 2300 < Re < 4000.
    result = calc.calculate_pressure_drop(
        **_base_kwargs(flow_rate_kg_s=0.05, viscosity_pa_s=2.0e-4)
    )
    assert 2300 <= result.reynolds_number < 4000
    assert result.flow_regime == "Transitional"
    assert result.friction_factor == pytest.approx(0.03)


def test_sutherland_viscosity_fallback_used(calc: PressureDropCalculator) -> None:
    kwargs = _base_kwargs()
    kwargs.pop("viscosity_pa_s")
    result = calc.calculate_pressure_drop(**kwargs)
    # Sutherland (air) at 300 K is near 1.85e-5 Pa·s.
    assert result.viscosity == pytest.approx(1.85e-5, rel=0.1)


def test_zero_flow_is_degenerate(calc: PressureDropCalculator) -> None:
    with pytest.raises(ValueError, match="Reynolds number is zero"):
        calc.calculate_pressure_drop(**_base_kwargs(flow_rate_kg_s=0.0))


def test_nonpositive_diameter_raises(calc: PressureDropCalculator) -> None:
    with pytest.raises(ValueError, match="pipe_diameter_m must be > 0"):
        calc.calculate_pressure_drop(**_base_kwargs(pipe_diameter_m=0.0))


def test_nonpositive_temperature_raises(calc: PressureDropCalculator) -> None:
    with pytest.raises(ValueError, match="temperature_k must be > 0"):
        calc.calculate_pressure_drop(**_base_kwargs(temperature_k=0.0))


def test_non_numeric_diameter_raises_type_error(calc: PressureDropCalculator) -> None:
    with pytest.raises(TypeError, match="pipe_diameter_m must be a number"):
        calc.calculate_pressure_drop(**_base_kwargs(pipe_diameter_m="big"))


def test_non_numeric_temperature_raises_type_error(
    calc: PressureDropCalculator,
) -> None:
    with pytest.raises(TypeError, match="temperature_k must be a number"):
        calc.calculate_pressure_drop(**_base_kwargs(temperature_k="hot"))


def test_reference_tables_present() -> None:
    assert PIPE_DIMENSIONS_SCH40['4"'] == pytest.approx(0.10226)
    assert ROUGHNESS_VALUES["Commercial Steel"] == pytest.approx(4.5e-5)
    assert all(v > 0 for v in PIPE_DIMENSIONS_SCH40.values())
