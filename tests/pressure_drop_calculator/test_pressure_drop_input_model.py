"""Tests for the pressure-drop calculator Pydantic API models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.pressure_drop_calculator import PressureDropInput, PressureDropOutput


def _valid_input(**overrides: object) -> PressureDropInput:
    values: dict[str, object] = {
        "pipe_diameter_m": 0.1,
        "pipe_length_m": 100.0,
        "mass_flow_rate_kg_s": 0.5,
        "inlet_pressure_pa": 101_325.0,
        "inlet_temperature_k": 300.0,
    }
    values.update(overrides)
    return PressureDropInput(**values)


def test_pressure_drop_input_exports_from_package_surface() -> None:
    request = _valid_input()
    response = PressureDropOutput(
        pressure_drop_pa=100.0,
        pressure_drop_bar=0.001,
        pressure_drop_psi=0.0145,
        outlet_temperature_k=300.0,
    )

    assert request.gas_composition == {"N2": 0.79, "O2": 0.21}
    assert response.success is True


def test_pressure_drop_input_rejects_composition_sum_outside_tolerance() -> None:
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        _valid_input(gas_composition={"N2": 0.7, "O2": 0.2})


def test_pressure_drop_input_rejects_negative_mole_fraction() -> None:
    with pytest.raises(ValidationError, match="between 0 and 1"):
        _valid_input(gas_composition={"N2": 1.1, "O2": -0.1})
