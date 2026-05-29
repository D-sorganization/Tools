"""Unit tests for the monolithic ``engine._flow_calculations`` module.

``_flow_calculations`` is the older single-file implementation that predates the
split engine modules (``flow_properties``/``friction_factors``/...). It exposes
the same public correlation API and is still imported by some callers, so it
needs its own coverage. Tests mirror the split-module assertions.
"""

from __future__ import annotations

import pytest
from sidekick.process_calculators.pressure_drop_calculator.engine import (
    _flow_calculations as fc,
)
from sidekick.process_calculators.pressure_drop_calculator.models.pressure_drop_data_models import (  # noqa: E501
    GasComposition,
    PipeFitting,
    PressureDropInputs,
)


@pytest.mark.parametrize(
    ("reynolds", "expected"),
    [
        (500.0, "laminar"),
        (2300.0, "transitional"),
        (4000.0, "turbulent"),
    ],
)
def test_classify_flow_regime(reynolds: float, expected: str) -> None:
    assert fc.classify_flow_regime(reynolds) == expected


def test_frictional_pressure_drop_known_value() -> None:
    expected = 0.02 * (100.0 / 0.1) * (0.5 * 1.2 * 10.0**2)
    assert fc.calculate_frictional_pressure_drop(
        0.02, 100.0, 0.1, 1.2, 10.0
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("f", "length", "diameter", "density", "velocity"),
    [
        (0.0, 100.0, 0.1, 1.2, 10.0),
        (0.02, 0.0, 0.1, 1.2, 10.0),
        (0.02, 100.0, 0.0, 1.2, 10.0),
        (0.02, 100.0, 0.1, 0.0, 10.0),
        (0.02, 100.0, 0.1, 1.2, 0.0),
    ],
)
def test_frictional_pressure_drop_rejects_nonpositive(
    f: float, length: float, diameter: float, density: float, velocity: float
) -> None:
    with pytest.raises(ValueError):
        fc.calculate_frictional_pressure_drop(f, length, diameter, density, velocity)


def test_elevation_pressure_drop_known_value() -> None:
    assert fc.calculate_elevation_pressure_drop(1000.0, 10.0) == pytest.approx(
        1000.0 * fc.GRAVITY * 10.0
    )


def test_erosional_velocity_intermittent_higher_than_continuous() -> None:
    cont = fc.calculate_erosional_velocity(50.0, "continuous")
    inter = fc.calculate_erosional_velocity(50.0, "intermittent")
    assert inter > cont > 0


def test_fitting_pressure_drop_uses_provided_k() -> None:
    fitting = PipeFitting(fitting_type="totally_made_up", quantity=2, k_factor=1.5)
    velocity_head = 0.5 * 1.2 * 8.0**2
    dp = fc.calculate_fitting_pressure_drop([fitting], 1.2, 8.0, 1.0e5, 4.0)
    assert dp == pytest.approx(3.0 * velocity_head)


def test_expansion_factor_bounded() -> None:
    y = fc.calculate_expansion_factor(1.0e5, 4.0e4, 0.02, 1000.0)
    assert 0.0 <= y <= 1.0


def test_compressible_flow_correction_runs() -> None:
    corrected_dp, p2 = fc.calculate_compressible_flow_correction(
        inlet_pressure=5.0e5,
        outlet_pressure=4.8e5,
        length=50.0,
        diameter=0.1,
        mass_flow_rate=2.0,
        temperature=300.0,
        molecular_weight=18.0,
        compressibility_factor=0.98,
        friction_factor=0.02,
    )
    assert corrected_dp >= 0
    assert 0 < p2 <= 5.0e5


def test_calculate_flow_properties_end_to_end() -> None:
    inputs = PressureDropInputs(
        pipe_diameter=0.1,
        pipe_length=100.0,
        pipe_roughness=4.5e-5,
        mass_flow_rate=1.0,
        inlet_pressure=5.0e5,
        inlet_temperature=300.0,
        gas_composition=GasComposition(components={"CH4": 1.0}),
    )
    props = fc.calculate_flow_properties(inputs)
    assert props.velocity > 0
    assert props.reynolds_number > 0
    assert 0 <= props.mach_number < 50


def test_calculate_flow_properties_rejects_bad_diameter() -> None:
    inputs = PressureDropInputs(
        pipe_diameter=0.0,
        pipe_length=100.0,
        pipe_roughness=4.5e-5,
        mass_flow_rate=1.0,
        inlet_pressure=5.0e5,
        inlet_temperature=300.0,
        gas_composition=GasComposition(components={"CH4": 1.0}),
    )
    with pytest.raises(ValueError):
        fc.calculate_flow_properties(inputs)
