"""Unit tests for the pressure-drop engine correlation modules.

Covers the previously-untested pure-logic engine helpers:
``friction_factors``, ``flow_properties`` (free functions), ``fittings`` and
``compressible_flow``. These are deterministic numerical correlations with
explicit precondition guards, so each test asserts a known value, the regime
branches, and the DbC ValueError paths.
"""

from __future__ import annotations

import math

import pytest
from sidekick.process_calculators.pressure_drop_calculator.engine import (
    compressible_flow as cf,
)
from sidekick.process_calculators.pressure_drop_calculator.engine import (
    fittings as fit,
)
from sidekick.process_calculators.pressure_drop_calculator.engine import (
    flow_properties as fp,
)
from sidekick.process_calculators.pressure_drop_calculator.engine import (
    friction_factors as ff,
)
from sidekick.process_calculators.pressure_drop_calculator.models.pressure_drop_data_models import (  # noqa: E501
    GasComposition,
    PipeFitting,
    PressureDropInputs,
)

# ---------------------------------------------------------------------------
# friction_factors
# ---------------------------------------------------------------------------


def test_friction_factor_laminar_known_value() -> None:
    # f = 64 / Re
    assert ff.friction_factor_laminar(2000.0) == pytest.approx(64.0 / 2000.0)


def test_friction_factor_laminar_nonpositive_returns_default() -> None:
    # Re <= 0 logs an error and returns the default laminar factor.
    assert ff.friction_factor_laminar(0.0) == pytest.approx(0.064)
    assert ff.friction_factor_laminar(-5.0) == pytest.approx(0.064)


@pytest.mark.parametrize(
    "func",
    [
        ff.friction_factor_swamee_jain,
        ff.friction_factor_haaland,
        ff.friction_factor_colebrook,
    ],
)
def test_explicit_methods_fall_back_to_laminar(func) -> None:
    # Below the laminar upper bound (2300) every method delegates to laminar.
    assert func(1000.0, 0.0001) == pytest.approx(64.0 / 1000.0)


@pytest.mark.parametrize(
    "func",
    [
        ff.friction_factor_swamee_jain,
        ff.friction_factor_haaland,
        ff.friction_factor_churchill,
        ff.friction_factor_colebrook,
    ],
)
def test_turbulent_friction_factor_is_physical(func) -> None:
    f = func(1.0e5, 0.0002)
    # Typical turbulent Darcy friction factors sit in this band.
    assert 0.005 < f < 0.1


def test_colebrook_converges_close_to_swamee_jain() -> None:
    re, rr = 5.0e5, 0.0005
    colebrook = ff.friction_factor_colebrook(re, rr)
    swamee = ff.friction_factor_swamee_jain(re, rr)
    # Swamee-Jain is an explicit approximation of Colebrook; within ~5%.
    assert colebrook == pytest.approx(swamee, rel=0.05)


def test_churchill_handles_creeping_flow() -> None:
    # Re < 1 returns the laminar constant (64.0).
    assert ff.friction_factor_churchill(0.5, 0.001) == pytest.approx(64.0)


@pytest.mark.parametrize(
    "method",
    ["colebrook", "swamee-jain", "swamee_jain", "churchill", "haaland", "CHURCHILL"],
)
def test_select_friction_factor_method_dispatch(method: str) -> None:
    f = ff.select_friction_factor_method(method, 1.0e5, 0.0002)
    assert f > 0


def test_select_friction_factor_method_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown friction factor method"):
        ff.select_friction_factor_method("invalid", 1.0e5, 0.0002)


def test_select_friction_factor_method_none_raises() -> None:
    with pytest.raises(ValueError, match="method must be provided"):
        ff.select_friction_factor_method(None, 1.0e5, 0.0002)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# flow_properties (free functions)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reynolds", "expected"),
    [
        (500.0, "laminar"),
        (2299.0, "laminar"),
        (2300.0, "transitional"),
        (3999.0, "transitional"),
        (4000.0, "turbulent"),
        (1.0e6, "turbulent"),
    ],
)
def test_classify_flow_regime(reynolds: float, expected: str) -> None:
    assert fp.classify_flow_regime(reynolds) == expected


def test_frictional_pressure_drop_darcy_weisbach() -> None:
    # ΔP = f (L/D) (0.5 ρ v²)
    f, length, diameter, density, velocity = 0.02, 100.0, 0.1, 1.2, 10.0
    expected = f * (length / diameter) * (0.5 * density * velocity**2)
    assert fp.calculate_frictional_pressure_drop(
        f, length, diameter, density, velocity
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
        fp.calculate_frictional_pressure_drop(f, length, diameter, density, velocity)


def test_elevation_pressure_drop_known_value() -> None:
    dp = fp.calculate_elevation_pressure_drop(1000.0, 10.0)
    assert dp == pytest.approx(1000.0 * fp.GRAVITY * 10.0)


def test_elevation_pressure_drop_negative_change_is_negative() -> None:
    assert fp.calculate_elevation_pressure_drop(1000.0, -5.0) < 0


def test_elevation_pressure_drop_none_raises() -> None:
    with pytest.raises(ValueError, match="density must be provided"):
        fp.calculate_elevation_pressure_drop(None, 10.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("service", "expect_higher"),
    [("continuous", False), ("intermittent", True), ("unknown_service", False)],
)
def test_erosional_velocity_service_types(service: str, expect_higher: bool) -> None:
    continuous = fp.calculate_erosional_velocity(50.0, "continuous")
    value = fp.calculate_erosional_velocity(50.0, service)
    assert value > 0
    if expect_higher:
        # Intermittent service permits a higher erosional velocity (C=125 vs 100).
        assert value > continuous
    else:
        assert value == pytest.approx(continuous)


def test_erosional_velocity_none_raises() -> None:
    with pytest.raises(ValueError, match="density must be provided"):
        fp.calculate_erosional_velocity(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# fittings
# ---------------------------------------------------------------------------


def test_fitting_pressure_drop_uses_provided_k_when_unknown_type() -> None:
    # An unknown fitting type falls through both lookup paths to the provided K.
    fitting = PipeFitting(fitting_type="totally_made_up", quantity=2, k_factor=1.5)
    density, velocity = 1.2, 8.0
    velocity_head = 0.5 * density * velocity**2
    dp = fit.calculate_fitting_pressure_drop(
        [fitting], density, velocity, reynolds_number=1.0e5, diameter_inches=4.0
    )
    # total_K = 1.5 * 2 = 3.0
    assert dp == pytest.approx(3.0 * velocity_head)


def test_fitting_pressure_drop_empty_list_is_zero() -> None:
    assert fit.calculate_fitting_pressure_drop(
        [], 1.2, 8.0, 1.0e5, 4.0
    ) == pytest.approx(0)


def test_fitting_pressure_drop_none_raises() -> None:
    with pytest.raises(ValueError, match="fittings must be provided"):
        fit.calculate_fitting_pressure_drop(None, 1.2, 8.0, 1.0e5, 4.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compressible_flow
# ---------------------------------------------------------------------------


def test_expansion_factor_small_drop_near_unity() -> None:
    # A tiny pressure drop yields an expansion factor at the y>=0.99 ceiling.
    y = cf.calculate_expansion_factor(1.0e6, 100.0, 0.02, 1000.0)
    assert y == pytest.approx(1.0)


def test_expansion_factor_bounded_between_zero_and_one() -> None:
    y = cf.calculate_expansion_factor(1.0e5, 4.0e4, 0.02, 1000.0)
    assert 0.0 <= y <= 1.0


def test_expansion_factor_nonpositive_inlet_returns_one() -> None:
    assert cf.calculate_expansion_factor(0.0, 100.0, 0.02, 1000.0) == 1.0


def test_expansion_factor_negative_drop_returns_one() -> None:
    assert cf.calculate_expansion_factor(1.0e5, -10.0, 0.02, 1000.0) == 1.0


def test_expansion_factor_drop_exceeds_inlet_returns_zero() -> None:
    # pressure_ratio <= 0 -> 0.0
    assert cf.calculate_expansion_factor(1.0e5, 2.0e5, 0.02, 1000.0) == 0.0


def test_expansion_factor_none_inlet_raises() -> None:
    with pytest.raises(ValueError, match="inlet_pressure must be provided"):
        cf.calculate_expansion_factor(None, 100.0, 0.02, 1000.0)  # type: ignore[arg-type]


def test_compressible_flow_correction_returns_drop_and_outlet() -> None:
    corrected_dp, p2 = cf.calculate_compressible_flow_correction(
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


@pytest.mark.parametrize(
    ("diameter", "temperature", "molecular_weight"),
    [(0.0, 300.0, 18.0), (0.1, 0.0, 18.0), (0.1, 300.0, 0.0)],
)
def test_compressible_flow_correction_rejects_nonpositive(
    diameter: float, temperature: float, molecular_weight: float
) -> None:
    with pytest.raises(ValueError):
        cf.calculate_compressible_flow_correction(
            inlet_pressure=5.0e5,
            outlet_pressure=4.8e5,
            length=50.0,
            diameter=diameter,
            mass_flow_rate=2.0,
            temperature=temperature,
            molecular_weight=molecular_weight,
            compressibility_factor=0.98,
            friction_factor=0.02,
        )


def _methane_inputs() -> PressureDropInputs:
    """A physically reasonable natural-gas pipe flow case."""
    return PressureDropInputs(
        pipe_diameter=0.1,
        pipe_length=100.0,
        pipe_roughness=4.5e-5,
        mass_flow_rate=1.0,
        inlet_pressure=5.0e5,
        inlet_temperature=300.0,
        gas_composition=GasComposition(components={"CH4": 1.0}),
    )


def test_calculate_flow_properties_end_to_end() -> None:
    props = fp.calculate_flow_properties(_methane_inputs())
    assert props.density > 0
    assert props.viscosity > 0
    assert props.velocity > 0
    assert props.reynolds_number > 0
    assert 0 <= props.mach_number < 50
    assert props.volumetric_flow_rate > 0


@pytest.mark.parametrize("field_name", ["pipe_diameter", "mass_flow_rate"])
def test_calculate_flow_properties_rejects_nonpositive(field_name: str) -> None:
    inputs = _methane_inputs()
    setattr(inputs, field_name, 0.0)
    with pytest.raises(ValueError):
        fp.calculate_flow_properties(inputs)


def test_module_constants_exposed() -> None:
    assert cf.PI == pytest.approx(math.pi)
    assert fp.PI == pytest.approx(math.pi)
    assert cf.R_UNIVERSAL > 0
