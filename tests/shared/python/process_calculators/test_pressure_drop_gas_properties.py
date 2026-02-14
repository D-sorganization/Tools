#!/usr/bin/env python3
"""Tests for pressure drop gas property utilities."""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.pressure_drop_calculator.utils.gas_properties import (
    DEFAULT_GAMMA_DIATOMIC,
    GAS_DATABASE,
    calculate_compressibility_factor,
    calculate_gas_properties,
    calculate_heat_capacity_ratio,
    calculate_ideal_gas_cp,
    calculate_ideal_gas_density,
    calculate_mixture_cp,
    calculate_mixture_molecular_weight,
    calculate_mixture_viscosity_simple,
    calculate_mixture_viscosity_wilke,
    calculate_pure_gas_viscosity_lucas,
    calculate_pure_gas_viscosity_sutherland,
    calculate_real_gas_density,
    calculate_speed_of_sound,
)


@pytest.fixture
def syngas() -> dict[str, float]:
    return {"H2": 0.30, "CO": 0.35, "CO2": 0.15, "N2": 0.15, "CH4": 0.05}


def test_calculate_ideal_gas_cp_known_component() -> None:
    cp = calculate_ideal_gas_cp("CO2", 600.0)
    assert cp > 0


def test_calculate_ideal_gas_cp_unknown_component_falls_back() -> None:
    cp_unknown = calculate_ideal_gas_cp("NOT_A_GAS", 500.0)
    cp_air = calculate_ideal_gas_cp("Air", 500.0)
    assert cp_unknown == pytest.approx(cp_air)


def test_calculate_mixture_cp(syngas: dict[str, float]) -> None:
    cp_mix = calculate_mixture_cp(syngas, 700.0)
    assert cp_mix > 0


def test_calculate_heat_capacity_ratio_physical_range(syngas: dict[str, float]) -> None:
    gamma = calculate_heat_capacity_ratio(syngas, 700.0)
    assert 1.0 <= gamma <= 1.7


def test_calculate_heat_capacity_ratio_invalid_cp_returns_default(
    monkeypatch: pytest.MonkeyPatch, syngas: dict[str, float]
) -> None:
    from upstream_drift_tools.process_calculators.pressure_drop_calculator.utils import (
        gas_properties as gp,
    )

    monkeypatch.setattr(gp, "calculate_mixture_cp", lambda *_: 1.0)
    gamma = calculate_heat_capacity_ratio(syngas, 600.0)
    assert gamma == pytest.approx(float(DEFAULT_GAMMA_DIATOMIC))


def test_calculate_speed_of_sound(syngas: dict[str, float]) -> None:
    a = calculate_speed_of_sound(syngas, 700.0)
    assert a > 0


def test_calculate_mixture_molecular_weight(syngas: dict[str, float]) -> None:
    mw = calculate_mixture_molecular_weight(syngas)
    assert 0 < mw < 50


def test_calculate_mixture_molecular_weight_skips_unknown() -> None:
    mw = calculate_mixture_molecular_weight({"H2": 0.5, "UNKNOWN": 0.5})
    assert mw == pytest.approx(0.5 * GAS_DATABASE["H2"].molecular_weight)


def test_ideal_and_real_gas_density_relationship() -> None:
    mw = 28.97
    t = 400.0
    p = 3e5
    rho_ideal = calculate_ideal_gas_density(mw, t, p)
    rho_real = calculate_real_gas_density(mw, t, p, compressibility=0.9)
    assert rho_real > rho_ideal


def test_calculate_compressibility_factor_bounds(syngas: dict[str, float]) -> None:
    z = calculate_compressibility_factor(syngas, temperature=600.0, pressure=8e6)
    assert 0.1 <= z <= 1.5


def test_pure_gas_viscosity_sutherland_scaling() -> None:
    mu_low = calculate_pure_gas_viscosity_sutherland(300.0)
    mu_high = calculate_pure_gas_viscosity_sutherland(600.0)
    assert mu_low > 0
    assert mu_high > mu_low


def test_pure_gas_viscosity_lucas_positive() -> None:
    props = GAS_DATABASE["CO2"]
    mu = calculate_pure_gas_viscosity_lucas(
        temperature=600.0, pressure=5e5, props=props
    )
    assert mu > 0


def test_mixture_viscosity_wilke_known_components(syngas: dict[str, float]) -> None:
    mu_mix = calculate_mixture_viscosity_wilke(syngas, temperature=700.0, pressure=1e5)
    assert mu_mix > 0


def test_mixture_viscosity_wilke_unknown_component_fallback() -> None:
    mu_mix = calculate_mixture_viscosity_wilke(
        {"H2": 0.5, "UNKNOWN": 0.5}, temperature=650.0, pressure=2e5
    )
    assert mu_mix > 0


def test_mixture_viscosity_simple(syngas: dict[str, float]) -> None:
    mu_simple = calculate_mixture_viscosity_simple(syngas, temperature=650.0)
    assert mu_simple > 0


def test_calculate_gas_properties_with_compressibility(
    syngas: dict[str, float],
) -> None:
    props = calculate_gas_properties(syngas, temperature=700.0, pressure=20e5)
    assert props["molecular_weight"] > 0
    assert props["density"] > 0
    assert props["viscosity"] > 0
    assert 0.1 <= props["compressibility_factor"] <= 1.5
    assert 1.0 <= props["heat_capacity_ratio"] <= 1.7
    assert props["speed_of_sound"] > 0
    assert props["cp"] > 0


def test_calculate_gas_properties_without_compressibility(
    syngas: dict[str, float],
) -> None:
    props = calculate_gas_properties(
        syngas, temperature=700.0, pressure=20e5, use_compressibility=False
    )
    assert props["compressibility_factor"] == pytest.approx(1.0)
