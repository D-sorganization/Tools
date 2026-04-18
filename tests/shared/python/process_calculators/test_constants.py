"""Tests for upstream_drift_tools.process_calculators.constants module.

Covers:
- Temperature conversion functions (C↔K, F↔K)
- Pressure conversion functions (bar↔Pa, psi↔Pa)
- Molecular weight lookup
- Stefan-Boltzmann constant value
- Roundtrip conversion accuracy
"""

from __future__ import annotations

import pytest
pytest.importorskip("numpy")
from upstream_drift_tools.process_calculators.constants import (
    STEFAN_BOLTZMANN,
    bar_to_pa,
    celsius_to_kelvin,
    fahrenheit_to_kelvin,
    get_molecular_weight,
    kelvin_to_celsius,
    kelvin_to_fahrenheit,
    pa_to_bar,
    pa_to_psi,
    psi_to_pa,
)

# ── Temperature Conversions ─────────────────────────────────────────────


class TestCelsiusKelvin:
    """Test Celsius ↔ Kelvin conversions."""

    def test_freezing_point(self) -> None:
        assert celsius_to_kelvin(0.0) == pytest.approx(273.15)

    def test_boiling_point(self) -> None:
        assert celsius_to_kelvin(100.0) == pytest.approx(373.15)

    def test_absolute_zero(self) -> None:
        assert kelvin_to_celsius(0.0) == pytest.approx(-273.15)

    def test_roundtrip(self) -> None:
        assert kelvin_to_celsius(celsius_to_kelvin(42.0)) == pytest.approx(42.0)


class TestFahrenheitKelvin:
    """Test Fahrenheit ↔ Kelvin conversions."""

    def test_freezing_point(self) -> None:
        assert fahrenheit_to_kelvin(32.0) == pytest.approx(273.15)

    def test_boiling_point(self) -> None:
        assert fahrenheit_to_kelvin(212.0) == pytest.approx(373.15)

    def test_roundtrip(self) -> None:
        assert kelvin_to_fahrenheit(fahrenheit_to_kelvin(98.6)) == pytest.approx(98.6)


# ── Pressure Conversions ────────────────────────────────────────────────


class TestBarPascal:
    """Test bar ↔ Pascal conversions."""

    def test_one_bar(self) -> None:
        assert bar_to_pa(1.0) == pytest.approx(100_000.0)

    def test_roundtrip(self) -> None:
        assert pa_to_bar(bar_to_pa(5.0)) == pytest.approx(5.0)


class TestPsiPascal:
    """Test psi ↔ Pascal conversions."""

    def test_one_atmosphere_approx(self) -> None:
        """1 atm ≈ 14.696 psi ≈ 101325 Pa."""
        assert psi_to_pa(14.696) == pytest.approx(101325.0, rel=0.01)

    def test_roundtrip(self) -> None:
        assert pa_to_psi(psi_to_pa(30.0)) == pytest.approx(30.0)


# ── Molecular Weight Lookup ─────────────────────────────────────────────


class TestMolecularWeightLookup:
    """Test get_molecular_weight function."""

    def test_hydrogen(self) -> None:
        mw = get_molecular_weight("H2")
        assert mw == pytest.approx(0.00201588, rel=0.01)

    def test_carbon_dioxide(self) -> None:
        mw = get_molecular_weight("CO2")
        assert mw == pytest.approx(0.04401, rel=0.01)

    def test_water(self) -> None:
        mw = get_molecular_weight("H2O")
        assert mw > 0

    def test_unknown_returns_air_default(self) -> None:
        """Unknown species returns air MW ≈ 0.029."""
        mw = get_molecular_weight("XenonNonExistent")
        assert mw == pytest.approx(0.029, rel=0.01)


# ── Physical Constants ──────────────────────────────────────────────────


class TestPhysicalConstants:
    """Test notable constants are correct."""

    def test_stefan_boltzmann(self) -> None:
        assert STEFAN_BOLTZMANN == pytest.approx(5.670374419e-8, rel=1e-6)
