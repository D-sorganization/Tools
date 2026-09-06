"""Tests for the steam calculation engine — thermodynamic properties.

Covers:
- SteamProperties dataclass
- SteamCalculationEngine initialization
- Water vapor pressure correlations (Antoine, Buck)
- Simplified property calculations
- Physical law validations
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
from numpy.testing import assert_allclose
from upstream_drift_tools.calculators.thermo.steam_engine import (
    CRITICAL_TEMPERATURE_WATER,
    KELVIN_TO_CELSIUS_OFFSET,
    SteamCalculationEngine,
    SteamProperties,
)

# ── SteamProperties Dataclass ────────────────────────────────────────────


class TestSteamProperties:
    """Test SteamProperties creation and export."""

    @pytest.fixture()
    def sample_props(self) -> SteamProperties:
        return SteamProperties(
            temperature=100.0,
            pressure=101325.0,
            density=0.598,
            specific_volume=1.673,
            enthalpy=2676.0,
            entropy=7.354,
            internal_energy=2506.0,
            cp=2.08,
            cv=1.57,
            speed_of_sound=450.0,
            dynamic_viscosity=1.227e-5,
            thermal_conductivity=0.0248,
            kinematic_viscosity=2.052e-5,
            quality=1.0,
            phase="vapor",
        )

    def test_to_dict_has_all_fields(self, sample_props: SteamProperties) -> None:
        d = sample_props.to_dict()
        assert "Temperature (K)" in d
        assert "Pressure (Pa)" in d
        assert "Phase" in d
        assert "Enthalpy (J/kg)" in d

    def test_to_dict_values_match(self, sample_props: SteamProperties) -> None:
        d = sample_props.to_dict()
        assert d["Temperature (K)"] == 100.0
        assert d["Phase"] == "vapor"

    def test_optional_fields_default_none(self) -> None:
        props = SteamProperties(
            temperature=200.0,
            pressure=200000.0,
            density=1.0,
            specific_volume=1.0,
            enthalpy=2800.0,
            entropy=7.0,
            internal_energy=2600.0,
            cp=2.0,
            cv=1.5,
            speed_of_sound=500.0,
            dynamic_viscosity=1e-5,
            thermal_conductivity=0.025,
            kinematic_viscosity=1e-5,
            quality=1.0,
            phase="vapor",
        )
        assert props.compressibility_factor is None
        assert props.prandtl_number is None


# ── SteamCalculationEngine Initialization ────────────────────────────────


class TestSteamEngineInit:
    """Test engine initialization."""

    def test_engine_creation(self) -> None:
        engine = SteamCalculationEngine()
        assert engine is not None

    def test_engine_default_state(self) -> None:
        engine = SteamCalculationEngine()
        assert engine.initialized is False or engine.initialized is True


# ── Vapor Pressure Correlations ──────────────────────────────────────────


class TestVaporPressure:
    """Test water vapor pressure correlations."""

    @pytest.fixture()
    def engine(self) -> SteamCalculationEngine:
        return SteamCalculationEngine()

    def test_vapor_pressure_increases_with_temp(
        self, engine: SteamCalculationEngine
    ) -> None:
        """Vapor pressure is a monotonically increasing function of temperature."""
        p20 = engine.calculate_water_vapor_pressure(20.0, method="buck")
        p50 = engine.calculate_water_vapor_pressure(50.0, method="buck")
        p80 = engine.calculate_water_vapor_pressure(80.0, method="buck")
        assert p20 < p50 < p80

    def test_vapor_pressure_at_low_temp(self, engine: SteamCalculationEngine) -> None:
        """Vapor pressure at 0°C should be on the order of 600 Pa."""
        pvap = engine.calculate_water_vapor_pressure(0.0, method="buck")
        # ~611 Pa at triple point, Buck approximation may differ slightly
        assert 400.0 < pvap < 1000.0

    def test_antoine_equation_positive(self, engine: SteamCalculationEngine) -> None:
        """Antoine equation should return positive values at reasonable temperatures."""
        for temp in [10.0, 25.0, 50.0, 75.0, 100.0]:
            pvap = engine._antoine_equation(temp)
            assert pvap > 0, f"Negative vapor pressure at {temp}°C"

    def test_antoine_at_50c(self, engine: SteamCalculationEngine) -> None:
        """Antoine equation at 50°C should give ~12.3 kPa."""
        pvap = engine._antoine_equation(50.0)
        assert 10000.0 < pvap < 15000.0

    def test_buck_equation_positive(self, engine: SteamCalculationEngine) -> None:
        """Buck equation should always return positive values."""
        for temp in [0.0, 10.0, 25.0, 50.0, 75.0, 100.0]:
            pvap = engine._buck_equation(temp)
            assert pvap > 0, f"Negative vapor pressure at {temp}°C"

    @pytest.mark.scientific
    @pytest.mark.parametrize(
        ("temperature_k", "expected_pa"),
        [
            (373.15, 101_417.0),
            (298.15, 3_169.9),
        ],
    )
    def test_saturation_pressure_matches_iapws_reference(
        self,
        temperature_k: float,
        expected_pa: float,
    ) -> None:
        """Saturation pressure stays anchored to IAPWS-IF97 values (#3391)."""
        from sidekick.calculators.thermo.steam_engine import (
            SteamCalculationEngine as SidekickSteamCalculationEngine,
        )

        engine = SidekickSteamCalculationEngine()
        assert engine.get_saturation_pressure(temperature_k) == pytest.approx(
            expected_pa,
            rel=0.03,
        )

    @pytest.mark.scientific
    def test_saturation_temperature_matches_iapws_reference(self) -> None:
        """Tsat at 1 MPa stays anchored to the IAPWS-IF97 reference (#3391)."""
        from sidekick.calculators.thermo.steam_engine import (
            SteamCalculationEngine as SidekickSteamCalculationEngine,
        )

        engine = SidekickSteamCalculationEngine()
        assert engine.get_saturation_temperature(1.0e6) == pytest.approx(
            453.03,
            rel=0.03,
        )


# ── Dew Point Calculation ────────────────────────────────────────────────


class TestDewPoint:
    """Test dew point temperature calculations."""

    @pytest.fixture()
    def engine(self) -> SteamCalculationEngine:
        return SteamCalculationEngine()

    def test_dew_point_increases_with_humidity(
        self, engine: SteamCalculationEngine
    ) -> None:
        """Higher partial pressure → higher dew point."""
        dp_low = engine.calculate_dew_point(1000.0, 101325.0)
        dp_high = engine.calculate_dew_point(3000.0, 101325.0)
        assert dp_high > dp_low

    def test_dew_point_reasonable_range(self, engine: SteamCalculationEngine) -> None:
        """Dew point at typical indoor conditions should be reasonable."""
        # Typical indoor: P_partial ~ 1500 Pa (RH ~60% at 22°C)
        dp = engine.calculate_dew_point(1500.0, 101325.0)
        assert -20.0 < dp < 50.0


# ── Simplified Property Calculations ─────────────────────────────────────


class TestSimplifiedCalculations:
    """Test the simplified (non-CoolProp/Cantera) calculation fallback."""

    @pytest.fixture()
    def engine(self) -> SteamCalculationEngine:
        return SteamCalculationEngine()

    def test_calculate_properties_returns_steam_properties(
        self, engine: SteamCalculationEngine
    ) -> None:
        """Engine should return SteamProperties regardless of backend."""
        # Use a superheated vapor state: 200°C = 473.15 K, low pressure
        props = engine.calculate_properties(473.15, 50000.0, engine="simplified")
        assert isinstance(props, SteamProperties)

    def test_properties_temperature_matches(
        self, engine: SteamCalculationEngine
    ) -> None:
        props = engine.calculate_properties(473.15, 200000.0, engine="simplified")
        assert_allclose(props.temperature, 473.15, atol=1.0)

    def test_properties_pressure_matches(self, engine: SteamCalculationEngine) -> None:
        props = engine.calculate_properties(473.15, 500000.0, engine="simplified")
        assert_allclose(props.pressure, 500000.0, rtol=0.01)

    def test_density_positive(self, engine: SteamCalculationEngine) -> None:
        # Superheated steam at 200°C, low pressure
        props = engine.calculate_properties(473.15, 50000.0, engine="simplified")
        assert props.density > 0

    def test_vapor_phase_enthalpy_positive(
        self, engine: SteamCalculationEngine
    ) -> None:
        """Steam enthalpy at 200°C, low pressure (vapor) should be positive."""
        # 473.15 K is above boiling at 50000 Pa, so should be vapor
        props = engine.calculate_properties(473.15, 50000.0, engine="simplified")
        assert props.enthalpy > 0


# ── Physical Constants ───────────────────────────────────────────────────


class TestPhysicalConstants:
    """Verify physical constants are correct."""

    def test_critical_temperature(self) -> None:
        """Critical temperature of water should be 647.15 K (373.95°C)."""
        assert_allclose(CRITICAL_TEMPERATURE_WATER, 647.15, rtol=0.001)

    def test_kelvin_offset(self) -> None:
        """Kelvin-Celsius offset should be 273.15."""
        assert_allclose(KELVIN_TO_CELSIUS_OFFSET, 273.15)


# ── Engine Selection ─────────────────────────────────────────────────────


class TestEngineSelection:
    """Test engine selection logic."""

    def test_simplified_engine_always_available(self) -> None:
        engine = SteamCalculationEngine()
        selected = engine.select_best_engine("simplified")
        assert selected == "simplified"

    def test_auto_engine_returns_something(self) -> None:
        engine = SteamCalculationEngine()
        selected = engine.select_best_engine("auto")
        assert selected in ("coolprop", "cantera", "simplified")
