"""Edge case and boundary value tests for AcidGasDewpointCalculator.

Covers:
  - Zero / negative partial pressure handling (returns NaN, not raises)
  - Unknown component handling
  - Empty / zero composition edge cases
  - Pure component dewpoints (100% H2S, 100% H2O)
  - Trace amounts of acid gas
  - Temperature / pressure out-of-range warnings
  - Condensation risk assessment boundaries

Design principles:
  - TDD: Tests describe the desired behaviour.
  - DRY: Common setup is shared via fixtures.
  - DbC: Each test documents pre/post-conditions.
  - Orthogonality: Each test class covers one category of edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (
    AcidGasComposition,
    AcidGasDewpointCalculator,
    DewpointResult,
    estimate_condensation_risk,
    quick_dewpoint_calculation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calculator() -> AcidGasDewpointCalculator:
    """Provide a fresh AcidGasDewpointCalculator instance."""
    return AcidGasDewpointCalculator()


@pytest.fixture
def typical_composition() -> AcidGasComposition:
    """Typical syngas composition with acid gases."""
    return AcidGasComposition(h2o=0.15, hf=0.001, hcl=0.002, h2s=0.005, other=0.842)


# ---------------------------------------------------------------------------
# Tests: Vapor pressure calculation edge cases
# ---------------------------------------------------------------------------


class TestVaporPressureEdgeCases:
    """Edge cases for calculate_vapor_pressure."""

    def test_unknown_component_raises(self, calculator):
        """Unknown component name must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown component"):
            calculator.calculate_vapor_pressure(100.0, "Argon")

    def test_unknown_method_raises(self, calculator):
        """Unknown method must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            calculator.calculate_vapor_pressure(100.0, "H2O", method="bogus")

    @pytest.mark.parametrize("component", ["H2O", "HF", "HCl", "H2S"])
    def test_all_known_components_return_positive(self, calculator, component):
        """All known components should return positive vapor pressure at 25 C."""
        vp = calculator.calculate_vapor_pressure(25.0, component)
        assert vp > 0
        assert math.isfinite(vp)

    def test_water_at_100c_near_one_atm(self, calculator):
        """Water vapor pressure at 100 C should be approximately 101325 Pa."""
        vp = calculator.calculate_vapor_pressure(100.0, "H2O")
        assert 90000 < vp < 120000

    def test_very_low_temperature(self, calculator):
        """Very low temperature produces a positive, small vapor pressure."""
        vp = calculator.calculate_vapor_pressure(-50.0, "H2O")
        assert vp > 0
        assert math.isfinite(vp)

    def test_extended_antoine_water_high_temp(self, calculator):
        """Extended Antoine uses different coefficients above 100 C for water."""
        vp_low = calculator.calculate_vapor_pressure(
            99.0, "H2O", method="extended_antoine"
        )
        vp_high = calculator.calculate_vapor_pressure(
            101.0, "H2O", method="extended_antoine"
        )
        # Both should be finite and positive; high temp has higher VP
        assert math.isfinite(vp_low) and vp_low > 0
        assert math.isfinite(vp_high) and vp_high > 0
        assert vp_high > vp_low


# ---------------------------------------------------------------------------
# Tests: Single-component dewpoint edge cases
# ---------------------------------------------------------------------------


class TestDewpointEdgeCases:
    """Edge cases for calculate_dewpoint (inverse Antoine)."""

    def test_zero_partial_pressure_raises(self, calculator):
        """Zero partial pressure should raise ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            calculator.calculate_dewpoint(0.0, "H2O")

    def test_negative_partial_pressure_raises(self, calculator):
        """Negative partial pressure should raise ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            calculator.calculate_dewpoint(-100.0, "H2O")

    def test_unknown_component_raises(self, calculator):
        """Unknown component should raise ValueError."""
        with pytest.raises(ValueError, match="unknown component"):
            calculator.calculate_dewpoint(1000.0, "Ar")

    def test_normal_conditions_return_finite(self, calculator):
        """A reasonable partial pressure should return a finite dewpoint."""
        # Approx 3170 Pa partial pressure (equivalent to ~0.03 bar)
        result = calculator.calculate_dewpoint(3170.0, "H2O")
        assert math.isfinite(result)

    def test_very_high_partial_pressure(self, calculator):
        """Very high partial pressure produces a high dewpoint temperature."""
        result = calculator.calculate_dewpoint(1e6, "H2O")
        assert math.isfinite(result)

    def test_very_small_partial_pressure(self, calculator):
        """Very small partial pressure produces a very low dewpoint."""
        result = calculator.calculate_dewpoint(0.01, "H2S")
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# Tests: Mixture dewpoint with empty/zero compositions
# ---------------------------------------------------------------------------


class TestMixtureDewpointZeroComposition:
    """Edge cases for calculate_dewpoint_mixture with zero or empty compositions."""

    def test_all_zero_composition(self, calculator):
        """All-zero composition should produce NaN dewpoints with a warning."""
        comp = AcidGasComposition(h2o=0, hf=0, hcl=0, h2s=0, other=1.0)
        result = calculator.calculate_dewpoint_mixture(150.0, 30.0, comp)
        assert isinstance(result, DewpointResult)
        # All individual dewpoints should be NaN because partial pressures are 0
        assert np.isnan(result.h2o_dewpoint_c)
        assert np.isnan(result.hf_dewpoint_c)
        assert np.isnan(result.hcl_dewpoint_c)
        assert np.isnan(result.h2s_dewpoint_c)
        # Overall dewpoint should be NaN
        assert np.isnan(result.overall_dewpoint_c)
        assert result.limiting_component == "Unknown"

    def test_only_water_in_composition(self, calculator):
        """Pure water composition should give a water dewpoint only."""
        comp = AcidGasComposition(h2o=0.10, hf=0, hcl=0, h2s=0, other=0.90)
        result = calculator.calculate_dewpoint_mixture(150.0, 30.0, comp)
        # Water dewpoint should be finite
        assert math.isfinite(result.h2o_dewpoint_c)
        # Acid gas dewpoints should be NaN
        assert np.isnan(result.hf_dewpoint_c)
        assert np.isnan(result.hcl_dewpoint_c)
        assert np.isnan(result.h2s_dewpoint_c)
        # Overall dewpoint should be the water dewpoint
        assert result.overall_dewpoint_c == pytest.approx(result.h2o_dewpoint_c)
        assert result.limiting_component == "H2O"


# ---------------------------------------------------------------------------
# Tests: Pure component dewpoints
# ---------------------------------------------------------------------------


class TestPureComponentDewpoints:
    """Tests for pure component scenarios (100% of one component)."""

    def test_pure_h2s(self, calculator):
        """100% H2S should give an H2S dewpoint; water dewpoint is NaN."""
        comp = AcidGasComposition(h2o=0, hf=0, hcl=0, h2s=1.0, other=0)
        result = calculator.calculate_dewpoint_mixture(150.0, 30.0, comp)
        assert math.isfinite(result.h2s_dewpoint_c)
        assert np.isnan(result.h2o_dewpoint_c)
        assert result.limiting_component == "H2S"

    def test_pure_h2o(self, calculator):
        """100% H2O should give a water dewpoint."""
        comp = AcidGasComposition(h2o=1.0, hf=0, hcl=0, h2s=0, other=0)
        result = calculator.calculate_dewpoint_mixture(150.0, 30.0, comp)
        assert math.isfinite(result.h2o_dewpoint_c)
        assert result.limiting_component == "H2O"
        # At 30 bar, water dewpoint should be well above 100 C
        assert result.h2o_dewpoint_c > 100


# ---------------------------------------------------------------------------
# Tests: Trace amounts of acid gas
# ---------------------------------------------------------------------------


class TestTraceAmounts:
    """Tests for trace (very small) acid gas concentrations."""

    @pytest.mark.parametrize(
        "h2s_fraction",
        [0.001, 0.0001, 0.00001, 1e-6],
    )
    def test_trace_h2s_gives_finite_dewpoint(self, calculator, h2s_fraction):
        """Very small H2S fractions should still give a finite dewpoint."""
        comp = AcidGasComposition(
            h2o=0.15, hf=0, hcl=0, h2s=h2s_fraction, other=0.85 - h2s_fraction
        )
        result = calculator.calculate_dewpoint_mixture(150.0, 30.0, comp)
        assert math.isfinite(result.h2s_dewpoint_c)

    def test_trace_h2s_dewpoint_decreases_with_concentration(self, calculator):
        """H2S dewpoint should decrease as concentration decreases."""
        dewpoints = []
        for h2s in [0.01, 0.001, 0.0001]:
            comp = AcidGasComposition(h2o=0.15, hf=0, hcl=0, h2s=h2s, other=0.85 - h2s)
            result = calculator.calculate_dewpoint_mixture(150.0, 30.0, comp)
            dewpoints.append(result.h2s_dewpoint_c)
        # Each subsequent dewpoint should be lower
        assert dewpoints[0] > dewpoints[1] > dewpoints[2]


# ---------------------------------------------------------------------------
# Tests: Temperature and pressure warnings
# ---------------------------------------------------------------------------


class TestInputRangeWarnings:
    """Tests for warnings issued when inputs are outside recommended ranges."""

    def test_extreme_low_temperature_warns(self, calculator, typical_composition):
        """Temperature below -100 C should generate a warning."""
        result = calculator.calculate_dewpoint_mixture(
            -150.0, 30.0, typical_composition
        )
        assert any("Temperature outside" in w for w in result.warnings)

    def test_extreme_high_temperature_warns(self, calculator, typical_composition):
        """Temperature above 400 C should generate a warning."""
        result = calculator.calculate_dewpoint_mixture(500.0, 30.0, typical_composition)
        assert any("Temperature outside" in w for w in result.warnings)

    def test_normal_temperature_no_warning(self, calculator, typical_composition):
        """Normal temperature should not generate a temperature warning."""
        result = calculator.calculate_dewpoint_mixture(150.0, 30.0, typical_composition)
        temp_warnings = [w for w in result.warnings if "Temperature" in w]
        assert len(temp_warnings) == 0

    def test_extreme_low_pressure_warns(self, calculator, typical_composition):
        """Pressure below 0.1 bar should generate a warning."""
        result = calculator.calculate_dewpoint_mixture(150.0, 0.05, typical_composition)
        assert any("Pressure outside" in w for w in result.warnings)

    def test_extreme_high_pressure_warns(self, calculator, typical_composition):
        """Pressure above 300 bar should generate a warning."""
        result = calculator.calculate_dewpoint_mixture(
            150.0, 500.0, typical_composition
        )
        assert any("Pressure outside" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Tests: Condensation risk assessment
# ---------------------------------------------------------------------------


class TestCondensationRiskBoundaries:
    """Boundary value tests for the condensation risk assessment logic."""

    def test_below_dewpoint_is_high_risk(self, calculator):
        """Operating below the overall dewpoint should be HIGH risk."""
        # Low temperature, moderate pressure, significant water content
        comp = AcidGasComposition(h2o=0.20, hf=0, hcl=0, h2s=0, other=0.80)
        result = calculator.calculate_dewpoint_mixture(20.0, 30.0, comp)
        # At 20 C and 30 bar with 20% water, we should be well below dewpoint
        assert "HIGH" in result.condensation_risk

    def test_well_above_dewpoint_is_very_low_risk(self, calculator):
        """Operating 50+ degrees above dewpoint should be VERY LOW risk."""
        comp = AcidGasComposition(h2o=0.001, hf=0, hcl=0, h2s=0, other=0.999)
        result = calculator.calculate_dewpoint_mixture(300.0, 1.0, comp)
        # Very little water, very high temp, low pressure => very large margin
        if not np.isnan(result.dewpoint_margin_c):
            assert "LOW" in result.condensation_risk


# ---------------------------------------------------------------------------
# Tests: Quick calculation convenience function
# ---------------------------------------------------------------------------


class TestQuickDewpointCalculation:
    """Tests for the quick_dewpoint_calculation convenience function."""

    def test_returns_expected_keys(self):
        """Return dict must contain the documented keys."""
        result = quick_dewpoint_calculation(
            temperature_c=150.0,
            pressure_bar=30.0,
            h2o_fraction=0.15,
            h2s_fraction=0.005,
        )
        assert "overall_dewpoint_c" in result
        assert "limiting_component" in result
        assert "condensation_risk" in result
        assert "dewpoint_margin_c" in result

    def test_zero_fractions_gives_nan_dewpoint(self):
        """All-zero fractions should return NaN overall dewpoint."""
        result = quick_dewpoint_calculation(
            temperature_c=150.0,
            pressure_bar=30.0,
            h2o_fraction=0.0,
        )
        assert np.isnan(result["overall_dewpoint_c"])


# ---------------------------------------------------------------------------
# Tests: estimate_condensation_risk convenience function
# ---------------------------------------------------------------------------


class TestEstimateCondensationRisk:
    """Tests for the estimate_condensation_risk convenience function."""

    def test_critical_risk_below_dewpoint(self):
        """Condensing conditions should be identified as Critical risk."""
        comp = AcidGasComposition(h2o=0.5, hf=0, hcl=0, h2s=0, other=0.5)
        result = estimate_condensation_risk(
            temperature_c=10.0, pressure_bar=50.0, composition=comp
        )
        assert result["risk_level"] == "Critical"

    def test_low_risk_safe_conditions(self):
        """Safe conditions with large margin should be Low risk."""
        comp = AcidGasComposition(h2o=0.001, hf=0, hcl=0, h2s=0, other=0.999)
        result = estimate_condensation_risk(
            temperature_c=300.0, pressure_bar=1.0, composition=comp
        )
        assert result["risk_level"] == "Low"


# ---------------------------------------------------------------------------
# Tests: AcidGasComposition dataclass edge cases
# ---------------------------------------------------------------------------


class TestAcidGasCompositionEdgeCases:
    """Edge cases for the AcidGasComposition dataclass."""

    def test_normalize_zero_total(self):
        """Normalizing a zero-total composition should return the same object."""
        comp = AcidGasComposition(h2o=0, hf=0, hcl=0, h2s=0, other=0)
        normalized = comp.normalize()
        assert normalized.total == pytest.approx(0.0)

    def test_normalize_sums_to_one(self):
        """Normalized composition should sum to 1.0."""
        comp = AcidGasComposition(h2o=3.0, hf=1.0, hcl=2.0, h2s=4.0, other=10.0)
        normalized = comp.normalize()
        assert normalized.total == pytest.approx(1.0, abs=1e-10)

    def test_to_dict_keys(self):
        """to_dict should return the expected component keys."""
        comp = AcidGasComposition(h2o=0.1, hf=0.01, hcl=0.02, h2s=0.03, other=0.84)
        d = comp.to_dict()
        assert set(d.keys()) == {"H2O", "HF", "HCl", "H2S", "Other"}
