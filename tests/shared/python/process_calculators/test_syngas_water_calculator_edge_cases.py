"""Edge case and boundary value tests for SyngasWaterCalculator.

Complements test_syngas_water_overflow.py by covering:
  - SyngasComposition dataclass edge cases
  - Dew point calculation boundaries
  - Water content at extreme pressure / temperature
  - Method selection (auto, explicit)
  - Vapor pressure fast lookup boundaries
  - Condensation risk convenience function edge cases
  - Curve generation edge cases

Design principles:
  - TDD: Tests describe the desired behaviour.
  - DRY: Common setup is shared via fixtures.
  - DbC: Each test documents pre/post-conditions.
  - Orthogonality: Each test class covers one category of edge cases.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("numpy")
import numpy as np
from upstream_drift_tools.process_calculators.syngas_water_calculator import (
    SyngasComposition,
    SyngasWaterCalculator,
    WaterContentResult,
    estimate_condensation_risk,
    quick_water_content,
)

from contracts import PreconditionError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calculator() -> SyngasWaterCalculator:
    """Provide a fresh SyngasWaterCalculator instance."""
    return SyngasWaterCalculator()


# ---------------------------------------------------------------------------
# Tests: SyngasComposition dataclass edge cases
# ---------------------------------------------------------------------------


class TestSyngasCompositionEdgeCases:
    """Edge cases for the SyngasComposition dataclass."""

    def test_normalize_zero_total_returns_self(self):
        """Normalizing a zero-total composition returns the original."""
        comp = SyngasComposition()
        normalized = comp.normalize()
        assert normalized.total == pytest.approx(0.0)

    def test_normalize_sums_to_one(self):
        """Normalized composition should sum to 1.0."""
        comp = SyngasComposition(h2=5.0, co=3.0, co2=2.0, n2=10.0)
        normalized = comp.normalize()
        assert normalized.total == pytest.approx(1.0, abs=1e-10)

    def test_normalize_preserves_ratios(self):
        """Normalization should preserve mole fraction ratios."""
        comp = SyngasComposition(h2=2.0, co=4.0)
        normalized = comp.normalize()
        # CO should be double H2
        assert normalized.co == pytest.approx(2 * normalized.h2, rel=1e-10)

    def test_to_dict_keys(self):
        """to_dict should return the expected component keys."""
        comp = SyngasComposition(h2=0.3, co=0.3, co2=0.15)
        d = comp.to_dict()
        expected_keys = {"H2", "CO", "CO2", "CH4", "N2", "AR", "H2O", "Other"}
        assert set(d.keys()) == expected_keys

    def test_total_property(self):
        """Total property should sum all components."""
        comp = SyngasComposition(h2=0.1, co=0.2, co2=0.3, n2=0.4)
        assert comp.total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: Vapor pressure method selection
# ---------------------------------------------------------------------------


class TestVaporPressureMethodSelection:
    """Tests for explicit and auto method selection."""

    @pytest.mark.parametrize("method", ["antoine", "buck", "iapws", "magnus"])
    def test_explicit_method_returns_positive(self, calculator, method):
        """All explicit methods should return positive vapor pressure at 50 C."""
        vp, method_name = calculator.calculate_vapor_pressure(50.0, method)
        assert vp > 0
        assert math.isfinite(vp)
        assert isinstance(method_name, str)

    def test_auto_method_at_50c(self, calculator):
        """Auto selection at 50 C should pick a valid method."""
        vp, method_name = calculator.calculate_vapor_pressure(50.0, "auto")
        assert vp > 0
        assert math.isfinite(vp)
        assert "auto" in method_name.lower()

    def test_auto_method_below_zero(self, calculator):
        """Auto selection below 0 C should use Buck equation."""
        vp, method_name = calculator.calculate_vapor_pressure(-10.0, "auto")
        assert vp > 0
        assert "Buck" in method_name

    def test_auto_method_above_100(self, calculator):
        """Auto selection above 100 C should use IAPWS."""
        vp, method_name = calculator.calculate_vapor_pressure(200.0, "auto")
        assert vp > 0
        assert "IAPWS" in method_name

    def test_magnus_rejects_out_of_range(self, calculator):
        """Magnus equation should raise ValueError for T < 0 or T > 100."""
        with pytest.raises(ValueError, match="Magnus equation valid"):
            calculator.calculate_vapor_pressure(-1.0, "magnus")
        with pytest.raises(ValueError, match="Magnus equation valid"):
            calculator.calculate_vapor_pressure(101.0, "magnus")

    def test_iapws_rejects_below_triple_point(self, calculator):
        """IAPWS should raise ValueError below the triple point temperature."""
        with pytest.raises(ValueError, match="Temperature out of IAPWS"):
            calculator.calculate_vapor_pressure(-10.0, "iapws")


# ---------------------------------------------------------------------------
# Tests: Dew point calculation edge cases
# ---------------------------------------------------------------------------


class TestDewPointEdgeCases:
    """Edge cases for calculate_dew_point."""

    def test_zero_partial_pressure(self, calculator):
        """Zero partial pressure should be rejected by the DbC precondition."""
        with pytest.raises(
            PreconditionError,
            match="partial_pressure_pa must be positive",
        ):
            calculator.calculate_dew_point(0.0, 101325.0)

    def test_normal_conditions(self, calculator):
        """At partial pressure of ~2.3 kPa (sat. at 20 C), dew point ~ 20 C."""
        # Buck equation at 20 C gives approximately 2338 Pa
        vp_20c = calculator._buck_equation(20.0)
        dp = calculator.calculate_dew_point(vp_20c, 101325.0)
        assert dp == pytest.approx(20.0, abs=1.0)

    def test_buck_equation_matches_buck_1981_reference(self, calculator):
        """_buck_equation must match the Buck (1981) reference curve it cites.

        Regression test for issue #3867: the above-freezing branch passed its
        C and D coefficients to the shared ``buck_pressure_pa`` kernel in the
        kernel's native "syngas" argument order, which computes
        ``(b - t/d) * t / (c + t)`` rather than the standard Buck formula
        ``(b - t/c) * t / (d + t)``. That produced 2636.34 Pa at 20 C instead
        of the correct ~2338.34 Pa (a ~12.7% error). ``steam_engine.py``
        already swaps C and D at its call site to get the correct curve from
        the same kernel; this asserts ``SyngasWaterCalculator`` does too.
        """
        vp_20c = calculator._buck_equation(20.0)
        assert vp_20c == pytest.approx(2338.34, rel=1e-4)

    def test_high_partial_pressure(self, calculator):
        """High partial pressure should give a high dew point."""
        # Saturated at ~80 C -> ~47 kPa
        vp_80c = calculator._buck_equation(80.0)
        dp = calculator.calculate_dew_point(vp_80c, 101325.0)
        assert dp == pytest.approx(80.0, abs=2.0)


# ---------------------------------------------------------------------------
# Tests: Water content at extreme conditions
# ---------------------------------------------------------------------------


class TestWaterContentExtremeConditions:
    """Tests for calculate_water_content with extreme inputs."""

    def test_very_high_pressure_low_water(self, calculator):
        """At very high pressure, water mole fraction should be very small."""
        result = calculator.calculate_water_content(
            temperature_c=25.0, pressure_bar=200.0
        )
        assert isinstance(result, WaterContentResult)
        assert result.mole_fraction_water > 0
        assert result.mole_fraction_water < 0.01  # Very small at high pressure

    def test_low_pressure_high_water(self, calculator):
        """At low pressure, water mole fraction should be larger."""
        result = calculator.calculate_water_content(
            temperature_c=80.0, pressure_bar=0.5
        )
        assert isinstance(result, WaterContentResult)
        assert result.mole_fraction_water > 0.1

    def test_condensation_warning_when_vp_exceeds_pressure(self, calculator):
        """When vapor pressure exceeds total pressure, a warning should be issued."""
        # At 100 C, water VP is ~1 bar; set total pressure to 0.5 bar
        result = calculator.calculate_water_content(
            temperature_c=100.0, pressure_bar=0.5, method="buck"
        )
        assert any("condensation" in w.lower() for w in result.warnings)

    def test_water_content_increases_with_temperature(self, calculator):
        """Water content should increase monotonically with temperature."""
        contents = []
        for temp in [10, 30, 50, 70, 90]:
            result = calculator.calculate_water_content(
                temperature_c=float(temp), pressure_bar=10.0
            )
            contents.append(result.mole_fraction_water)
        for i in range(1, len(contents)):
            assert contents[i] > contents[i - 1]

    def test_water_content_decreases_with_pressure(self, calculator):
        """Water content should decrease as pressure increases (at constant T)."""
        contents = []
        for pressure in [1.0, 5.0, 10.0, 50.0]:
            result = calculator.calculate_water_content(
                temperature_c=40.0, pressure_bar=pressure
            )
            contents.append(result.mole_fraction_water)
        for i in range(1, len(contents)):
            assert contents[i] < contents[i - 1]


# ---------------------------------------------------------------------------
# Tests: Custom composition
# ---------------------------------------------------------------------------


class TestCustomComposition:
    """Tests for passing a SyngasComposition object directly."""

    def test_custom_composition_accepted(self, calculator):
        """Passing a SyngasComposition object should work."""
        comp = SyngasComposition(h2=0.5, co=0.3, co2=0.1, n2=0.1, name="Test Gas")
        result = calculator.calculate_water_content(
            temperature_c=40.0, pressure_bar=10.0, gas_composition=comp
        )
        assert isinstance(result, WaterContentResult)
        assert result.gas_composition == "Test Gas"

    def test_unknown_preset_falls_back_gracefully(self, calculator):
        """Unknown preset name should fall back to typical_syngas composition.

        The code uses SYNGAS_PRESETS.get(name, typical), so it uses the typical
        composition but records the originally requested name.
        """
        result = calculator.calculate_water_content(
            temperature_c=40.0, pressure_bar=10.0, gas_composition="nonexistent_preset"
        )
        assert isinstance(result, WaterContentResult)
        # The calculator still produces a valid result (using typical syngas values)
        assert math.isfinite(result.mole_fraction_water)
        assert result.mole_fraction_water > 0


# ---------------------------------------------------------------------------
# Tests: Vapor pressure fast lookup
# ---------------------------------------------------------------------------


class TestVaporPressureFastLookup:
    """Tests for the fast interpolation-based vapor pressure lookup."""

    def test_at_373k_near_one_atm(self, calculator):
        """At 373 K (~100 C), fast lookup should be near 101325 Pa."""
        vp = calculator.vapor_pressure_fast(373.15)
        assert 90000 < vp < 115000

    def test_out_of_range_returns_nan(self, calculator):
        """Temperatures outside the table range should return NaN."""
        vp = calculator.vapor_pressure_fast(200.0)  # Well below table range (273 K)
        assert np.isnan(vp)

    def test_fast_agrees_with_iapws(self, calculator):
        """Fast lookup should approximately agree with IAPWS at common temps."""
        for temp_k in [300.0, 350.0, 400.0, 500.0]:
            temp_c = temp_k - 273.15
            vp_fast = calculator.vapor_pressure_fast(temp_k)
            vp_iapws = calculator._iapws_equation(temp_c)
            if math.isfinite(vp_fast) and math.isfinite(vp_iapws):
                assert vp_fast == pytest.approx(vp_iapws, rel=0.01)


# ---------------------------------------------------------------------------
# Tests: quick_water_content convenience function
# ---------------------------------------------------------------------------


class TestQuickWaterContent:
    """Tests for the quick_water_content convenience function."""

    def test_returns_expected_keys(self):
        """Return dict must contain the documented keys."""
        result = quick_water_content(temperature_c=25.0, pressure_bar=1.0)
        assert "water_content_mg_nm3" in result
        assert "water_content_ppmv" in result
        assert "dew_point_c" in result
        assert "mole_fraction" in result

    def test_all_values_finite(self):
        """All returned values should be finite."""
        result = quick_water_content(temperature_c=25.0, pressure_bar=1.0)
        for key, value in result.items():
            assert math.isfinite(value), f"{key} is not finite: {value}"


# ---------------------------------------------------------------------------
# Tests: estimate_condensation_risk convenience function
# ---------------------------------------------------------------------------


class TestEstimateCondensationRisk:
    """Tests for the estimate_condensation_risk convenience function."""

    def test_critical_risk_below_dewpoint(self):
        """Operating below dew point should be 'Critical - Condensation occurring'."""
        result = estimate_condensation_risk(temperature_c=-10.0, pressure_bar=1.0)
        assert "Critical" in result["condensation_risk"]
        assert result["condensation_occurring"] is True

    def test_low_risk_high_temp(self):
        """Operating well above dew point should be 'Low' risk.

        At 200 C and 1 bar, the dew point margin is ~110 C, which easily
        exceeds the default 10 C safety margin.
        """
        result = estimate_condensation_risk(temperature_c=200.0, pressure_bar=1.0)
        assert result["condensation_risk"] == "Low"
        assert result["condensation_occurring"] is False

    def test_recommended_temperature_above_dewpoint(self):
        """Recommended temperature should be dew point + safety margin."""
        result = estimate_condensation_risk(
            temperature_c=25.0, pressure_bar=1.0, safety_margin_c=15.0
        )
        expected_recommended = result["dew_point_c"] + 15.0
        assert result["recommended_temperature_c"] == pytest.approx(
            expected_recommended, rel=1e-6
        )


# ---------------------------------------------------------------------------
# Tests: Water content curve generation
# ---------------------------------------------------------------------------


class TestWaterContentCurveGeneration:
    """Tests for generate_water_content_curve."""

    def test_returns_dataframe_with_expected_columns(self, calculator):
        """Generated curve should have the expected column names."""
        df = calculator.generate_water_content_curve(pressure_bar=10.0, num_points=10)
        expected_cols = {
            "temperature_c",
            "water_content_mg_nm3",
            "water_content_ppmv",
            "water_mole_fraction",
            "vapor_pressure_bar",
        }
        assert set(df.columns) == expected_cols

    def test_correct_number_of_points(self, calculator):
        """Generated curve should have exactly num_points rows."""
        df = calculator.generate_water_content_curve(pressure_bar=10.0, num_points=25)
        assert len(df) == 25

    def test_temperature_range_respected(self, calculator):
        """Temperature values in the curve should span the requested range."""
        df = calculator.generate_water_content_curve(
            pressure_bar=10.0, temp_range=(0, 50), num_points=10
        )
        assert df["temperature_c"].min() == pytest.approx(0.0)
        assert df["temperature_c"].max() == pytest.approx(50.0)

    def test_water_content_increases_along_curve(self, calculator):
        """Water content should increase with temperature in the curve."""
        df = calculator.generate_water_content_curve(
            pressure_bar=10.0, temp_range=(10, 80), num_points=20
        )
        ppmv_values = df["water_content_ppmv"].tolist()
        for i in range(1, len(ppmv_values)):
            assert ppmv_values[i] >= ppmv_values[i - 1]


# ---------------------------------------------------------------------------
# Tests: Mixture molecular weight calculation
# ---------------------------------------------------------------------------


class TestMixtureMolecularWeight:
    """Tests for _calculate_mixture_mw."""

    def test_pure_hydrogen(self, calculator):
        """Pure H2 should give MW ~ 2.016 g/mol."""
        comp = SyngasComposition(h2=1.0)
        mw = calculator._calculate_mixture_mw(comp)
        assert mw == pytest.approx(2.016, abs=0.01)

    def test_pure_co(self, calculator):
        """Pure CO should give MW ~ 28.01 g/mol."""
        comp = SyngasComposition(co=1.0)
        mw = calculator._calculate_mixture_mw(comp)
        assert mw == pytest.approx(28.01, abs=0.01)

    def test_zero_composition_returns_default(self, calculator):
        """All-zero composition should return the default MW."""
        comp = SyngasComposition()
        mw = calculator._calculate_mixture_mw(comp)
        assert mw == pytest.approx(15.0, abs=0.1)  # MW_SYNGAS_TYPICAL_GMOL

    def test_fifty_fifty_h2_co(self, calculator):
        """50/50 H2/CO should give average of their molecular weights."""
        comp = SyngasComposition(h2=0.5, co=0.5)
        mw = calculator._calculate_mixture_mw(comp)
        expected = 0.5 * 2.016 + 0.5 * 28.01
        assert mw == pytest.approx(expected, abs=0.01)
