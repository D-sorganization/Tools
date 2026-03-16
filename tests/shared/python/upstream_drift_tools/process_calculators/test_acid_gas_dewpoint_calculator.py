"""Unit tests for acid_gas_dewpoint_calculator.py.

Imports directly from the source module (not the package __init__)
to avoid pulling in scipy/matplotlib via ode_solver at import time.
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup: point at the src tree so we can import without package install
# ---------------------------------------------------------------------------
_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../src/shared/python")
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Import directly from the module, not via the package __init__
from upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (  # noqa: E402
    ACID_GAS_PRESETS,
    AcidGasComposition,
    AcidGasDewpointCalculator,
    DewpointResult,
    estimate_condensation_risk,
    quick_dewpoint_calculation,
)

# ===========================================================================
# AcidGasComposition tests
# ===========================================================================


class TestAcidGasComposition:
    def test_normalize_sums_to_one(self) -> None:
        comp = AcidGasComposition(h2o=0.5, hf=0.5, hcl=0.0, h2s=0.0, other=1.0)
        norm = comp.normalize()
        assert math.isclose(norm.total, 1.0, rel_tol=1e-9)

    def test_normalize_preserves_ratios(self) -> None:
        comp = AcidGasComposition(h2o=1.0, hf=1.0, hcl=0.0, h2s=0.0, other=0.0)
        norm = comp.normalize()
        assert math.isclose(norm.h2o, 0.5, rel_tol=1e-9)
        assert math.isclose(norm.hf, 0.5, rel_tol=1e-9)

    def test_normalize_already_normalized(self) -> None:
        comp = AcidGasComposition(h2o=0.4, hf=0.1, hcl=0.2, h2s=0.1, other=0.2)
        norm = comp.normalize()
        assert math.isclose(norm.total, 1.0, rel_tol=1e-9)

    def test_normalize_zero_total_returns_original(self) -> None:
        comp = AcidGasComposition(h2o=0.0)
        norm = comp.normalize()
        assert norm.total == 0.0

    def test_to_dict_keys(self) -> None:
        comp = AcidGasComposition(h2o=0.1, hf=0.2, hcl=0.3, h2s=0.4, other=0.0)
        d = comp.to_dict()
        assert set(d.keys()) == {"H2O", "HF", "HCl", "H2S", "Other"}

    def test_to_dict_values(self) -> None:
        comp = AcidGasComposition(h2o=0.1, hf=0.2, hcl=0.3, h2s=0.4, other=0.0)
        d = comp.to_dict()
        assert d["H2O"] == pytest.approx(0.1)
        assert d["HF"] == pytest.approx(0.2)

    def test_total_property(self) -> None:
        comp = AcidGasComposition(h2o=0.1, hf=0.2, hcl=0.3, h2s=0.4, other=0.5)
        assert comp.total == pytest.approx(1.5)

    def test_presets_exist(self) -> None:
        assert "typical_syngas" in ACID_GAS_PRESETS
        assert "biomass_gasification" in ACID_GAS_PRESETS

    def test_preset_is_acid_gas_composition(self) -> None:
        for preset in ACID_GAS_PRESETS.values():
            assert isinstance(preset, AcidGasComposition)


# ===========================================================================
# DewpointResult tests
# ===========================================================================


class TestDewpointResult:
    def _make_result(self) -> DewpointResult:
        return DewpointResult(
            temperature_c=150.0,
            temperature_k=423.15,
            pressure_bar=30.0,
            pressure_pa=3_000_000.0,
            composition=AcidGasComposition(h2o=0.15),
            h2o_dewpoint_c=60.0,
            hf_dewpoint_c=None,
            hcl_dewpoint_c=None,
            h2s_dewpoint_c=None,
            overall_dewpoint_c=60.0,
            limiting_component="H2O",
            h2o_vapor_pressure_pa=500_000.0,
            hf_vapor_pressure_pa=None,
            hcl_vapor_pressure_pa=None,
            h2s_vapor_pressure_pa=None,
            h2o_partial_pressure_pa=450_000.0,
            hf_partial_pressure_pa=None,
            hcl_partial_pressure_pa=None,
            h2s_partial_pressure_pa=None,
            dewpoint_margin_c=90.0,
            condensation_risk="Low",
            calculation_method="antoine",
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            warnings=[],
            sources=["Antoine equation"],
        )

    def test_to_dict_has_input_key(self) -> None:
        d = self._make_result().to_dict()
        assert "input" in d

    def test_to_dict_has_dewpoints_key(self) -> None:
        d = self._make_result().to_dict()
        assert "dewpoints" in d

    def test_to_dict_has_safety_key(self) -> None:
        d = self._make_result().to_dict()
        assert "safety" in d

    def test_to_dict_temperature(self) -> None:
        d = self._make_result().to_dict()
        assert d["input"]["temperature_c"] == pytest.approx(150.0)

    def test_to_dict_overall_dewpoint(self) -> None:
        d = self._make_result().to_dict()
        assert d["dewpoints"]["overall"] == pytest.approx(60.0)

    def test_to_dict_risk_level(self) -> None:
        d = self._make_result().to_dict()
        assert d["safety"]["condensation_risk"] == "Low"

    def test_to_dict_warnings_list(self) -> None:
        d = self._make_result().to_dict()
        assert isinstance(d["warnings"], list)


# ===========================================================================
# AcidGasDewpointCalculator tests
# ===========================================================================


@pytest.fixture
def calc() -> AcidGasDewpointCalculator:
    return AcidGasDewpointCalculator()


class TestCalculateVaporPressure:
    def test_water_at_100c_approx_1atm(self, calc: AcidGasDewpointCalculator) -> None:
        p = calc.calculate_vapor_pressure(100.0, "H2O", "antoine")
        # At 100°C, water VP should be close to 101325 Pa (1 atm)
        assert 90_000 < p < 115_000

    def test_pressure_increases_with_temperature(
        self, calc: AcidGasDewpointCalculator
    ) -> None:
        p_low = calc.calculate_vapor_pressure(50.0, "H2O", "antoine")
        p_high = calc.calculate_vapor_pressure(150.0, "H2O", "antoine")
        assert p_high > p_low

    def test_extended_antoine_returns_positive(
        self, calc: AcidGasDewpointCalculator
    ) -> None:
        for component in ["H2O", "HF", "HCl", "H2S"]:
            p = calc.calculate_vapor_pressure(50.0, component, "extended_antoine")
            assert p > 0, f"VP for {component} should be positive"

    def test_invalid_component_raises(self, calc: AcidGasDewpointCalculator) -> None:
        with pytest.raises(ValueError, match="Unknown component"):
            calc.calculate_vapor_pressure(100.0, "CO2", "antoine")

    def test_invalid_method_raises(self, calc: AcidGasDewpointCalculator) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            calc.calculate_vapor_pressure(100.0, "H2O", "fake_method")

    def test_hf_vapor_pressure(self, calc: AcidGasDewpointCalculator) -> None:
        p = calc.calculate_vapor_pressure(20.0, "HF", "antoine")
        assert p > 0

    def test_hcl_vapor_pressure(self, calc: AcidGasDewpointCalculator) -> None:
        p = calc.calculate_vapor_pressure(20.0, "HCl", "antoine")
        assert p > 0

    def test_h2s_vapor_pressure(self, calc: AcidGasDewpointCalculator) -> None:
        p = calc.calculate_vapor_pressure(20.0, "H2S", "antoine")
        assert p > 0


class TestCalculateDewpoint:
    def test_water_dewpoint_at_1atm(self, calc: AcidGasDewpointCalculator) -> None:
        # partial pressure = 1 atm → dewpoint ≈ 100°C
        t = calc.calculate_dewpoint(101_325.0, "H2O")
        assert 98.0 < t < 102.0

    def test_lower_partial_pressure_lower_dewpoint(
        self, calc: AcidGasDewpointCalculator
    ) -> None:
        t_low = calc.calculate_dewpoint(5000.0, "H2O")
        t_high = calc.calculate_dewpoint(50_000.0, "H2O")
        assert t_low < t_high

    def test_zero_pressure_raises(self, calc: AcidGasDewpointCalculator) -> None:
        with pytest.raises(ValueError, match="partial_pressure_pa must be > 0"):
            calc.calculate_dewpoint(0.0, "H2O")

    def test_negative_pressure_raises(self, calc: AcidGasDewpointCalculator) -> None:
        with pytest.raises(ValueError, match="partial_pressure_pa must be > 0"):
            calc.calculate_dewpoint(-100.0, "H2O")

    def test_unknown_component_raises(self, calc: AcidGasDewpointCalculator) -> None:
        with pytest.raises(ValueError, match="unknown component"):
            calc.calculate_dewpoint(1000.0, "NO2")


class TestCalculateDewpointMixture:
    def test_typical_syngas_returns_result(
        self, calc: AcidGasDewpointCalculator
    ) -> None:
        comp = ACID_GAS_PRESETS["typical_syngas"]
        result = calc.calculate_dewpoint_mixture(
            temperature_c=150.0, pressure_bar=30.0, composition=comp
        )
        assert isinstance(result, DewpointResult)
        assert result.overall_dewpoint_c is not None
        assert not math.isnan(result.overall_dewpoint_c)

    def test_margin_is_positive_well_above_dewpoint(
        self, calc: AcidGasDewpointCalculator
    ) -> None:
        comp = ACID_GAS_PRESETS["typical_syngas"]
        result = calc.calculate_dewpoint_mixture(
            temperature_c=300.0, pressure_bar=10.0, composition=comp
        )
        assert result.dewpoint_margin_c > 0

    def test_zero_pressure_raises(self, calc: AcidGasDewpointCalculator) -> None:
        comp = AcidGasComposition(h2o=0.1)
        with pytest.raises(ValueError):
            calc.calculate_dewpoint_mixture(150.0, 0.0, comp)

    def test_unphysical_temperature_raises(
        self, calc: AcidGasDewpointCalculator
    ) -> None:
        comp = AcidGasComposition(h2o=0.1)
        with pytest.raises(ValueError):
            calc.calculate_dewpoint_mixture(-300.0, 10.0, comp)

    def test_warns_on_out_of_range_temperature(
        self, calc: AcidGasDewpointCalculator
    ) -> None:
        comp = AcidGasComposition(h2o=0.01)  # very low fractions
        result = calc.calculate_dewpoint_mixture(
            temperature_c=-120.0,  # below expected range
            pressure_bar=0.005,  # also below range
            composition=comp,
        )
        # Should have generated at least one warning
        assert len(result.warnings) > 0

    def test_limiting_component_is_known_species(
        self, calc: AcidGasDewpointCalculator
    ) -> None:
        comp = ACID_GAS_PRESETS["biomass_gasification"]
        result = calc.calculate_dewpoint_mixture(
            temperature_c=200.0, pressure_bar=20.0, composition=comp
        )
        if result.limiting_component is not None:
            assert result.limiting_component in ("H2O", "HF", "HCl", "H2S")

    def test_composition_dict_stored_in_result(
        self, calc: AcidGasDewpointCalculator
    ) -> None:
        comp = ACID_GAS_PRESETS["typical_syngas"]
        result = calc.calculate_dewpoint_mixture(
            temperature_c=150.0, pressure_bar=30.0, composition=comp
        )
        assert isinstance(result.composition, AcidGasComposition)


class TestGenerateDewpointCurves:
    def test_returns_dataframe(self, calc: AcidGasDewpointCalculator) -> None:
        comp = ACID_GAS_PRESETS["typical_syngas"]
        df = calc.generate_dewpoint_curves(
            pressure_bar=10.0,
            composition=comp,
            temp_range=(80, 200),
            num_points=5,
        )
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self, calc: AcidGasDewpointCalculator) -> None:
        comp = ACID_GAS_PRESETS["typical_syngas"]
        df = calc.generate_dewpoint_curves(
            pressure_bar=10.0,
            composition=comp,
            temp_range=(80, 200),
            num_points=7,
        )
        assert len(df) == 7

    def test_overall_dewpoint_column_present(
        self, calc: AcidGasDewpointCalculator
    ) -> None:
        comp = ACID_GAS_PRESETS["typical_syngas"]
        df = calc.generate_dewpoint_curves(
            pressure_bar=10.0, composition=comp, temp_range=(80, 200), num_points=5
        )
        assert "Overall_Dewpoint_C" in df.columns

    def test_temperature_column_present(self, calc: AcidGasDewpointCalculator) -> None:
        comp = ACID_GAS_PRESETS["typical_syngas"]
        df = calc.generate_dewpoint_curves(
            pressure_bar=10.0, composition=comp, temp_range=(80, 200), num_points=5
        )
        assert "Temperature_C" in df.columns


# ===========================================================================
# Standalone helper functions
# ===========================================================================


class TestQuickDewpointCalculation:
    def test_returns_dict(self) -> None:
        result = quick_dewpoint_calculation(
            temperature_c=120.0,
            pressure_bar=15.0,
            h2o_fraction=0.2,
            hf_fraction=0.01,
        )
        assert isinstance(result, dict)

    def test_has_overall_dewpoint_key(self) -> None:
        result = quick_dewpoint_calculation(
            temperature_c=120.0,
            pressure_bar=15.0,
            h2o_fraction=0.2,
        )
        assert "overall_dewpoint_c" in result

    def test_limiting_component_is_valid(self) -> None:
        result = quick_dewpoint_calculation(
            temperature_c=120.0,
            pressure_bar=15.0,
            h2o_fraction=0.2,
            hf_fraction=0.01,
        )
        if result["limiting_component"] is not None:
            assert result["limiting_component"] in ("H2O", "HF", "HCl", "H2S")

    def test_water_dominated_mix(self) -> None:
        result = quick_dewpoint_calculation(
            temperature_c=200.0,
            pressure_bar=5.0,
            h2o_fraction=0.9,
            hf_fraction=0.001,
        )
        # H2O should dominate
        assert result["limiting_component"] == "H2O"


class TestEstimateCondensationRisk:
    def test_low_temperature_is_high_risk(self) -> None:
        comp = AcidGasComposition(h2o=0.9)
        result = estimate_condensation_risk(
            temperature_c=5.0,
            pressure_bar=10.0,
            composition=comp,
            safety_margin_c=10.0,
        )
        assert result["risk_level"] in ("High", "Critical")

    def test_high_temperature_is_low_risk(self) -> None:
        comp = AcidGasComposition(h2o=0.05, hf=0.001, hcl=0.001, h2s=0.001)
        result = estimate_condensation_risk(
            temperature_c=300.0,
            pressure_bar=5.0,
            composition=comp,
            safety_margin_c=10.0,
        )
        assert result["risk_level"] == "Low"

    def test_result_has_risk_level_key(self) -> None:
        comp = AcidGasComposition(h2o=0.1)
        result = estimate_condensation_risk(
            temperature_c=100.0, pressure_bar=10.0, composition=comp
        )
        assert "risk_level" in result

    def test_result_has_margin_key(self) -> None:
        comp = AcidGasComposition(h2o=0.1)
        result = estimate_condensation_risk(
            temperature_c=100.0, pressure_bar=10.0, composition=comp
        )
        assert "current_margin_c" in result


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-p", "no:cov", __file__]))
