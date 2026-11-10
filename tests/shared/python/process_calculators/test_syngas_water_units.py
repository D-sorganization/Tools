"""Golden-value tests for syngas water content unit conversions (#6004)."""

from __future__ import annotations

import math

import pytest
from upstream_drift_tools.process_calculators.constants import (
    MW_WATER_GMOL,
    NORMAL_PRESSURE_PA,
    NORMAL_TEMPERATURE_K,
    SCF_PER_LBMOL_60F_14696PSIA,
    STANDARD_BASIS_LB_MMSCF,
)
from upstream_drift_tools.process_calculators.syngas_water_calculator import (
    SyngasWaterCalculator,
)
from upstream_drift_tools.process_calculators.water_content_units import (
    water_content_units,
)
from upstream_drift_tools.utils.unit_constants import R_UNIVERSAL

pytestmark = pytest.mark.unit

_TOL = 0.01  # 1 % per issue acceptance


def _expected_g_m3(vapor_pressure_pa: float, temperature_k: float) -> float:
    # ρ_v = P_w·MW / (R·T); R_UNIVERSAL [J/(mol·K)], MW_WATER_GMOL [g/mol].
    return vapor_pressure_pa * MW_WATER_GMOL / (R_UNIVERSAL * temperature_k)


def _expected_lb_mmscf(ppmv: float) -> float:
    # lb/MMscf = ppmv·MW / (scf/lb-mol at 60 °F, 14.696 psia).
    return ppmv * MW_WATER_GMOL / SCF_PER_LBMOL_60F_14696PSIA


@pytest.fixture
def calculator() -> SyngasWaterCalculator:
    return SyngasWaterCalculator()


class TestSyngasWaterUnitGoldenValues:
    def test_saturated_40c_g_per_m3(self, calculator: SyngasWaterCalculator) -> None:
        """40 °C saturated at ~1 atm → ~51.1 g/m³ (Psat ≈ 7384 Pa)."""
        result = calculator.calculate_water_content(
            temperature_c=40.0,
            pressure_bar=1.01325,
            gas_composition="typical_syngas",
        )
        t_k = 40.0 + 273.15
        expected = _expected_g_m3(result.vapor_pressure_pa, t_k)
        assert result.vapor_pressure_pa == pytest.approx(7384.0, rel=0.02)
        assert result.water_content_g_per_m3 == pytest.approx(expected, rel=_TOL)
        assert result.water_content_g_per_m3 == pytest.approx(51.1, rel=_TOL)

    def test_1000_ppmv_lb_per_mmscf(self, calculator: SyngasWaterCalculator) -> None:
        """1000 ppmv → ~47.5 lb/MMscf."""
        y_water = 1000.0 / 1e6
        t_k = 313.15
        p_w = y_water * 101325.0
        units = calculator._convert_water_content_units(y_water, p_w, t_k)
        expected = _expected_lb_mmscf(1000.0)
        assert units.lb_mmscf == pytest.approx(expected, rel=_TOL)
        assert units.lb_mmscf == pytest.approx(47.5, rel=_TOL)

    def test_normal_conditions_mg_nm3_consistency(self) -> None:
        """g/m³ at normal P,T equals mg/Nm³ / 1000."""
        y_water = 0.05
        vapor_pa = y_water * NORMAL_PRESSURE_PA
        units = water_content_units(y_water, vapor_pa, NORMAL_TEMPERATURE_K)
        assert units.mg_nm3 == pytest.approx(units.g_m3 * 1000.0, rel=1e-6)

    def test_result_reports_standard_basis(
        self, calculator: SyngasWaterCalculator
    ) -> None:
        result = calculator.calculate_water_content(
            temperature_c=40.0,
            pressure_bar=1.0,
        )
        assert result.water_content_standard_basis == STANDARD_BASIS_LB_MMSCF


class TestEngineMatchesFallbackUnits:
    def test_kernel_matches_direct_formula(self) -> None:
        y = 0.01
        t_k = 300.0
        p_w = y * 5.0e5
        units = water_content_units(y, p_w, t_k)
        assert units.g_m3 == pytest.approx(_expected_g_m3(p_w, t_k), rel=1e-9)
        assert units.lb_mmscf == pytest.approx(_expected_lb_mmscf(units.ppmv), rel=1e-9)

    def test_engine_matches_shared_kernel(
        self, calculator: SyngasWaterCalculator
    ) -> None:
        """Full calculator and ``water_content_units`` share one code path."""
        result = calculator.calculate_water_content(
            temperature_c=25.0,
            pressure_bar=1.0,
            composition_key="typical_syngas",
        )
        units = water_content_units(
            result.mole_fraction_water,
            result.vapor_pressure_pa,
            result.temperature_k,
        )
        assert result.water_content_g_per_m3 == pytest.approx(units.g_m3, rel=1e-9)
        assert result.water_content_lb_per_mmscf == pytest.approx(
            units.lb_mmscf, rel=1e-9
        )
        assert result.water_content_mg_per_nm3 == pytest.approx(units.mg_nm3, rel=1e-9)
        assert result.water_content_ppmv == pytest.approx(units.ppmv, rel=1e-9)
        assert math.isfinite(result.water_content_g_per_m3)
