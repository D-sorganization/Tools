"""Golden-value tests for syngas water content unit conversions (Refs #6004)."""

from __future__ import annotations

import math

import pytest
from upstream_drift_tools.process_calculators.constants import (
    MW_WATER_GMOL,
    NORMAL_PRESSURE_PA,
    NORMAL_TEMPERATURE_K,
)
from upstream_drift_tools.process_calculators.syngas_water_calculator import (
    SyngasWaterCalculator,
    water_content_units,
)
from upstream_drift_tools.utils.unit_constants import R_UNIVERSAL

pytestmark = pytest.mark.unit

_TOL = 0.01  # 1 % per issue acceptance


def _expected_g_m3(vapor_pressure_pa: float, temperature_k: float) -> float:
    return vapor_pressure_pa * MW_WATER_GMOL / (R_UNIVERSAL * temperature_k)


def _expected_lb_mmscf(ppmv: float) -> float:
    return ppmv * MW_WATER_GMOL / 379.49


@pytest.fixture
def calculator() -> SyngasWaterCalculator:
    return SyngasWaterCalculator()


class TestSyngasWaterUnitGoldenValues:
    def test_saturated_40c_g_per_m3(self, calculator: SyngasWaterCalculator) -> None:
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
        y_water = 1000.0 / 1e6
        t_k = 313.15
        p_pa = 101325.0
        units = water_content_units(y_water=y_water, p_pa=p_pa, t_k=t_k)
        expected = _expected_lb_mmscf(1000.0)
        assert units.lb_mmscf == pytest.approx(expected, rel=_TOL)
        assert units.lb_mmscf == pytest.approx(47.5, rel=_TOL)

    def test_normal_conditions_mg_nm3_consistency(self) -> None:
        y_water = 0.05
        units = water_content_units(
            y_water=y_water,
            p_pa=NORMAL_PRESSURE_PA,
            t_k=NORMAL_TEMPERATURE_K,
        )
        assert units.mg_nm3 == pytest.approx(units.g_m3 * 1000.0, rel=1e-6)

    def test_result_reports_standard_basis(
        self, calculator: SyngasWaterCalculator
    ) -> None:
        result = calculator.calculate_water_content(
            temperature_c=40.0,
            pressure_bar=1.0,
        )
        assert result.standard_basis == "60F_14.696psia"


class TestEngineMatchesFallbackUnits:
    def test_kernel_matches_direct_formula(self) -> None:
        y = 0.01
        t_k = 300.0
        p_pa = 5.0e5
        p_w = y * p_pa
        units = water_content_units(y_water=y, p_pa=p_pa, t_k=t_k)
        assert units.g_m3 == pytest.approx(_expected_g_m3(p_w, t_k), rel=1e-9)
        assert units.lb_mmscf == pytest.approx(_expected_lb_mmscf(units.ppmv), rel=1e-9)

    def test_engine_matches_shared_kernel(
        self, calculator: SyngasWaterCalculator
    ) -> None:
        result = calculator.calculate_water_content(
            temperature_c=25.0,
            pressure_bar=1.0,
            gas_composition="typical_syngas",
        )
        units = water_content_units(
            y_water=result.mole_fraction_water,
            p_pa=result.pressure_pa,
            t_k=result.temperature_k,
        )
        assert result.water_content_g_per_m3 == pytest.approx(units.g_m3, rel=1e-9)
        assert result.water_content_lb_per_mmscf == pytest.approx(
            units.lb_mmscf, rel=1e-9
        )
        assert result.water_content_mg_per_nm3 == pytest.approx(units.mg_nm3, rel=1e-9)
        assert result.water_content_ppmv == pytest.approx(units.ppmv, rel=1e-9)
        assert math.isfinite(result.water_content_g_per_m3)
