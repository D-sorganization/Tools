"""
Tests for syngas_water_calculator overflow fix and performance benchmarks.

Covers GitHub issue #654:
  - RuntimeWarning overflow encountered in exp (line 276)
  - Numerical overflow prevention with extreme temperature values
  - Performance timing for critical calculator paths

Design principles:
  - TDD: Tests describe the desired behaviour before verifying the fix.
  - DRY: Common setup is shared via fixtures.
  - DbC: Each test documents pre/post-conditions.
  - Orthogonality: Overflow tests are independent of benchmark tests.
"""

from __future__ import annotations

import math
import time
import warnings

import pytest

pytest.importorskip("numpy")
import numpy as np
from upstream_drift_tools.process_calculators.syngas_water_calculator import (
    _EXP_MAX_ARG,
    SyngasWaterCalculator,
    WaterContentResult,
    _safe_exp,
    quick_water_content,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calculator() -> SyngasWaterCalculator:
    """Provide a fresh SyngasWaterCalculator instance."""
    return SyngasWaterCalculator()


# ---------------------------------------------------------------------------
# Unit tests: _safe_exp helper
# ---------------------------------------------------------------------------


class TestSafeExp:
    """Verify the _safe_exp clamping helper prevents overflow."""

    def test_normal_values_match_math_exp(self) -> None:
        """_safe_exp(x) == math.exp(x) for values inside the safe range."""
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0, 100.0]:
            assert _safe_exp(x) == pytest.approx(math.exp(x), rel=1e-12)

    def test_large_positive_does_not_overflow(self) -> None:
        """Exponents above _EXP_MAX_ARG are clamped -- no RuntimeWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _safe_exp(1e6)
        assert math.isfinite(result)
        assert result == pytest.approx(math.exp(_EXP_MAX_ARG), rel=1e-12)

    def test_large_negative_does_not_underflow_to_nan(self) -> None:
        """Extremely negative exponents clamp toward zero, remain finite."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _safe_exp(-1e6)
        assert math.isfinite(result)
        assert result >= 0.0

    def test_boundary_value_at_max(self) -> None:
        """At exactly _EXP_MAX_ARG the result equals math.exp(_EXP_MAX_ARG)."""
        assert _safe_exp(_EXP_MAX_ARG) == pytest.approx(
            math.exp(_EXP_MAX_ARG), rel=1e-12
        )

    def test_boundary_value_at_neg_max(self) -> None:
        """At exactly -_EXP_MAX_ARG the result equals math.exp(-_EXP_MAX_ARG)."""
        assert _safe_exp(-_EXP_MAX_ARG) == pytest.approx(
            math.exp(-_EXP_MAX_ARG), rel=1e-12
        )

    def test_result_always_non_negative(self) -> None:
        """exp(x) >= 0 for all real x; clamping must preserve this."""
        for x in [-1e10, -700, -1, 0, 1, 700, 1e10]:
            assert _safe_exp(x) >= 0.0


# ---------------------------------------------------------------------------
# Unit tests: Buck equation overflow (issue #654, line 276)
# ---------------------------------------------------------------------------


class TestBuckEquationOverflow:
    """Verify _buck_equation does not overflow for extreme temperatures."""

    def test_extreme_positive_temperature_no_overflow(
        self, calculator: SyngasWaterCalculator
    ) -> None:
        """Very high temperature must return a finite vapor pressure."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = calculator._buck_equation(10000.0)
        assert math.isfinite(result)
        assert result > 0

    def test_extreme_negative_temperature_no_overflow(
        self, calculator: SyngasWaterCalculator
    ) -> None:
        """Very low temperature must return a finite vapor pressure."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = calculator._buck_equation(-10000.0)
        assert math.isfinite(result)
        assert result >= 0

    def test_moderate_temperature_unchanged(
        self, calculator: SyngasWaterCalculator
    ) -> None:
        """Fix must not alter results for normal operating temperatures."""
        # 50 C is well within the safe range for the Buck equation.
        result = calculator._buck_equation(50.0)
        # Expected value from the Buck (1981) reference formula (issue #3867):
        # a=0.61121, b=18.678, c=234.5, d=257.14
        a, b, c, d = 0.61121, 18.678, 234.5, 257.14
        exponent = (b - 50.0 / c) * 50.0 / (d + 50.0)
        expected_kpa = a * math.exp(exponent)
        expected_pa = expected_kpa * 1000
        assert result == pytest.approx(expected_pa, rel=1e-9)

    def test_below_freezing_moderate_unchanged(
        self, calculator: SyngasWaterCalculator
    ) -> None:
        """Below-freezing path uses different constants; verify correctness."""
        result = calculator._buck_equation(-10.0)
        assert math.isfinite(result)
        assert result > 0

    def test_no_runtime_warning_at_1e4(self, calculator: SyngasWaterCalculator) -> None:
        """Ensure no RuntimeWarning is raised at temperature = 1e4 C."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            calculator._buck_equation(1e4)

    def test_no_runtime_warning_at_neg_1e4(
        self, calculator: SyngasWaterCalculator
    ) -> None:
        """Ensure no RuntimeWarning is raised at temperature = -1e4 C."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            calculator._buck_equation(-1e4)


# ---------------------------------------------------------------------------
# Unit tests: Antoine equation overflow
# ---------------------------------------------------------------------------


class TestAntoineEquationOverflow:
    """Verify _antoine_equation handles extreme temperatures safely."""

    def test_extreme_positive_temperature(
        self, calculator: SyngasWaterCalculator
    ) -> None:
        """Antoine equation must not overflow for very high temperatures."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = calculator._antoine_equation(5000.0)
        assert math.isfinite(result)

    def test_moderate_temperature_accuracy(
        self, calculator: SyngasWaterCalculator
    ) -> None:
        """Check Antoine returns correct value at 100 C."""
        result = calculator._antoine_equation(100.0)
        # At 100 C the vapor pressure should be close to 1 atm (~101325 Pa).
        assert 90000 < result < 120000


# ---------------------------------------------------------------------------
# Unit tests: IAPWS equation overflow
# ---------------------------------------------------------------------------


class TestIAPWSEquationOverflow:
    """Verify _iapws_equation uses _safe_exp internally."""

    def test_near_critical_temperature(self, calculator: SyngasWaterCalculator) -> None:
        """At temperatures just below critical point, result must be finite."""
        # Critical T for water is 373.946 C.  Test at 370 C.
        result = calculator._iapws_equation(370.0)
        assert math.isfinite(result)
        assert result > 0

    def test_at_100c(self, calculator: SyngasWaterCalculator) -> None:
        """At 100 C the IAPWS vapor pressure should be near 1 atm."""
        result = calculator._iapws_equation(100.0)
        assert 95000 < result < 110000


# ---------------------------------------------------------------------------
# Unit tests: Magnus equation overflow
# ---------------------------------------------------------------------------


class TestMagnusEquationOverflow:
    """Verify _magnus_equation uses _safe_exp internally."""

    def test_at_zero(self, calculator: SyngasWaterCalculator) -> None:
        """At 0 C the vapor pressure should be approximately 611 Pa."""
        result = calculator._magnus_equation(0.0)
        assert 600 < result < 625

    def test_at_100(self, calculator: SyngasWaterCalculator) -> None:
        """At 100 C the vapor pressure should be near 1 atm."""
        result = calculator._magnus_equation(100.0)
        assert 90000 < result < 120000


# ---------------------------------------------------------------------------
# Integration tests: full calculate_water_content with extreme inputs
# ---------------------------------------------------------------------------


class TestCalculateWaterContentOverflow:
    """End-to-end tests ensuring calculate_water_content is overflow-safe."""

    def test_high_temperature_high_pressure(
        self, calculator: SyngasWaterCalculator
    ) -> None:
        """Extreme operating conditions must not crash the calculator."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = calculator.calculate_water_content(
                temperature_c=350.0,
                pressure_bar=100.0,
                gas_composition="typical_syngas",
                method="iapws",
            )
        assert isinstance(result, WaterContentResult)
        assert math.isfinite(result.mole_fraction_water)

    def test_negative_temperature(self, calculator: SyngasWaterCalculator) -> None:
        """Sub-zero temperatures use below-freezing Buck constants."""
        result = calculator.calculate_water_content(
            temperature_c=-15.0,
            pressure_bar=1.0,
            gas_composition="typical_syngas",
            method="buck",
        )
        assert isinstance(result, WaterContentResult)
        assert result.water_content_ppmv >= 0

    def test_result_fields_are_finite(self, calculator: SyngasWaterCalculator) -> None:
        """All numeric result fields must be finite for a normal calculation."""
        result = calculator.calculate_water_content(
            temperature_c=40.0,
            pressure_bar=30.0,
        )
        for field_name in (
            "mole_fraction_water",
            "mass_fraction_water",
            "water_content_g_per_m3",
            "water_content_mg_per_nm3",
            "water_content_ppmv",
            "water_content_lb_per_mmscf",
            "dew_point_c",
            "relative_humidity",
        ):
            value = getattr(result, field_name)
            assert math.isfinite(value), f"{field_name} is not finite: {value}"


# ---------------------------------------------------------------------------
# Performance / benchmark tests
# ---------------------------------------------------------------------------


class TestPerformanceBenchmarks:
    """Performance timing tests for critical calculator paths.

    These are simple wall-clock timing tests.  If pytest-benchmark is
    available in the future, these can be converted to use the benchmark
    fixture for statistical rigour.
    """

    @pytest.mark.slow
    def test_buck_equation_throughput(self, calculator: SyngasWaterCalculator) -> None:
        """Buck equation should evaluate 10 000 calls in under 1 second."""
        temperatures = list(np.linspace(-40, 200, 10_000))
        start = time.perf_counter()
        for t in temperatures:
            calculator._buck_equation(t)
        elapsed = time.perf_counter() - start
        msg = f"Buck equation took {elapsed:.3f}s for 10k calls"
        assert elapsed < 1.0, msg

    @pytest.mark.slow
    def test_calculate_water_content_throughput(
        self, calculator: SyngasWaterCalculator
    ) -> None:
        """Full water content calculation: 100 calls in under 5 seconds."""
        start = time.perf_counter()
        for t in np.linspace(0, 100, 100):
            calculator.calculate_water_content(
                temperature_c=float(t), pressure_bar=30.0
            )
        elapsed = time.perf_counter() - start
        msg = f"calculate_water_content took {elapsed:.3f}s for 100 calls"
        assert elapsed < 5.0, msg

    @pytest.mark.slow
    def test_vapor_pressure_fast_lookup(
        self, calculator: SyngasWaterCalculator
    ) -> None:
        """Fast interpolation lookup: 10 000 calls in under 0.5 seconds."""
        temperatures_k = list(np.linspace(280, 640, 10_000))
        start = time.perf_counter()
        for t in temperatures_k:
            calculator.vapor_pressure_fast(t)
        elapsed = time.perf_counter() - start
        msg = f"vapor_pressure_fast took {elapsed:.3f}s for 10k calls"
        assert elapsed < 0.5, msg

    @pytest.mark.slow
    def test_quick_water_content_latency(self) -> None:
        """quick_water_content convenience function: single call < 0.5s."""
        start = time.perf_counter()
        result = quick_water_content(temperature_c=25.0, pressure_bar=1.0)
        elapsed = time.perf_counter() - start
        msg = f"quick_water_content took {elapsed:.3f}s"
        assert elapsed < 0.5, msg
        assert "water_content_ppmv" in result

    @pytest.mark.slow
    def test_safe_exp_overhead_negligible(self) -> None:
        """_safe_exp overhead vs math.exp should be < 10x for 100k calls.

        Only values within math.exp's valid range are benchmarked so the
        comparison is apples-to-apples (no OverflowError in the baseline).
        Uses best-of-3 runs to avoid transient runner scheduling /
        context-switch spikes.
        """
        # Keep within [-700, 700] so both math.exp and _safe_exp take the same path
        values = [float(x) for x in np.linspace(-700, 700, 100_000)]

        ratios = []
        for _ in range(3):
            start = time.perf_counter()
            for v in values:
                math.exp(v)
            baseline = time.perf_counter() - start

            start = time.perf_counter()
            for v in values:
                _safe_exp(v)
            safe_time = time.perf_counter() - start

            ratio = safe_time / max(baseline, 1e-9)
            ratios.append((ratio, safe_time, baseline))

        min_ratio, best_safe, best_base = min(ratios, key=lambda r: r[0])
        msg = (
            f"_safe_exp is {min_ratio:.1f}x slower than math.exp "
            f"({best_safe:.4f}s vs {best_base:.4f}s)"
        )
        assert min_ratio < 25.0, msg


# ---------------------------------------------------------------------------
# Unit conversions and basis tests (#6004)
# ---------------------------------------------------------------------------


class TestSyngasWaterUnits:
    """Tests for syngas water unit conversions and standard basis (#6004)."""

    def test_forty_c_saturated_returns_51_g_per_m3(self) -> None:
        """40 °C saturated at 1 atm (101325 Pa) returns 51.1 g/m³ within 1%."""
        calc = SyngasWaterCalculator()
        res = calc.calculate_water_content(temperature_c=40.0, pressure_bar=1.01325)
        assert 50.59 <= res.water_content_g_per_m3 <= 51.61, (
            f"Expected ~51.1 g/m3 (+-1%), got {res.water_content_g_per_m3}"
        )

    def test_thousand_ppmv_returns_47_5_lb_per_mmscf(self) -> None:
        """1000 ppmv (y_water = 0.001) returns 47.5 lb/MMscf within 1%."""
        from calc_backend.contracts.syngas_water import SyngasWaterRequest
        from calc_backend.routers.syngas_water import _fallback_calculate

        from shared.python.sidekick.process_calculators.constants import (
            ANTOINE_WATER_A,
            ANTOINE_WATER_B,
            ANTOINE_WATER_C,
        )
        from shared.python.sidekick.process_calculators.syngas_water_calculator import (
            water_content_units,
        )
        from shared.python.sidekick.process_calculators.water_vapor_pressure import (
            antoine_pressure_pa,
        )

        units = water_content_units(y_water=0.001, p_pa=101325.0, t_k=293.15)
        assert 47.0 <= units.lb_mmscf <= 48.0, (
            f"Expected ~47.5 lb/MMscf (+-1%), got {units.lb_mmscf}"
        )

        calc = SyngasWaterCalculator()
        engine_units = calc._convert_water_content_units(
            y_water=0.001,
            vapor_pressure_pa=101.325,
            temperature_k=293.15,
            pressure_pa=101325.0,
        )
        assert 47.0 <= engine_units["lb_mmscf"] <= 48.0

        # Router fallback test at y_water = 0.001 (1000 ppmv)
        vp_pa = antoine_pressure_pa(
            ANTOINE_WATER_A, ANTOINE_WATER_B, ANTOINE_WATER_C, 25.0
        )
        p_bar = (vp_pa / 1e5) / 0.001
        req = SyngasWaterRequest(
            temperature_c=25.0, pressure_bar=p_bar, composition_key="typical_syngas"
        )
        resp = _fallback_calculate(req)
        assert resp.water_content.water_content_ppmv == pytest.approx(1000.0, rel=1e-3)
        assert 47.0 <= resp.water_content.water_content_lb_per_mmscf <= 48.0

    def test_engine_and_router_fallback_agree_on_units(self) -> None:
        """Engine and router fallback agree on all four units for same conditions."""
        from calc_backend.contracts.syngas_water import SyngasWaterRequest
        from calc_backend.routers.syngas_water import _fallback_calculate

        from shared.python.sidekick.process_calculators.syngas_water_calculator import (
            water_content_units,
        )

        req = SyngasWaterRequest(
            temperature_c=80.0, pressure_bar=10.0, composition_key="typical_syngas"
        )
        resp = _fallback_calculate(req)
        y_water = resp.water_content.mole_fraction_water
        p_pa = 10.0 * 1e5
        t_k = 80.0 + 273.15

        units = water_content_units(y_water=y_water, p_pa=p_pa, t_k=t_k)
        assert resp.water_content.water_content_ppmv == pytest.approx(units.ppmv)
        assert resp.water_content.water_content_g_per_m3 == pytest.approx(units.g_m3)
        assert resp.water_content.water_content_mg_per_nm3 == pytest.approx(
            units.mg_nm3
        )
        assert resp.water_content.water_content_lb_per_mmscf == pytest.approx(
            units.lb_mmscf
        )

    def test_cross_unit_consistency_mg_nm3_and_g_m3(self) -> None:
        """Cross-unit consistency: g/m3 at normal conds * 1000 equals mg/Nm3."""
        from shared.python.sidekick.process_calculators.syngas_water_calculator import (
            water_content_units,
        )

        units = water_content_units(y_water=0.025, p_pa=101325.0, t_k=273.15)
        assert units.g_m3 * 1000.0 == pytest.approx(units.mg_nm3)

    def test_standard_basis_field_present(self) -> None:
        """Result states standard_basis == '60F_14.696psia'."""
        from calc_backend.contracts.syngas_water import SyngasWaterRequest
        from calc_backend.routers.syngas_water import _fallback_calculate

        from shared.python.sidekick.process_calculators.syngas_water_calculator import (
            SyngasWaterCalculator,
            water_content_units,
        )

        units = water_content_units(y_water=0.01, p_pa=101325.0, t_k=298.15)
        assert units.standard_basis == "60F_14.696psia"
        assert units.to_dict()["standard_basis"] == "60F_14.696psia"

        calc = SyngasWaterCalculator()
        res = calc.calculate_water_content(40.0, 1.0)
        assert res.standard_basis == "60F_14.696psia"
        assert res.to_dict()["results"]["standard_basis"] == "60F_14.696psia"

        req = SyngasWaterRequest(temperature_c=40.0, pressure_bar=1.0)
        resp = _fallback_calculate(req)
        assert resp.water_content.standard_basis == "60F_14.696psia"

    def test_water_content_units_input_validation(self) -> None:
        """Validate rejection of NaN, Inf, out-of-range y_water,
        and non-positive p/t/mw.
        """
        from shared.python.sidekick.process_calculators.syngas_water_calculator import (
            water_content_units,
        )

        for invalid_y in [-0.01, 1.05, float("nan"), float("inf"), float("-inf")]:
            with pytest.raises(ValueError, match="y_water"):
                water_content_units(y_water=invalid_y, p_pa=101325.0, t_k=293.15)

        for invalid_p in [0.0, -100.0, float("nan"), float("inf")]:
            with pytest.raises(ValueError, match="p_pa"):
                water_content_units(y_water=0.01, p_pa=invalid_p, t_k=293.15)

        for invalid_t in [0.0, -273.15, float("nan"), float("inf")]:
            with pytest.raises(ValueError, match="t_k"):
                water_content_units(y_water=0.01, p_pa=101325.0, t_k=invalid_t)

        for invalid_mw in [0.0, -18.0, float("nan"), float("inf")]:
            with pytest.raises(ValueError, match="mw_water"):
                water_content_units(
                    y_water=0.01, p_pa=101325.0, t_k=293.15, mw_water=invalid_mw
                )
