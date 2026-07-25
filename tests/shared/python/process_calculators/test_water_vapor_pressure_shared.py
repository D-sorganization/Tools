"""Tests for the shared water-vapor-pressure correlation kernel.

Covers the single-source Antoine / Buck / IAPWS / Magnus helpers introduced to
deduplicate the inline correlation copies that previously lived in
``syngas_water_calculator``, ``acid_gas_dewpoint_calculator``,
``calculators.thermo.steam_engine`` and the ``calc_backend`` syngas-water
router (issues #3675, #3677, #3678).
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("numpy")

from shared.python.sidekick.process_calculators.constants import (
    ANTOINE_WATER_A,
    ANTOINE_WATER_B,
    ANTOINE_WATER_C,
    BUCK_ABOVE_FREEZING_A,
    BUCK_ABOVE_FREEZING_B,
    BUCK_ABOVE_FREEZING_C,
    BUCK_ABOVE_FREEZING_D,
    MMHG_TO_PA_CONV,
    WATER_VAPOR_A,
    WATER_VAPOR_B,
    WATER_VAPOR_C,
    WATER_VAPOR_D,
)
from shared.python.sidekick.process_calculators.water_vapor_pressure import (
    antoine_pressure_pa,
    antoine_temperature_c,
    buck_pressure_pa,
    iapws_pressure_pa,
    magnus_pressure_pa,
    safe_exp,
)


class TestSafeExp:
    def test_matches_math_exp_in_range(self) -> None:
        for x in (-10.0, 0.0, 1.0, 10.0, 100.0):
            assert safe_exp(x) == pytest.approx(math.exp(x), rel=1e-12)

    def test_clamps_large_positive(self) -> None:
        # No overflow / RuntimeWarning; clamped to a finite value.
        assert math.isfinite(safe_exp(1e6))

    def test_clamps_large_negative(self) -> None:
        assert safe_exp(-1e6) == pytest.approx(0.0, abs=1e-300)

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError):
            safe_exp(None)  # type: ignore[arg-type]


class TestAntoineForward:
    def test_water_at_100c_near_1atm(self) -> None:
        """Antoine water at 100°C should be near 1 atm (101325 Pa)."""
        p = antoine_pressure_pa(
            ANTOINE_WATER_A, ANTOINE_WATER_B, ANTOINE_WATER_C, 100.0
        )
        assert p == pytest.approx(101325.0, rel=0.03)

    def test_matches_manual_formula(self) -> None:
        """Delegated result must equal the explicit log10/mmHg expression."""
        t = 55.0
        log10_p_mmhg = ANTOINE_WATER_A - ANTOINE_WATER_B / (ANTOINE_WATER_C + t)
        expected = 10**log10_p_mmhg * MMHG_TO_PA_CONV
        assert antoine_pressure_pa(
            ANTOINE_WATER_A, ANTOINE_WATER_B, ANTOINE_WATER_C, t
        ) == pytest.approx(expected, rel=1e-12)

    def test_monotonic(self) -> None:
        temps = [0, 20, 40, 60, 80, 100]
        ps = [
            antoine_pressure_pa(ANTOINE_WATER_A, ANTOINE_WATER_B, ANTOINE_WATER_C, t)
            for t in temps
        ]
        assert all(ps[i + 1] > ps[i] for i in range(len(ps) - 1))


class TestAntoineInverse:
    def test_roundtrip(self) -> None:
        """antoine_temperature_c is the inverse of antoine_pressure_pa."""
        for t in (5.0, 25.0, 75.0, 99.0):
            p = antoine_pressure_pa(
                ANTOINE_WATER_A, ANTOINE_WATER_B, ANTOINE_WATER_C, t
            )
            t_back = antoine_temperature_c(
                ANTOINE_WATER_A, ANTOINE_WATER_B, ANTOINE_WATER_C, p
            )
            assert t_back == pytest.approx(t, abs=1e-6)

    def test_non_positive_pressure_raises(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            antoine_temperature_c(
                ANTOINE_WATER_A, ANTOINE_WATER_B, ANTOINE_WATER_C, 0.0
            )

    def test_negative_pressure_raises(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            antoine_temperature_c(
                ANTOINE_WATER_A, ANTOINE_WATER_B, ANTOINE_WATER_C, -10.0
            )


class TestBuck:
    def test_buck_water_at_20c(self) -> None:
        p = buck_pressure_pa(0.61121, 18.678, 257.14, 234.5, 20.0)
        assert p == pytest.approx(2338.34, rel=5e-3)

    def test_buck_water_at_50c(self) -> None:
        p = buck_pressure_pa(0.61121, 18.678, 257.14, 234.5, 50.0)
        assert p == pytest.approx(12349.4, rel=5e-3)

    def test_positive(self) -> None:
        assert buck_pressure_pa(0.61121, 18.678, 257.14, 234.5, 25.0) > 0


class TestIapws:
    def test_at_100c_near_1atm(self) -> None:
        assert iapws_pressure_pa(100.0) == pytest.approx(101325.0, rel=0.01)

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="IAPWS"):
            iapws_pressure_pa(1000.0)


class TestMagnus:
    def test_at_0c(self) -> None:
        assert magnus_pressure_pa(0.0) == pytest.approx(611.0, rel=0.02)

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="Magnus"):
            magnus_pressure_pa(150.0)


class TestConsumersDelegate:
    """The previously-duplicated callers must agree with the shared kernel."""

    def test_syngas_antoine_equals_shared(self) -> None:
        from shared.python.sidekick.process_calculators.syngas_water_calculator import (
            SyngasWaterCalculator,
        )

        calc = SyngasWaterCalculator()
        shared = antoine_pressure_pa(
            ANTOINE_WATER_A, ANTOINE_WATER_B, ANTOINE_WATER_C, 60.0
        )
        assert calc._antoine_equation(60.0) == pytest.approx(shared, rel=1e-12)

    def test_steam_antoine_equals_shared(self) -> None:
        from shared.python.sidekick.calculators.thermo.steam_engine import (
            SteamCalculationEngine,
        )

        engine = SteamCalculationEngine()
        shared = antoine_pressure_pa(
            ANTOINE_WATER_A, ANTOINE_WATER_B, ANTOINE_WATER_C, 60.0
        )
        assert engine._antoine_equation(60.0) == pytest.approx(shared, rel=1e-12)

    def test_steam_buck_delegates_with_standard_coefficient_roles(self) -> None:
        """Steam Buck delegation must preserve the physical over-water curve."""
        from shared.python.sidekick.calculators.thermo.steam_engine import (
            BUCK_A,
            BUCK_B,
            BUCK_C,
            BUCK_D,
            MBAR_TO_KPA_FACTOR,
            SteamCalculationEngine,
        )

        engine = SteamCalculationEngine()
        for t in (10.0, 20.0, 50.0, 80.0):
            a_kpa = BUCK_A / MBAR_TO_KPA_FACTOR
            legacy = a_kpa * math.exp((BUCK_B - t / BUCK_C) * t / (t + BUCK_D)) * 1000.0
            assert engine._buck_equation(t) == pytest.approx(legacy, rel=1e-12)

    def test_syngas_buck_matches_reference_curve(self) -> None:
        """Syngas Buck delegation must use the physical over-water curve."""
        from shared.python.sidekick.process_calculators.syngas_water_calculator import (
            SyngasWaterCalculator,
        )

        calc = SyngasWaterCalculator()
        for t, expected in (
            (0.0, 611.21),
            (20.0, 2338.34),
            (50.0, 12349.4),
        ):
            assert calc._buck_equation(t) == pytest.approx(expected, rel=5e-3)

    def test_syngas_buck_below_freezing_unchanged(self) -> None:
        from shared.python.sidekick.process_calculators.syngas_water_calculator import (
            SyngasWaterCalculator,
        )

        calc = SyngasWaterCalculator()
        t = -20.0
        exponent = (WATER_VAPOR_B - t / WATER_VAPOR_D) * t / (WATER_VAPOR_C + t)
        expected = WATER_VAPOR_A * math.exp(exponent) * 1000.0
        assert calc._buck_equation(t) == pytest.approx(expected, rel=1e-12)

    def test_syngas_buck_is_continuous_at_freezing(self) -> None:
        from shared.python.sidekick.process_calculators.syngas_water_calculator import (
            SyngasWaterCalculator,
        )

        calc = SyngasWaterCalculator()
        assert calc._buck_equation(-0.001) == pytest.approx(
            calc._buck_equation(0.001), rel=5e-4
        )

    def test_syngas_buck_dew_point_roundtrip(self) -> None:
        from shared.python.sidekick.process_calculators.syngas_water_calculator import (
            SyngasWaterCalculator,
        )

        calc = SyngasWaterCalculator()
        for t in (20.0, 50.0):
            vapor_pressure_pa = calc._buck_equation(t)
            assert calc.calculate_dew_point(
                vapor_pressure_pa, 101325.0
            ) == pytest.approx(t, abs=1.0)

    def test_syngas_buck_shared_formula_uses_named_roles(self) -> None:
        from shared.python.sidekick.process_calculators.syngas_water_calculator import (
            SyngasWaterCalculator,
        )

        calc = SyngasWaterCalculator()
        for t in (10.0, 20.0, 50.0, 80.0):
            expected = (
                BUCK_ABOVE_FREEZING_A
                * math.exp(
                    (BUCK_ABOVE_FREEZING_B - t / BUCK_ABOVE_FREEZING_C)
                    * t
                    / (BUCK_ABOVE_FREEZING_D + t)
                )
                * 1000.0
            )
            assert calc._buck_equation(t) == pytest.approx(expected, rel=1e-12)

    def test_acid_gas_water_antoine_equals_shared(self) -> None:
        from shared.python.sidekick.process_calculators.acid_gas_dewpoint_calculator import (  # noqa: E501
            AcidGasDewpointCalculator,
        )

        calc = AcidGasDewpointCalculator()
        shared = antoine_pressure_pa(
            ANTOINE_WATER_A, ANTOINE_WATER_B, ANTOINE_WATER_C, 60.0
        )
        assert calc.calculate_vapor_pressure(60.0, "H2O", "antoine") == pytest.approx(
            shared, rel=1e-12
        )
