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
    MMHG_TO_PA_CONV,
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
        # Pin the consolidated Buck kernel output (a=0.61121 kPa) at 20°C in the
        # syngas coefficient order ``(b - t/d) * t / (c + t)``. Value matches the
        # pre-refactor SyngasWaterCalculator._buck_equation (2636.34 Pa).
        p = buck_pressure_pa(0.61121, 18.678, 234.5, 257.14, 20.0)
        assert p == pytest.approx(2636.34, rel=1e-4)

    def test_syngas_order_matches_legacy(self) -> None:
        """The kernel uses the syngas coefficient order, not the steam transpose."""
        a, b, c, d = 0.61121, 18.678, 234.5, 257.14
        legacy_syngas = a * math.exp((b - 20.0 / d) * 20.0 / (c + 20.0)) * 1000.0
        assert buck_pressure_pa(a, b, c, d, 20.0) == pytest.approx(
            legacy_syngas, rel=1e-12
        )

    def test_positive(self) -> None:
        assert buck_pressure_pa(0.61121, 18.678, 234.5, 257.14, 25.0) > 0


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

    def test_steam_buck_preserves_legacy_curve(self) -> None:
        """Steam Buck delegation must reproduce the pre-refactor steam curve.

        The steam engine historically used the coefficient order
        ``(b - T/c) * T / (T + d)`` (numerator divisor = C, denominator = D),
        which is the transpose of the syngas Buck order the shared kernel
        adopts.  Guard against silently re-introducing the wrong arg order
        (which shifts the steam saturation curve by ~13% at 20°C).
        """
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

    def test_syngas_buck_matches_buck_1981_reference_curve(self) -> None:
        """Syngas Buck delegation must reproduce the Buck (1981) reference curve.

        Was ``test_syngas_buck_preserves_legacy_curve``, pinned to the
        pre-fix transposed-coefficient curve. Issue #3867 found that curve
        was ~12.7% off the Buck (1981) source the constants cite (confirmed
        against steam-table values, e.g. ~2338 Pa not ~2636 Pa at 20 C) and
        fixed the call site to swap C and D, matching how
        ``calculators.thermo.steam_engine`` already calls the same kernel.
        """
        from shared.python.sidekick.process_calculators.constants import (
            BUCK_ABOVE_FREEZING_A,
            BUCK_ABOVE_FREEZING_B,
            BUCK_ABOVE_FREEZING_C,
            BUCK_ABOVE_FREEZING_D,
        )
        from shared.python.sidekick.process_calculators.syngas_water_calculator import (
            SyngasWaterCalculator,
        )

        calc = SyngasWaterCalculator()
        for t in (10.0, 20.0, 50.0, 80.0):
            reference = (
                BUCK_ABOVE_FREEZING_A
                * math.exp(
                    (BUCK_ABOVE_FREEZING_B - t / BUCK_ABOVE_FREEZING_C)
                    * t
                    / (BUCK_ABOVE_FREEZING_D + t)
                )
                * 1000.0
            )
            assert calc._buck_equation(t) == pytest.approx(reference, rel=1e-12)

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
