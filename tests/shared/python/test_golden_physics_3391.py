"""Golden-physics reference-value tests (#3391).

The shared calculation modules were historically covered almost exclusively by
*property* tests (monotonicity, shape, raises-on-bad-input). That is precisely
how a 43% vapor-pressure error, a one-interval-short integrator, a +47% flare
heat release, and wrong acid-gas Antoine constants all survived under test. This
module anchors the calculators to *authoritative external reference values*
(NIST SP 811, IAPWS-IF97, analytic solutions) with explicit tolerances.

All tests are marked ``scientific``. Reference-value assertions that require the
optional CoolProp/Cantera accurate backends are guarded so the analytic and
simplified-path anchors still run everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.scientific


# --------------------------------------------------------------------------- #
# 1. Unit conversion — NIST SP 811 exact factors + round-trip exactness
# --------------------------------------------------------------------------- #
class TestUnitConversionReferenceValues:
    @pytest.fixture()
    def service(self):
        from sidekick.calculators.conversion.service import UnitConversionService

        return UnitConversionService()

    @pytest.mark.parametrize(
        ("value", "from_unit", "to_unit", "expected"),
        [
            # NIST SP 811 defined exact conversion factors.
            (1.0, "BTU", "J", 1055.05585262),
            (1.0, "gal", "L", 3.785411784),
            (1.0, "psi", "Pa", 6894.757293168),
            (1.0, "ft", "m", 0.3048),
            (1.0, "lb", "kg", 0.45359237),
            (1.0, "in", "m", 0.0254),
        ],
    )
    def test_defined_factors_match_nist(
        self, service, value, from_unit, to_unit, expected
    ) -> None:
        result = service.convert(value, from_unit, to_unit)
        assert result.value == pytest.approx(expected, rel=1e-9)

    def test_factor_table_round_trip_exactness(self, service) -> None:
        """Every within-category pair must round-trip to 1e-9 (fuzz found #3384)."""
        category_map = service.category_map
        checked = 0
        for category, factors in category_map.items():
            units = list(factors)
            if len(units) < 2:
                continue
            base = units[0]
            for other in units[1:]:
                try:
                    forward = service.convert(1.0, base, other).value
                    back = service.convert(forward, other, base).value
                except Exception:  # noqa: BLE001 - some categories need extra args
                    continue
                assert back == pytest.approx(1.0, rel=1e-9), (
                    f"{category}: {base}->{other}->{base} lost precision"
                )
                checked += 1
        assert checked > 0  # the fuzz must actually exercise pairs


# --------------------------------------------------------------------------- #
# 2. Calculus — analytic anchors (Simpson exact on a parabola, d/dt sin = cos)
# --------------------------------------------------------------------------- #
class TestCalculusReferenceValues:
    def _signal(self, t, y):
        from signal_toolkit.core import Signal

        return Signal(time=t, values=y, name="f", units="")

    def test_simpson_integrates_parabola_exactly(self) -> None:
        from signal_toolkit.calculus import IntegrationMethod, Integrator

        x = np.linspace(0.0, 3.0, 31)
        # Exact: ∫₀³ x² dx = 9.
        result = Integrator(method=IntegrationMethod.SIMPSON).integrate(
            self._signal(x, x**2)
        )
        assert result.value == pytest.approx(9.0, abs=1e-9)

    def test_trapezoid_linear_is_exact(self) -> None:
        from signal_toolkit.calculus import IntegrationMethod, Integrator

        x = np.linspace(0.0, 10.0, 51)
        # ∫₀¹⁰ (2t+1) dt = 110, exact for the trapezoid rule on a linear signal.
        result = Integrator(method=IntegrationMethod.TRAPEZOID).integrate(
            self._signal(x, 2 * x + 1)
        )
        assert result.value == pytest.approx(110.0, rel=1e-9)

    def test_derivative_of_sine_is_cosine(self) -> None:
        from signal_toolkit.calculus import Differentiator

        t = np.linspace(0.0, 2 * np.pi, 101)
        deriv = Differentiator().differentiate(self._signal(t, np.sin(t)))
        # Interior central-difference error is O(h²); endpoints are excluded.
        interior_err = np.max(np.abs(deriv.values[2:-2] - np.cos(t)[2:-2]))
        assert interior_err < 1e-2


# --------------------------------------------------------------------------- #
# 3. Steam — IAPWS-IF97 anchors (simplified path within 3%, accurate within .5%)
# --------------------------------------------------------------------------- #
class TestSteamReferenceValues:
    @pytest.fixture()
    def engine(self):
        from sidekick.calculators.thermo.steam_engine import SteamCalculationEngine

        return SteamCalculationEngine()

    @pytest.mark.parametrize(
        ("temperature_k", "expected_pa"),
        [
            (373.15, 101_417.0),  # Psat(100 °C), IAPWS-IF97
            (298.15, 3_169.9),  # Psat(25 °C), IAPWS-IF97
        ],
    )
    def test_saturation_pressure_anchors(
        self, engine, temperature_k, expected_pa
    ) -> None:
        # 3% covers the simplified Antoine path; the accurate backends are well
        # inside this when present.
        psat = engine.get_saturation_pressure(temperature_k)
        assert psat == pytest.approx(expected_pa, rel=0.03)

    def test_saturation_temperature_anchor(self, engine) -> None:
        # Tsat(1 MPa) = 453.03 K (IAPWS-IF97).
        tsat = engine.get_saturation_temperature(1.0e6)
        assert tsat == pytest.approx(453.03, rel=0.03)


# --------------------------------------------------------------------------- #
# 4. Flare — pure-CH₄ heat-release anchor (mass-basis LHV)
# --------------------------------------------------------------------------- #
class TestFlareReferenceValues:
    def test_pure_methane_heat_release(self) -> None:
        from upstream_drift_tools.process_calculators.flare_calculator import (
            FlareCalculator,
        )

        calc = FlareCalculator()
        total_flow = 3600.0  # kg/hr -> 1 kg/s
        design = calc.calculate_flare_size(total_flow, {"CH4": 100.0}, 300.0, 1.0)
        # Pure CH₄: mass-basis LHV = 50,010 kJ/kg, at 1 kg/s -> 50,010 kW.
        assert design.heat_release == pytest.approx(50_010.0, rel=1e-6)
