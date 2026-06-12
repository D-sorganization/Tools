"""Regression tests for the wave-1 process-calculator numeric fixes (#3103).

Covers the compressible-flow consolidation/expansion factor (F1/F2), the WGS
K↔composition self-consistency and T>0 guard (F3/F8), the assert→ValueError
conversions (F4), the laminar Re≤0 raise (F6), and the compressibility
div-zero guard (F7).
"""

from __future__ import annotations

import math

import pytest
from sidekick.process_calculators.pressure_drop_calculator.engine import (
    _flow_calculations,
    compressible_flow,
)
from sidekick.process_calculators.pressure_drop_calculator.engine._friction_factors import (  # noqa: E501
    friction_factor_laminar,
)
from sidekick.process_calculators.pressure_drop_calculator.utils.gas_properties import (
    calculate_compressibility_factor,
)
from sidekick.process_calculators.wgs_reactor_calculator import WGSReactorEngine


# --------------------------------------------------------------------------- #
# F1 / F2 — compressible flow: single solver + sane expansion factor
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_compressible_solver_is_single_source() -> None:
    """``_flow_calculations`` re-exports the canonical solver (F1)."""
    assert (
        _flow_calculations.calculate_expansion_factor
        is compressible_flow.calculate_expansion_factor
    )
    assert (
        _flow_calculations.calculate_compressible_flow_correction
        is compressible_flow.calculate_compressible_flow_correction
    )


@pytest.mark.unit
@pytest.mark.parametrize("pressure_drop", [1e3, 1e4, 5e4, 1e5])
def test_expansion_factor_bounded(pressure_drop: float) -> None:
    """Expansion factor Y stays in (0, 1] for pr < 1 (F2)."""
    y = compressible_flow.calculate_expansion_factor(
        inlet_pressure=2e5,
        pressure_drop=pressure_drop,
        friction_factor=0.02,
        length_over_diameter=100.0,
    )
    assert 0.0 < y <= 1.0


# --------------------------------------------------------------------------- #
# F3 / F8 — WGS reactor
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_wgs_reported_k_matches_solved_composition() -> None:
    """Reported K equals K recovered from the solved composition (F3)."""
    engine = WGSReactorEngine()
    temperature = 623.15  # 350 °C
    result = engine.calculate_equilibrium_composition(
        {"CO": 1.0, "H2O": 0.0, "CO2": 0.0, "H2": 0.0},
        temperature,
        pressure=1.0,
        steam_ratio=2.0,
    )
    comp = result["composition"]
    recovered_k = (comp["CO2"] * comp["H2"]) / (comp["CO"] * comp["H2O"])
    assert recovered_k == pytest.approx(result["equilibrium_constant"], rel=1e-6)


@pytest.mark.unit
def test_wgs_extent_solver_hand_value() -> None:
    """Hand-solved quadratic extent for a symmetric feed at K=4."""
    # K=4, n_CO=n_H2O=1, no products: K = x^2 / (1-x)^2 -> sqrt(K) = x/(1-x)
    # -> 2 = x/(1-x) -> 2 - 2x = x -> x = 2/3.
    x = WGSReactorEngine._solve_extent_from_k(4.0, 1.0, 1.0, 0.0, 0.0)
    assert x == pytest.approx(2.0 / 3.0, rel=1e-9)


@pytest.mark.unit
def test_wgs_negative_temperature_raises() -> None:
    """Van't Hoff guard rejects non-positive Kelvin temperature (F8)."""
    engine = WGSReactorEngine()
    with pytest.raises(ValueError, match="positive"):
        engine.calculate_equilibrium_constant(-50.0)
    with pytest.raises(ValueError, match="positive"):
        engine.calculate_equilibrium_constant(0.0)


# --------------------------------------------------------------------------- #
# F6 — laminar friction factor raises on Re <= 0
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_laminar_friction_raises_on_nonpositive_re() -> None:
    with pytest.raises(ValueError, match="positive"):
        friction_factor_laminar(0.0)
    with pytest.raises(ValueError, match="positive"):
        friction_factor_laminar(-100.0)


@pytest.mark.unit
def test_laminar_friction_known_value() -> None:
    assert friction_factor_laminar(1000.0) == pytest.approx(64.0 / 1000.0)


# --------------------------------------------------------------------------- #
# F7 — compressibility factor ideal-gas fallback for unknown-only composition
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_compressibility_unknown_only_returns_unity() -> None:
    """No recognised components -> Z = 1.0, not a ZeroDivisionError (F7)."""
    z = calculate_compressibility_factor(
        composition={"unobtanium": 1.0},
        temperature=300.0,
        pressure=1.0e5,
    )
    assert z == pytest.approx(1.0)
    assert math.isfinite(z)


# --------------------------------------------------------------------------- #
# #3386 — WGS equilibrium constant uses the Moe (1962) correlation, which
# tracks the NIST-JANAF temperature dependence across the shift window.
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.scientific
@pytest.mark.parametrize(
    ("temperature", "expected_k"),
    [
        # Moe correlation K = exp(4577.8/T - 4.33). These anchor the curve and
        # are within a few percent of NIST-JANAF across 600-1200 K.
        (600.0, 27.104),
        (800.0, 4.024),
        (1000.0, 1.281),
        (1200.0, 0.597),
    ],
)
def test_wgs_equilibrium_matches_moe_correlation(
    temperature: float, expected_k: float
) -> None:
    engine = WGSReactorEngine()
    assert engine.calculate_equilibrium_constant(temperature) == pytest.approx(
        expected_k, rel=1e-3
    )


@pytest.mark.unit
@pytest.mark.scientific
def test_wgs_k_unity_crossover_near_janaf() -> None:
    """K=1 crossover lands near ~1025-1090 K (JANAF), not the old ~979 K.

    The previous constant-coefficient Van't Hoff form crossed K=1 at 979 K; the
    Moe correlation crosses near 1057 K, far closer to the JANAF value.
    """
    engine = WGSReactorEngine()
    assert engine.calculate_equilibrium_constant(1000.0) > 1.0
    assert engine.calculate_equilibrium_constant(1090.0) < 1.0
    # Old (buggy) form gave K<1 already at 1000 K; the corrected curve does not.
    assert engine.calculate_equilibrium_constant(1000.0) > 1.27


# --------------------------------------------------------------------------- #
# #3390 — compressible solver reports the physical choked state instead of
# silently echoing the unachievable requested outlet pressure.
# --------------------------------------------------------------------------- #
def _choking_kwargs(outlet_pressure: float) -> dict[str, float]:
    """Inputs whose critical pressure exceeds the requested outlet pressure."""
    return {
        "inlet_pressure": 5.0e5,
        "outlet_pressure": outlet_pressure,
        "length": 100.0,
        "diameter": 0.05,
        "mass_flow_rate": 5.0,
        "temperature": 300.0,
        "molecular_weight": 0.016,  # kg/mol (methane)
        "compressibility_factor": 1.0,
        "friction_factor": 0.02,
    }


@pytest.mark.unit
@pytest.mark.scientific
def test_compressible_choked_reports_critical_pressure() -> None:
    """When choked, P2 is the critical pressure, not the requested outlet (#3390)."""
    # Drive the request well below the critical pressure to force choking.
    dp, p2 = compressible_flow.calculate_compressible_flow_correction(
        **_choking_kwargs(outlet_pressure=1.0e3)
    )
    # Critical pressure P2_crit = G * sqrt(Z R T / M); reconstruct it here.
    area = compressible_flow.PI * (0.05**2) / 4.0
    mass_flux = 5.0 / area
    coeff = mass_flux**2 * (1.0 * compressible_flow.R_UNIVERSAL * 300.0) / 0.016
    p2_crit = math.sqrt(coeff)

    assert p2 == pytest.approx(p2_crit, rel=1e-9)
    # Reported P2 must NOT be the unachievable requested outlet pressure.
    assert p2 > 1.0e3
    assert dp == pytest.approx(5.0e5 - p2_crit, rel=1e-9)


@pytest.mark.unit
def test_compressible_unchoked_returns_solved_outlet() -> None:
    """A comfortably sub-critical request solves normally (not flagged choked)."""
    # Small mass flow -> low critical pressure -> request is achievable.
    dp, p2 = compressible_flow.calculate_compressible_flow_correction(
        inlet_pressure=5.0e5,
        outlet_pressure=4.5e5,
        length=10.0,
        diameter=0.2,
        mass_flow_rate=0.1,
        temperature=300.0,
        molecular_weight=0.016,
        compressibility_factor=1.0,
        friction_factor=0.02,
    )
    assert 0.0 < p2 <= 5.0e5
    assert dp == pytest.approx(5.0e5 - p2, rel=1e-9)
