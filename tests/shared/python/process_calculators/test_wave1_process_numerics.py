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
