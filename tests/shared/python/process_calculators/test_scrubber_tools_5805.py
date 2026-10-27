"""Unit tests for #5805: Scrubber Eckert flooding flow parameter (TOOLS-20).

Verifies that calculate_flooding_velocity computes the Eckert flow parameter using
the actual gas mass flux G, matching GM-10:
    flow_param = (L / G) * sqrt(rho_G / rho_L)

And enforces DbC preconditions:
    Pre: G > 0
    Post: flow parameter dimensionless and proportional to 1/G
"""

from __future__ import annotations

import math

import pytest
from sidekick.process_calculators.scrubber_calculator import (
    PACKING_DATABASE,
    WATER_DENSITY,
    WATER_VISCOSITY,
    calculate_flooding_velocity,
)

from contracts import PreconditionError


def test_eckert_flow_param_uses_gas_flux() -> None:
    """Flooding velocity must vary with gas mass flux G (GM-10 / TOOLS-20).

    Shared vector:
        L = 5.0 kg/(m^2 * s)
        G_low = 2.0 kg/(m^2 * s)
        G_high = 4.0 kg/(m^2 * s)
        rho_G = 1.2 kg/m^3
        rho_L = WATER_DENSITY (998.2 kg/m^3)
    """
    packing = PACKING_DATABASE["Metal Pall Rings"]
    liquid_mass_flux = 5.0
    gas_flux_low = 2.0
    gas_flux_high = 4.0
    gas_density = 1.2
    liquid_density = WATER_DENSITY

    v_flood_low = calculate_flooding_velocity(
        liquid_mass_flux=liquid_mass_flux,
        gas_density=gas_density,
        liquid_density=liquid_density,
        packing=packing,
        liquid_viscosity=WATER_VISCOSITY,
        gas_mass_flux=gas_flux_low,
    )

    v_flood_high = calculate_flooding_velocity(
        liquid_mass_flux=liquid_mass_flux,
        gas_density=gas_density,
        liquid_density=liquid_density,
        packing=packing,
        liquid_viscosity=WATER_VISCOSITY,
        gas_mass_flux=gas_flux_high,
    )

    assert v_flood_low > 0.0
    assert v_flood_high > 0.0
    # As G increases from 2.0 to 4.0, L/G decreases, flow_param decreases,
    # so flooding velocity increases.
    assert v_flood_high > v_flood_low, (
        f"Higher gas mass flux must yield higher flooding velocity due to reduced L/G. "
        f"Got v_flood_low={v_flood_low}, v_flood_high={v_flood_high}"
    )

    # Ratio must differ significantly from 1.0 (on main, ratio is 1.0)
    ratio = v_flood_high / v_flood_low
    assert not math.isclose(ratio, 1.0, rel_tol=1e-3), (
        f"Flooding velocity invariant to gas_mass_flux: ratio={ratio}"
    )


def test_flooding_velocity_gas_mass_flux_preconditions() -> None:
    """DbC Precondition: gas_mass_flux must be positive (G > 0)."""
    packing = PACKING_DATABASE["Metal Pall Rings"]

    # Zero gas mass flux must raise
    with pytest.raises((PreconditionError, ValueError), match="gas_mass_flux"):
        calculate_flooding_velocity(
            liquid_mass_flux=5.0,
            gas_density=1.2,
            liquid_density=WATER_DENSITY,
            packing=packing,
            gas_mass_flux=0.0,
        )

    # Negative gas mass flux must raise
    with pytest.raises((PreconditionError, ValueError), match="gas_mass_flux"):
        calculate_flooding_velocity(
            liquid_mass_flux=5.0,
            gas_density=1.2,
            liquid_density=WATER_DENSITY,
            packing=packing,
            gas_mass_flux=-2.0,
        )

    # None gas mass flux must raise ValueError
    with pytest.raises((PreconditionError, ValueError), match="gas_mass_flux"):
        calculate_flooding_velocity(
            liquid_mass_flux=5.0,
            gas_density=1.2,
            liquid_density=WATER_DENSITY,
            packing=packing,
            gas_mass_flux=None,  # type: ignore[arg-type]
        )
