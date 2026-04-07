"""Compressible flow calculations for pressure drop."""

import logging
import math

from ....utils.unit_constants import R_UNIVERSAL_KMOL

logger = logging.getLogger(__name__)

R_UNIVERSAL = R_UNIVERSAL_KMOL  # J/(kmol·K)
PI = math.pi


def _iterate_compressible_pressure(
    P1: float,
    P2_initial: float,
    coeff: float,
    resistance: float,
    max_iterations: int = 50,
    tolerance: float = 1.0,
) -> tuple[float, bool]:
    """Iteratively solve the isothermal compressible flow equation for P2."""
    if not (P1 is not None):
        raise ValueError("P1 must be provided")
    P2 = P2_initial

    for iteration in range(max_iterations):
        P2_old = P2

        ln_term = 2.0 * math.log(P1 / P2) if P2 > 0 and P1 > P2 else 0.0
        rhs = coeff * (resistance + ln_term)
        P2_squared = P1**2 - rhs

        if P2_squared <= 0:
            logger.warning(
                "Compressible flow calculation indicates choked flow condition"
            )
            return P2, True

        P2 = math.sqrt(P2_squared)

        if abs(P2 - P2_old) < tolerance:
            logger.debug(f"Compressible flow converged in {iteration + 1} iterations")
            break

    return P2, False


def calculate_compressible_flow_correction(
    inlet_pressure: float,
    outlet_pressure: float,
    length: float,
    diameter: float,
    mass_flow_rate: float,
    temperature: float,
    molecular_weight: float,
    compressibility_factor: float,
    friction_factor: float,
    total_k_factor: float = 0.0,
) -> tuple[float, float]:
    """Calculate pressure drop accounting for compressibility effects."""
    if not (diameter > 0):
        raise ValueError(f"diameter must be positive, got {diameter}")
    if not (temperature > 0):
        raise ValueError(f"temperature must be positive (K), got {temperature}")
    if not (molecular_weight > 0):
        raise ValueError(f"molecular_weight must be positive, got {molecular_weight}")

    area = PI * (diameter**2) / 4.0
    G = mass_flow_rate / area
    resistance = friction_factor * (length / diameter) + total_k_factor

    coeff = (
        (G**2) * (compressibility_factor * R_UNIVERSAL * temperature) / molecular_weight
    )

    P2, is_choked = _iterate_compressible_pressure(
        inlet_pressure,
        outlet_pressure,
        coeff,
        resistance,
    )

    if is_choked:
        return inlet_pressure - outlet_pressure, outlet_pressure

    corrected_dp = inlet_pressure - P2

    pressure_ratio = P2 / inlet_pressure
    if pressure_ratio > 0:
        expansion_factor = math.sqrt(
            pressure_ratio * (1 - pressure_ratio**2) / (1 - pressure_ratio)
            if pressure_ratio < 1
            else 1.0
        )
    else:
        expansion_factor = 1.0

    logger.debug(
        f"Compressible flow correction: ΔP_incomp={inlet_pressure - outlet_pressure:.0f} Pa, "
        f"ΔP_comp={corrected_dp:.0f} Pa, Y={expansion_factor:.3f}"
    )

    return corrected_dp, P2


def calculate_expansion_factor(
    inlet_pressure: float,
    pressure_drop: float,
    friction_factor: float,
    length_over_diameter: float,
    gamma: float = 1.4,
) -> float:
    """Calculate gas expansion factor Y for compressible flow."""
    if not (inlet_pressure is not None):
        raise ValueError("inlet_pressure must be provided")
    if inlet_pressure <= 0 or pressure_drop < 0:
        return 1.0

    # Pressure ratio
    pressure_ratio = (inlet_pressure - pressure_drop) / inlet_pressure

    if pressure_ratio <= 0:
        # Choked flow condition
        return 0.0

    if pressure_ratio >= 0.99:
        # Nearly incompressible
        return 1.0

    r = pressure_ratio
    k = gamma

    try:
        numerator = k * (r ** (2.0 / k)) * (1.0 - r ** ((k - 1.0) / k))
        denominator = (k - 1.0) * (1.0 - r)

        Y = 1.0 if denominator <= 0 else math.sqrt(numerator / denominator)
    except (ValueError, ZeroDivisionError):
        # Fallback to simplified formula
        Y = 1.0 - pressure_drop / (3.0 * gamma * inlet_pressure)

    # Bound Y to physical limits
    Y = max(0.0, min(Y, 1.0))

    return Y
