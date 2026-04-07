"""Friction factor calculations for pipe flow."""

import logging
import math

from ...constants import (
    CHURCHILL_B_COEFF,
    COLEBROOK_ROUGHNESS_COEFF,
    FRICTION_FACTOR_DEFAULT_LAMINAR,
    LAMINAR_FRICTION_CONSTANT,
    RE_LAMINAR_UPPER,
    SWAMEE_JAIN_COEFF,
)

logger = logging.getLogger(__name__)


def friction_factor_laminar(reynolds_number: float) -> float:
    """Calculate friction factor for laminar flow (Re < 2300).

    f = 64 / Re  (Hagen-Poiseuille equation)

    Args:
        reynolds_number: Reynolds number (must be positive)

    Returns:
        Darcy friction factor (always positive)
    """
    if reynolds_number <= 0:
        logger.error("Reynolds number must be positive")
        return FRICTION_FACTOR_DEFAULT_LAMINAR  # Default for Re ~ 1000

    result = LAMINAR_FRICTION_CONSTANT / reynolds_number
    if not (result > 0):
        raise ValueError(f"Friction factor must be positive, got {result}")
    return result


def friction_factor_colebrook(
    reynolds_number: float,
    relative_roughness: float,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> float:
    """Calculate friction factor using Colebrook-White equation (implicit)."""
    if not (reynolds_number is not None):
        raise ValueError("reynolds_number must be provided")
    if reynolds_number < RE_LAMINAR_UPPER:
        return friction_factor_laminar(reynolds_number)

    # Initial guess using Swamee-Jain as starting point
    f = friction_factor_swamee_jain(reynolds_number, relative_roughness)

    # Newton-Raphson iteration
    for i in range(max_iterations):
        f_old = f

        # Colebrook-White equation rearranged
        term1 = relative_roughness / COLEBROOK_ROUGHNESS_COEFF
        term2 = 2.51 / (reynolds_number * math.sqrt(f))
        f_new = 0.25 / (math.log10(term1 + term2) ** 2)

        # Check convergence
        if abs(f_new - f_old) < tolerance:
            logger.debug(f"Colebrook converged in {i + 1} iterations: f = {f_new:.6f}")
            return f_new

        f = f_new

    logger.warning(f"Colebrook did not converge in {max_iterations} iterations")
    return f


def friction_factor_swamee_jain(
    reynolds_number: float, relative_roughness: float
) -> float:
    """Calculate friction factor using Swamee-Jain explicit approximation."""
    if not (reynolds_number is not None):
        raise ValueError("reynolds_number must be provided")
    if reynolds_number < RE_LAMINAR_UPPER:
        return friction_factor_laminar(reynolds_number)

    # Swamee-Jain equation
    term1 = relative_roughness / COLEBROOK_ROUGHNESS_COEFF
    term2 = SWAMEE_JAIN_COEFF / (reynolds_number**0.9)

    f = 0.25 / (math.log10(term1 + term2) ** 2)

    logger.debug(
        f"Swamee-Jain: Re={reynolds_number:.0f}, ε/D={relative_roughness:.6f}, f={f:.6f}"
    )
    return f


def friction_factor_churchill(
    reynolds_number: float, relative_roughness: float
) -> float:
    """Calculate friction factor using Churchill explicit correlation."""
    if not (reynolds_number is not None):
        raise ValueError("reynolds_number must be provided")
    Re = reynolds_number

    if Re < 1:
        return LAMINAR_FRICTION_CONSTANT  # Avoid division by zero

    # Churchill correlation
    term1 = (7.0 / Re) ** 0.9 + 0.27 * relative_roughness
    A = (-2.457 * math.log(term1)) ** 16

    B = (CHURCHILL_B_COEFF / Re) ** 16

    term2 = (8.0 / Re) ** 12
    term3 = 1.0 / ((A + B) ** 1.5)

    f = 8.0 * ((term2 + term3) ** (1.0 / 12.0))

    logger.debug(f"Churchill: Re={Re:.0f}, ε/D={relative_roughness:.6f}, f={f:.6f}")
    return float(f)


def friction_factor_haaland(reynolds_number: float, relative_roughness: float) -> float:
    """Calculate friction factor using Haaland explicit approximation."""
    if not (reynolds_number is not None):
        raise ValueError("reynolds_number must be provided")
    if reynolds_number < RE_LAMINAR_UPPER:
        return friction_factor_laminar(reynolds_number)

    term1 = (relative_roughness / COLEBROOK_ROUGHNESS_COEFF) ** 1.11
    term2 = 6.9 / reynolds_number

    inv_sqrt_f = -1.8 * math.log10(term1 + term2)
    f = 1.0 / (inv_sqrt_f**2)

    return f


def select_friction_factor_method(
    method: str, reynolds_number: float, relative_roughness: float
) -> float:
    """Select and calculate friction factor using specified method."""
    if not (method is not None):
        raise ValueError("method must be provided")
    method = method.lower()

    if method == "colebrook":
        return friction_factor_colebrook(reynolds_number, relative_roughness)
    elif method == "swamee-jain" or method == "swamee_jain":
        return friction_factor_swamee_jain(reynolds_number, relative_roughness)
    elif method == "churchill":
        return friction_factor_churchill(reynolds_number, relative_roughness)
    elif method == "haaland":
        return friction_factor_haaland(reynolds_number, relative_roughness)
    else:
        available = ["colebrook", "swamee-jain", "churchill", "haaland"]
        raise ValueError(
            f"Unknown friction factor method '{method}'. Available: {available}"
        )
