"""Fitting pressure drop calculations."""

import logging

from ..models.pressure_drop_data_models import PipeFitting
from ..utils.fitting_loss_coefficients import (
    calculate_two_k_factor,
    get_fitting_k_factor,
)

logger = logging.getLogger(__name__)


def calculate_fitting_pressure_drop(
    fittings: list[PipeFitting],
    density: float,
    velocity: float,
    reynolds_number: float,
    diameter_inches: float,
) -> float:
    """Calculate total pressure drop across fittings and valves.

    ΔP_fitting = Σ(K_i) × (ρV²/2)

    Args:
        fittings: List of PipeFitting objects
        density: Fluid density (kg/m³)
        velocity: Flow velocity (m/s)
        reynolds_number: Reynolds number (for Two-K method)
        diameter_inches: Pipe diameter (inches) (for Two-K method)

    Returns:
        Total fitting pressure drop (Pa)

    Reference:
        Crane TP-410, Chapter 2: Resistance of Valves and Fittings
    """
    if not (fittings is not None):
        raise ValueError("fittings must be provided")
    total_k = 0.0
    velocity_head = 0.5 * density * (velocity**2)

    for fitting in fittings:
        # Try to use Two-K method if available
        fitting_type_2k = fitting.fitting_type + "_2k"

        try:
            # Use Two-K method for better accuracy
            k_factor = calculate_two_k_factor(
                fitting_type_2k, reynolds_number, diameter_inches
            )
            logger.debug(
                f"Using Two-K method for {fitting.fitting_type}: K = {k_factor:.3f}"
            )
        except (ValueError, KeyError):
            # Fall back to standard K-factor
            try:
                k_factor = get_fitting_k_factor(fitting.fitting_type)
                logger.debug(
                    f"Using standard K for {fitting.fitting_type}: K = {k_factor:.3f}"
                )
            except ValueError:
                # Use provided K-factor
                k_factor = fitting.k_factor
                logger.warning(
                    f"Using provided K-factor for {fitting.fitting_type}: K = {k_factor:.3f}"
                )

        total_k += k_factor * fitting.quantity

    dp_fitting = total_k * velocity_head

    logger.info(f"Fitting losses: Total K = {total_k:.1f}, ΔP = {dp_fitting:.1f} Pa")
    return dp_fitting
