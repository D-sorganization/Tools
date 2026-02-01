"""Glass Properties Interface Module

Interface for external glass property calculators with default models.
Extracted from electrode_advisor.py for better organization.

Author: Chemical Equilibrium Calculator Team
Date: July 8, 2025
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class GlassPropertiesInterface:
    """Interface for external glass property calculators"""

    def __init__(self, external_calculator: Callable | None = None) -> None:
        """Initialize the class."""
        self.external_calculator = external_calculator
        self._default_properties = {
            "base_conductivity": 1.0,  # S/m at reference
            "activation_energy": 80000,  # J/mol
            "reference_temp": 1473.15,  # K (1200°C)
            "metal_conductivity": 10000.0,  # S/m - Very high for molten metal
        }
        self._temperature_dependent_data: dict[tuple[float, Any, float], float] = (
            {}
        )  # Cache for temperature-dependent properties
        self._current_properties: dict[str, Any] = {}  # Store current glass properties

    def get_conductivity(
        self,
        temperature_celsius: float,
        composition: dict[str, float] | None = None,
        power_density: float = 0,
        is_metal: bool = False,
    ) -> float:
        """Get electrical conductivity from external calculator or default model"""
        if is_metal:
            # Metal has very high conductivity, relatively constant with temperature
            return self._default_properties["metal_conductivity"]

        # Check cache first
        # Convert composition dict to a hashable type (tuple of sorted items) for caching
        comp_key = tuple(sorted(composition.items())) if composition else None

        cache_key = (
            temperature_celsius,
            comp_key,
            power_density,
        )
        if cache_key in self._temperature_dependent_data:
            return self._temperature_dependent_data[cache_key]

        if self.external_calculator:
            # Use external calculator if available
            try:
                conductivity = self.external_calculator(
                    temperature_celsius,
                    composition,
                    power_density,
                )
            except Exception as e:
                logger.warning(f"External calculator failed: {e}. Using default model.")
                conductivity = self._default_conductivity_model(
                    temperature_celsius,
                    power_density,
                )

        else:
            # Use default Arrhenius model for glass
            conductivity = self._default_conductivity_model(
                temperature_celsius,
                power_density,
            )

        # Cache the result
        self._temperature_dependent_data[cache_key] = conductivity
        return conductivity

    def set_external_calculator(self, calculator: Callable) -> None:
        """Set external calculator function"""
        self.external_calculator = calculator
        # Clear cache when calculator changes
        self._temperature_dependent_data.clear()

    def update_properties(self, properties: dict) -> None:
        """Update current glass properties"""
        self._current_properties.update(properties)

    def get_current_properties(self) -> dict:
        """Get current glass properties"""
        return self._current_properties.copy()

    def _default_conductivity_model(
        self,
        temperature_celsius: float,
        power_density: float = 0,
    ) -> float:
        """Default conductivity model using Arrhenius equation"""
        temp_kelvin = temperature_celsius + 273.15
        props = self._default_properties

        # Base Arrhenius equation
        # Simulating Arrhenius behavior: sigma = sigma0 * exp(-Ea / RT)
        # We model it relative to a reference conductivity at a reference temp
        # ln(sigma) = ln(sigma_ref) - (Ea/R) * (1/T - 1/T_ref)

        exponent = (-props["activation_energy"] / 8.314) * (
            1 / temp_kelvin - 1 / props["reference_temp"]
        )
        base_sigma = props["base_conductivity"] * np.exp(exponent)

        # Power density heating effect (simplified local heating)
        if power_density > 0:
            # Assume power density causes a local temperature rise that increases conductivity
            delta_t = power_density * 0.0001  # Simplified coefficient
            temp_effective = temp_kelvin + delta_t

            exponent_effective = (-props["activation_energy"] / 8.314) * (
                1 / temp_effective - 1 / props["reference_temp"]
            )
            base_sigma_effective = props["base_conductivity"] * np.exp(
                exponent_effective
            )

            # Use the effective conductivity
            base_sigma = base_sigma_effective

        return float(base_sigma)

    def get_resistivity(
        self,
        temperature_celsius: float,
        composition: dict[str, float] | None = None,
        power_density: float = 0,
        is_metal: bool = False,
    ) -> float:
        """Get electrical resistivity (1/conductivity)"""
        conductivity = self.get_conductivity(
            temperature_celsius,
            composition,
            power_density,
            is_metal,
        )
        return 1.0 / conductivity if conductivity > 0 else float("inf")

    def clear_cache(self) -> None:
        """Clear temperature-dependent data cache"""
        self._temperature_dependent_data.clear()
