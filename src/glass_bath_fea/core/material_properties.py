"""Material property models for Glass Bath FEA.

This module provides temperature and composition dependent material
property calculations for molten glass and metal, compatible with
the electrode adviser's GlassPropertiesInterface.

Models implemented:
- Electrical conductivity: Arrhenius equation with composition correction
- Viscosity: Fulcher equation
- Resistivity: Inverse of conductivity
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import GlassComposition


# Physical constants
GAS_CONSTANT = 8.314  # J/(mol·K)

# Default Arrhenius parameters (from electrode adviser)
DEFAULT_BASE_CONDUCTIVITY = 1.0  # S/m at reference temperature
DEFAULT_ACTIVATION_ENERGY = 80000  # J/mol
DEFAULT_REFERENCE_TEMP_K = 1473.15  # K (1200°C)

# Metal conductivity (relatively temperature-independent)
DEFAULT_METAL_CONDUCTIVITY = 10000.0  # S/m

# Fulcher equation parameters for soda-lime glass viscosity
FULCHER_A = -2.0
FULCHER_B = 4500.0  # K
FULCHER_T0 = 250.0  # K


class GlassMaterialModel:
    """Temperature and composition dependent glass material model.

    Provides electrical conductivity, resistivity, and viscosity
    calculations based on the Arrhenius and Fulcher equations.

    Attributes:
        composition: Glass composition specification
    """

    def __init__(
        self,
        composition: GlassComposition,
        base_conductivity: float = DEFAULT_BASE_CONDUCTIVITY,
        activation_energy: float = DEFAULT_ACTIVATION_ENERGY,
        reference_temp_k: float = DEFAULT_REFERENCE_TEMP_K,
    ) -> None:
        """Initialize material model with composition.

        Args:
            composition: Glass composition specification
            base_conductivity: Base conductivity at reference temp (S/m)
            activation_energy: Activation energy for conduction (J/mol)
            reference_temp_k: Reference temperature in Kelvin
        """
        self.composition = composition
        self._base_conductivity = base_conductivity
        self._activation_energy = activation_energy
        self._reference_temp_k = reference_temp_k

        # Pre-compute Arrhenius factor for efficiency
        self._arrhenius_factor = -self._activation_energy / GAS_CONSTANT

    def get_conductivity(
        self,
        temperature_celsius: float,
        power_density: float = 0.0,
    ) -> float:
        """Calculate temperature and composition dependent conductivity.

        Uses modified Arrhenius equation:
            σ(T) = σ₀ × C_comp × exp(-Ea/R × (1/T - 1/T_ref))

        Where C_comp is a composition correction factor.

        Args:
            temperature_celsius: Temperature in degrees Celsius
            power_density: Local power density (W/m³) for heating effect

        Returns:
            Electrical conductivity in S/m
        """
        # Convert to Kelvin
        temp_k = temperature_celsius + 273.15

        # Apply small heating effect from power dissipation
        if power_density > 0:
            temp_k += 0.0001 * power_density  # Minor local heating

        # Arrhenius temperature dependence
        arrhenius_term = math.exp(
            self._arrhenius_factor * (1.0 / temp_k - 1.0 / self._reference_temp_k)
        )

        # Composition correction factors
        comp_factor = self._get_composition_factor()

        return self._base_conductivity * comp_factor * arrhenius_term

    def _get_composition_factor(self) -> float:
        """Calculate composition correction factor for conductivity.

        Na2O increases ionic mobility, Fe2O3 increases electronic conduction.

        Returns:
            Multiplicative correction factor
        """
        # Reference composition: Na2O = 13%, Fe2O3 = 0.1%
        na_factor = 1.0 + 0.02 * (self.composition.na2o - 13.0)
        fe_factor = 1.0 + 0.5 * self.composition.fe2o3

        return na_factor * fe_factor

    def get_resistivity(
        self,
        temperature_celsius: float,
        power_density: float = 0.0,
    ) -> float:
        """Calculate electrical resistivity.

        Args:
            temperature_celsius: Temperature in degrees Celsius
            power_density: Local power density (W/m³)

        Returns:
            Electrical resistivity in Ω·m
        """
        sigma = self.get_conductivity(temperature_celsius, power_density)
        return 1.0 / sigma

    def get_viscosity(self, temperature_celsius: float) -> float:
        """Calculate temperature dependent viscosity using Fulcher equation.

        log₁₀(η) = A + B/(T - T₀)

        Args:
            temperature_celsius: Temperature in degrees Celsius

        Returns:
            Dynamic viscosity in Pa·s
        """
        temp_k = temperature_celsius + 273.15

        # Fulcher equation
        log_viscosity = FULCHER_A + FULCHER_B / (temp_k - FULCHER_T0)

        return 10.0**log_viscosity

    def get_arrhenius_params(self) -> dict[str, float]:
        """Get Arrhenius equation parameters for MATLAB export.

        Returns:
            Dictionary with base_conductivity, activation_energy,
            reference_temp, and composition_factor.
        """
        return {
            "base_conductivity": self._base_conductivity,
            "activation_energy": self._activation_energy,
            "reference_temp": self._reference_temp_k,
            "composition_factor": self._get_composition_factor(),
        }


def get_metal_conductivity(temperature_celsius: float) -> float:
    """Get metal layer conductivity.

    Metal conductivity is much higher than glass and relatively
    temperature-independent over the operating range.

    Args:
        temperature_celsius: Temperature in degrees Celsius

    Returns:
        Electrical conductivity in S/m
    """
    # Small temperature correction (metals decrease slightly with temperature)
    temp_factor = 1.0 - 0.0001 * (temperature_celsius - 1200.0)
    return DEFAULT_METAL_CONDUCTIVITY * temp_factor


def export_material_data(
    model: GlassMaterialModel,
    output_path: Path | str,
    include_metal: bool = True,
) -> None:
    """Export material property data to MATLAB .mat file.

    Creates a .mat file containing all parameters needed for
    the MATLAB PDE Toolbox solver.

    Args:
        model: Glass material model instance
        output_path: Path to output .mat file
        include_metal: Whether to include metal properties
    """
    from scipy.io import savemat

    params = model.get_arrhenius_params()

    data = {
        "base_conductivity": np.array([params["base_conductivity"]]),
        "activation_energy": np.array([params["activation_energy"]]),
        "reference_temp": np.array([params["reference_temp"]]),
        "composition_factor": np.array([params["composition_factor"]]),
        "gas_constant": np.array([GAS_CONSTANT]),
    }

    if include_metal:
        data["metal_conductivity"] = np.array([DEFAULT_METAL_CONDUCTIVITY])

    # Add Fulcher parameters for viscosity
    data["fulcher_A"] = np.array([FULCHER_A])
    data["fulcher_B"] = np.array([FULCHER_B])
    data["fulcher_T0"] = np.array([FULCHER_T0])

    savemat(str(output_path), data, format="5")  # MATLAB v5 format
