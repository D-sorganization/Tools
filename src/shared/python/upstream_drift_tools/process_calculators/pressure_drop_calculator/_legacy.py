"""Legacy API for pressure_drop_calculator.

This module contains the original ``PressureDropCalculator`` and
``PressureDropResult`` interface that was used before the modular
``PressureDropCalculationEngine`` was introduced.

These classes are retained for backwards compatibility. New code
should use the modern API via ``PressureDropCalculationEngine``
and ``PressureDropInputs`` / ``PressureDropResults``.

Extracted from ``__init__.py`` per issue #1696 to eliminate the
god-module anti-pattern (calculation logic in an init file).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from upstream_drift_tools.utils.unit_constants import R_UNIVERSAL

# Standard Pipe Dimensions (Schedule 40) - Inner Diameter in meters
PIPE_DIMENSIONS_SCH40: Final[dict[str, float]] = {
    '1/2"': 0.01575,
    '3/4"': 0.02093,
    '1"': 0.02664,
    '1-1/4"': 0.03505,
    '1-1/2"': 0.04089,
    '2"': 0.05250,
    '2-1/2"': 0.06271,
    '3"': 0.07793,
    '4"': 0.10226,
    '6"': 0.15405,
    '8"': 0.20272,
    '10"': 0.25450,
    '12"': 0.30328,
    '14"': 0.33655,
    '16"': 0.38735,
    '18"': 0.43815,
    '20"': 0.48895,
    '24"': 0.59055,
}

# Standard Roughness Values in meters
ROUGHNESS_VALUES: Final[dict[str, float]] = {
    "Commercial Steel": 0.000045,
    "Drawn Tubing": 0.0000015,
    "Stainless Steel": 0.000015,
    "Cast Iron": 0.00026,
    "Concrete": 0.001,
}


@dataclass
class PressureDropResult:
    """Pressure drop calculation result (legacy API)."""

    pressure_drop_pa: float
    reynolds_number: float
    friction_factor: float
    velocity: float  # m/s
    flow_regime: str
    density: float  # kg/m^3
    viscosity: float  # Pa*s


class PressureDropCalculator:
    """Core pressure drop calculation engine (legacy API)."""

    def __init__(self) -> None:
        """Initialize the calculator."""

    def calculate_pressure_drop(
        self,
        pipe_diameter_m: float,
        pipe_length_m: float,
        roughness_m: float,
        flow_rate_kg_s: float,
        temperature_k: float,
        pressure_pa: float,
        molecular_weight_kg_mol: float,
    ) -> PressureDropResult:
        """Calculate pressure drop using Darcy-Weisbach equation.

        Args:
            pipe_diameter_m: Inner pipe diameter in metres. Must be > 0.
            pipe_length_m: Pipe length in metres.
            roughness_m: Absolute pipe roughness in metres.
            flow_rate_kg_s: Mass flow rate in kg/s.
            temperature_k: Gas temperature in Kelvin. Must be > 0.
            pressure_pa: Inlet pressure in Pascals. Must be > 0.
            molecular_weight_kg_mol: Gas molecular weight in kg/mol.

        Returns:
            PressureDropResult with pressure drop, flow regime, and
            fluid properties.

        Raises:
            TypeError: If any argument is not a float or int.
            ValueError: If pipe_diameter_m or temperature_k <= 0.
        """
        if not isinstance(pipe_diameter_m, (int, float)):
            raise TypeError("pipe_diameter_m must be a number")
        if not isinstance(temperature_k, (int, float)):
            raise TypeError("temperature_k must be a number")
        if pipe_diameter_m <= 0:
            raise ValueError("pipe_diameter_m must be > 0")
        if temperature_k <= 0:
            raise ValueError("temperature_k must be > 0")

        # Calculate gas properties (Z=1.0 assumption - Ideal Gas)
        Z = 1.0
        density = (pressure_pa * molecular_weight_kg_mol) / (
            Z * R_UNIVERSAL * temperature_k
        )

        # Estimate viscosity using Sutherland's formula
        T_ref = 291.15  # K
        mu_ref = 1.827e-5  # Pa·s
        S = 120  # K
        viscosity = (
            mu_ref
            * ((T_ref + S) / (temperature_k + S))
            * (temperature_k / T_ref) ** 1.5
        )

        # Calculate volumetric flow and velocity
        vol_flow = flow_rate_kg_s / density if density > 0 else 0.0
        area = math.pi * (pipe_diameter_m / 2) ** 2
        velocity = vol_flow / area if area > 0 else 0.0

        # Calculate Reynolds number
        Re = (
            (density * velocity * pipe_diameter_m) / viscosity if viscosity > 0 else 0.0
        )

        # Calculate friction factor
        rel_roughness = roughness_m / pipe_diameter_m if pipe_diameter_m > 0 else 0.0

        if Re > 4000:
            # Swamee-Jain explicit approximation
            A = rel_roughness / 3.7
            B = 5.74 / (Re**0.9) if Re > 0 else 0.01
            try:
                friction_factor = 0.25 / (math.log10(A + B) ** 2)
            except ValueError:
                friction_factor = 0.02  # Fallback
        elif Re > 2300:
            friction_factor = 0.03  # Transition
        else:
            friction_factor = 64 / Re if Re > 0 else 0.05

        # Calculate pressure drop (Darcy-Weisbach)
        pressure_drop_pa = (
            friction_factor
            * (pipe_length_m / pipe_diameter_m)
            * (density * velocity**2 / 2)
        )

        # Determine flow regime
        if Re < 2300:
            flow_regime = "Laminar"
        elif Re < 4000:
            flow_regime = "Transitional"
        else:
            flow_regime = "Turbulent"

        return PressureDropResult(
            pressure_drop_pa=pressure_drop_pa,
            reynolds_number=Re,
            friction_factor=friction_factor,
            velocity=velocity,
            flow_regime=flow_regime,
            density=density,
            viscosity=viscosity,
        )
