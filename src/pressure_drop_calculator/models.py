"""Pydantic data models for Pressure Drop Calculator API.

Provides strongly-typed input/output models with validation for the pressure drop
calculation pipeline. These models ensure type safety across the API boundary
and provide clear, validated request/response contracts.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PressureDropInput(BaseModel):
    """Validated input model for pressure drop calculations.

    All dimensions in SI units (meters, Pascals, Kelvin, kg/s) unless otherwise noted.

    Attributes:
        pipe_diameter_m: Internal pipe diameter in meters. Must be positive.
        pipe_length_m: Total pipe length in meters. Must be positive.
        mass_flow_rate_kg_s: Mass flow rate in kg/s. Must be positive.
        inlet_pressure_pa: Inlet pressure (absolute) in Pascals. Must be positive.
        inlet_temperature_k: Inlet temperature in Kelvin. Must be positive.
        pipe_roughness_m: Absolute pipe roughness in meters. Defaults to 0.000045 m (drawn tubing).
        elevation_change_m: Elevation change in meters (positive = upward). Defaults to 0.
        gas_composition: Dictionary of gas component names to mole fractions. Must sum to 1.0.
            Defaults to air (N2: 0.79, O2: 0.21).
        friction_method: Friction factor correlation method. One of "colebrook",
            "churchill", "swamee-jain". Defaults to "colebrook".
        apply_compressibility: Whether to apply compressible flow corrections. Defaults to True.
    """  # noqa: E501

    pipe_diameter_m: float = Field(
        ..., gt=0, description="Internal pipe diameter in meters"
    )
    pipe_length_m: float = Field(..., gt=0, description="Pipe length in meters")
    mass_flow_rate_kg_s: float = Field(..., gt=0, description="Mass flow rate in kg/s")
    inlet_pressure_pa: float = Field(
        ..., gt=0, description="Inlet absolute pressure in Pa"
    )
    inlet_temperature_k: float = Field(..., gt=0, description="Inlet temperature in K")
    pipe_roughness_m: float = Field(
        default=0.000045, ge=0, description="Absolute roughness in meters"
    )
    elevation_change_m: float = Field(
        default=0.0, description="Elevation change in meters (positive upward)"
    )
    gas_composition: dict[str, float] = Field(
        default_factory=lambda: {"N2": 0.79, "O2": 0.21},
        description="Gas composition as mole fractions (must sum to 1.0)",
    )
    friction_method: str = Field(
        default="colebrook",
        description="Friction factor method: 'colebrook', 'churchill', or 'swamee-jain'",  # noqa: E501
    )
    apply_compressibility: bool = Field(
        default=True, description="Apply compressible flow corrections"
    )

    class Config:
        """Pydantic config."""

        str_strip_whitespace = True
        validate_assignment = True

    def validate_composition(self) -> None:
        """Validate that gas composition sums to 1.0 within tolerance.

        Raises:
            ValueError: If composition does not sum to 1.0 (±1%).
        """
        total = sum(self.gas_composition.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Gas composition mole fractions must sum to 1.0, got {total:.4f}"
            )
        if any(x < 0 or x > 1 for x in self.gas_composition.values()):
            raise ValueError("All mole fractions must be between 0 and 1")


class PressureDropOutput(BaseModel):
    """Validated output model for pressure drop calculations.

    All results in SI units (Pascals, m/s, etc.) unless otherwise noted.

    Attributes:
        pressure_drop_pa: Total pressure drop in Pascals.
        pressure_drop_bar: Total pressure drop in bar (1 bar = 100000 Pa).
        pressure_drop_psi: Total pressure drop in psi (1 psi = 6894.76 Pa).
        friction_pressure_drop_pa: Pressure drop due to friction alone (Pa).
        acceleration_pressure_drop_pa: Pressure drop due to flow acceleration/deceleration (Pa).
        elevation_pressure_drop_pa: Pressure drop due to elevation change (Pa).
        inlet_velocity_m_s: Inlet velocity in m/s.
        outlet_velocity_m_s: Outlet velocity in m/s.
        reynolds_number: Reynolds number at inlet conditions (dimensionless).
        friction_factor: Darcy friction factor at inlet conditions (dimensionless).
        outlet_pressure_pa: Outlet pressure (absolute) in Pascals.
        outlet_temperature_k: Outlet temperature in Kelvin (accounting for expansion/compression).
        average_density_kg_m3: Average gas density along the pipe (kg/m³).
        mach_number: Mach number at outlet (for compressible flow analysis).
        compressibility_factor: Compressibility factor (Z) used in calculations.
        calculation_method: Method used for friction factor ("colebrook", "churchill", etc.).
    """  # noqa: E501

    pressure_drop_pa: float = Field(..., ge=0, description="Total pressure drop in Pa")
    pressure_drop_bar: float = Field(
        ..., ge=0, description="Total pressure drop in bar"
    )
    pressure_drop_psi: float = Field(
        ..., ge=0, description="Total pressure drop in psi"
    )
    friction_pressure_drop_pa: float = Field(
        default=0.0, ge=0, description="Friction pressure drop in Pa"
    )
    acceleration_pressure_drop_pa: float = Field(
        default=0.0, description="Acceleration pressure drop in Pa"
    )
    elevation_pressure_drop_pa: float = Field(
        default=0.0, description="Elevation pressure drop in Pa"
    )
    inlet_velocity_m_s: float = Field(
        default=0.0, ge=0, description="Inlet velocity in m/s"
    )
    outlet_velocity_m_s: float = Field(
        default=0.0, ge=0, description="Outlet velocity in m/s"
    )
    reynolds_number: float = Field(
        default=0.0, ge=0, description="Reynolds number (dimensionless)"
    )
    friction_factor: float = Field(
        default=0.0, ge=0, description="Darcy friction factor (dimensionless)"
    )
    outlet_pressure_pa: float = Field(
        default=0.0, ge=0, description="Outlet absolute pressure in Pa"
    )
    outlet_temperature_k: float = Field(
        default=0.0, gt=0, description="Outlet temperature in K"
    )
    average_density_kg_m3: float = Field(
        default=0.0, ge=0, description="Average gas density in kg/m³"
    )
    mach_number: float = Field(
        default=0.0, ge=0, description="Mach number at outlet (dimensionless)"
    )
    compressibility_factor: float = Field(
        default=1.0, gt=0, description="Compressibility factor Z (dimensionless)"
    )
    calculation_method: str = Field(
        default="", description="Friction factor correlation method used"
    )
    success: bool = Field(default=True, description="Whether calculation succeeded")
    error_message: str = Field(
        default="", description="Error message if calculation failed"
    )

    class Config:
        """Pydantic config."""

        validate_assignment = True
