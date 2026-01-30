"""Physical Constants and Conversion Factors for Process Calculators.

This module provides NIST-standard physical constants and conversion factors
needed by the standalone process calculators. Values are sourced from:
- NIST Special Publication 811 (2008 Edition)
- CODATA 2018 recommended values
- Perry's Chemical Engineers' Handbook, 9th ed.
"""

from typing import Final

# =============================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS (CODATA 2018)
# =============================================================================

# Universal gas constant [J/(mol·K)]
R_UNIVERSAL: Final[float] = 8.314462618
R_GAS_J_MOL_K: Final[float] = R_UNIVERSAL  # Alias

# Universal gas constant [kJ/(kmol·K)]
R_UNIVERSAL_KMOL: Final[float] = R_UNIVERSAL * 1000.0

# Standard gravity [m/s²]
STANDARD_GRAVITY: Final[float] = 9.80665
G: Final[float] = STANDARD_GRAVITY  # Alias

# Avogadro constant [1/mol]
AVOGADRO_NUMBER: Final[float] = 6.02214076e23
NA: Final[float] = AVOGADRO_NUMBER  # Alias

# Boltzmann constant [J/K]
BOLTZMANN_CONSTANT: Final[float] = 1.380649e-23
KB: Final[float] = BOLTZMANN_CONSTANT  # Alias

# Stefan-Boltzmann constant [W/(m²·K⁴)]
STEFAN_BOLTZMANN: Final[float] = 5.670374419e-8

# =============================================================================
# STANDARD CONDITIONS
# =============================================================================

# Standard Temperature and Pressure (IUPAC - Since 1982)
STP_TEMPERATURE_K: Final[float] = 273.15  # 0°C
STP_PRESSURE_PA: Final[float] = 100000.0  # 1 bar (100 kPa)
STP_PRESSURE_KPA: Final[float] = 100.0  # 1 bar

# Standard Atmosphere
ATM_PA: Final[float] = 101325.0  # 1 atm
ATM_KPA: Final[float] = 101.325  # 1 atm

# Normal Temperature and Pressure (NTP)
NTP_TEMPERATURE_K: Final[float] = 293.15  # 20°C
NTP_PRESSURE_PA: Final[float] = 101325.0  # 1 atm

# Standard Ambient Temperature and Pressure (SATP)
SATP_TEMPERATURE_K: Final[float] = 298.15  # 25°C
SATP_TEMPERATURE_C: Final[float] = 25.0
SATP_PRESSURE_PA: Final[float] = 100000.0  # 1 bar

# Reference conditions for thermodynamic calculations
T_REF_K: Final[float] = 298.15  # Standard reference temperature (25°C)
P_REF_PA: Final[float] = 101325.0  # Standard reference pressure (1 atm)
P_REF_KPA: Final[float] = 101.325

# SCFM Standard Conditions (US Engineering)
SCFM_60F_TEMPERATURE_K: Final[float] = 288.706  # 60°F
SCFM_70F_TEMPERATURE_K: Final[float] = 294.261  # 70°F

# =============================================================================
# MOLAR VOLUMES
# =============================================================================

# Ideal gas molar volume at STP [m³/mol]
MOLAR_VOLUME_STP: Final[float] = 0.02271095  # At 273.15 K, 100 kPa
MOLAR_VOLUME_STP_ATM: Final[float] = 0.022413969545  # At 273.15 K, 1 atm

# =============================================================================
# TEMPERATURE CONVERSIONS
# =============================================================================

CELSIUS_TO_KELVIN_OFFSET: Final[float] = 273.15
FAHRENHEIT_TO_RANKINE_OFFSET: Final[float] = 459.67
RANKINE_PER_KELVIN: Final[float] = 1.8

# =============================================================================
# PRESSURE CONVERSIONS (to Pa)
# =============================================================================

BAR_TO_PA: Final[float] = 100000.0
KPA_TO_PA: Final[float] = 1000.0
MPA_TO_PA: Final[float] = 1000000.0
ATM_TO_PA: Final[float] = 101325.0
PSI_TO_PA: Final[float] = 6894.757293168361
TORR_TO_PA: Final[float] = 133.32236842105263
MMHG_TO_PA: Final[float] = 133.322387415  # At 0°C
INH2O_TO_PA: Final[float] = 249.08890833333  # At 4°C

# =============================================================================
# ENERGY CONVERSIONS (to J)
# =============================================================================

KJ_TO_J: Final[float] = 1000.0
CAL_TO_J: Final[float] = 4.184  # Thermochemical calorie
KCAL_TO_J: Final[float] = 4184.0
BTU_TO_J: Final[float] = 1055.05585262

# =============================================================================
# FLOW CONVERSIONS
# =============================================================================

# Volumetric flow
CFM_TO_M3_S: Final[float] = 0.000471947443
LPM_TO_M3_S: Final[float] = 0.0000166667

# Mass flow
LB_HR_TO_KG_S: Final[float] = 0.000125998
KG_HR_TO_KG_S: Final[float] = 1 / 3600

# =============================================================================
# VISCOSITY
# =============================================================================

CP_TO_PA_S: Final[float] = 0.001  # Centipoise to Pascal-second
POISE_TO_PA_S: Final[float] = 0.1

# =============================================================================
# NUMERICAL TOLERANCES
# =============================================================================

ATOL_ZERO: Final[float] = 1e-12  # Absolute tolerance for zero comparisons
RTOL_DEFAULT: Final[float] = 1e-6  # Default relative tolerance

# =============================================================================
# COMMON MOLECULAR WEIGHTS [kg/mol]
# =============================================================================

MW_H2: Final[float] = 0.00201588
MW_O2: Final[float] = 0.031998
MW_N2: Final[float] = 0.0280134
MW_CO: Final[float] = 0.02801
MW_CO2: Final[float] = 0.04401
MW_H2O: Final[float] = 0.01801528
MW_CH4: Final[float] = 0.01604
MW_NH3: Final[float] = 0.01703
MW_H2S: Final[float] = 0.03408
MW_HCL: Final[float] = 0.036461
MW_HF: Final[float] = 0.02001
MW_SO2: Final[float] = 0.064066
MW_AIR: Final[float] = 0.028964  # Dry air average

# Dictionary form for lookup
MOLECULAR_WEIGHTS: dict[str, float] = {
    "H2": MW_H2,
    "O2": MW_O2,
    "N2": MW_N2,
    "CO": MW_CO,
    "CO2": MW_CO2,
    "H2O": MW_H2O,
    "CH4": MW_CH4,
    "NH3": MW_NH3,
    "H2S": MW_H2S,
    "HCl": MW_HCL,
    "HF": MW_HF,
    "SO2": MW_SO2,
    "Air": MW_AIR,
}

# =============================================================================
# WATER/STEAM PROPERTIES
# =============================================================================

# Water properties at 25°C
WATER_DENSITY_25C: Final[float] = 997.05  # kg/m³
WATER_VISCOSITY_25C: Final[float] = 0.00089  # Pa·s

# Triple point
WATER_TRIPLE_POINT_T: Final[float] = 273.16  # K
WATER_TRIPLE_POINT_P: Final[float] = 611.657  # Pa

# Critical point
WATER_CRITICAL_T: Final[float] = 647.096  # K
WATER_CRITICAL_P: Final[float] = 22064000.0  # Pa
WATER_CRITICAL_RHO: Final[float] = 322.0  # kg/m³

# Heat of vaporization at 100°C [J/kg]
WATER_HEAT_VAP_100C: Final[float] = 2256400.0

# Antoine equation coefficients for water (pressure in mmHg, temperature in °C)
# Valid range: 1°C to 100°C
ANTOINE_WATER_A: Final[float] = 8.07131
ANTOINE_WATER_B: Final[float] = 1730.63
ANTOINE_WATER_C: Final[float] = 233.426


def get_molecular_weight(species: str) -> float:
    """Get molecular weight for a species [kg/mol].

    Args:
        species: Chemical species name or formula

    Returns:
        Molecular weight in kg/mol, or 0.029 (air) if not found
    """
    # Normalize common variations
    normalized = species.upper().replace(" ", "")
    lookup = {k.upper(): v for k, v in MOLECULAR_WEIGHTS.items()}
    return lookup.get(normalized, MW_AIR)


def celsius_to_kelvin(temp_c: float) -> float:
    """Convert temperature from Celsius to Kelvin."""
    return temp_c + CELSIUS_TO_KELVIN_OFFSET


def kelvin_to_celsius(temp_k: float) -> float:
    """Convert temperature from Kelvin to Celsius."""
    return temp_k - CELSIUS_TO_KELVIN_OFFSET


def fahrenheit_to_kelvin(temp_f: float) -> float:
    """Convert temperature from Fahrenheit to Kelvin."""
    return (temp_f + FAHRENHEIT_TO_RANKINE_OFFSET) / RANKINE_PER_KELVIN


def kelvin_to_fahrenheit(temp_k: float) -> float:
    """Convert temperature from Kelvin to Fahrenheit."""
    return temp_k * RANKINE_PER_KELVIN - FAHRENHEIT_TO_RANKINE_OFFSET


def bar_to_pa(pressure_bar: float) -> float:
    """Convert pressure from bar to Pascal."""
    return pressure_bar * BAR_TO_PA


def pa_to_bar(pressure_pa: float) -> float:
    """Convert pressure from Pascal to bar."""
    return pressure_pa / BAR_TO_PA


def psi_to_pa(pressure_psi: float) -> float:
    """Convert pressure from psi to Pascal."""
    return pressure_psi * PSI_TO_PA


def pa_to_psi(pressure_pa: float) -> float:
    """Convert pressure from Pascal to psi."""
    return pressure_pa / PSI_TO_PA


# =============================================================================
# ADDITIONAL WATER PROPERTIES FOR PROCESS CALCULATORS
# =============================================================================

# Liquid water heat capacity [J/(kg·K)] at 25°C
CP_WATER_LIQUID: Final[float] = 4182.0

# Liquid water density at standard conditions [kg/m³]
DENSITY_WATER_STD: Final[float] = 997.0

# Heat of vaporization of water [J/kg] at 100°C
H_VAP_WATER: Final[float] = 2256400.0

# Mass conversion
KG_TO_LB: Final[float] = 2.20462262185

# =============================================================================
# WATER VAPOR PRESSURE COEFFICIENTS (Modified Buck Equation)
# =============================================================================
# Source: Buck, A.L. (1981). "New Equations for Computing Vapor Pressure"

WATER_VAPOR_A: Final[float] = 0.61115  # [kPa]
WATER_VAPOR_B: Final[float] = 23.036  # [K]
WATER_VAPOR_C: Final[float] = 279.82  # [K]
WATER_VAPOR_D: Final[float] = 333.7  # [K]
