"""
Shared physical and mathematical constants for the Tools monorepo.

This module provides NIST-standard values for constants that appear in
multiple tool packages. Domain-specific constants (astronomical values,
tissue densities, UI sizing) remain in their respective modules.

Source: NIST CODATA 2018 recommended values.
"""

from __future__ import annotations

from typing import Final

# ─────────────────────────────────────────────────────────────────────────────
# Universal physical constants
# ─────────────────────────────────────────────────────────────────────────────

#: Gravitational acceleration at Earth's surface (m/s²)
STANDARD_GRAVITY: Final[float] = 9.80665

#: Universal gas constant (J/mol/K)
R_UNIVERSAL: Final[float] = 8.314_462_618

#: Boltzmann constant (J/K)
BOLTZMANN_CONSTANT: Final[float] = 1.380_649e-23

#: Avogadro's number (mol⁻¹)
AVOGADRO_NUMBER: Final[float] = 6.022_140_76e23

#: Speed of light in vacuum (m/s)
SPEED_OF_LIGHT: Final[float] = 299_792_458.0

# ─────────────────────────────────────────────────────────────────────────────
# Thermodynamic reference conditions
# ─────────────────────────────────────────────────────────────────────────────

#: Standard pressure, 1 atm (Pa)
STANDARD_PRESSURE_PA: Final[float] = 101_325.0

#: Standard temperature, 0°C in Kelvin
STANDARD_TEMPERATURE_K: Final[float] = 273.15

#: Standard temperature, 15°C in Kelvin (ISO 13443)
ISO_STANDARD_TEMPERATURE_K: Final[float] = 288.15

#: Celsius to Kelvin offset (add to °C to get K)
CELSIUS_TO_KELVIN: Final[float] = 273.15

# ─────────────────────────────────────────────────────────────────────────────
# Conversion factors
# ─────────────────────────────────────────────────────────────────────────────

#: Degrees to radians (multiply angle in ° by this)
DEG_TO_RAD: Final[float] = 3.141_592_653_589_793 / 180.0

#: Radians to degrees (multiply angle in rad by this)
RAD_TO_DEG: Final[float] = 180.0 / 3.141_592_653_589_793

#: Kilopascal to Pascal
KPA_TO_PA: Final[float] = 1_000.0

#: Megapascal to Pascal
MPA_TO_PA: Final[float] = 1_000_000.0

#: Bar to Pascal
BAR_TO_PA: Final[float] = 100_000.0

#: Atmosphere to Pascal
ATM_TO_PA: Final[float] = STANDARD_PRESSURE_PA

#: PSI to Pascal
PSI_TO_PA: Final[float] = 6_894.757_293_168

#: kJ to J
KJ_TO_J: Final[float] = 1_000.0

#: MJ to J
MJ_TO_J: Final[float] = 1_000_000.0

#: kWh to J
KWH_TO_J: Final[float] = 3_600_000.0

#: Hours to seconds
HOURS_TO_SECONDS: Final[float] = 3_600.0

#: Minutes to seconds
MINUTES_TO_SECONDS: Final[float] = 60.0
