# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Unit Converter - Python backend using NIST-standard conversion factors.

Mirrors the conversion logic from the PyQt6 flow rate converter and the
shared upstream_drift_tools unit_constants module, providing a unified
API for all unit categories.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS (sourced from shared/python/upstream_drift_tools/utils/unit_constants.py)
# =============================================================================

CELSIUS_OFFSET = 273.15
RANKINE_RATIO = 5.0 / 9.0

# Standard conditions for gas flow
STANDARD_CONDITIONS: dict[str, dict[str, float | str]] = {
    "STP": {
        "temp": 273.15,
        "pressure": 101325.0,
        "label": "STP (0 deg C, 101.325 kPa)",
    },
    "SCFM_60F": {
        "temp": 288.706,
        "pressure": 101325.0,
        "label": "SCFM at 60 deg F, 14.696 psia",
    },
    "SCFM_70F": {
        "temp": 294.261,
        "pressure": 101325.0,
        "label": "SCFM at 70 deg F, 14.696 psia",
    },
    "NTP": {
        "temp": 293.15,
        "pressure": 101325.0,
        "label": "NTP (20 deg C, 101.325 kPa)",
    },
    "SATP": {"temp": 298.15, "pressure": 100000.0, "label": "SATP (25 deg C, 1 bar)"},
}

# Gas database with physical properties
GAS_DATABASE: dict[str, dict[str, float | str]] = {
    "air": {"name": "Air", "mw": 28.97, "density_stp": 1.2922, "k": 1.4},
    "nitrogen": {"name": "Nitrogen", "mw": 28.014, "density_stp": 1.2506, "k": 1.4},
    "oxygen": {"name": "Oxygen", "mw": 31.999, "density_stp": 1.4289, "k": 1.395},
    "hydrogen": {"name": "Hydrogen", "mw": 2.016, "density_stp": 0.08988, "k": 1.405},
    "methane": {"name": "Methane", "mw": 16.043, "density_stp": 0.7168, "k": 1.321},
    "co": {"name": "Carbon Monoxide", "mw": 28.01, "density_stp": 1.25, "k": 1.4},
    "co2": {"name": "Carbon Dioxide", "mw": 44.01, "density_stp": 1.9768, "k": 1.289},
    "h2o": {"name": "Water Vapor", "mw": 18.015, "density_stp": 0.00485, "k": 1.33},
}

SCFM_TO_CU_METER_PER_HOUR_AT_60F = 1.699010795

# =============================================================================
# CONVERSION FACTORS - NIST-standard values
# All factors convert to a base SI unit for each category.
# =============================================================================

CONVERSION_FACTORS: dict[str, dict[str, float]] = {
    "length": {
        "m": 1.0,
        "cm": 0.01,
        "mm": 0.001,
        "um": 1.0e-6,
        "nm": 1.0e-9,
        "km": 1000.0,
        "in": 0.0254,
        "ft": 0.3048,
        "yd": 0.9144,
        "mi": 1609.344,
        "mil": 2.54e-5,
    },
    "mass": {
        "kg": 1.0,
        "g": 0.001,
        "mg": 1.0e-6,
        "tonne": 1000.0,
        "lb": 0.45359237,
        "oz": 0.028349523125,
        "ton": 907.18474,
        "long_ton": 1016.0469088,
        "slug": 14.59390294,
        "grain": 6.479891e-5,
    },
    "volume": {
        "m3": 1.0,
        "L": 0.001,
        "mL": 1.0e-6,
        "cm3": 1.0e-6,
        "mm3": 1.0e-9,
        "ft3": 0.028316846592,
        "in3": 1.6387064e-5,
        "gal": 0.003785411784,
        "imp_gal": 0.00454609,
        "qt": 0.000946352946,
        "pt": 0.000473176473,
        "fl_oz": 2.95735295625e-5,
        "bbl": 0.158987294928,
    },
    "pressure": {
        "Pa": 1.0,
        "kPa": 1000.0,
        "MPa": 1.0e6,
        "GPa": 1.0e9,
        "bar": 100000.0,
        "atm": 101325.0,
        "mbar": 100.0,
        "psi": 6894.757293168,
        "torr": 133.322387415,
        "mmHg": 133.322387415,
        "inHg": 3386.389,
        "inH2O": 249.082,
        "ftH2O": 2989.07,
        "cmH2O": 98.0665,
    },
    "energy": {
        "J": 1.0,
        "kJ": 1000.0,
        "MJ": 1.0e6,
        "GJ": 1.0e9,
        "BTU": 1055.05585262,
        "cal": 4.184,
        "kcal": 4184.0,
        "Wh": 3600.0,
        "kWh": 3.6e6,
        "MWh": 3.6e9,
        "therm": 105505585.262,
        "erg": 1.0e-7,
        "eV": 1.602176634e-19,
    },
    "power": {
        "W": 1.0,
        "kW": 1000.0,
        "MW": 1.0e6,
        "GW": 1.0e9,
        "hp": 745.69987158227022,
        "metric_hp": 735.49875,
        "BTU/hr": 0.29307107017222,
        "cal/s": 4.184,
        "kcal/hr": 1.163,
    },
    "mass_flow": {
        "kg/s": 1.0,
        "kg/min": 1.0 / 60.0,
        "kg/hr": 1.0 / 3600.0,
        "kg/day": 1.0 / 86400.0,
        "g/s": 0.001,
        "g/min": 0.001 / 60.0,
        "g/hr": 0.001 / 3600.0,
        "g/day": 0.001 / 86400.0,
        "lb/hr": 0.45359237 / 3600.0,
        "lb/min": 0.45359237 / 60.0,
        "lb/s": 0.45359237,
        "lb/day": 0.45359237 / 86400.0,
        "tonne/hr": 1000.0 / 3600.0,
        "tonne/day": 1000.0 / 86400.0,
        "ton/hr": 907.18474 / 3600.0,
        "ton/day": 907.18474 / 86400.0,
    },
    "area": {
        "m2": 1.0,
        "cm2": 1.0e-4,
        "mm2": 1.0e-6,
        "km2": 1.0e6,
        "in2": 6.4516e-4,
        "ft2": 0.09290304,
        "yd2": 0.83612736,
        "acre": 4046.8564224,
        "hectare": 10000.0,
    },
    "time": {
        "s": 1.0,
        "min": 60.0,
        "hr": 3600.0,
        "day": 86400.0,
    },
    "volumetric_flow": {
        "m3/s": 1.0,
        "m3/min": 1.0 / 60.0,
        "m3/hr": 1.0 / 3600.0,
        "m3/day": 1.0 / 86400.0,
        "L/s": 0.001,
        "L/min": 0.001 / 60.0,
        "L/hr": 0.001 / 3600.0,
        "L/day": 0.001 / 86400.0,
        "ft3/s": 0.028316846592,
        "ft3/min": 0.028316846592 / 60.0,
        "ft3/hr": 0.028316846592 / 3600.0,
        "ft3/day": 0.028316846592 / 86400.0,
        "gal/min": 0.003785411784 / 60.0,
        "gpm": 0.003785411784 / 60.0,
        "gal/hr": 0.003785411784 / 3600.0,
        "gal/day": 0.003785411784 / 86400.0,
        "imp_gal/min": 0.00454609 / 60.0,
        "imp_gal/hr": 0.00454609 / 3600.0,
        "imp_gal/day": 0.00454609 / 86400.0,
        "bbl/day": 0.158987294928 / 86400.0,
        "bbl/hr": 0.158987294928 / 3600.0,
    },
    "density": {
        "kg/m3": 1.0,
        "g/L": 1.0,
        "g/cm3": 1000.0,
        "lb/ft3": 16.01846337396,
        "lb/gal": 119.8264273,
        "kg/L": 1000.0,
    },
    "dynamic_viscosity": {
        "Pa.s": 1.0,
        "mPa.s": 0.001,
        "cP": 0.001,
        "P": 0.1,
        "lb/ft.s": 1.4881639436,
    },
    "kinematic_viscosity": {
        "m2/s": 1.0,
        "cSt": 1.0e-6,
        "cm2/s": 1.0e-4,
        "St": 1.0e-4,
        "ft2/s": 0.09290304,
    },
    "thermal_conductivity": {
        "W/m.K": 1.0,
        "BTU/(ft.hr.F)": 1.7307346664,
        "cal/(cm.s.C)": 418.4,
    },
    "heat_transfer": {
        "W/m2.K": 1.0,
        "BTU/(ft2.hr.F)": 5.6782633411,
    },
    "specific_heat": {
        "J/kg.K": 1.0,
        "kJ/kg.K": 1000.0,
        "BTU/lb.F": 4186.8,
        "cal/g.C": 4186.8,
    },
}

# Heating value conversions (mass-based to MJ/kg)
HEATING_VALUE_FACTORS: dict[str, float | None] = {
    "MJ/kg": 1.0,
    "kJ/kg": 0.001,
    "J/kg": 1e-6,
    "cal/g": 0.004184,
    "kcal/kg": 0.004184,
    "BTU/lb": 0.00232444,
    "kWh/kg": 3.6,
    "MJ/Nm3": None,
    "BTU/scf": None,
    "kWh/Nm3": None,
}

# Category display labels
CATEGORY_LABELS: dict[str, str] = {
    "length": "Length",
    "mass": "Mass",
    "volume": "Volume",
    "temperature": "Temperature",
    "pressure": "Pressure",
    "energy": "Energy",
    "power": "Power",
    "gas_flow": "Gas Flow",
    "heating_value": "Heating Value",
    "mass_flow": "Mass Flow",
    "volumetric_flow": "Volumetric Flow",
    "area": "Area",
    "time": "Time",
    "density": "Density",
    "dynamic_viscosity": "Dynamic Viscosity",
    "kinematic_viscosity": "Kinematic Viscosity",
    "thermal_conductivity": "Thermal Conductivity",
    "heat_transfer": "Heat Transfer",
    "specific_heat": "Specific Heat",
}


@dataclass
class ConversionResult:
    """Result of a unit conversion."""

    value: float
    from_unit: str
    to_unit: str
    result: float
    category: str


class UnitConverter:
    """NIST-compliant unit converter with support for all engineering categories."""

    def get_categories(self) -> list[str]:
        """Return all available conversion categories."""
        return [
            "temperature",
            "gas_flow",
            "heating_value",
            *list(CONVERSION_FACTORS.keys()),
        ]

    def get_units_for_category(self, category: str) -> list[str]:
        """Return all units available in a category."""
        if not (category is not None):
            raise ValueError("category must be provided")
        if category == "temperature":
            return ["K", "C", "F", "R"]
        if category == "gas_flow":
            return ["SCFM", "ACFM", "Nm3/hr"]
        if category == "heating_value":
            return list(HEATING_VALUE_FACTORS.keys())
        if category in CONVERSION_FACTORS:
            return list(CONVERSION_FACTORS[category].keys())
        return []

    def get_category_label(self, category: str) -> str:
        """Return the display label for a category."""
        return CATEGORY_LABELS.get(category, category)

    def get_category(self, unit: str) -> str | None:
        """Determine which category a unit belongs to."""
        # Temperature (highest priority)
        if not (unit is not None):
            raise ValueError("unit must be provided")
        if unit.upper() in ("K", "C", "F", "R"):
            return "temperature"

        # Gas flow
        if unit.upper() in ("SCFM", "ACFM", "NM3/HR"):
            return "gas_flow"

        # Heating value
        if unit in HEATING_VALUE_FACTORS:
            return "heating_value"

        # Standard categories
        for cat, units in CONVERSION_FACTORS.items():
            if unit in units:
                return cat

        return None

    def convert(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        *,
        temperature: float | None = None,
        pressure: float | None = None,
        gas_type: str = "air",
        standard_condition: str = "SCFM_60F",
        gas_density_stp: float | None = None,
    ) -> ConversionResult:
        """Perform a unit conversion.

        Args:
            value: The numeric value to convert.
            from_unit: Source unit.
            to_unit: Target unit.
            temperature: Temperature in K (for gas flow ACFM conversions).
            pressure: Pressure in Pa (for gas flow ACFM conversions).
            gas_type: Gas type key (for gas flow conversions).
            standard_condition: Standard condition key (for gas flow).
            gas_density_stp: Gas density at STP in kg/m3 (for heating value).

        Returns:
            ConversionResult with the converted value.

        Raises:
            ValueError: If units are incompatible or unknown.
        """
        from_category = self.get_category(from_unit)
        to_category = self.get_category(to_unit)

        if from_category is None:
            raise ValueError(f"Unknown unit: {from_unit}")
        if to_category is None:
            raise ValueError(f"Unknown unit: {to_unit}")

        if from_category != to_category:
            # Allow mass_flow <-> gas_flow cross-category
            is_gas_flow = (
                from_category == "gas_flow" and to_category == "mass_flow"
            ) or (  # noqa: E501
                from_category == "mass_flow" and to_category == "gas_flow"
            )
            if not is_gas_flow:
                raise ValueError(
                    f"Cannot convert {from_unit} ({from_category}) "
                    f"to {to_unit} ({to_category})"  # noqa: E501
                )

        category = from_category

        if category == "temperature":
            result = self._convert_temperature(value, from_unit, to_unit)
        elif category == "gas_flow" or to_category == "gas_flow":
            result = self._convert_gas_flow(
                value,
                from_unit,
                to_unit,
                temperature=temperature,
                pressure=pressure,
                gas_type=gas_type,
                standard_condition=standard_condition,
            )
        elif category == "heating_value":
            result = self._convert_heating_value(
                value, from_unit, to_unit, gas_density_stp
            )  # noqa: E501
        else:
            result = self._convert_linear(value, from_unit, to_unit, category)

        return ConversionResult(
            value=value,
            from_unit=from_unit,
            to_unit=to_unit,
            result=result,
            category=category,
        )

    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert temperature units via Kelvin as intermediate."""
        if not (value is not None):
            raise ValueError("value must be provided")
        fu = from_unit.upper()
        tu = to_unit.upper()

        if fu == tu:
            return value

        # Convert to Kelvin
        if fu == "K":
            kelvin = value
        elif fu == "C":
            kelvin = value + CELSIUS_OFFSET
        elif fu == "F":
            kelvin = (value - 32) * RANKINE_RATIO + CELSIUS_OFFSET
        elif fu == "R":
            kelvin = value * RANKINE_RATIO
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")

        # Convert from Kelvin
        if tu == "K":
            return kelvin
        elif tu == "C":
            return kelvin - CELSIUS_OFFSET
        elif tu == "F":
            return (kelvin - CELSIUS_OFFSET) / RANKINE_RATIO + 32
        elif tu == "R":
            return kelvin / RANKINE_RATIO
        else:
            raise ValueError(f"Unknown temperature unit: {to_unit}")

    def _convert_linear(
        self, value: float, from_unit: str, to_unit: str, category: str
    ) -> float:  # noqa: E501
        """Convert units with a simple linear factor through base SI unit."""
        factors = CONVERSION_FACTORS[category]
        from_factor = factors.get(from_unit)
        to_factor = factors.get(to_unit)

        if from_factor is None or to_factor is None:
            raise ValueError(
                f"Conversion factors not found for {from_unit} to {to_unit}"
            )  # noqa: E501

        base_value = value * from_factor
        return base_value / to_factor

    def _convert_gas_flow(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        *,
        temperature: float | None = None,
        pressure: float | None = None,
        gas_type: str = "air",
        standard_condition: str = "SCFM_60F",
    ) -> float:
        """Convert gas flow rates via standard m3/hr as intermediate."""
        gas_props = GAS_DATABASE.get(gas_type.lower(), GAS_DATABASE["air"])
        standard = STANDARD_CONDITIONS.get(
            standard_condition, STANDARD_CONDITIONS["SCFM_60F"]
        )  # noqa: E501

        fu = from_unit.upper()
        tu = to_unit.upper()

        # Validate ACFM requires T/P
        if (fu == "ACFM" or tu == "ACFM") and (temperature is None or pressure is None):
            raise ValueError(
                "Temperature and pressure are required for ACFM conversions"
            )  # noqa: E501

        # Convert to standard m3/hr
        std_temp = float(standard["temp"])
        std_pressure = float(standard["pressure"])
        stp = STANDARD_CONDITIONS["STP"]
        stp_temp = float(stp["temp"])
        stp_pressure = float(stp["pressure"])
        density = float(gas_props["density_stp"])

        if fu == "SCFM":
            m3_hr_std = value * SCFM_TO_CU_METER_PER_HOUR_AT_60F
            if std_temp != stp_temp or std_pressure != stp_pressure:
                m3_hr_std = (
                    m3_hr_std * (stp_temp / std_temp) * (std_pressure / stp_pressure)
                )  # noqa: E501
        elif fu == "ACFM":
            if not (temperature is not None and pressure is not None):
                raise ValueError("DbC Blocked: Precondition failed.")
            scfm = value * (std_temp / temperature) * (pressure / std_pressure)
            m3_hr_std = scfm * SCFM_TO_CU_METER_PER_HOUR_AT_60F
            if std_temp != stp_temp or std_pressure != stp_pressure:
                m3_hr_std = (
                    m3_hr_std * (stp_temp / std_temp) * (std_pressure / stp_pressure)
                )  # noqa: E501
        elif fu in ("NM3/HR", "NM3/HR"):
            m3_hr_std = value
        elif from_unit.lower() in CONVERSION_FACTORS.get("mass_flow", {}):
            kg_s = value * CONVERSION_FACTORS["mass_flow"][from_unit.lower()]
            kg_hr = kg_s * 3600.0
            m3_hr_std = kg_hr / density
        else:
            raise ValueError(f"Unknown gas flow unit: {from_unit}")

        # Convert from standard m3/hr to target
        if tu == "SCFM":
            m3_hr_at_scfm = m3_hr_std
            if stp_temp != std_temp or stp_pressure != std_pressure:
                m3_hr_at_scfm = (
                    m3_hr_std * (std_temp / stp_temp) * (stp_pressure / std_pressure)
                )  # noqa: E501
            return m3_hr_at_scfm / SCFM_TO_CU_METER_PER_HOUR_AT_60F
        elif tu == "ACFM":
            if not (temperature is not None and pressure is not None):
                raise ValueError("DbC Blocked: Precondition failed.")
            m3_hr_at_scfm = m3_hr_std
            if stp_temp != std_temp or stp_pressure != std_pressure:
                m3_hr_at_scfm = (
                    m3_hr_std * (std_temp / stp_temp) * (stp_pressure / std_pressure)
                )  # noqa: E501
            scfm_val = m3_hr_at_scfm / SCFM_TO_CU_METER_PER_HOUR_AT_60F
            return scfm_val * (std_pressure / pressure) * (temperature / std_temp)
        elif tu in ("NM3/HR", "NM3/HR"):
            return m3_hr_std
        elif to_unit.lower() in CONVERSION_FACTORS.get("mass_flow", {}):
            kg_hr = m3_hr_std * density
            kg_s = kg_hr / 3600.0
            return kg_s / CONVERSION_FACTORS["mass_flow"][to_unit.lower()]
        else:
            raise ValueError(f"Unknown gas flow unit: {to_unit}")

    def _convert_heating_value(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        gas_density_stp: float | None = None,
    ) -> float:
        """Convert heating value units via MJ/kg as intermediate."""
        from_key = from_unit.lower()
        to_key = to_unit.lower()

        if from_key == to_key:
            return value

        from_factor = HEATING_VALUE_FACTORS.get(from_unit)
        to_factor = HEATING_VALUE_FACTORS.get(to_unit)

        if from_factor is None and from_unit not in HEATING_VALUE_FACTORS:
            raise ValueError(f"Unknown heating value unit: {from_unit}")
        if to_factor is None and to_unit not in HEATING_VALUE_FACTORS:
            raise ValueError(f"Unknown heating value unit: {to_unit}")

        # Convert to MJ/kg
        if HEATING_VALUE_FACTORS[from_unit] is None:
            if gas_density_stp is None:
                raise ValueError(f"Gas density required for {from_unit} conversion")
            if from_key in ("mj/nm3",):
                mj_per_kg = value / gas_density_stp
            elif from_key == "btu/scf":
                mj_nm3 = value * 0.0372589
                mj_per_kg = mj_nm3 / gas_density_stp
            elif from_key in ("kwh/nm3",):
                mj_nm3 = value * 3.6
                mj_per_kg = mj_nm3 / gas_density_stp
            else:
                raise ValueError(f"Conversion from {from_unit} not implemented")
        else:
            mj_per_kg = value * float(HEATING_VALUE_FACTORS[from_unit])  # type: ignore[arg-type]

        # Convert from MJ/kg to target
        if HEATING_VALUE_FACTORS[to_unit] is None:
            if gas_density_stp is None:
                raise ValueError(f"Gas density required for {to_unit} conversion")
            if to_key in ("mj/nm3",):
                return mj_per_kg * gas_density_stp
            elif to_key == "btu/scf":
                mj_nm3 = mj_per_kg * gas_density_stp
                return mj_nm3 / 0.0372589
            elif to_key in ("kwh/nm3",):
                mj_nm3 = mj_per_kg * gas_density_stp
                return mj_nm3 / 3.6
            else:
                raise ValueError(f"Conversion to {to_unit} not implemented")
        else:
            return mj_per_kg / float(HEATING_VALUE_FACTORS[to_unit])  # type: ignore[arg-type]

    def format_number(self, num: float) -> str:
        """Format a number for display, matching the PyQt6 style."""
        if not (num is not None):
            raise ValueError("num must be provided")
        if math.isnan(num) or math.isinf(num):
            return str(num)

        if abs(num) >= 1e10 or (abs(num) < 1e-6 and num != 0):
            return f"{num:.6e}"

        # Use up to 10 significant digits
        formatted = f"{num:.10g}"
        return formatted
