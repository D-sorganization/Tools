"""Baghouse Calculator
===================

Core calculation engine for baghouse filter performance, solid removal, and drum sizing.

This is a standalone calculator that can work with or without the full thermodynamic
engine. When the thermo module is not available, it uses simplified ideal gas calculations.
"""

from dataclasses import dataclass
from typing import Any, Optional

from .constants import (
    R_UNIVERSAL,
    CELSIUS_TO_KELVIN_OFFSET,
    ATM_PA,
    STP_TEMPERATURE_K,
)

# Try to import thermo module, use simplified calculations if not available
try:
    from tools.thermo import (
        FlowUnit,
        GasStream,
        ThermodynamicCalculator,
    )
    from tools.unit_converter import convert
    HAS_THERMO = True
except ImportError:
    HAS_THERMO = False
    FlowUnit = None
    GasStream = None
    ThermodynamicCalculator = None

    def convert(value: float, from_unit: str, to_unit: str) -> float:
        """Simple temperature conversion fallback."""
        if from_unit == "K" and to_unit == "C":
            return value - CELSIUS_TO_KELVIN_OFFSET
        elif from_unit == "C" and to_unit == "K":
            return value + CELSIUS_TO_KELVIN_OFFSET
        return value


@dataclass
class BaghouseResult:
    """Result data for baghouse calculation."""

    carbon_removed_rate: float  # kg/hr
    ash_removed_rate: float  # kg/hr
    total_solids_removed_rate: float  # kg/hr

    drum_fill_time_hours: float
    drum_fill_time_days: float
    carbon_only_fill_time_hours: float
    ash_only_fill_time_hours: float

    clean_gas_flow_rate: float  # kg/hr
    flow_acfm: float
    flow_scfm: float
    air_to_cloth_ratio: float  # ft/min

    outlet_temperature_c: float
    ash_stream_composition: dict[str, float]

    removal_efficiency: dict[str, float]


class BaghouseCalculator:
    """Core baghouse calculation engine.

    This calculator determines baghouse filter performance including solid removal
    rates, drum fill times, and air-to-cloth ratios. It can operate in two modes:

    1. Full mode (with thermo module): Uses detailed thermodynamic calculations
    2. Simplified mode (standalone): Uses ideal gas approximations
    """

    def __init__(self, thermo_calc: Optional[Any] = None) -> None:
        """Initialize the baghouse calculator.

        Args:
            thermo_calc: Optional thermodynamic calculator instance. If not provided
                        and thermo module is available, one will be created.
        """
        if thermo_calc is not None:
            self.thermo_calc = thermo_calc
        elif HAS_THERMO and ThermodynamicCalculator is not None:
            self.thermo_calc = ThermodynamicCalculator()
        else:
            self.thermo_calc = None

    def _estimate_cp_ideal(self, composition: dict[str, float]) -> float:
        """Estimate Cp for gas mixture using ideal gas approximation.

        Uses constant Cp values for common syngas species.

        Args:
            composition: Mole fraction composition

        Returns:
            Estimated Cp in J/(kg·K)
        """
        # Approximate Cp values at ~500K in J/(mol·K)
        cp_data = {
            "H2": 29.1, "CO": 29.2, "CO2": 41.3, "H2O": 35.5,
            "N2": 29.5, "CH4": 44.5, "O2": 30.1, "Ar": 20.8,
        }
        # Molecular weights in kg/mol
        mw_data = {
            "H2": 0.002, "CO": 0.028, "CO2": 0.044, "H2O": 0.018,
            "N2": 0.028, "CH4": 0.016, "O2": 0.032, "Ar": 0.040,
        }

        cp_mol = 0.0
        mw_avg = 0.0
        for species, frac in composition.items():
            cp_mol += frac * cp_data.get(species, 30.0)
            mw_avg += frac * mw_data.get(species, 0.028)

        if mw_avg > 0:
            return cp_mol / mw_avg  # J/(kg·K)
        return 1000.0  # Default fallback

    def _estimate_volume_flow(
        self,
        mass_flow_kg_s: float,
        temperature_k: float,
        pressure_pa: float,
        composition: dict[str, float],
    ) -> tuple[float, float]:
        """Estimate volumetric flow rates using ideal gas law.

        Returns:
            (acfm, scfm) - Actual and standard cubic feet per minute
        """
        # Molecular weights in kg/mol
        mw_data = {
            "H2": 0.002, "CO": 0.028, "CO2": 0.044, "H2O": 0.018,
            "N2": 0.028, "CH4": 0.016, "O2": 0.032, "Ar": 0.040,
        }

        mw_avg = sum(frac * mw_data.get(species, 0.028)
                     for species, frac in composition.items())

        if mw_avg > 0:
            molar_flow = mass_flow_kg_s / mw_avg  # mol/s
        else:
            molar_flow = mass_flow_kg_s / 0.028  # Assume N2-like

        # Actual volume flow (ideal gas)
        vol_actual_m3_s = molar_flow * R_UNIVERSAL * temperature_k / pressure_pa

        # Standard volume flow (at STP: 273.15 K, 101325 Pa)
        vol_std_m3_s = molar_flow * R_UNIVERSAL * STP_TEMPERATURE_K / ATM_PA

        # Convert to cfm (1 m³/s = 2118.88 cfm)
        acfm = vol_actual_m3_s * 2118.88
        scfm = vol_std_m3_s * 2118.88

        return acfm, scfm

    def calculate(
        self,
        gas_flow_kg_s: float,
        inlet_temp_k: float,
        pressure_pa: float,
        composition: dict[str, float],
        solid_carbon_in_kg_hr: float,
        ash_in_kg_hr: float,
        carbon_removal_efficiency: float,  # 0-1
        ash_removal_efficiency: float,  # 0-1
        heat_loss_w: float,
        drum_volume_m3: float,
        solid_density_kg_m3: float,
        bag_area_ft2: float,
    ) -> BaghouseResult:
        """Calculate baghouse performance.

        Args:
            gas_flow_kg_s: Gas mass flow rate [kg/s]
            inlet_temp_k: Inlet temperature [K]
            pressure_pa: Pressure [Pa]
            composition: Gas composition dictionary
            solid_carbon_in_kg_hr: Solid carbon input rate [kg/hr]
            ash_in_kg_hr: Ash input rate [kg/hr]
            carbon_removal_efficiency: Carbon removal efficiency (0-1)
            ash_removal_efficiency: Ash removal efficiency (0-1)
            heat_loss_w: Heat loss rate [W]
            drum_volume_m3: Collection drum volume [m³]
            solid_density_kg_m3: Density of collected solids [kg/m³]
            bag_area_ft2: Total bag filter area [ft²]

        Returns:
            BaghouseResult object
        """
        # Calculate temperature drop from heat loss
        if self.thermo_calc is not None and HAS_THERMO:
            # Use full thermodynamic calculation
            try:
                stream = GasStream(
                    flow_rate=gas_flow_kg_s,
                    flow_unit=FlowUnit.MASS,
                    temperature=inlet_temp_k,
                    pressure=pressure_pa,
                    composition=composition,
                )
                props = self.thermo_calc.calculate_stream_properties(stream)
                cp_mass = props.cp  # J/kg-K

                if gas_flow_kg_s > 0 and cp_mass > 0:
                    temp_drop_k = heat_loss_w / (gas_flow_kg_s * cp_mass)
                else:
                    temp_drop_k = 0.0

                outlet_temp_k = max(inlet_temp_k - temp_drop_k, 0.0)
                outlet_temp_c = convert(outlet_temp_k, "K", "C")

                # Re-calculate at outlet for volume flow
                outlet_stream = GasStream(
                    flow_rate=gas_flow_kg_s,
                    flow_unit=FlowUnit.MASS,
                    temperature=outlet_temp_k,
                    pressure=pressure_pa,
                    composition=composition,
                )
                outlet_props = self.thermo_calc.calculate_stream_properties(outlet_stream)
                flow_acfm = outlet_props.acfm_flow
                flow_scfm = outlet_props.scfm_flow

            except Exception:
                # Fall through to simplified calculation
                self.thermo_calc = None

        if self.thermo_calc is None:
            # Simplified calculation without thermo module
            cp_mass = self._estimate_cp_ideal(composition)

            if gas_flow_kg_s > 0 and cp_mass > 0:
                temp_drop_k = heat_loss_w / (gas_flow_kg_s * cp_mass)
            else:
                temp_drop_k = 0.0

            outlet_temp_k = max(inlet_temp_k - temp_drop_k, 0.0)
            outlet_temp_c = outlet_temp_k - CELSIUS_TO_KELVIN_OFFSET

            flow_acfm, flow_scfm = self._estimate_volume_flow(
                gas_flow_kg_s, outlet_temp_k, pressure_pa, composition
            )

        # Solids Removal
        carbon_removed = solid_carbon_in_kg_hr * carbon_removal_efficiency
        ash_removed = ash_in_kg_hr * ash_removal_efficiency
        total_solids_removed = carbon_removed + ash_removed

        # Drum Sizing
        drum_mass_capacity = solid_density_kg_m3 * drum_volume_m3

        if total_solids_removed > 0:
            fill_time_hours = drum_mass_capacity / total_solids_removed
        else:
            fill_time_hours = float("inf")

        fill_time_days = (
            fill_time_hours / 24.0 if fill_time_hours != float("inf") else float("inf")
        )

        c_fill = (
            drum_mass_capacity / carbon_removed if carbon_removed > 0 else float("inf")
        )
        a_fill = drum_mass_capacity / ash_removed if ash_removed > 0 else float("inf")

        # Air to Cloth Ratio
        air_to_cloth = flow_acfm / bag_area_ft2 if bag_area_ft2 > 0 else 0.0

        # Mass flow in kg/hr
        gas_flow_kg_hr = gas_flow_kg_s * 3600.0

        ash_stream_comp = {
            "carbon_fraction": (
                carbon_removed / total_solids_removed
                if total_solids_removed > 0
                else 0.0
            ),
            "ash_fraction": (
                ash_removed / total_solids_removed if total_solids_removed > 0 else 0.0
            ),
        }

        return BaghouseResult(
            carbon_removed_rate=carbon_removed,
            ash_removed_rate=ash_removed,
            total_solids_removed_rate=total_solids_removed,
            drum_fill_time_hours=fill_time_hours,
            drum_fill_time_days=fill_time_days,
            carbon_only_fill_time_hours=c_fill,
            ash_only_fill_time_hours=a_fill,
            clean_gas_flow_rate=gas_flow_kg_hr,
            flow_acfm=flow_acfm,
            flow_scfm=flow_scfm,
            air_to_cloth_ratio=air_to_cloth,
            outlet_temperature_c=outlet_temp_c,
            ash_stream_composition=ash_stream_comp,
            removal_efficiency={
                "carbon": carbon_removal_efficiency * 100.0,
                "ash": ash_removal_efficiency * 100.0,
            },
        )
