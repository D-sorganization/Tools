"""
results/extractor.py
====================
Pulls simulation results from a solved DWSIM flowsheet and returns them
as plain Python dicts and numbers — no DWSIM objects needed downstream.

Why this matters:
    Without an extractor, running the model gives you no programmatic access
    to the results.  This module makes results available to the CLI, GUI,
    parametric sweep engine, and report generator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# DWSIM property string constants — avoids magic strings scattered through code
_PROP_TEMPERATURE = "Temperature"  # Kelvin
_PROP_PRESSURE = "Pressure"  # Pa
_PROP_MASSFLOW = "MassFlow"  # kg/s
_PROP_MOLFRAC = "MoleFraction."  # prefix: "MoleFraction.Hydrogen"
_PROP_MASSFRAC = "MassFraction."  # prefix: "MassFraction.CO"
_PROP_ENTHALPY = "SpecificEnthalpy"  # J/kg
_PROP_ENERGY_FLOW = "EnergyFlow"  # W


@dataclass
class StreamResult:
    """Results extracted from a single DWSIM material stream."""

    name: str
    temperature_C: float = 0.0
    pressure_kPa: float = 0.0
    mass_flow_kg_s: float = 0.0
    mole_fractions: dict[str, float] = field(default_factory=dict)
    mass_fractions: dict[str, float] = field(default_factory=dict)
    specific_enthalpy_kJ_kg: float = 0.0
    # Derived
    volumetric_flow_Nm3_h: float = 0.0  # Calculated from ideal gas law at NTP


@dataclass
class EnergyStreamResult:
    """Results extracted from a DWSIM energy stream."""

    name: str
    energy_flow_kW: float = 0.0


@dataclass
class FlowsheetResults:
    """All extracted results from a solved gasification flowsheet."""

    # Stream results keyed by stream name
    streams: dict[str, StreamResult] = field(default_factory=dict)
    energy_streams: dict[str, EnergyStreamResult] = field(default_factory=dict)

    # Convergence info
    converged: bool = False
    errors: list[str] = field(default_factory=list)

    # Performance metrics (populated by MetricsCalculator)
    metrics: dict[str, Any] = field(default_factory=dict)

    def get_stream(self, name: str) -> StreamResult | None:
        return self.streams.get(name)

    def to_dict(self) -> dict:
        """Convert to a plain dict for JSON serialisation."""
        return {
            "converged": self.converged,
            "errors": self.errors,
            "streams": {
                k: {
                    "temperature_C": v.temperature_C,
                    "pressure_kPa": v.pressure_kPa,
                    "mass_flow_kg_s": v.mass_flow_kg_s,
                    "mole_fractions": v.mole_fractions,
                    "mass_fractions": v.mass_fractions,
                    "specific_enthalpy_kJ_kg": v.specific_enthalpy_kJ_kg,
                    "volumetric_flow_Nm3_h": v.volumetric_flow_Nm3_h,
                }
                for k, v in self.streams.items()
            },
            "energy_streams": {
                k: {"energy_flow_kW": v.energy_flow_kW}
                for k, v in self.energy_streams.items()
            },
            "metrics": self.metrics,
        }


class ResultsExtractor:
    """
    Pulls values from a solved DWSIM flowsheet into FlowsheetResults.

    Usage
    -----
    After running model.run(), call:

        extractor = ResultsExtractor(compound_names)
        results = extractor.extract(builder)
        print(results.metrics["cold_gas_efficiency"])

    Parameters
    ----------
    compound_names:
        List of compound names in the simulation.  Needed to iterate
        over mole fractions.
    key_streams:
        If provided, only extract these stream names.  If None,
        extract all streams in builder.materials.
    """

    KELVIN_OFFSET = 273.15

    def __init__(
        self,
        compound_names: list[str] | None = None,
        key_streams: list[str] | None = None,
    ):
        self.compound_names = compound_names or []
        self.key_streams = key_streams

    # ─────────────────────────────────────────────────────────────────────────

    def extract(self, builder, converged: bool | None = None) -> FlowsheetResults:
        """
        Extract all results from a solved flowsheet.

        Parameters
        ----------
        builder:
            A FlowsheetBuilder instance that has been built and run.
        converged:
            Optional explicit convergence flag from the DWSIM solver.
            If provided, this takes precedence over the extraction-error heuristic.
            If None, falls back to: len(errors) == 0 (may be inaccurate — see Bug #5).

        Returns
        -------
        FlowsheetResults
            All extracted stream data and derived metrics.
        """
        results = FlowsheetResults()

        streams_to_extract = (
            {
                n: builder.materials[n]
                for n in self.key_streams
                if n in builder.materials
            }
            if self.key_streams
            else builder.materials
        )

        for name, stream_obj in streams_to_extract.items():
            try:
                results.streams[name] = self._extract_material_stream(name, stream_obj)
            except Exception as exc:
                msg = f"Could not extract stream '{name}': {exc}"
                logger.warning(msg)
                results.errors.append(msg)

        for name, e_obj in builder.energy_streams.items():
            try:
                results.energy_streams[name] = self._extract_energy_stream(name, e_obj)
            except Exception as exc:
                logger.debug(f"Could not extract energy stream '{name}': {exc}")

        # fix: extraction errors != solver non-convergence (Bug #5).
        # Use the explicit converged flag if provided; fall back to heuristic.
        if converged is not None:
            results.converged = converged
        else:
            results.converged = len(results.errors) == 0

        logger.info(
            f"Extraction complete: {len(results.streams)} streams, "
            f"{len(results.energy_streams)} energy streams, "
            f"converged={results.converged}"
        )
        return results

    # ─────────────────────────────────────────────────────────────────────────

    def _extract_material_stream(self, name: str, stream_obj) -> StreamResult:
        """Extract all properties from a single material stream."""
        result = StreamResult(name=name)

        # Temperature (DWSIM returns K; we store °C)
        t_k = self._get_prop(stream_obj, _PROP_TEMPERATURE, default=0.0)
        result.temperature_C = t_k - self.KELVIN_OFFSET if t_k else 0.0

        # Pressure (Pa → kPa)
        p_pa = self._get_prop(stream_obj, _PROP_PRESSURE, default=0.0)
        result.pressure_kPa = p_pa / 1000.0 if p_pa else 0.0

        # Mass flow
        result.mass_flow_kg_s = self._get_prop(stream_obj, _PROP_MASSFLOW, default=0.0)

        # Specific enthalpy (J/kg → kJ/kg)
        h = self._get_prop(stream_obj, _PROP_ENTHALPY, default=0.0)
        result.specific_enthalpy_kJ_kg = (h / 1000.0) if h else 0.0

        # Mole and mass fractions
        for compound in self.compound_names:
            mf = self._get_prop(stream_obj, f"{_PROP_MOLFRAC}{compound}", default=0.0)
            if mf and mf > 1e-9:
                result.mole_fractions[compound] = mf

            wf = self._get_prop(stream_obj, f"{_PROP_MASSFRAC}{compound}", default=0.0)
            if wf and wf > 1e-9:
                result.mass_fractions[compound] = wf

        # Volumetric flow at NTP (0°C, 101.325 kPa) using ideal gas
        # V_dot = m_dot * R * T / (MW_mix * P)
        result.volumetric_flow_Nm3_h = self._calc_volumetric_flow(
            mass_flow_kg_s=result.mass_flow_kg_s,
            mole_fractions=result.mole_fractions,
            pressure_Pa=p_pa or 101325.0,
        )

        return result

    def _extract_energy_stream(self, name: str, e_obj) -> EnergyStreamResult:
        """Extract energy flow from an energy stream."""
        w = self._get_prop(e_obj, _PROP_ENERGY_FLOW, default=0.0)
        return EnergyStreamResult(name=name, energy_flow_kW=(w / 1000.0) if w else 0.0)

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_prop(obj, prop_name: str, default=None):
        """
        Safely read a DWSIM property value.

        DWSIM properties can be read as attributes or via GetPropertyValue().
        We try both to be resilient across API versions.
        """
        try:
            val = obj.GetPropertyValue(prop_name)
            if val is not None:
                return float(val)
        except Exception as exc:
            logger.debug(
                f"Exception getting property via GetPropertyValue: {exc}"
            )  # AUTO-FIXED
        try:
            val = getattr(obj, prop_name)
            if val is not None:
                return float(val)
        except Exception as exc:
            logger.debug(f"Exception getting property via getattr: {exc}")  # AUTO-FIXED
        return default

    @staticmethod
    def _calc_volumetric_flow(
        mass_flow_kg_s: float,
        mole_fractions: dict[str, float],
        pressure_Pa: float,
    ) -> float:
        """
        Estimate volumetric flow at Normal Temperature and Pressure
        (NTP: 0°C = 273.15 K, 101325 Pa) in Nm³/h using ideal gas law.

        V_dot [m³/s] = (m_dot / MW_mix) * R * T_NTP / P_NTP
        """
        from dwsim_model.chemistry.biomass_decomposer import _MW as MW_TABLE

        if mass_flow_kg_s <= 0 or not mole_fractions:
            return 0.0

        R = 8.314  # J/mol/K
        T_NTP = 273.15  # K
        P_NTP = 101325.0  # Pa

        # MW mix (g/mol) from mole fractions
        mw_mix = 0.0
        for compound, xf in mole_fractions.items():
            # Look for compound MW in our table (simplified name matching)
            for key, mw in MW_TABLE.items():
                if key.lower() in compound.lower() or compound.lower() in key.lower():
                    mw_mix += xf * mw
                    break

        if mw_mix <= 0:
            return 0.0

        mw_mix_kg = mw_mix / 1000.0  # kg/mol
        vol_flow_m3_s = (mass_flow_kg_s / mw_mix_kg) * R * T_NTP / P_NTP
        return vol_flow_m3_s * 3600.0  # m³/s → Nm³/h
