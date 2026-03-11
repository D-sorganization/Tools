"""
results/metrics.py
==================
Calculates key performance indicators (KPIs) for the gasification process
from extracted flowsheet results.

All KPIs are computed from the FlowsheetResults object returned by
ResultsExtractor — no DWSIM objects needed here.  That separation means
you can test metrics calculations without DWSIM installed.

Key KPIs
--------
Cold Gas Efficiency (CGE):
    Chemical energy (LHV) in product syngas divided by chemical energy
    (LHV) in the biomass feed.  Target: > 65%.

Carbon Conversion Efficiency (CCE):
    Fraction of feed carbon that ends up in gaseous products (CO, CO2, CH4,
    hydrocarbons).  Target: > 90%.

H2/CO Ratio:
    Molar ratio of H2 to CO in the final syngas.  Determines suitability
    for downstream use:
        ≈ 1.0  → direct combustion / power generation
        ≈ 1.8  → Fischer-Tropsch synthesis
        ≈ 2.0  → methanol synthesis
        ≈ 3.0  → SNG / methanation

Specific Energy Consumption (SEC):
    Electrical energy input (PEM plasma) per kg of waste processed.
    Units: kWh/t (kilowatt-hours per tonne of biomass feed).

Tar Loading:
    Mass of tar species in the product gas per Nm³ at standard conditions.
    Target for IC engine use: < 100 mg/Nm³.
    Target for fuel cell use: < 1 mg/Nm³.

Mass Balance Closure:
    (sum of outlet mass flows) / (sum of inlet mass flows).
    Should be 1.000 ± 0.001 for a well-converged simulation.

Energy Balance Closure:
    (sum of outlet enthalpies + heat losses) / (sum of inlet enthalpies + heat inputs).
    Should be 1.000 ± 0.01.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Lower Heating Values (MJ/kg) for key fuel species
# Sources: Perry's Chemical Engineers Handbook, 8th Ed.
LHV_MJ_KG: dict[str, float] = {
    "Hydrogen": 119.96,
    "Carbon monoxide": 10.10,
    "Methane": 50.05,
    "Ethylene": 47.19,
    "Ethane": 47.48,
    "Acetylene": 48.22,
    "Naphthalene": 39.85,
    "Toluene": 40.94,
}

# Molecular weights (g/mol) — same subset
MW_G_MOL: dict[str, float] = {
    "Hydrogen": 2.016,
    "Carbon monoxide": 28.010,
    "Methane": 16.043,
    "Carbon dioxide": 44.010,
    "Water": 18.015,
    "Nitrogen": 28.014,
    "Oxygen": 32.000,
    "Helium": 4.003,
    "Ethylene": 28.053,
    "Ethane": 30.069,
    "Acetylene": 26.038,
    "Naphthalene": 128.174,
    "Toluene": 92.140,
    "Hydrogen sulfide": 34.081,
    "Ammonia": 17.031,
}

# Carbon mass fraction by species for surrogate and product streams.
CARBON_MASS_FRACTION: dict[str, float] = {
    "Carbon monoxide": 12.011 / 28.010,
    "Carbon dioxide": 12.011 / 44.010,
    "Methane": 12.011 / 16.043,
    "Ethylene": 24.022 / 28.053,
    "Ethane": 24.022 / 30.069,
    "Acetylene": 24.022 / 26.038,
    "Naphthalene": 120.110 / 128.174,
    "Toluene": 84.077 / 92.140,
}

# Typical biomass LHV (MJ/kg, as-received) — used as fallback if not in config
DEFAULT_BIOMASS_LHV_MJ_KG = 15.0

# Tar surrogate species
TAR_SPECIES = {"Naphthalene", "Toluene", "Phenol", "Benzene"}

# Inlet and outlet stream names (matches STREAM_NAMES in constants.py)
INLET_STREAMS = [
    "Gasifier_Biomass_Feed",
    "Gasifier_Oxygen_Feed",
    "Gasifier_Steam_Feed",
    "Gasifier_Solids_Feed",
    "Gasifier_Cooling_Water_In",
    "PEM_Solids_Feed",
    "PEM_Oxygen_Feed",
    "PEM_Steam_Feed",
    "TRC_Solids_Feed",
    "TRC_Oxygen_Feed",
    "TRC_Steam_Feed",
    "Quench_Water_Injection",
    "Quench_Nitrogen",
    "Quench_Steam",
]

OUTLET_STREAMS = [
    "Final_Syngas",
    "Gasifier_Glass_Out",
    "PEM_Glass_Out",
    "Baghouse_Solids_Out",
    "Scrubber_Blowdown",
    "Gasifier_Cooling_Steam_Out",
]

ENERGY_INLET_STREAMS = [
    "E_PEM_AC_Power",
    "E_PEM_DC_Power",
    "E_Blower",
]

ENERGY_OUTLET_STREAMS = [
    "E_Gasifier_HeatLoss",
    "E_PEM_HeatLoss",
    "E_TRC_HeatLoss",
    "E_Gasifier_Flux_to_CW",
]


@dataclass
class GasificationMetrics:
    """All computed KPIs for one simulation run."""

    cold_gas_efficiency: float = 0.0  # dimensionless (0–1)
    carbon_conversion_efficiency: float = 0.0  # dimensionless (0–1)
    h2_co_ratio: float = 0.0
    specific_energy_consumption_kWh_t: float = 0.0
    tar_loading_mg_Nm3: float = 0.0

    syngas_lhv_mj_kg: float = 0.0
    syngas_lhv_mj_nm3: float = 0.0  # LHV on a volumetric (Nm³) basis
    syngas_mass_flow_kg_s: float = 0.0
    syngas_volumetric_flow_Nm3_h: float = 0.0
    syngas_temperature_C: float = 0.0

    feed_mass_flow_kg_s: float = 0.0
    biomass_lhv_mj_kg: float = DEFAULT_BIOMASS_LHV_MJ_KG

    mass_balance_closure: float = 0.0  # should be ~1.0
    energy_balance_closure: float = 0.0  # should be ~1.0

    warnings: list[str] = field(default_factory=list)

    @staticmethod
    def _r(value, ndigits: int):
        """Round value or return None if value is None."""
        return round(value, ndigits) if value is not None else None

    def to_dict(self) -> dict[str, Any]:
        r = self._r
        return {
            "cold_gas_efficiency": r(self.cold_gas_efficiency, 4),
            "carbon_conversion_efficiency": r(self.carbon_conversion_efficiency, 4),
            "h2_co_ratio": r(self.h2_co_ratio, 3),
            "specific_energy_consumption_kWh_t": r(
                self.specific_energy_consumption_kWh_t, 1
            ),
            "tar_loading_mg_Nm3": r(self.tar_loading_mg_Nm3, 2),
            "syngas_lhv_mj_nm3": r(self.syngas_lhv_mj_nm3, 3),
            "syngas_lhv_mj_kg": r(self.syngas_lhv_mj_kg, 3),
            "syngas_mass_flow_kg_s": r(self.syngas_mass_flow_kg_s, 4),
            "syngas_volumetric_flow_Nm3_h": r(self.syngas_volumetric_flow_Nm3_h, 1),
            "syngas_temperature_C": r(self.syngas_temperature_C, 1),
            "feed_mass_flow_kg_s": r(self.feed_mass_flow_kg_s, 4),
            "biomass_lhv_mj_kg": r(self.biomass_lhv_mj_kg, 2),
            "mass_balance_closure": r(self.mass_balance_closure, 5),
            "energy_balance_closure": r(self.energy_balance_closure, 4),
            "warnings": self.warnings,
        }

    def check_targets(self, targets: dict) -> list[str]:
        """
        Compare metrics against scenario targets.

        Returns a list of failure messages (empty list = all targets met).
        This makes it easy to check: ``if not metrics.check_targets(targets): print("OK")``.
        """
        failures = []

        def _safe_ge(val, threshold, label):
            """Greater-than-or-equal check that handles None gracefully."""
            if val is None:
                failures.append(f"{label}: metric not computed (None).")
                return
            if val < threshold:
                failures.append(f"{label}: {val:.3g} < target {threshold:.3g}.")

        def _safe_le(val, threshold, label):
            if val is None:
                failures.append(f"{label}: metric not computed (None).")
                return
            if val > threshold:
                failures.append(f"{label}: {val:.3g} > target {threshold:.3g}.")

        if "cold_gas_efficiency_min" in targets:
            _safe_ge(
                self.cold_gas_efficiency,
                targets["cold_gas_efficiency_min"],
                "Cold Gas Efficiency",
            )
        if "carbon_conversion_min" in targets:
            _safe_ge(
                self.carbon_conversion_efficiency,
                targets["carbon_conversion_min"],
                "Carbon Conversion",
            )
        if "h2_co_ratio_target" in targets:
            target = targets["h2_co_ratio_target"]
            val = self.h2_co_ratio
            if val is None:
                failures.append("H2/CO ratio: not computed (None).")
            elif abs(val - target) / max(target, 1e-6) > 0.15:
                failures.append(
                    f"H2/CO ratio: {val:.2f} deviates >15% from target {target:.2f}."
                )
        if "tar_loading_mg_Nm3_max" in targets:
            _safe_le(
                self.tar_loading_mg_Nm3,
                targets["tar_loading_mg_Nm3_max"],
                "Tar Loading",
            )

        return failures


class MetricsCalculator:
    """
    Computes all gasification KPIs from a FlowsheetResults object.

    Parameters
    ----------
    biomass_lhv_mj_kg:
        LHV of the biomass feed in MJ/kg (as-received).  Used to compute
        cold gas efficiency.  Default: 15.0 MJ/kg (typical MSW).
    """

    def __init__(
        self,
        biomass_lhv_mj_kg: float = DEFAULT_BIOMASS_LHV_MJ_KG,
        biomass_carbon_mass_fraction: float | None = None,
    ):
        if biomass_carbon_mass_fraction is not None and not (
            0.0 <= biomass_carbon_mass_fraction <= 1.0
        ):
            raise ValueError(
                "biomass_carbon_mass_fraction must be between 0.0 and 1.0."
            )
        self.biomass_lhv_mj_kg = biomass_lhv_mj_kg
        self.biomass_carbon_mass_fraction = biomass_carbon_mass_fraction

    # ─────────────────────────────────────────────────────────────────────────

    def calculate(self, results) -> GasificationMetrics:
        """
        Compute all KPIs.

        Parameters
        ----------
        results:
            FlowsheetResults from ResultsExtractor.extract()

        Returns
        -------
        GasificationMetrics
        """
        m = GasificationMetrics(biomass_lhv_mj_kg=self.biomass_lhv_mj_kg)

        syngas = results.get_stream("Final_Syngas")
        biomass = results.get_stream("Gasifier_Biomass_Feed")

        if syngas is None:
            m.warnings.append(
                "Final_Syngas stream not found — most metrics will be zero."
            )
            logger.warning(
                "Final_Syngas not in results — skipping metric calculations."
            )
            return m

        if biomass is None:
            m.warnings.append(
                "Gasifier_Biomass_Feed not found — CGE and CCE unavailable."
            )

        # ── Basic syngas properties ──────────────────────────────────────────
        m.syngas_mass_flow_kg_s = syngas.mass_flow_kg_s
        m.syngas_temperature_C = syngas.temperature_C
        m.syngas_volumetric_flow_Nm3_h = syngas.volumetric_flow_Nm3_h

        if biomass:
            m.feed_mass_flow_kg_s = biomass.mass_flow_kg_s

        # ── Syngas LHV ───────────────────────────────────────────────────────
        m.syngas_lhv_mj_kg = self._calc_syngas_lhv(syngas.mass_fractions)
        m.syngas_lhv_mj_nm3 = self._calc_syngas_lhv_volumetric(
            syngas, m.syngas_lhv_mj_kg
        )

        # ── Cold Gas Efficiency ──────────────────────────────────────────────
        if biomass and biomass.mass_flow_kg_s > 0 and self.biomass_lhv_mj_kg > 0:
            syngas_energy = syngas.mass_flow_kg_s * m.syngas_lhv_mj_kg  # MW
            biomass_energy = biomass.mass_flow_kg_s * self.biomass_lhv_mj_kg  # MW
            m.cold_gas_efficiency = syngas_energy / biomass_energy
            if m.cold_gas_efficiency > 1.5:
                m.warnings.append(
                    f"CGE = {m.cold_gas_efficiency:.2%} > 150% — possible unit error "
                    "in biomass LHV or mass flows."
                )

        # ── Carbon Conversion Efficiency ─────────────────────────────────────
        carbon_conversion = self._calc_carbon_conversion(results, biomass)
        if carbon_conversion is None:
            m.warnings.append(
                "Carbon conversion carbon basis unavailable for Gasifier_Biomass_Feed."
            )
            m.carbon_conversion_efficiency = 0.0
        else:
            m.carbon_conversion_efficiency = carbon_conversion

        # ── H2/CO Ratio ──────────────────────────────────────────────────────
        m.h2_co_ratio = self._calc_ratio(
            syngas.mole_fractions, "Hydrogen", "Carbon monoxide"
        )

        # ── Specific Energy Consumption ──────────────────────────────────────
        m.specific_energy_consumption_kWh_t = self._calc_sec(results, biomass)

        # ── Tar Loading ──────────────────────────────────────────────────────
        m.tar_loading_mg_Nm3 = self._calc_tar_loading(syngas)

        # ── Mass Balance Closure ─────────────────────────────────────────────
        m.mass_balance_closure = self._calc_mass_balance(results)

        # ── Energy Balance Closure ───────────────────────────────────────────
        m.energy_balance_closure = self._calc_energy_balance(results)

        # ── Sanity warnings ──────────────────────────────────────────────────
        if abs(m.mass_balance_closure - 1.0) > 0.01:
            m.warnings.append(
                f"Mass balance not closed: {m.mass_balance_closure:.4f} "
                f"(deviation = {abs(m.mass_balance_closure - 1) * 100:.2f}%). "
                "Check that all inlet/outlet streams were solved."
            )
        if m.h2_co_ratio == 0 and syngas.mass_flow_kg_s > 0:
            m.warnings.append("H2/CO ratio is 0 — syngas may contain no CO or H2.")

        logger.info(
            f"Metrics: CGE={m.cold_gas_efficiency:.1%}, "
            f"CCE={m.carbon_conversion_efficiency:.1%}, "
            f"H2/CO={m.h2_co_ratio:.2f}, "
            f"MB={m.mass_balance_closure:.4f}"
        )

        return m

    # ─────────────────────────────────────────────────────────────────────────
    # Private calculation methods
    # ─────────────────────────────────────────────────────────────────────────

    def _calc_syngas_lhv(self, mass_fractions: dict[str, float]) -> float:
        """
        Compute syngas LHV (MJ/kg) from mass fractions using component LHVs.

        LHV_mix = Σ (w_i * LHV_i)
        """
        lhv = 0.0
        for compound, wf in mass_fractions.items():
            if compound in LHV_MJ_KG:
                lhv += wf * LHV_MJ_KG[compound]
        return lhv

    @staticmethod
    def _calc_syngas_lhv_volumetric(syngas_stream, syngas_lhv_mj_kg: float) -> float:
        """Convert syngas LHV from a mass basis (MJ/kg) to a volumetric basis (MJ/Nm3)."""
        if (
            syngas_stream.mass_flow_kg_s <= 0
            or syngas_stream.volumetric_flow_Nm3_h <= 0
        ):
            return 0.0
        energy_flow_mj_h = syngas_stream.mass_flow_kg_s * syngas_lhv_mj_kg * 3600.0
        return energy_flow_mj_h / syngas_stream.volumetric_flow_Nm3_h

    def _calc_carbon_conversion(self, results, biomass_stream) -> float | None:
        """
        Carbon Conversion Efficiency = C in gas products / C in feed.

        Carbon in gas outlets: sum over CO, CO2, CH4, C2 species.
        Carbon in feed: from Gasifier_Biomass_Feed mass flow × carbon fraction.
        """
        if biomass_stream is None or biomass_stream.mass_flow_kg_s <= 0:
            return 0.0

        carbon_feed_kg_s = self._calc_feed_carbon_mass_flow(biomass_stream)
        if carbon_feed_kg_s is None or carbon_feed_kg_s <= 0:
            logger.debug(
                "Carbon basis unavailable in biomass feed — CCE not calculable."
            )
            return None

        # Sum carbon in all outlet gas streams
        carbon_gas_kg_s = 0.0

        for stream_name in ["Final_Syngas", "Syngas_Pre_PEM", "Syngas_Pre_TRC"]:
            s = results.get_stream(stream_name)
            if s is None or s.mass_flow_kg_s <= 0:
                continue
            carbon_gas_kg_s += self._calc_stream_carbon_mass_flow(s)
            break  # Use only the first (most downstream) stream found

        if carbon_feed_kg_s > 0:
            return min(carbon_gas_kg_s / carbon_feed_kg_s, 1.0)
        return 0.0

    def _calc_feed_carbon_mass_flow(self, biomass_stream) -> float | None:
        """Compute biomass-feed carbon mass flow from an explicit basis or a surrogate stream."""
        if self.biomass_carbon_mass_fraction is not None:
            return biomass_stream.mass_flow_kg_s * self.biomass_carbon_mass_fraction

        inferred_carbon_kg_s = self._calc_stream_carbon_mass_flow(biomass_stream)
        if inferred_carbon_kg_s > 0:
            return inferred_carbon_kg_s
        return None

    @staticmethod
    def _calc_stream_carbon_mass_flow(stream) -> float:
        """Compute the carbon mass flow in a stream from its carbon-bearing species."""
        carbon_mass_flow = 0.0
        for compound, carbon_fraction in CARBON_MASS_FRACTION.items():
            wf = stream.mass_fractions.get(compound, 0.0)
            carbon_mass_flow += stream.mass_flow_kg_s * wf * carbon_fraction
        return carbon_mass_flow

    @staticmethod
    def _calc_ratio(mole_fractions: dict, numerator: str, denominator: str) -> float:
        """Compute molar ratio of two species. Returns 0 if denominator is near zero."""
        num = mole_fractions.get(numerator, 0.0)
        den = mole_fractions.get(denominator, 0.0)
        if den < 1e-9:
            return 0.0
        return num / den

    def _calc_sec(self, results, biomass_stream) -> float:
        """
        Specific Energy Consumption = total electrical input / biomass feed rate.
        Units: kWh per tonne of biomass (kWh/t).
        """
        if biomass_stream is None or biomass_stream.mass_flow_kg_s <= 0:
            return 0.0

        total_elec_kW = 0.0
        for e_name in ["E_PEM_AC_Power", "E_PEM_DC_Power", "E_Blower"]:
            e = results.energy_streams.get(e_name)
            if e:
                total_elec_kW += abs(e.energy_flow_kW)

        feed_t_h = biomass_stream.mass_flow_kg_s * 3.6  # kg/s → t/h
        if feed_t_h <= 0:
            return 0.0

        return total_elec_kW / feed_t_h  # kW / (t/h) = kWh/t

    @staticmethod
    def _calc_tar_loading(syngas_stream) -> float:
        """
        Tar loading in mg/Nm³.

        Uses mass fractions of tar surrogate species and the volumetric
        flow at NTP.
        """
        if syngas_stream.volumetric_flow_Nm3_h <= 0:
            return 0.0

        tar_kg_s = 0.0
        for species in TAR_SPECIES:
            wf = syngas_stream.mass_fractions.get(species, 0.0)
            tar_kg_s += wf * syngas_stream.mass_flow_kg_s

        # Convert: kg/s → mg/h, then divide by Nm³/h
        tar_mg_h = tar_kg_s * 1e6 * 3600.0
        vol_Nm3_h = syngas_stream.volumetric_flow_Nm3_h
        return tar_mg_h / vol_Nm3_h if vol_Nm3_h > 0 else 0.0

    @staticmethod
    def _calc_mass_balance(results) -> float:
        """
        Mass balance closure = total outlet mass flow / total inlet mass flow.
        """
        inlet_total = sum(
            results.streams[n].mass_flow_kg_s
            for n in INLET_STREAMS
            if n in results.streams
        )
        outlet_total = sum(
            results.streams[n].mass_flow_kg_s
            for n in OUTLET_STREAMS
            if n in results.streams
        )
        if inlet_total <= 0:
            return 0.0
        return outlet_total / inlet_total

    @staticmethod
    def _calc_energy_balance(results) -> float:
        """
        Energy balance closure = total outlet enthalpy / total inlet enthalpy.
        Includes energy stream contributions.
        """
        # Enthalpy flows (kW) from material streams
        inlet_h = sum(
            results.streams[n].mass_flow_kg_s
            * results.streams[n].specific_enthalpy_kJ_kg
            for n in INLET_STREAMS
            if n in results.streams
        )
        outlet_h = sum(
            results.streams[n].mass_flow_kg_s
            * results.streams[n].specific_enthalpy_kJ_kg
            for n in OUTLET_STREAMS
            if n in results.streams
        )

        # Add electrical energy inputs
        energy_in = sum(
            abs(results.energy_streams[n].energy_flow_kW)
            for n in ENERGY_INLET_STREAMS
            if n in results.energy_streams
        )
        energy_out = sum(
            abs(results.energy_streams[n].energy_flow_kW)
            for n in ENERGY_OUTLET_STREAMS
            if n in results.energy_streams
        )

        total_in = inlet_h + energy_in
        total_out = outlet_h + energy_out

        if abs(total_in) < 1e-6:
            return 0.0
        return total_out / total_in
