"""Gasification Equilibrium Engine - thin orchestrator.

SRP: Wires together feed building, solving, and metrics.
DIP: Depends on abstractions (solver, metrics modules), not concretions.
OCP: New injection types are handled by feed.py; engine doesn't change.

This is the primary public API for library consumers.
"""

from dataclasses import dataclass, field

import numpy as np

from .feed import (
    FeedComposition,
    build_total_feed,
)
from .metrics import (
    carbon_conversion,
    cold_gas_efficiency,
    composition_dict,
    dry_mole_fractions,
    gas_mole_fractions,
    h2_co_ratio,
)
from .solver import ElementMatrix, solve_equilibrium
from .sweeps import surface_sweep, temperature_sweep
from .thermo_data import SPECIES_DB, get_all_species


@dataclass
class EquilibriumResult:
    """Result of an equilibrium calculation.

    Invariant: if converged, element_balance_error < tolerance
    """

    temperature: float
    pressure: float
    species: list
    mole_fractions: np.ndarray
    moles: np.ndarray
    total_moles: float
    gibbs_energy: float
    converged: bool
    iterations: int
    element_balance_error: float
    h2_co_ratio: float = 0.0
    cold_gas_efficiency: float = 0.0
    carbon_conversion: float = 0.0
    feed_elements: dict = field(default_factory=dict)

    def composition_dict(self):
        return composition_dict(self.mole_fractions, self.species)

    def moles_dict(self):
        return dict(zip(self.species, self.moles, strict=True))

    def dry_mole_fractions(self):
        return dry_mole_fractions(self.composition_dict())


class GasificationEngine:
    """Orchestrator: combines feed + solver + metrics.

    All calculation logic is delegated to focused modules.
    This class provides the unified API and manages shared state
    (species list, element matrix).
    """

    def __init__(self, species_keys=None):
        """Initialize with species set.

        Precondition: all species_keys must exist in SPECIES_DB
        """
        if species_keys is None:
            species_keys = get_all_species()
        for k in species_keys:
            assert k in SPECIES_DB, f"Unknown species: {k}"
        self.species_keys = list(species_keys)
        self.matrix = ElementMatrix.from_species(species_keys)

    @property
    def n_species(self):
        return self.matrix.n_species

    def solve(
        self,
        temperature,
        pressure=101325.0,
        feed=None,
        feed_mass=None,
        process_inputs=None,
        steam_carbon_ratio=0.0,
        oxygen_carbon_ratio=0.0,
        equivalence_ratio=None,
        tolerance=None,
        warm_start=None,
    ):
        """Solve for equilibrium composition.

        Supports three feed modes:
            1. feed={element: moles} - direct element specification
            2. feed_mass={element: mass_frac} - mass fraction conversion
            3. process_inputs=ProcessInputs - full process model with injections

        Legacy params (steam_carbon_ratio, oxygen_carbon_ratio) are converted
        to ProcessInputs internally for backward compatibility.

        Precondition: temperature > 0, pressure > 0
        Postcondition: EquilibriumResult with valid composition
        """
        feed_elements = self._build_feed(
            feed,
            feed_mass,
            process_inputs,
            steam_carbon_ratio,
            oxygen_carbon_ratio,
            equivalence_ratio,
        )

        raw = solve_equilibrium(
            T=temperature,
            P=pressure,
            feed_elements=feed_elements,
            matrix=self.matrix,
            tolerance=tolerance,
            warm_start=warm_start,
        )

        return self._build_result(raw, temperature, pressure, feed_elements)

    def _build_feed(
        self,
        feed,
        feed_mass,
        process_inputs,
        steam_carbon_ratio,
        oxygen_carbon_ratio,
        equivalence_ratio,
    ):
        """Construct total element balance from inputs.

        Handles all three input modes and legacy compatibility.
        """
        if process_inputs is not None:
            base = self._resolve_base_feed(feed, feed_mass)
            return build_total_feed(base, process_inputs)

        if feed_mass is not None:
            base = FeedComposition.from_mass_fractions(feed_mass)
        elif feed is not None:
            base = FeedComposition.from_dict(feed)
        else:
            base = FeedComposition(C=1.0, H=0.1, O=0.5)

        if self._has_legacy_params(
            steam_carbon_ratio, oxygen_carbon_ratio, equivalence_ratio
        ):
            return self._apply_legacy_ratios(
                base,
                steam_carbon_ratio,
                oxygen_carbon_ratio,
                equivalence_ratio,
            )

        return base.as_dict()

    def _resolve_base_feed(self, feed, feed_mass):
        """Convert raw feed args to FeedComposition."""
        if feed_mass is not None:
            return FeedComposition.from_mass_fractions(feed_mass)
        if feed is not None:
            return FeedComposition.from_dict(feed)
        return FeedComposition(C=1.0, H=0.1, O=0.5)

    @staticmethod
    def _has_legacy_params(sc, oc, er):
        return sc > 0 or oc > 0 or (er is not None and er > 0)

    @staticmethod
    def _apply_legacy_ratios(
        base, steam_carbon_ratio, oxygen_carbon_ratio, equivalence_ratio
    ):
        """Convert legacy ratio params to element additions.

        This preserves backward compatibility with the old API.
        """
        elements = base.as_dict()
        c_moles = elements.get("C", 0.0)

        if steam_carbon_ratio > 0 and c_moles > 0:
            elements["H"] = elements.get("H", 0.0) + steam_carbon_ratio * c_moles * 2
            elements["O"] = elements.get("O", 0.0) + steam_carbon_ratio * c_moles

        if oxygen_carbon_ratio > 0 and c_moles > 0:
            elements["O"] = elements.get("O", 0.0) + oxygen_carbon_ratio * c_moles * 2

        if equivalence_ratio is not None and equivalence_ratio > 0:
            stoich_o = elements.get("C", 0) * 2 + elements.get("H", 0) * 0.5
            elements["O"] = stoich_o * (1.0 / equivalence_ratio)

        return elements

    def _build_result(self, raw, temperature, pressure, feed_elements):
        """Convert SolverResult to EquilibriumResult with metrics."""
        mole_fracs, total_gas = gas_mole_fractions(raw.moles, self.species_keys)

        return EquilibriumResult(
            temperature=temperature,
            pressure=pressure,
            species=list(self.species_keys),
            mole_fractions=mole_fracs,
            moles=raw.moles,
            total_moles=total_gas,
            gibbs_energy=raw.gibbs_energy,
            converged=raw.converged,
            iterations=raw.iterations,
            element_balance_error=raw.balance_error,
            h2_co_ratio=h2_co_ratio(raw.moles, self.species_keys),
            cold_gas_efficiency=cold_gas_efficiency(
                mole_fracs,
                total_gas,
                self.species_keys,
                feed_elements,
            ),
            carbon_conversion=carbon_conversion(
                raw.moles,
                self.species_keys,
                feed_elements,
            ),
            feed_elements=dict(feed_elements),
        )

    def temperature_sweep(self, t_start, t_end, n_points=50, **kwargs):
        """Delegate to sweeps module."""
        return temperature_sweep(self, t_start, t_end, n_points, **kwargs)

    def surface_sweep(self, t_range, param_name, param_range, **kwargs):
        """Delegate to sweeps module."""
        return surface_sweep(self, t_range, param_name, param_range, **kwargs)
