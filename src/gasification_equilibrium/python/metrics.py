"""Post-solve metrics computation.

SRP: Computes derived quantities from equilibrium solution.
     No feed processing, no optimization, no plotting.

All functions are pure (no side effects, no state).
"""

import numpy as np

from .thermo_data import HEATING_VALUES_HHV, SPECIES_DB


def gas_mole_fractions(moles, species_keys):
    """Compute gas-phase mole fractions from absolute moles.

    Postcondition: fractions sum to ~1.0, all >= 0
    """
    gas_mask = np.array([SPECIES_DB[k]["phase"] == "gas" for k in species_keys])
    n_gas = moles * gas_mask
    total = max(n_gas.sum(), 1e-15)
    return n_gas / total, total


def h2_co_ratio(moles, species_keys):
    """Compute H2/CO molar ratio.

    Returns 0.0 if CO is negligible.
    """
    comp = dict(zip(species_keys, moles, strict=True))
    h2 = comp.get("H2", 0.0)
    co = comp.get("CO", 0.0)
    return h2 / co if co > 1e-12 else 0.0


def carbon_conversion(moles, species_keys, feed_elements):
    """Fraction of feed carbon converted to gas phase.

    Postcondition: 0.0 <= result <= 1.0
    """
    c_feed = feed_elements.get("C", 0.0)
    if c_feed <= 1e-12:
        return 1.0
    try:
        c_solid_idx = species_keys.index("C_solid")
        c_solid = moles[c_solid_idx]
    except ValueError:
        c_solid = 0.0
    return float(np.clip(1.0 - c_solid / c_feed, 0.0, 1.0))


def cold_gas_efficiency(mole_fractions, total_gas_moles, species_keys, feed_elements):
    """Cold gas efficiency on HHV basis.

    CGE = (energy in product gas) / (energy in feed)

    Postcondition: 0.0 <= result (can exceed 1.0 with external energy input)
    """
    syngas_energy = sum(
        mole_fractions[i] * total_gas_moles * HEATING_VALUES_HHV.get(sp, 0.0)
        for i, sp in enumerate(species_keys)
    )
    c_feed = feed_elements.get("C", 0.0)
    h_feed = feed_elements.get("H", 0.0)
    feed_energy = c_feed * 393.5 + h_feed * 0.5 * 285.8
    if feed_energy <= 0:
        return 0.0
    return min(syngas_energy / feed_energy, 2.0)


def composition_dict(mole_fractions, species_keys):
    """Convert arrays to {species: fraction} dict."""
    return dict(zip(species_keys, mole_fractions, strict=True))


def dry_mole_fractions(comp_dict):
    """Remove H2O and renormalize.

    Postcondition: 'H2O' not in result, values sum to ~1.0
    """
    h2o = comp_dict.get("H2O", 0.0)
    if h2o >= 1.0:
        return dict(comp_dict)
    return {k: v / (1.0 - h2o) for k, v in comp_dict.items() if k != "H2O"}
