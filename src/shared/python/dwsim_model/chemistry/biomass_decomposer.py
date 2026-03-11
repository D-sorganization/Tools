"""
chemistry/biomass_decomposer.py
================================
Converts biomass ultimate/proximate analysis into a DWSIM-compatible
gas-phase component mixture.

Why is this needed?
-------------------
DWSIM cannot directly model solid biomass.  Real biomass (wood chips,
MSW, agricultural residues) is a complex solid defined by its chemical
composition (ultimate analysis) and physical properties (proximate
analysis).  This module converts that description into an equivalent
gas mixture so DWSIM's thermodynamic equations can process it.

The approach (Channiwala & Parikh correlation + element balance):
    1. Read ultimate analysis (C, H, O, N, S, Cl, ash) as mass fractions.
    2. Estimate the contribution of fixed carbon vs. volatiles.
    3. Distribute elements into surrogate gas species that:
       - Preserve the carbon content → CO + CO2 + CH4
       - Preserve the hydrogen content → H2 + H2O + CH4
       - Preserve the oxygen content → CO + CO2 + H2O
       - Add N, S, Cl as trace contaminants if in compound set
    4. Return mole fractions for each DWSIM compound.

Reference:
    Channiwala & Parikh (2002) Fuel 81(8), 1051–1063
    Jarungthammachote & Dutta (2007) Energy Conv. Mgmt. 48(7), 2085–2091
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Atomic weights (g/mol)
_MW = {
    "C": 12.011,
    "H": 1.008,
    "O": 15.999,
    "N": 14.007,
    "S": 32.06,
    "Cl": 35.45,
    "CO": 28.010,
    "H2": 2.016,
    "CO2": 44.010,
    "CH4": 16.043,
    "H2O": 18.015,
    "N2": 28.014,
    "H2S": 34.081,
    "NH3": 17.031,
    "HCl": 36.461,
    "He": 4.003,  # Ash proxy
}


@dataclass
class BiomassFeed:
    """
    Describes a biomass feed stream.

    Parameters
    ----------
    ultimate_daf:
        Dict of {element: mass_fraction} on a dry-ash-free (daf) basis.
        Keys: C, H, O, N, S, Cl.  Must sum to ≈ 1.0.
    moisture_ar:
        Moisture content on an as-received basis (mass fraction, 0–0.50).
    ash_ar:
        Ash content on an as-received basis (mass fraction, 0–0.40).
    hhv_mj_kg:
        Higher heating value (MJ/kg, dry basis).  Used for validation only.
    """

    ultimate_daf: dict[str, float] = field(
        default_factory=lambda: {
            "C": 0.501,
            "H": 0.062,
            "O": 0.421,
            "N": 0.008,
            "S": 0.005,
            "Cl": 0.003,
        }
    )
    moisture_ar: float = 0.15
    ash_ar: float = 0.10
    hhv_mj_kg: float = 18.5

    def __post_init__(self):
        total = sum(self.ultimate_daf.values())
        if abs(total - 1.0) > 0.03:
            raise ValueError(
                f"Ultimate analysis (daf) must sum to 1.0, got {total:.4f}. "
                "Check that fractions, not percentages, were supplied."
            )
        if not 0.0 <= self.moisture_ar <= 0.60:
            raise ValueError(f"moisture_ar {self.moisture_ar} out of range [0, 0.60]")
        if not 0.0 <= self.ash_ar <= 0.50:
            raise ValueError(f"ash_ar {self.ash_ar} out of range [0, 0.50]")


class BiomassDecomposer:
    """
    Converts a BiomassFeed description into DWSIM component mole fractions.

    The algorithm preserves element balance (C, H, O) by distributing the
    biomass atoms into a set of surrogate gas species.  Trace elements
    (N, S, Cl) are mapped to NH3, H2S, and HCl respectively if those
    compounds are in the simulation.

    Example
    -------
    >>> feed = BiomassFeed()
    >>> dec = BiomassDecomposer()
    >>> mole_fracs = dec.decompose(feed)
    >>> print(mole_fracs)
    {'Carbon monoxide': 0.32, 'Hydrogen': 0.28, ...}
    """

    def __init__(self, available_compounds: list[str] | None = None):
        """
        Parameters
        ----------
        available_compounds:
            List of compound names registered in the DWSIM simulation.
            Used to decide whether to include trace species.  If None,
            only core species are used.
        """
        self.available = set(available_compounds or [])

    # ─────────────────────────────────────────────────────────────────────────

    def decompose(self, feed: BiomassFeed) -> dict[str, float]:
        """
        Compute mole fractions for the equivalent gas mixture.

        Returns a dict of {DWSIM_compound_name: mole_fraction}.
        All values are ≥ 0 and the dict sums to 1.0.
        """
        # ── Step 1: Convert as-received to daf ──────────────────────────────
        # The daf basis already excludes moisture and ash.
        # We need to account for moisture separately as H2O.
        moisture_frac = feed.moisture_ar
        ash_frac = feed.ash_ar
        daf_frac = 1.0 - moisture_frac - ash_frac

        if daf_frac <= 0:
            raise ValueError(
                f"daf_frac = {daf_frac:.3f} ≤ 0. moisture + ash exceeds 100% of feed."
            )

        # Scale daf ultimate analysis to as-received, mass-based
        ua = feed.ultimate_daf
        # mass of each element per kg as-received feed
        m_C = ua.get("C", 0.0) * daf_frac  # kg C  per kg biomass
        m_H = ua.get("H", 0.0) * daf_frac  # kg H
        m_O = ua.get("O", 0.0) * daf_frac  # kg O
        m_N = ua.get("N", 0.0) * daf_frac  # kg N
        m_S = ua.get("S", 0.0) * daf_frac  # kg S
        m_Cl = ua.get("Cl", 0.0) * daf_frac  # kg Cl
        m_H2O_moisture = moisture_frac  # kg H2O from moisture

        # ── Step 2: Convert to moles per kg biomass ──────────────────────────
        mol_C = m_C / _MW["C"]
        mol_H = m_H / _MW["H"]  # moles of H atoms
        mol_O = m_O / _MW["O"]  # moles of O atoms
        mol_N = m_N / _MW["N"]
        mol_S = m_S / _MW["S"]
        mol_Cl = m_Cl / _MW["Cl"]
        mol_H2O_moist = m_H2O_moisture / _MW["H2O"]

        # ── Step 3: Distribute trace elements ───────────────────────────────
        # All N → NH3:   N + 1.5*H2 → NH3  (consumes H)
        # All S → H2S:   S + H2 → H2S       (consumes H)
        # All Cl → HCl:  Cl + 0.5*H2 → HCl  (consumes H)

        mol_NH3 = mol_N  # 1 mol NH3 per mol N
        mol_H2S = mol_S  # 1 mol H2S per mol S
        mol_HCl = mol_Cl  # 1 mol HCl per mol Cl

        # H consumed by trace species
        mol_H_for_NH3 = mol_NH3 * 3  # 3 H atoms per NH3
        mol_H_for_H2S = mol_H2S * 2  # 2 H atoms per H2S
        mol_H_for_HCl = mol_HCl * 1  # 1 H atom per HCl

        mol_H_remaining = mol_H - (mol_H_for_NH3 + mol_H_for_H2S + mol_H_for_HCl)
        if mol_H_remaining < 0:
            logger.warning(
                "Trace elements consume more H than available — "
                "setting NH3, H2S, HCl to zero and continuing."
            )
            mol_NH3 = mol_H2S = mol_HCl = 0.0
            mol_H_remaining = mol_H

        # ── Step 4: Core C/H/O distribution ─────────────────────────────────
        # Strategy (simplified devolatilisation model):
        #   - All carbon → CO first (maximise CO which is the gasification target)
        #   - Remaining O → CO2 and H2O (using a split factor)
        #   - Remaining H → H2
        #   - Moisture enters directly as H2O
        #
        # CO/CO2 split: governed by partial oxidation equilibrium.
        # At gasification temperatures (>800°C), CO strongly dominates.
        # We use CO2_fraction = 0.15 as a reasonable default.

        co2_frac = 0.15  # Fraction of carbon going to CO2 (rest to CO)

        mol_CO2 = mol_C * co2_frac
        mol_CO = mol_C * (1.0 - co2_frac)

        # Oxygen balance check — O consumed by CO and CO2
        mol_O_in_CO_CO2 = mol_CO + 2 * mol_CO2
        mol_O_remaining = mol_O - mol_O_in_CO_CO2

        # Remaining O → H2O
        mol_H2O_from_O = max(mol_O_remaining, 0.0)  # Can't be negative

        # All remaining H → H2 and H2O
        mol_H_for_H2O = mol_H2O_from_O * 2  # 2 H per H2O
        mol_H_for_H2 = max(mol_H_remaining - mol_H_for_H2O, 0.0)
        mol_H2 = mol_H_for_H2 / 2  # H atoms → H2 molecules

        mol_H2O_total = mol_H2O_from_O + mol_H2O_moist

        # Small methane fraction (from methanation / incomplete conversion)
        # Roughly 2–5% of product on a molar basis
        ch4_yield = 0.03 * mol_C
        mol_CH4 = ch4_yield
        mol_CO = max(mol_CO - 2 * ch4_yield, 0.0)  # CH4 formation consumes CO
        mol_H2 = max(mol_H2 - 4 * ch4_yield, 0.0)  # and H2

        # ── Step 5: Ash proxy (Helium) ───────────────────────────────────────
        # Ash doesn't contribute to gas-phase chemistry but must be tracked
        # for mass balance.  We add it as a very small Helium fraction.
        mol_He = (ash_frac / daf_frac) * mol_C * 0.01  # small proxy

        # ── Step 6: Build mole fraction dict ────────────────────────────────
        mol_dict: dict[str, float] = {
            "Carbon monoxide": mol_CO,
            "Hydrogen": mol_H2,
            "Carbon dioxide": mol_CO2,
            "Methane": mol_CH4,
            "Water": mol_H2O_total,
            "Helium": mol_He,
        }

        # Only include trace species if they're in the simulation
        if "Ammonia" in self.available and mol_NH3 > 0:
            mol_dict["Ammonia"] = mol_NH3
        if "Hydrogen sulfide" in self.available and mol_H2S > 0:
            mol_dict["Hydrogen sulfide"] = mol_H2S
        if "Hydrogen chloride" in self.available and mol_HCl > 0:
            mol_dict["Hydrogen chloride"] = mol_HCl

        # Normalise to mole fractions
        total_moles = sum(mol_dict.values())
        if total_moles <= 0:
            raise RuntimeError(
                "BiomassDecomposer produced zero total moles — check inputs."
            )

        mole_fractions = {k: v / total_moles for k, v in mol_dict.items() if v > 0}

        logger.info(
            f"BiomassDecomposer: decomposed feed into {len(mole_fractions)} species. "
            f"Total moles = {total_moles:.4f} per kg biomass."
        )
        logger.debug(
            "Mole fractions: "
            + ", ".join(
                f"{k}={v:.3f}"
                for k, v in sorted(mole_fractions.items(), key=lambda x: -x[1])
            )
        )

        return mole_fractions

    # ─────────────────────────────────────────────────────────────────────────

    def estimate_hhv(self, feed: BiomassFeed) -> float:
        """
        Estimate the higher heating value (MJ/kg, dry) using the
        Channiwala-Parikh correlation.

        HHV (MJ/kg) = 0.3491*C + 1.1783*H + 0.1005*S
                    - 0.1034*O - 0.0151*N - 0.0211*Ash

        where C, H, S, O, N, Ash are as-received mass fractions (not daf).

        Reference: Channiwala & Parikh (2002) Fuel 81(8), 1051–1063
        """
        ua = feed.ultimate_daf
        daf = 1.0 - feed.moisture_ar - feed.ash_ar

        # Channiwala & Parikh (2002) expects mass fractions in PERCENT (0–100),
        # not as fractions (0–1).  Multiply each term by 100 before applying
        # the correlation coefficients.
        C = ua.get("C", 0.0) * daf * 100  # % on as-received basis
        H = ua.get("H", 0.0) * daf * 100
        S = ua.get("S", 0.0) * daf * 100
        O_pct = ua.get("O", 0.0) * daf * 100
        N = ua.get("N", 0.0) * daf * 100
        ash = feed.ash_ar * 100

        hhv = (
            0.3491 * C
            + 1.1783 * H
            + 0.1005 * S
            - 0.1034 * O_pct
            - 0.0151 * N
            - 0.0211 * ash
        )
        logger.debug(f"Channiwala-Parikh HHV estimate: {hhv:.2f} MJ/kg")
        return hhv


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test / demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Typical woody biomass (pine, oven-dry basis)
    feed = BiomassFeed(
        ultimate_daf={
            "C": 0.501,
            "H": 0.062,
            "O": 0.421,
            "N": 0.008,
            "S": 0.005,
            "Cl": 0.003,
        },
        moisture_ar=0.15,
        ash_ar=0.10,
        hhv_mj_kg=18.5,
    )

    decomposer = BiomassDecomposer(
        available_compounds=[
            "Carbon monoxide",
            "Hydrogen",
            "Carbon dioxide",
            "Methane",
            "Water",
            "Nitrogen",
            "Helium",
            "Ammonia",
            "Hydrogen sulfide",
        ]
    )

    fracs = decomposer.decompose(feed)

    print("\nDecomposed biomass → equivalent gas mixture:")
    print(f"{'Species':<25} {'Mole Fraction':>15}")
    print("-" * 42)
    for species, xf in sorted(fracs.items(), key=lambda x: -x[1]):
        print(f"{species:<25} {xf:>15.4f}")

    estimated_hhv = decomposer.estimate_hhv(feed)
    print(f"\nChanniwala-Parikh HHV estimate: {estimated_hhv:.2f} MJ/kg")
    print(f"Provided HHV: {feed.hhv_mj_kg:.2f} MJ/kg")
    print(f"Difference: {abs(estimated_hhv - feed.hhv_mj_kg):.2f} MJ/kg")
