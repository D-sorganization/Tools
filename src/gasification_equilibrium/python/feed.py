"""Feed composition builder with blending and process injection support.

SRP: This module handles ONLY feed composition construction.
OCP: New injection types are added by composing Injection dataclasses,
     not by modifying existing methods.

Design by Contract:
    - All flows are non-negative
    - Element totals are always the sum of contributions
    - Feed rate converts mass flow to molar element flow
"""

from dataclasses import dataclass, field

from .thermo_data import ATOMIC_WEIGHTS

# ─── Injection compounds (element breakdowns) ──────────────────────────────────
# Each maps to {element: atoms_per_molecule}

COMPOUND_ELEMENTS = {
    "H2O": {"H": 2, "O": 1},
    "O2": {"O": 2},
    "N2": {"N": 2},
    "CH4": {"C": 1, "H": 4},
    "C3H8": {"C": 3, "H": 8},
    "natural_gas": {"C": 1.05, "H": 4.16, "N": 0.04},  # ~95% CH4, 3% C2H6, 2% N2
}

AIR_O2_FRACTION = 0.2095
AIR_N2_FRACTION = 0.7808


@dataclass
class Injection:
    """A single process stream injection.

    Invariant: flow >= 0
    """

    name: str
    flow: float = 0.0  # mol/s (or mol per unit feed)
    elements: dict = field(default_factory=dict)

    def element_contribution(self):
        """Return {element: moles} added by this injection.

        Postcondition: all values >= 0
        """
        return {e: self.flow * count for e, count in self.elements.items()}


@dataclass
class OxidantConfig:
    """Oxidant selection: pure O2 or air.

    Invariant: o2_flow >= 0
    """

    use_air: bool = False
    o2_flow: float = 0.0  # mol O2 per unit feed

    def element_contribution(self):
        """Return elements from oxidant (O2 + N2 if air).

        Air composition: 20.95% O2, 78.08% N2, 0.93% Ar (by mole).
        """
        if self.o2_flow <= 0:
            return {}
        result = {"O": self.o2_flow * 2}
        if self.use_air:
            air_moles = self.o2_flow / AIR_O2_FRACTION
            result["N"] = air_moles * AIR_N2_FRACTION * 2  # N2 has 2 N atoms
        return result


@dataclass
class FeedComposition:
    """CHONS elemental feed composition.

    Represents the solid/liquid feedstock before any process injections.

    Invariant: all element values >= 0
    """

    C: float = 0.0
    H: float = 0.0
    O: float = 0.0  # noqa: E741
    N: float = 0.0
    S: float = 0.0

    def as_dict(self):
        """Return {element: moles} for non-zero elements."""
        return {
            k: v
            for k, v in {
                "C": self.C,
                "H": self.H,
                "O": self.O,
                "N": self.N,
                "S": self.S,
            }.items()
            if v > 0
        }

    def total_moles(self):
        return self.C + self.H + self.O + self.N + self.S

    @classmethod
    def from_dict(cls, d):
        """Create from {element: value} dict."""
        return cls(
            C=d.get("C", 0.0),
            H=d.get("H", 0.0),
            O=d.get("O", 0.0),
            N=d.get("N", 0.0),
            S=d.get("S", 0.0),
        )

    @classmethod
    def from_mass_fractions(cls, mass_fracs):
        """Convert mass fractions to molar amounts (per kg basis).

        Precondition: mass_fracs values >= 0, keys are element symbols
        Postcondition: returns FeedComposition with moles/kg
        """
        total = sum(
            v for k, v in mass_fracs.items() if k != "Ash" and k in ATOMIC_WEIGHTS
        )
        if total <= 0:
            return cls()
        moles = {}
        for elem, frac in mass_fracs.items():
            if elem == "Ash" or elem not in ATOMIC_WEIGHTS:
                continue
            moles[elem] = (frac / total) / (ATOMIC_WEIGHTS[elem] / 1000.0)
        return cls.from_dict(moles)


@dataclass
class ProcessInputs:
    """All process injection streams.

    OCP: Add new injection types by adding fields here;
    build_total_feed() automatically picks them up.
    """

    oxidant: OxidantConfig = field(default_factory=OxidantConfig)
    steam: Injection = field(
        default_factory=lambda: Injection("Steam", elements=COMPOUND_ELEMENTS["H2O"])
    )
    n2_purge: Injection = field(
        default_factory=lambda: Injection("N2 Purge", elements=COMPOUND_ELEMENTS["N2"])
    )
    ch4_injection: Injection = field(
        default_factory=lambda: Injection("CH4", elements=COMPOUND_ELEMENTS["CH4"])
    )
    c3h8_injection: Injection = field(
        default_factory=lambda: Injection("C3H8", elements=COMPOUND_ELEMENTS["C3H8"])
    )
    natural_gas: Injection = field(
        default_factory=lambda: Injection(
            "Natural Gas", elements=COMPOUND_ELEMENTS["natural_gas"]
        )
    )
    feed_rate_kg_hr: float = 100.0  # kg/hr of solid feed

    def all_injections(self):
        """Return list of all injection streams."""
        return [
            self.steam,
            self.n2_purge,
            self.ch4_injection,
            self.c3h8_injection,
            self.natural_gas,
        ]


def build_total_feed(base_feed, process_inputs):
    """Combine base feed with all process injections into total element balance.

    This is the central feed-building function. It is pure (no side effects).

    Args:
        base_feed: FeedComposition (CHONS of solid feedstock)
        process_inputs: ProcessInputs (all injection streams)

    Returns:
        dict of {element: total_moles}

    Precondition: base_feed and process_inputs are valid
    Postcondition: all returned values >= 0
    """
    total = base_feed.as_dict()

    # Add oxidant (O2 or air)
    _merge_elements(total, process_inputs.oxidant.element_contribution())

    # Add each injection stream
    for injection in process_inputs.all_injections():
        _merge_elements(total, injection.element_contribution())

    return total


def _merge_elements(target, source):
    """Merge source element dict into target (in-place addition).

    Precondition: source values >= 0
    """
    for elem, amount in source.items():
        target[elem] = target.get(elem, 0.0) + amount


# ─── Feed presets ───────────────────────────────────────────────────────────────

FEED_PRESETS = {
    "Bituminous Coal": {
        "mass_fractions": {
            "C": 0.75,
            "H": 0.05,
            "O": 0.08,
            "N": 0.015,
            "S": 0.01,
            "Ash": 0.095,
        },
        "description": "Typical bituminous coal (dry basis)",
    },
    "Sub-bituminous Coal": {
        "mass_fractions": {
            "C": 0.60,
            "H": 0.04,
            "O": 0.15,
            "N": 0.01,
            "S": 0.005,
            "Ash": 0.195,
        },
        "description": "Sub-bituminous coal (PRB-like)",
    },
    "Lignite": {
        "mass_fractions": {
            "C": 0.45,
            "H": 0.03,
            "O": 0.20,
            "N": 0.008,
            "S": 0.01,
            "Ash": 0.302,
        },
        "description": "Low-rank lignite coal",
    },
    "Biomass (Wood)": {
        "mass_fractions": {
            "C": 0.50,
            "H": 0.06,
            "O": 0.42,
            "N": 0.002,
            "S": 0.001,
            "Ash": 0.017,
        },
        "description": "Typical woody biomass",
    },
    "Petcoke": {
        "mass_fractions": {
            "C": 0.88,
            "H": 0.04,
            "O": 0.01,
            "N": 0.015,
            "S": 0.05,
            "Ash": 0.005,
        },
        "description": "Petroleum coke",
    },
    "MSW (Municipal Waste)": {
        "mass_fractions": {
            "C": 0.35,
            "H": 0.05,
            "O": 0.25,
            "N": 0.01,
            "S": 0.005,
            "Ash": 0.335,
        },
        "description": "Municipal solid waste",
    },
    "Natural Gas (CH4)": {
        "elements": {"C": 1.0, "H": 4.0},
        "description": "Pure methane equivalent",
    },
    "Custom": {
        "elements": {"C": 1.0, "H": 1.0, "O": 0.5},
        "description": "User-defined CHONS composition",
    },
}


def feed_from_preset(name):
    """Create FeedComposition from a named preset.

    Precondition: name is a key in FEED_PRESETS
    Postcondition: returns valid FeedComposition
    """
    preset = FEED_PRESETS[name]
    if "mass_fractions" in preset:
        return FeedComposition.from_mass_fractions(preset["mass_fractions"])
    return FeedComposition.from_dict(preset.get("elements", {}))
