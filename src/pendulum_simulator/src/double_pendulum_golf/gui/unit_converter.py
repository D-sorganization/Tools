"""
Unit conversion system for the Pendulum Simulator (#1137).

Provides bidirectional conversion between SI (meters, kg, N·m) and
Imperial (inches, lbs, lbf·in) units.  Designed for integration with
the controls widgets — all physics remain in SI internally.

Design by Contract
------------------
- UnitConverter is stateless; all methods are pure functions.
- ``system`` must be one of ``"SI"`` or ``"Imperial"``.
- Conversion factors are exact (based on NIST reference values).

DRY
---
Single source of truth for all conversion constants.  GUI widgets call
``display_length()`` / ``parse_length()`` rather than scattering factors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (exact NIST values)
# ---------------------------------------------------------------------------
_METERS_PER_INCH = 0.0254
_KG_PER_LB = 0.45359237
_NM_PER_LBFIN = 0.1129848290276167  # 1 lbf·in = 0.1130 N·m


class UnitSystem(Enum):
    """Supported unit systems."""

    SI = "SI"
    IMPERIAL = "Imperial"


@dataclass(frozen=True)
class UnitConverter:
    """Bidirectional unit converter between SI and Imperial.

    All physics computations use SI internally.  This class converts
    *display* values (what the user sees) to/from SI.

    Contract
    --------
    - ``system`` must be a valid ``UnitSystem``.
    - All ``to_si_*`` methods accept display-unit values and return SI.
    - All ``from_si_*`` methods accept SI values and return display-unit.
    """

    system: UnitSystem = UnitSystem.SI

    # ------------------------------------------------------------------
    # Length  (m ↔ in)
    # ------------------------------------------------------------------

    def to_si_length(self, value: float) -> float:
        """Convert a length from display units to meters.

        Pre: value is finite.
        Post: result is in meters.
        """
        if self.system == UnitSystem.IMPERIAL:
            return value * _METERS_PER_INCH
        return value

    def from_si_length(self, meters: float) -> float:
        """Convert a length from meters to display units.

        Pre: meters is finite.
        Post: result is in display units (m or in).
        """
        if self.system == UnitSystem.IMPERIAL:
            return meters / _METERS_PER_INCH
        return meters

    # ------------------------------------------------------------------
    # Mass  (kg ↔ lb)
    # ------------------------------------------------------------------

    def to_si_mass(self, value: float) -> float:
        """Convert a mass from display units to kg."""
        if self.system == UnitSystem.IMPERIAL:
            return value * _KG_PER_LB
        return value

    def from_si_mass(self, kg: float) -> float:
        """Convert a mass from kg to display units."""
        if self.system == UnitSystem.IMPERIAL:
            return kg / _KG_PER_LB
        return kg

    # ------------------------------------------------------------------
    # Torque  (N·m ↔ lbf·in)
    # ------------------------------------------------------------------

    def to_si_torque(self, value: float) -> float:
        """Convert a torque from display units to N·m."""
        if self.system == UnitSystem.IMPERIAL:
            return value * _NM_PER_LBFIN
        return value

    def from_si_torque(self, nm: float) -> float:
        """Convert a torque from N·m to display units."""
        if self.system == UnitSystem.IMPERIAL:
            return nm / _NM_PER_LBFIN
        return nm

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    @property
    def length_unit(self) -> str:
        """Display-unit label for length."""
        return "in" if self.system == UnitSystem.IMPERIAL else "m"

    @property
    def mass_unit(self) -> str:
        """Display-unit label for mass."""
        return "lb" if self.system == UnitSystem.IMPERIAL else "kg"

    @property
    def torque_unit(self) -> str:
        """Display-unit label for torque."""
        return "lbf·in" if self.system == UnitSystem.IMPERIAL else "N·m"

    @property
    def velocity_unit(self) -> str:
        """Display-unit label for angular velocity (always rad/s)."""
        return "rad/s"

    @property
    def angle_unit(self) -> str:
        """Display-unit label for angles (always degrees in UI)."""
        return "°"
