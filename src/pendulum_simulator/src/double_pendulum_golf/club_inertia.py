"""Single source of truth for the distal (club) segment's inertial properties.

Canonical convention (owner ruling on Tools #5431, 2026-10-11): the club is a
**uniform shaft** of mass ``m2`` and length ``L2`` (COM at ``L2/2``, centroidal
inertia ``m2*L2**2/12``) plus a **head point mass** ``mClub`` at ``L2``.

Every consumer (mass matrix, Coriolis, gravity, forces, energies, transfer
diagnostics, objectives) reads the same three numbers from here instead of
re-deriving them:

* ``mass_kg``                   Mc = m2 + mh
* ``first_moment_kg_m``         s  = m2*L2/2 + mh*L2      (= Mc * COM distance)
* ``inertia_about_pivot_kg_m2`` Jw = m2*L2**2/3 + mh*L2**2 (= Icm + Mc * COM**2)
* ``centroidal_inertia_kg_m2``  Icm = m2*L2**2*(m2 + 4*mh) / (12*Mc), the closed form
  of ``Jw - s**2/Mc``. It is never computed as a difference of near-equal
  terms: at tiny ``m2`` that subtraction cancels catastrophically and can go
  negative.

The old lumped model (all of ``m2 + mClub`` at the tip) is the special case
``s = Mc*L2`` and ``Jw = Mc*L2**2``.

Cross-language parity (Rust ``DistalInertia``, TypeScript ``distalInertia``) is
pinned by ``tests/fixtures/distal_inertia_cases.json``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["DistalSegmentInertia", "distal_inertia"]


@dataclass(frozen=True, slots=True)
class DistalSegmentInertia:
    """Mass, first moment and pivot inertia of the club about the wrist."""

    mass_kg: float
    first_moment_kg_m: float
    inertia_about_pivot_kg_m2: float
    centroidal_inertia_kg_m2: float

    @property
    def com_m(self) -> float:
        """Distance from the wrist to the club's centre of mass."""
        return self.first_moment_kg_m / self.mass_kg

    @property
    def icm_kg_m2(self) -> float:
        """Centroidal inertia of shaft plus head about the club COM."""
        return self.centroidal_inertia_kg_m2


def distal_inertia(
    shaft_mass_kg: float, head_mass_kg: float, length_m: float
) -> DistalSegmentInertia:
    """Return the canonical club inertia for a uniform shaft plus head mass.

    Pre: masses finite and non-negative, total mass > 0, length finite and > 0.
    Post: ``Icm >= 0`` by construction (closed form, no cancellation).
    """
    for name, value in (
        ("shaft_mass_kg", shaft_mass_kg),
        ("head_mass_kg", head_mass_kg),
        ("length_m", length_m),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    if shaft_mass_kg < 0.0 or head_mass_kg < 0.0:
        raise ValueError("shaft and head masses must be non-negative")
    if shaft_mass_kg + head_mass_kg <= 0.0:
        raise ValueError("total club mass must be positive")
    if length_m <= 0.0:
        raise ValueError(f"length_m must be positive, got {length_m}")
    return DistalSegmentInertia(
        mass_kg=shaft_mass_kg + head_mass_kg,
        first_moment_kg_m=(0.5 * shaft_mass_kg + head_mass_kg) * length_m,
        inertia_about_pivot_kg_m2=(shaft_mass_kg / 3.0 + head_mass_kg) * length_m**2,
        centroidal_inertia_kg_m2=(
            shaft_mass_kg
            * length_m**2
            * (shaft_mass_kg + 4.0 * head_mass_kg)
            / (12.0 * (shaft_mass_kg + head_mass_kg))
        ),
    )
