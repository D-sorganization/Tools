"""Composite club inertial model: head + shaft + grip.

The club is modeled as three rigid components on a common shaft axis,
with the grip butt as the origin and distance measured along the shaft
toward the head:

* **Head** — a point mass ``m_h`` at the full club length ``L`` (its
  own MOI about its CG is negligible next to ``m_h L²`` — under 0.5%
  for every library club — and is documented as excluded). About the
  shaft axis the head contributes its spec'd scalar MOI directly.
* **Shaft** — a uniform thin rod of mass ``m_s`` spanning 0..L.
  About its end (the grip axis): ``I = m_s L² / 3``; CG at ``L/2``.
* **Grip** — a uniform sleeve of mass ``m_g`` over the top
  ``GRIP_LENGTH_M``: ``I = m_g l_g² / 3`` about the butt; CG at
  ``l_g/2``.

Formulas (all from the parallel-axis theorem and standard rod/tube
moments; any mechanics text):

    total mass      M = m_h + m_s + m_g
    balance point   d = (m_h·L + m_s·L/2 + m_g·l_g/2) / M
    MOI about grip  I_g = m_h·L² + m_s·L²/3 + m_g·l_g²/3
    MOI about shaft I_s = I_head_spec + m_s·r_s² + m_g·r_g²

where ``r_s``/``r_g`` are thin-tube radii for the shaft and grip. The
default component masses and radii are typical published values (steel
or graphite shafts 60-110 g, grips ~50 g, shaft OD ~12 mm, grip OD
~22 mm); callers with measured components should pass their own.

TypeScript twin: ``web/src/model/club.ts`` (``clubInertia``), pinned
against the numeric cases in ``tests/rate_of_closure/test_club.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from rate_of_closure._contracts import ensure, require

from .types import ClubSpec

__all__ = [
    "DEFAULT_GRIP_MASS_KG",
    "DEFAULT_SHAFT_MASS_KG",
    "GRIP_LENGTH_M",
    "GRIP_TUBE_RADIUS_M",
    "SHAFT_TUBE_RADIUS_M",
    "ClubInertia",
    "club_inertia",
]

#: Typical published component values (see module docstring).
DEFAULT_SHAFT_MASS_KG = 0.075
DEFAULT_GRIP_MASS_KG = 0.050
GRIP_LENGTH_M = 0.25
SHAFT_TUBE_RADIUS_M = 0.006
GRIP_TUBE_RADIUS_M = 0.011


@dataclass(frozen=True)
class ClubInertia:
    """Composite inertial properties of an assembled club.

    Attributes:
        total_mass_kg: Head + shaft + grip mass.
        balance_point_m: Whole-club CG distance from the grip butt,
            along the shaft.
        moi_about_grip_kg_m2: MOI about a swing axis through the grip
            butt, perpendicular to the shaft.
        moi_about_shaft_kg_m2: MOI about the shaft's own long axis
            (the axis whose rotation closes the face).
    """

    total_mass_kg: float
    balance_point_m: float
    moi_about_grip_kg_m2: float
    moi_about_shaft_kg_m2: float


def club_inertia(
    spec: ClubSpec,
    shaft_mass_kg: float = DEFAULT_SHAFT_MASS_KG,
    grip_mass_kg: float = DEFAULT_GRIP_MASS_KG,
) -> ClubInertia:
    """Compose head, shaft, and grip into whole-club inertia.

    Args:
        spec: The club to assemble.
        shaft_mass_kg: Shaft mass; default is a typical published
            mid-weight shaft.
        grip_mass_kg: Grip mass; default is a typical published
            standard grip.

    Returns:
        The composite :class:`ClubInertia` (see module docstring for
        the formulas).
    """
    require(0.0 < shaft_mass_kg <= 0.25, "shaft_mass_kg out of range", shaft_mass_kg)
    require(0.0 < grip_mass_kg <= 0.15, "grip_mass_kg out of range", grip_mass_kg)

    length = spec.length_m
    m_h, m_s, m_g = spec.head_mass_kg, shaft_mass_kg, grip_mass_kg
    total = m_h + m_s + m_g
    balance = (m_h * length + m_s * length / 2.0 + m_g * GRIP_LENGTH_M / 2.0) / total
    moi_grip = m_h * length**2 + m_s * length**2 / 3.0 + m_g * GRIP_LENGTH_M**2 / 3.0
    moi_shaft = (
        spec.moi_about_shaft_kg_m2
        + m_s * SHAFT_TUBE_RADIUS_M**2
        + m_g * GRIP_TUBE_RADIUS_M**2
    )

    result = ClubInertia(
        total_mass_kg=total,
        balance_point_m=balance,
        moi_about_grip_kg_m2=moi_grip,
        moi_about_shaft_kg_m2=moi_shaft,
    )
    ensure(0.0 < result.balance_point_m < length, "balance point on the shaft")
    ensure(result.moi_about_grip_kg_m2 > 0.0, "grip MOI positive")
    ensure(
        result.moi_about_shaft_kg_m2 > spec.moi_about_shaft_kg_m2,
        "shaft-axis MOI grows with components",
    )
    return result
