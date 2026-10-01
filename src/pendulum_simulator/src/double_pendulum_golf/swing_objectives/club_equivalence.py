"""Converting a real golf club into the point-mass-at-tip club this model uses.

:func:`double_pendulum_golf.physics.mass_matrix` treats segment 2 as a point
mass at the tip of the shaft. A real club is not that: a driver is about 0.31 kg
with its centre of mass roughly three quarters of the way down. Those two facts
have to be reconciled, and reconciling them **wrongly** is what produced the
incorrect conclusion recorded in #4785.

The mistake and the fix
-----------------------
The shipped preset lumped the real club's *mass* at the tip. That is the wrong
invariant. What governs the swing is the club's inertia about the wrist,

.. code-block:: text

    delta = I_com + m * r^2          (parallel axis)

and the arm/club coupling ``mu = m * L1 * r`` that appears in every centrifugal
and Coriolis term. Putting a driver's full 0.31 kg at the tip of a 1.10 m shaft
gives ``delta = 0.605`` against the real ``0.288`` — **2.1x too much** — and
doubles the coupling with it. Since the coupling is what drives the wrist *open*
and fights the release, doubling it forces the optimizer into heavy hub-torque
reversal, which stops the hands. That artifact was reported as a structural limit
of the model.

Matching the inertia instead gives an equivalent tip mass

.. code-block:: text

    me = delta_real / L2_model^2

which for a driver on a 1.10 m modelled shaft is **0.238 kg**, not 0.50. With
that correction the same model, optimizer and objective produce 50.8 m/s of
clubhead speed with 7.95 m/s of hand speed and a club/arm rate ratio of 3.18 —
all three inside the bands in
:mod:`double_pendulum_golf.swing_objectives.reference_kinematics`.

The equivalent mass is a *modelling quantity*, not a claim about how much a club
weighs. It is the mass that, placed at the tip, swings like the real club.

Club measurements follow the clubfitting convention of reporting inertia about
the butt of the grip; see `Jorgensen 1970
<https://doi.org/10.1119/1.1976419>`_ and `Cochran & Stobbs 1968
<https://archive.org/details/searchforperfect0000coch>`_ for the double-pendulum
context, and `MacKenzie & Sprigings 2009
<https://doi.org/10.1007/s12283-009-0020-9>`_ for club properties in a
forward-dynamics golfer.

Closes #4785.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RealClubSpec",
    "DRIVER_SPEC",
    "SEVEN_IRON_SPEC",
    "wrist_inertia",
    "equivalent_tip_mass",
]


@dataclass(frozen=True, slots=True)
class RealClubSpec:
    """Measured properties of an actual golf club.

    Attributes:
        name: Human-readable club name.
        mass_kg: Total club mass.
        length_m: Overall club length, butt to sole.
        com_m: Centre of mass distance from the butt of the grip. For a driver
            this sits near 76-80% of the length, because the head dominates.
        inertia_about_com_kgm2: Moment of inertia about the club's own centre of
            mass.
    """

    name: str
    mass_kg: float
    length_m: float
    com_m: float
    inertia_about_com_kgm2: float

    def __post_init__(self) -> None:
        """Validate that the club is physically buildable.

        Pre: none.
        Post: every field is finite and inside its physical range.
        """
        if not (self.mass_kg > 0.0 and np.isfinite(self.mass_kg)):
            raise ValueError(f"mass_kg must be positive, got {self.mass_kg}")
        if not (self.length_m > 0.0 and np.isfinite(self.length_m)):
            raise ValueError(f"length_m must be positive, got {self.length_m}")
        if not (0.0 <= self.com_m <= self.length_m):
            raise ValueError(
                f"com_m must lie on the club (0 to {self.length_m}), got {self.com_m}"
            )
        if not (
            self.inertia_about_com_kgm2 >= 0.0 and np.isfinite(self.inertia_about_com_kgm2)
        ):
            raise ValueError(
                f"inertia_about_com_kgm2 must be non-negative, "
                f"got {self.inertia_about_com_kgm2}"
            )

    @property
    def inertia_about_wrist_kgm2(self) -> float:
        """Inertia about the grip, by the parallel axis theorem."""
        return float(self.inertia_about_com_kgm2 + self.mass_kg * self.com_m**2)


#: Modern driver. Head ~0.20 kg at the tip, shaft ~0.06 kg, grip ~0.05 kg, which
#: puts the COM near 76% of length and gives ~0.29 kg*m^2 about the butt — the
#: figure clubfitters measure.
DRIVER_SPEC = RealClubSpec(
    name="Driver",
    mass_kg=0.310,
    length_m=1.143,
    com_m=0.867,
    inertia_about_com_kgm2=0.0551,
)

#: Modern 7-iron: heavier head, shorter shaft, COM slightly further down.
SEVEN_IRON_SPEC = RealClubSpec(
    name="7-iron",
    mass_kg=0.415,
    length_m=0.940,
    com_m=0.734,
    inertia_about_com_kgm2=0.0437,
)


def wrist_inertia(club: RealClubSpec) -> float:
    """Return the club's moment of inertia about the wrist, in kg*m^2.

    This is the quantity a point-mass-at-tip model has to reproduce, because it
    is what sets both the wrist-row mass-matrix term and the arm/club coupling.

    Args:
        club: Measured club properties.

    Returns:
        Inertia about the grip.

    Post: strictly positive.
    """
    return club.inertia_about_wrist_kgm2


def equivalent_tip_mass(club: RealClubSpec, shaft_length_m: float) -> float:
    """Return the tip mass that makes a point-mass club swing like a real one.

    A point mass ``me`` at distance ``shaft_length_m`` from the wrist has inertia
    ``me * shaft_length_m**2`` about that wrist. Setting that equal to the real
    club's wrist inertia gives the equivalent mass.

    This is deliberately **not** the club's actual mass. For a driver on a 1.10 m
    modelled shaft it is 0.238 kg against a real 0.310 kg; using the real mass
    would overstate the wrist inertia and the arm/club coupling by roughly 2.1x
    and drive the optimizer into the artifact described in #4785.

    Args:
        club: Measured club properties.
        shaft_length_m: Length of the modelled second link, wrist to clubhead.

    Returns:
        Equivalent point mass in kg, to be placed at the tip.

    Pre: ``shaft_length_m`` is positive and finite.
    Post: strictly positive and finite.
    """
    if not (shaft_length_m > 0.0 and np.isfinite(shaft_length_m)):
        raise ValueError(f"shaft_length_m must be positive, got {shaft_length_m}")
    return float(wrist_inertia(club) / shaft_length_m**2)
