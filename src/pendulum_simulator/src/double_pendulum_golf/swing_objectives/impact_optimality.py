"""Why a speed-optimal downswing stops the hands, in closed form.

The objective comparison in epic #4766 converges to swings that brake the arms
to a standstill at impact. That is not a solver artifact — it is the exact
optimum of the model, and this module states why.

The result
----------
At impact the club is in line with the arms, so both angular rates drive the
clubhead along the same perpendicular:

.. code-block:: text

    v_head = (L1 + L2) * omega1  +  L2 * phidot

Maximising ``v_head`` subject to a fixed kinetic energy
``0.5 * qdot^T M qdot`` is a linear objective on a quadratic form, so the
optimum lies along ``qdot* ∝ M^-1 c`` with ``c = (L1 + L2, L2)``. Taking the arm
component of that solution and simplifying gives

.. code-block:: text

    omega1*  ∝  L1 * [ I2 - m2 * r2 * (L2 - r2) ]

which is what :func:`impact_hand_speed_coefficient` returns.

Three regimes follow:

* **Point-mass clubhead** (``r2 = L2``, ``I2 = 0``) — the model shipped in
  :mod:`double_pendulum_golf.physics` — the bracket is **identically zero for
  every parameter value**. All of segment 2's mass sits at the tip, so the
  club's kinetic energy *is* ``0.5 * me * v_head**2`` and any arm motion is
  energy that never reached the clubhead. The optimizer stops the hands because
  stopping them is optimal.
* **A real driver** (``m2 = 0.31 kg``, ``r2 = 0.89 m``, ``L2 = 1.143 m``,
  ``I2 = 0.043 kg m^2``) gives a **negative** bracket: the optimum wants the
  hands moving *backward* through impact.
* A **forward** optimum needs ``r2`` near 1.0 m or ``I2`` above 0.2 kg m^2 —
  roughly five times a real driver's. No club anyone swings lands there.

The practical consequence is that **distributed club inertia is not the fix**.
Real golfers keep 6-9 m/s of hand speed at impact
(`Nesbit 2005 <https://www.jssm.org/jssm-04-499.xml.xml>`_) not because it is
speed-optimal but because their actuation is limited: torque capacity falls with
joint angular velocity (`Hill 1938 <https://doi.org/10.1098/rspb.1938.0050>`_),
and the arms are attached to a torso that cannot be stopped on demand. The fix
therefore belongs in :mod:`double_pendulum_golf.swing_objectives.actuation`.

Related reading: `Jorgensen 1970
<https://doi.org/10.1119/1.1976433>`_ for the canonical double-pendulum golf
model, and `Miura 2001 <https://doi.org/10.1007/BF02844309>`_ for the
"parametric acceleration" inward hand pull, which is the real, much milder
version of the effect this model exaggerates.

Closes #4776.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from double_pendulum_golf.physics import PendulumParams, mass_matrix

__all__ = [
    "EnergyOptimalRates",
    "impact_hand_speed_coefficient",
    "optimal_hand_speed_sign",
    "energy_optimal_rates",
]

FloatArray = npt.NDArray[np.float64]

#: Below this magnitude the coefficient is reported as an exact zero. The
#: point-mass cancellation is algebraic, so anything larger is a real effect.
_ZERO_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class EnergyOptimalRates:
    """The rate split that maximises clubhead speed for a fixed kinetic energy.

    Attributes:
        arm_rate_rad_s: Arm angular velocity at impact.
        uncock_rate_rad_s: Wrist uncocking rate at impact.
        hand_speed_m_s: Resulting hand speed, ``L1 * arm_rate``.
        clubhead_speed_m_s: Resulting clubhead speed.
        kinetic_energy_j: The energy budget these rates carry.
    """

    arm_rate_rad_s: float
    uncock_rate_rad_s: float
    hand_speed_m_s: float
    clubhead_speed_m_s: float
    kinetic_energy_j: float


def _club_properties(
    params: PendulumParams,
    club_mass_kg: float | None,
    club_com_m: float | None,
    club_inertia_kgm2: float | None,
) -> tuple[float, float, float]:
    """Resolve club mass, COM distance and inertia, defaulting to the shipped model.

    The shipped model treats segment 2 as a point mass at the tip, so the
    defaults are ``r2 = L2`` and ``I2 = 0``.

    Pre: any supplied override is non-negative, and the COM lies on the club.
    Post: returns ``(mass, com, inertia)`` with mass positive.
    """
    mass = params.m2 + params.mClub if club_mass_kg is None else club_mass_kg
    com = params.L2 if club_com_m is None else club_com_m
    inertia = 0.0 if club_inertia_kgm2 is None else club_inertia_kgm2

    if not (mass > 0.0 and np.isfinite(mass)):
        raise ValueError(f"club_mass_kg must be positive, got {mass}")
    if not (com >= 0.0 and np.isfinite(com)):
        raise ValueError(f"club_com_m must be non-negative, got {com}")
    if not (inertia >= 0.0 and np.isfinite(inertia)):
        raise ValueError(f"club_inertia_kgm2 must be non-negative, got {inertia}")
    return mass, com, inertia


def impact_hand_speed_coefficient(
    params: PendulumParams,
    club_mass_kg: float | None = None,
    club_com_m: float | None = None,
    club_inertia_kgm2: float | None = None,
) -> float:
    """Return ``L1 * [ I2 - m2 * r2 * (L2 - r2) ]``, in kg*m^3.

    The sign of this quantity is the sign of the energy-optimal hand speed at
    impact; its zero is the reason a point-mass-clubhead optimizer stops the
    hands. See the module docstring for the derivation.

    Args:
        params: Double pendulum parameters. Supplies ``L1`` and ``L2``, and the
            default point-mass club properties.
        club_mass_kg: Club mass override. Defaults to ``m2 + mClub``.
        club_com_m: Club centre-of-mass distance from the wrist. Defaults to
            ``L2``, the shipped point-mass-at-the-tip assumption.
        club_inertia_kgm2: Club inertia about its own COM. Defaults to zero.

    Returns:
        The coefficient. Exactly zero for the shipped model.

    Pre: overrides are non-negative and finite.
    Post: the result is finite.
    """
    mass, com, inertia = _club_properties(params, club_mass_kg, club_com_m, club_inertia_kgm2)
    bracket = inertia - mass * com * (params.L2 - com)
    return float(params.L1 * bracket)


def optimal_hand_speed_sign(
    params: PendulumParams,
    club_mass_kg: float | None = None,
    club_com_m: float | None = None,
    club_inertia_kgm2: float | None = None,
) -> str:
    """Name the regime the coefficient puts this club in.

    Args:
        params: Double pendulum parameters.
        club_mass_kg: Club mass override.
        club_com_m: Club centre-of-mass distance override.
        club_inertia_kgm2: Club inertia override.

    Returns:
        ``"stopped"``, ``"backward"`` or ``"forward"`` — what the speed-optimal
        swing wants the hands doing at impact.
    """
    coefficient = impact_hand_speed_coefficient(
        params, club_mass_kg, club_com_m, club_inertia_kgm2
    )
    if abs(coefficient) <= _ZERO_TOLERANCE:
        return "stopped"
    return "forward" if coefficient > 0.0 else "backward"


def energy_optimal_rates(
    params: PendulumParams, kinetic_energy_j: float
) -> EnergyOptimalRates:
    """Compute the clubhead-speed-optimal rate split for a kinetic energy budget.

    This is the theoretical ceiling the collocation optimizer converges toward:
    the fastest clubhead an impact posture can produce with a given amount of
    kinetic energy in the system, ignoring how the golfer got there.

    Args:
        params: Double pendulum parameters.
        kinetic_energy_j: Total system kinetic energy at impact, in joules.

    Returns:
        The optimal rates and the speeds they produce.

    Pre: ``kinetic_energy_j`` is positive and finite.
    Post: the returned rates carry exactly ``kinetic_energy_j``.
    """
    if not (kinetic_energy_j > 0.0 and np.isfinite(kinetic_energy_j)):
        raise ValueError(f"kinetic_energy_j must be positive, got {kinetic_energy_j}")

    # Impact posture: club in line with the arms.
    mass = mass_matrix(0.0, params)
    gradient = np.array([params.L1 + params.L2, params.L2], dtype=np.float64)
    direction = np.linalg.solve(mass, gradient)

    scale = np.sqrt(2.0 * kinetic_energy_j / (direction @ mass @ direction))
    arm_rate, uncock_rate = direction * scale
    return EnergyOptimalRates(
        arm_rate_rad_s=float(arm_rate),
        uncock_rate_rad_s=float(uncock_rate),
        hand_speed_m_s=float(params.L1 * arm_rate),
        clubhead_speed_m_s=float(gradient @ np.array([arm_rate, uncock_rate])),
        kinetic_energy_j=float(kinetic_energy_j),
    )
