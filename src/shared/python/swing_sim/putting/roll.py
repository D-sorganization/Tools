"""Skid -> pure-roll model with stimpmeter green speed (#4125, H3).

Phase 1 — skid (first principles)
---------------------------------
A struck putt leaves the face sliding: its forward speed ``v0``
exceeds the surface speed ``omega0 * r`` of its (back)spin. Kinetic
friction ``mu_k`` acts backward at the contact point, decelerating the
ball and — through the torque ``mu_k m g r`` about the center, with
``I = (2/5) m r^2`` — spinning it up toward rolling::

    dv/dt     = -mu_k g                 =>  v(t)     = v0 - mu_k g t
    domega/dt = +(5/2) mu_k g / r       =>  omega(t) = omega0 + (5/2) mu_k g t / r

Pure roll starts when the contact point stops slipping, ``v = omega r``::

    t_skid = (v0 - omega0 r) / ((7/2) mu_k g)
    v_roll = v(t_skid) = (5 v0 + 2 omega0 r) / 7

(the classic 5/7 result for a sliding ball with no initial spin), and
the skid distance is ``v0 t_skid - mu_k g t_skid^2 / 2``.

The sliding friction coefficient is a turf property independent of
green speed; 0.40 is the typical published grass sliding value (also
the green-contact friction default in UpstreamDrift's ``contact.rs``).

Phase 2 — pure roll and the stimpmeter
--------------------------------------
A rolling ball decelerates by rolling resistance, modeled as a
constant deceleration ``a = mu_r g`` (small-angle turf deformation
drag), so a ball rolling at ``v`` stops after ``v^2 / (2 mu_r g)``.

``mu_r`` is parameterized by the **stimpmeter**, whose geometry is
openly documented by the USGA: a 36 in (0.9144 m) ramp with a ball
notch 30 in (0.762 m) from the lower end; the ball releases when the
ramp reaches ~20 degrees. Rolling down the ramp, energy balance gives
the release speed::

    m g L sin(20 deg) = (1/2) m v^2 (1 + I / (m r_c^2))

where ``r_c`` is the contact radius. The ball rides the edges of the
ramp's V-groove, so ``r_c < r`` and the effective inertia exceeds the
free-rolling 2/5. With the groove's contact radius at ~0.87 r
(consistent with the ~20 mm groove width), this yields the widely
quoted release speed of ~6.0 ft/s::

    v_release = sqrt(2 g L sin20 / (1 + (2/5) / 0.87^2)) ~= 1.83 m/s

A green "stimps" ``S`` feet when that release speed rolls out ``S``
feet, which inverts to the rolling coefficient::

    mu_r = v_release^2 / (2 g S)

Sanity: stimp 10 (3.048 m) gives ``mu_r ~= 0.056``, inside the
published 0.05-0.07 band for tournament greens. The stimp -> mu_r ->
roll-out chain round-trips exactly (test-enforced).

Credit: parameterizing green speed as a stimp-derived friction number
follows UpstreamDrift's ``turf_properties.py`` concept; the derivation
here is redone from the USGA geometry (their constant assumed a
different release speed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shared.python.contracts import ensure, require, require_finite

__all__ = [
    "DEFAULT_SLIDING_MU",
    "GRAVITY_M_S2",
    "STIMP_RELEASE_SPEED_MPS",
    "SkidSolution",
    "roll_out_distance",
    "roll_time_s",
    "rolling_mu_to_stimp",
    "solve_skid",
    "stimp_to_rolling_mu",
]

#: Standard gravity [m/s^2].
GRAVITY_M_S2 = 9.80665

#: Typical published grass sliding friction (also UpstreamDrift's
#: green-contact default in ``contact.rs``).
DEFAULT_SLIDING_MU = 0.40

#: Meters per foot.
_FOOT_M = 0.3048

#: Stimpmeter ramp travel from the ball notch to the lip [m] (30 in).
_STIMP_RAMP_TRAVEL_M = 0.762

#: Stimpmeter release angle [rad] (~20 deg, USGA documented).
_STIMP_RELEASE_ANGLE_RAD = math.radians(20.0)

#: V-groove contact radius as a fraction of the ball radius —
#: consistent with the ~20 mm groove width; see module derivation.
_GROOVE_CONTACT_FRACTION = 0.87

#: Stimpmeter release speed [m/s], derived in the module docstring
#: (~1.83 m/s = 6.0 ft/s, the widely quoted value).
STIMP_RELEASE_SPEED_MPS = math.sqrt(
    2.0
    * GRAVITY_M_S2
    * _STIMP_RAMP_TRAVEL_M
    * math.sin(_STIMP_RELEASE_ANGLE_RAD)
    / (1.0 + (2.0 / 5.0) / _GROOVE_CONTACT_FRACTION**2)
)


def stimp_to_rolling_mu(stimp_ft: float) -> float:
    """Rolling-resistance coefficient for a green speed.

    ``mu_r = v_release^2 / (2 g S)`` — see the module derivation.

    Args:
        stimp_ft: Stimpmeter reading [feet]; greens span ~4-16.

    Returns:
        Dimensionless rolling deceleration coefficient.

    Raises:
        ValueError: If the stimp reading is out of range.
    """
    require_finite(stimp_ft, "stimp_ft")
    require(3.0 <= stimp_ft <= 16.0, "stimp must be in [3, 16] ft", stimp_ft)
    mu = STIMP_RELEASE_SPEED_MPS**2 / (2.0 * GRAVITY_M_S2 * stimp_ft * _FOOT_M)
    ensure(0.0 < mu < 0.2, "rolling mu must be physically small", mu)
    return mu


def rolling_mu_to_stimp(mu_r: float) -> float:
    """Inverse of :func:`stimp_to_rolling_mu` (round-trip exact).

    Args:
        mu_r: Rolling-resistance coefficient.

    Returns:
        Stimpmeter reading [feet].

    Raises:
        ValueError: If the coefficient is out of range.
    """
    require_finite(mu_r, "mu_r")
    require(0.0 < mu_r < 0.2, "rolling mu must be in (0, 0.2)", mu_r)
    return STIMP_RELEASE_SPEED_MPS**2 / (2.0 * GRAVITY_M_S2 * mu_r * _FOOT_M)


@dataclass(frozen=True)
class SkidSolution:
    """Closed-form skid phase (flat green; see module derivation).

    Attributes:
        duration_s: Time until pure roll begins [s].
        distance_m: Ground covered while skidding [m].
        exit_speed_mps: Speed when pure roll begins,
            ``(5 v0 + 2 omega0 r) / 7`` [m/s].
    """

    duration_s: float
    distance_m: float
    exit_speed_mps: float


def solve_skid(
    speed_mps: float,
    spin_rad_s: float,
    ball_radius_m: float,
    mu_slide: float = DEFAULT_SLIDING_MU,
) -> SkidSolution:
    """Closed-form flat-green skid phase.

    Args:
        speed_mps: Initial forward speed ``v0`` [m/s].
        spin_rad_s: Initial spin ``omega0`` [rad/s], topspin positive
            (a struck putt starts with backspin, negative).
        ball_radius_m: Ball radius [m].
        mu_slide: Sliding friction coefficient.

    Returns:
        The :class:`SkidSolution`; zero-duration when the ball is
        already rolling (``v0 <= omega0 r``).

    Raises:
        ValueError: If inputs are out of range.
    """
    require_finite(speed_mps, "speed_mps")
    require(speed_mps > 0.0, "speed must be positive", speed_mps)
    require_finite(spin_rad_s, "spin_rad_s")
    require_finite(ball_radius_m, "ball_radius_m")
    require(
        0.01 <= ball_radius_m <= 0.05,
        "ball radius must be plausible [m]",
        ball_radius_m,
    )
    require_finite(mu_slide, "mu_slide")
    require(0.0 < mu_slide <= 1.5, "mu_slide must be in (0, 1.5]", mu_slide)

    surface_speed = spin_rad_s * ball_radius_m
    if speed_mps <= surface_speed:
        return SkidSolution(0.0, 0.0, speed_mps)
    duration = (speed_mps - surface_speed) / (3.5 * mu_slide * GRAVITY_M_S2)
    distance = speed_mps * duration - 0.5 * mu_slide * GRAVITY_M_S2 * duration**2
    exit_speed = (5.0 * speed_mps + 2.0 * surface_speed) / 7.0
    ensure(exit_speed > 0.0, "roll must start moving forward", exit_speed)
    ensure(exit_speed <= speed_mps, "skid cannot speed the ball up", exit_speed)
    ensure(distance >= 0.0, "skid distance is non-negative", distance)
    return SkidSolution(
        duration_s=duration, distance_m=distance, exit_speed_mps=exit_speed
    )


def roll_out_distance(speed_mps: float, mu_roll: float) -> float:
    """Flat-green pure-roll stopping distance ``v^2 / (2 mu_r g)``.

    Args:
        speed_mps: Rolling speed [m/s].
        mu_roll: Rolling-resistance coefficient.

    Returns:
        Distance to rest [m].

    Raises:
        ValueError: If inputs are out of range.
    """
    require_finite(speed_mps, "speed_mps")
    require(speed_mps >= 0.0, "speed must be non-negative", speed_mps)
    require_finite(mu_roll, "mu_roll")
    require(0.0 < mu_roll < 0.2, "mu_roll must be in (0, 0.2)", mu_roll)
    return speed_mps**2 / (2.0 * mu_roll * GRAVITY_M_S2)


def roll_time_s(speed_mps: float, mu_roll: float) -> float:
    """Flat-green pure-roll time to rest ``v / (mu_r g)``.

    Args:
        speed_mps: Rolling speed [m/s].
        mu_roll: Rolling-resistance coefficient.

    Returns:
        Time to rest [s].

    Raises:
        ValueError: If inputs are out of range.
    """
    require_finite(speed_mps, "speed_mps")
    require(speed_mps >= 0.0, "speed must be non-negative", speed_mps)
    require_finite(mu_roll, "mu_roll")
    require(0.0 < mu_roll < 0.2, "mu_roll must be in (0, 0.2)", mu_roll)
    return speed_mps / (mu_roll * GRAVITY_M_S2)
