"""Putter-ball impact: low-speed impulse with loft (epic #4125, H3).

Model
-----
The putter face is a plane tilted back from vertical by the effective
loft ``delta`` (static loft plus shaft lean; ~3 degrees for putters).
The head travels horizontally at ``v`` through a sub-millisecond
contact, so gravity and the turf reaction are ignored during contact
(documented assumption; contact ~0.5 ms, gravity changes speeds by
~0.005 m/s over that window).

Decompose along the face normal ``n = (cos d, sin d)`` and the
in-face tangential (up-the-face) direction ``t = (-sin d, cos d)``
in the vertical putt plane (x = putt line, y = up):

* **Normal**: 1-D impulse-momentum with coefficient of restitution
  ``e`` between a free head of mass ``M`` and ball of mass ``m``::

      v_ball_n = v cos d * (1 + e) * M / (M + m)

  Putter-face COR at putt speeds: 0.78 is the typical published value
  for a steel putter face (also the default in UpstreamDrift's
  ``putter_stroke.py`` and the green-contact default in its
  ``contact.rs``); milled/insert faces span roughly 0.73-0.82.

* **Tangential**: the face surface moves up-the-face relative to the
  ball at ``u = v sin d``. Sliding friction acts on the ball's contact
  point until the contact point matches the face surface speed
  (no-slip). With tangential impulse ``P``: ``v_t = P/m`` and
  ``omega = P r / I`` where ``I = (2/5) m r^2``, and the contact-point
  speed is ``v_t + omega r``; no-slip gives ``P = (2/7) m u``, so::

      v_ball_t = (2/7) u        omega r = (5/7) u   (backspin)

  This is the standard 2/7 rolling-cap for a struck sphere — the same
  factor used by ``swing_sim.impact`` (``SPHERE_ROLLING_CAP_FACTOR``).
  The spin direction is backspin: friction drags the ball's back
  surface upward, rotating the top of the ball toward the face.

Recomposing gives the launch speed, a launch angle slightly above the
effective loft, and the initial backspin — the "slide" state that the
skid phase of :mod:`.roll` starts from.

Sign conventions: ``spin_rad_s`` is about the transverse (left-
pointing) axis with **topspin positive**, so a freshly struck putt has
``spin_rad_s < 0`` (backspin) and pure roll is ``v = omega r`` with
``omega > 0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shared.python.contracts import ensure, require, require_finite
from shared.python.swing_sim.impact import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
)

__all__ = [
    "DEFAULT_PUTTER_COR",
    "MINIMAL_PUTTERS",
    "PuttLaunch",
    "PutterSpec",
    "clubhead_speed_from_backstroke",
    "strike",
]

#: Typical published putter-face COR at putt speeds (steel face; the
#: value UpstreamDrift uses for both its putter stroke model and its
#: green-contact default). Milled/insert faces span ~0.73-0.82.
DEFAULT_PUTTER_COR = 0.78

#: Standard gravity [m/s^2].
_GRAVITY_M_S2 = 9.80665

#: 2/7 tangential rolling cap for a struck sphere (see module docs).
_ROLLING_CAP = 2.0 / 7.0


@dataclass(frozen=True)
class PutterSpec:
    """Minimal putter description for the putting vertical.

    NOTE (H1 reconciliation, epic #4125): this is a deliberately
    minimal H3-local spec. The H1 club-library putters
    (``rate_of_closure.club.library``) carry the full geometry; UIs
    should build a ``PutterSpec`` from a library ``ClubSpec`` when one
    is available and fall back to :data:`MINIMAL_PUTTERS` otherwise.

    Attributes:
        name: Display name.
        head_mass_kg: Head mass [kg]; putters are typically 0.33-0.37.
        loft_deg: Static face loft [deg]; putters are typically 2-4.
        cor: Face coefficient of restitution at putt speeds (0-1).
    """

    name: str
    head_mass_kg: float
    loft_deg: float
    cor: float = DEFAULT_PUTTER_COR

    def __post_init__(self) -> None:
        require(bool(self.name.strip()), "putter name must be non-empty")
        require_finite(self.head_mass_kg, "head_mass_kg")
        require(
            0.1 <= self.head_mass_kg <= 1.0,
            "head mass must be plausible [kg]",
            self.head_mass_kg,
        )
        require_finite(self.loft_deg, "loft_deg")
        require(
            -2.0 <= self.loft_deg <= 10.0,
            "putter loft must be in [-2, 10] deg",
            self.loft_deg,
        )
        require_finite(self.cor, "cor")
        require(0.0 < self.cor < 1.0, "COR must be in (0, 1)", self.cor)


#: H3-local minimal putter specs — marked for reconciliation with the
#: H1 club-library putters (see :class:`PutterSpec` docstring). Head
#: masses and lofts are typical published catalogue values.
MINIMAL_PUTTERS: dict[str, PutterSpec] = {
    spec.name: spec
    for spec in (
        PutterSpec(name="Blade Putter", head_mass_kg=0.350, loft_deg=3.0),
        PutterSpec(name="Mallet Putter", head_mass_kg=0.360, loft_deg=3.0),
    )
}


@dataclass(frozen=True)
class PuttLaunch:
    """Ball state immediately after putter impact.

    Attributes:
        ball_speed_mps: Launch speed magnitude [m/s].
        launch_angle_deg: Launch angle above the horizontal [deg].
        horizontal_speed_mps: Speed component along the green [m/s] —
            the initial condition for the skid/roll model.
        spin_rad_s: Spin about the transverse axis [rad/s], topspin
            positive; a struck putt starts with backspin (negative).
        effective_loft_deg: Loft actually presented at impact [deg]
            (static loft + shaft lean).
    """

    ball_speed_mps: float
    launch_angle_deg: float
    horizontal_speed_mps: float
    spin_rad_s: float
    effective_loft_deg: float


def strike(
    putter: PutterSpec,
    clubhead_speed_mps: float,
    shaft_lean_deg: float = 0.0,
) -> PuttLaunch:
    """Solve the putter-ball impact (see module docstring derivation).

    Args:
        putter: Putter head description.
        clubhead_speed_mps: Horizontal head speed at impact [m/s];
            putts span roughly 0.2-4 m/s.
        shaft_lean_deg: Forward press (negative reduces effective
            loft) [deg].

    Returns:
        The post-impact :class:`PuttLaunch`.

    Raises:
        ValueError: If any input is out of its physical range.
    """
    require_finite(clubhead_speed_mps, "clubhead_speed_mps")
    require(
        0.0 < clubhead_speed_mps <= 10.0,
        "clubhead speed must be in (0, 10] m/s",
        clubhead_speed_mps,
    )
    require_finite(shaft_lean_deg, "shaft_lean_deg")
    require(
        abs(shaft_lean_deg) <= 10.0,
        "shaft lean must be within +/-10 deg",
        shaft_lean_deg,
    )
    effective_loft_deg = putter.loft_deg + shaft_lean_deg
    require(
        -2.0 <= effective_loft_deg <= 15.0,
        "effective loft must stay in [-2, 15] deg",
        effective_loft_deg,
    )
    delta = math.radians(effective_loft_deg)
    mass_ratio = putter.head_mass_kg / (putter.head_mass_kg + GOLF_BALL_MASS_KG)
    transfer = (1.0 + putter.cor) * mass_ratio

    v_normal = transfer * clubhead_speed_mps * math.cos(delta)
    u_tangential = clubhead_speed_mps * math.sin(delta)
    v_tangential = _ROLLING_CAP * u_tangential
    # Backspin: contact-surface speed (5/7)u, topspin-positive sign.
    spin_rad_s = -(1.0 - _ROLLING_CAP) * u_tangential / GOLF_BALL_RADIUS_M

    horizontal = v_normal * math.cos(delta) - v_tangential * math.sin(delta)
    vertical = v_normal * math.sin(delta) + v_tangential * math.cos(delta)
    ball_speed = math.hypot(horizontal, vertical)
    launch_angle_deg = math.degrees(math.atan2(vertical, horizontal))

    ensure(ball_speed > 0.0, "ball must leave the face", ball_speed)
    ensure(
        ball_speed <= 2.0 * clubhead_speed_mps,
        "smash factor is bounded by 2 (equal-mass elastic limit)",
        ball_speed / clubhead_speed_mps,
    )
    ensure(horizontal > 0.0, "putt must move toward the hole", horizontal)
    return PuttLaunch(
        ball_speed_mps=ball_speed,
        launch_angle_deg=launch_angle_deg,
        horizontal_speed_mps=horizontal,
        spin_rad_s=spin_rad_s,
        effective_loft_deg=effective_loft_deg,
    )


def clubhead_speed_from_backstroke(
    backstroke_m: float, putter_length_m: float = 0.889
) -> float:
    """Head speed at the bottom of a pendulum stroke.

    Proxy model (documented simplification): the putting stroke is a
    simple pendulum of length ``L`` released from a backstroke arc
    amplitude ``A``. Small-angle SHM gives the bottom-of-arc speed::

        v = A * omega_p = A * sqrt(g / L)

    Args:
        backstroke_m: Backstroke arc length [m]; typical putts use
            0.1-0.6 m.
        putter_length_m: Pendulum length [m]; default is the standard
            35 in putter (0.889 m).

    Returns:
        Clubhead speed at impact [m/s].

    Raises:
        ValueError: If inputs are out of range.
    """
    require_finite(backstroke_m, "backstroke_m")
    require(
        0.0 < backstroke_m <= 1.5,
        "backstroke must be in (0, 1.5] m",
        backstroke_m,
    )
    require_finite(putter_length_m, "putter_length_m")
    require(
        0.5 <= putter_length_m <= 1.5,
        "putter length must be in [0.5, 1.5] m",
        putter_length_m,
    )
    speed = backstroke_m * math.sqrt(_GRAVITY_M_S2 / putter_length_m)
    ensure(speed > 0.0, "pendulum speed must be positive", speed)
    return speed
