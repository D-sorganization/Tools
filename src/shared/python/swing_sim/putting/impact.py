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

2-D stroke extension (epic #4800, P1)
-------------------------------------
:func:`strike` accepts the full delivered stroke, reusing the
``swing_sim.impact`` sign conventions verbatim (AffineDrift frame:
``x`` = target line, ``y`` = up, ``z`` = right; see
:mod:`shared.python.swing_sim.impact.delivery`):

* Club path (+) = in-to-out: the head travels right of the aim line.
* Face angle (+) = open: the face normal points right of the aim line.
* Attack angle (+) = hitting up.
* Strike offset: toe (+) and high (+) in millimetres on the face.
* Aim (+) = start line aimed right of the target line. Face and path
  are measured **relative to the aim line**, so a perfect stroke aimed
  2 deg left starts exactly 2 deg left.

The solve factorizes (documented small-angle assumption — putting
deliveries stay within a few degrees, so vertical/horizontal
cross-coupling terms are second order):

* **Stroke plane** (vertical): the H3 model above, generalized to a
  head velocity climbing at the attack angle ``a``; the spin loft is
  ``delta - a`` and the solved launch is rotated back by ``a``. The
  ``a = 0`` limit reproduces the 1-D results bit-for-bit.
* **Horizontal split** (face vs path): the normal COR impulse launches
  the ball along the face azimuth while the tangential 2/7 rolling-cap
  impulse (``SPHERE_ROLLING_CAP_FACTOR``, same constant and derivation
  as ``swing_sim.impact.models``) drags it toward the path:

      start = aim + face + atan2((2/7) sin(path - face),
                                 transfer * cos(path - face))

  For small angles ``start ~= (1 - k) face + k path`` with
  ``k = (2/7) / transfer`` (face-dominant, the established
  launch-monitor split). The same tangential impulse leaves sidespin
  ``(5/7) u / r`` about the up axis: **positive = draw-side** for a
  right-handed player (ball turns left; arises when the path is right
  of the face). This matches the full-swing convention where positive
  spin-axis *tilt* is fade-side (a fade axis has a negative up
  component).
* **Off-center strike**: the head's effective mass along the contact
  normal drops per the scalar reduction documented in
  ``swing_sim.impact.models.RigidBodyImpactModel``::

      M_eff = 1 / (1/M + r^2 / I)

  with ``r`` the in-face offset magnitude and ``I`` the head MOI about
  its CG. ``head_moi_kg_m2`` is the explicit P3 hook: epic #4800 P3
  will supply mesh-derived MOI tensors via ``golf_club``; until then
  the documented catalogue default :data:`DEFAULT_PUTTER_MOI_KG_M2`
  applies. Gear-effect face twist also arrives with P3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shared.python.contracts import ensure, require, require_finite
from shared.python.swing_sim.impact import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
)

__all__ = [
    "DEFAULT_PUTTER_COR",
    "DEFAULT_PUTTER_MOI_KG_M2",
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

#: Typical putter-head MOI about the CG vertical (heel-toe) axis
#: [kg m^2]. Published catalogue values: blades ~3.8-4.5e-4, mallets
#: ~5.0-7.0e-4 (spec sheets quote 3800-7000 g cm^2). Mid value; the
#: explicit ``head_moi_kg_m2`` hook on :func:`strike` overrides it and
#: is filled by mesh-derived MOI in epic #4800 P3.
DEFAULT_PUTTER_MOI_KG_M2 = 4.5e-4

#: Standard gravity [m/s^2].
_GRAVITY_M_S2 = 9.80665

#: 2/7 tangential rolling cap for a struck sphere (see module docs).
_ROLLING_CAP = 2.0 / 7.0

_MM_TO_M = 1e-3


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
        start_azimuth_deg: Horizontal start direction [deg] relative to
            the target line, + = right (AffineDrift frame, matching the
            ``swing_sim.impact`` conventions). ``horizontal_speed_mps``
            points along this line. Defaults to 0.0 (the 1-D limit).
        sidespin_rad_s: Spin about the vertical (up) axis [rad/s];
            + = draw-side for a right-handed player (ball turns left of
            the start line; arises when the path is right of the face).
            Defaults to 0.0 (the 1-D limit).
    """

    ball_speed_mps: float
    launch_angle_deg: float
    horizontal_speed_mps: float
    spin_rad_s: float
    effective_loft_deg: float
    start_azimuth_deg: float = 0.0
    sidespin_rad_s: float = 0.0


def strike(
    putter: PutterSpec,
    clubhead_speed_mps: float,
    shaft_lean_deg: float = 0.0,
    *,
    aim_deg: float = 0.0,
    face_angle_deg: float = 0.0,
    path_angle_deg: float = 0.0,
    attack_angle_deg: float = 0.0,
    strike_offset_toe_mm: float = 0.0,
    strike_offset_high_mm: float = 0.0,
    head_moi_kg_m2: float | None = None,
) -> PuttLaunch:
    """Solve the putter-ball impact (see module docstring derivation).

    All new parameters default to the square, centered, level 1-D
    stroke, whose results are bit-identical to the pre-#4800 model.
    Sign conventions follow ``swing_sim.impact`` verbatim (see the
    module docstring's *2-D stroke extension* section).

    Args:
        putter: Putter head description.
        clubhead_speed_mps: Head speed at impact [m/s]; putts span
            roughly 0.2-4 m/s.
        shaft_lean_deg: Forward press (negative reduces effective
            loft) [deg].
        aim_deg: Start-line aim relative to the target line [deg];
            + = right. Face and path are measured off the aim line.
        face_angle_deg: Face angle at impact [deg]; + = open (right of
            the aim line).
        path_angle_deg: Putter path at impact [deg]; + = in-to-out
            (head travelling right of the aim line).
        attack_angle_deg: Attack angle [deg]; + = hitting up.
        strike_offset_toe_mm: Strike location toward the toe [mm].
        strike_offset_high_mm: Strike location up the face [mm].
        head_moi_kg_m2: Putter-head MOI about its CG [kg m^2] for the
            off-center effective-mass reduction. **P3 hook** (epic
            #4800): mesh-derived MOI plugs in here; ``None`` selects
            the catalogue default :data:`DEFAULT_PUTTER_MOI_KG_M2`.
            Centered strikes are unaffected by this value.

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
    for name, value, bound in (
        ("aim_deg", aim_deg, 45.0),
        ("face_angle_deg", face_angle_deg, 20.0),
        ("path_angle_deg", path_angle_deg, 20.0),
        ("attack_angle_deg", attack_angle_deg, 10.0),
        ("strike_offset_toe_mm", strike_offset_toe_mm, 40.0),
        ("strike_offset_high_mm", strike_offset_high_mm, 20.0),
    ):
        require_finite(value, name)
        require(abs(value) <= bound, f"{name} must be within +/-{bound}", value)
    if head_moi_kg_m2 is not None:
        require_finite(head_moi_kg_m2, "head_moi_kg_m2")
        require(
            1e-5 <= head_moi_kg_m2 <= 1e-2,
            "head MOI must be plausible [kg m^2]",
            head_moi_kg_m2,
        )
    effective_loft_deg = putter.loft_deg + shaft_lean_deg
    require(
        -2.0 <= effective_loft_deg <= 15.0,
        "effective loft must stay in [-2, 15] deg",
        effective_loft_deg,
    )

    # Off-center strike: scalar effective-mass reduction, the same
    # formula as swing_sim.impact.models.RigidBodyImpactModel.
    offset_r_m = math.hypot(strike_offset_toe_mm, strike_offset_high_mm) * _MM_TO_M
    if offset_r_m > 0.0:
        moi = DEFAULT_PUTTER_MOI_KG_M2 if head_moi_kg_m2 is None else head_moi_kg_m2
        head_mass_eff = 1.0 / (1.0 / putter.head_mass_kg + offset_r_m**2 / moi)
    else:
        head_mass_eff = putter.head_mass_kg

    delta = math.radians(effective_loft_deg)
    alpha = math.radians(attack_angle_deg)
    # Spin loft: face-normal-to-velocity angle in the stroke plane.
    beta = delta - alpha
    mass_ratio = head_mass_eff / (head_mass_eff + GOLF_BALL_MASS_KG)
    transfer = (1.0 + putter.cor) * mass_ratio

    # Stroke-plane solve (H3 model in the velocity-aligned frame).
    v_normal = transfer * clubhead_speed_mps * math.cos(beta)
    u_tangential = clubhead_speed_mps * math.sin(beta)
    v_tangential = _ROLLING_CAP * u_tangential
    # Backspin: contact-surface speed (5/7)u, topspin-positive sign.
    spin_rad_s = -(1.0 - _ROLLING_CAP) * u_tangential / GOLF_BALL_RADIUS_M
    along = v_normal * math.cos(beta) - v_tangential * math.sin(beta)
    lift = v_normal * math.sin(beta) + v_tangential * math.cos(beta)
    # Rotate by the attack angle back to horizontal/vertical.
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    horizontal = along * cos_a - lift * sin_a
    vertical = along * sin_a + lift * cos_a

    # Horizontal face-vs-path split (module docstring): normal impulse
    # along the face azimuth, 2/7 tangential impulse toward the path.
    face_to_path = math.radians(path_angle_deg - face_angle_deg)
    sin_fp = math.sin(face_to_path)
    cos_fp = math.cos(face_to_path)
    deflection_rad = math.atan2(_ROLLING_CAP * sin_fp, transfer * cos_fp)
    start_azimuth_deg = aim_deg + face_angle_deg + math.degrees(deflection_rad)
    # Face-to-path mismatch trims the normal impulse (cosine) while the
    # tangential impulse adds in quadrature; exactly 1.0 when square.
    scale = math.hypot(transfer * cos_fp, _ROLLING_CAP * sin_fp) / transfer
    horizontal *= scale
    vertical *= scale
    sidespin_rad_s = (
        (1.0 - _ROLLING_CAP) * clubhead_speed_mps * cos_a * sin_fp / GOLF_BALL_RADIUS_M
    )

    ball_speed = math.hypot(horizontal, vertical)
    launch_angle_deg = math.degrees(math.atan2(vertical, horizontal))

    ensure(ball_speed > 0.0, "ball must leave the face", ball_speed)
    ensure(
        ball_speed <= 2.0 * clubhead_speed_mps,
        "smash factor is bounded by 2 (equal-mass elastic limit)",
        ball_speed / clubhead_speed_mps,
    )
    ensure(horizontal > 0.0, "putt must move toward the hole", horizontal)
    ball_ke = 0.5 * GOLF_BALL_MASS_KG * ball_speed**2 + (
        0.5 * GOLF_BALL_MOMENT_OF_INERTIA_KG_M2 * (spin_rad_s**2 + sidespin_rad_s**2)
    )
    ensure(
        ball_ke <= 0.5 * putter.head_mass_kg * clubhead_speed_mps**2,
        "impact cannot create energy",
        ball_ke,
    )
    return PuttLaunch(
        ball_speed_mps=ball_speed,
        launch_angle_deg=launch_angle_deg,
        horizontal_speed_mps=horizontal,
        spin_rad_s=spin_rad_s,
        effective_loft_deg=effective_loft_deg,
        start_azimuth_deg=start_azimuth_deg,
        sidespin_rad_s=sidespin_rad_s,
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
