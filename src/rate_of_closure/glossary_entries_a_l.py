"""Glossary entry data (apex through lie_angle) — see :mod:`rate_of_closure.glossary`.

Split from the glossary module to honor the 500-LOC file budget; the
public surface (``GLOSSARY``, ``FIELD_TO_TERM``, ``search_terms``)
lives in :mod:`rate_of_closure.glossary`, which merges these dicts and
validates them.
"""

from __future__ import annotations

from ._contracts import require
from .glossary_types import GlossaryEntry

__all__ = ["ENTRIES"]


def _entry(term: str, definition: str) -> GlossaryEntry:
    require(bool(term.strip()), "glossary term must be non-empty", term)
    require(
        len(definition.strip()) >= 60,
        "glossary definitions must be substantive",
        term,
    )
    return GlossaryEntry(term=term, definition=definition)


ENTRIES: dict[str, GlossaryEntry] = {
    "apex": _entry(
        "Apex Height",
        "The peak height of the ball's trajectory — the point where lift "
        "and gravity momentarily balance vertical motion. Typical driver "
        "apex is 25-40 m (launch-monitor norms; swing_sim.flight metrics).",
    ),
    "attack_angle": _entry(
        "Attack Angle (AoA)",
        "The vertical angle of the clubhead's velocity at impact: positive "
        "= hitting up on the ball. One of the launch-monitor delivery "
        "parameters (AffineDrift Launch Monitor Technology Review).",
    ),
    "ball_speed": _entry(
        "Ball Speed",
        "The speed of the ball immediately after impact, set by the "
        "effective-mass momentum exchange of the COR impulse model "
        "(swing_sim.impact). Divided by clubhead speed it gives the smash "
        "factor.",
    ),
    "bulge": _entry(
        "Bulge",
        "The horizontal (heel-toe) curvature of a wood's face. It starts "
        "toe strikes pointing further right so the gear-effect draw spin "
        "curves the ball back toward the target (club-design literature; "
        "rate_of_closure.club face model).",
    ),
    "carry": _entry(
        "Carry Distance",
        "The horizontal distance from launch to the first ground contact "
        "of the integrated trajectory — no roll-out included "
        "(swing_sim.flight terminal ground event).",
    ),
    "ccv": _entry(
        "Club Closure Velocity (CCV)",
        "The rate the face normal sweeps horizontally (closes), in deg/s: "
        "CCV = HTV sin(lie) + SPV cos(lie). Cheetham 2014 tour driver "
        "data puts the mean near 2,100 deg/s.",
    ),
    "cg_depth": _entry(
        "CG Depth",
        "How far the clubhead's center of gravity sits behind the face "
        "plane. A deeper CG lengthens the recoil lever arm of an "
        "off-center impulse and therefore strengthens gear-effect spin "
        "(swing_sim.impact.gear_effect derivation).",
    ),
    "closure_rate": _entry(
        "Closure Rate",
        "How fast the face angle changes as the club approaches impact — "
        "reported as CCV (deg/s), per foot of travel (deg/ft), per inch, "
        "or per millisecond. The speed-invariant deg/ft form equals "
        "1 / R_ISA (AffineDrift closure-rate derivation).",
    ),
    "club_path": _entry(
        "Club Path",
        "The horizontal direction of the clubhead's velocity at impact, "
        "relative to the target line: positive = in-to-out (right of "
        "target for a right-handed player). Standard launch-monitor sign "
        "convention (AffineDrift 02-parameters).",
    ),
    "contact_duration": _entry(
        "Contact Duration",
        "The time the ball stays compressed on the face — about 450 "
        "microseconds for a driver. The face keeps rotating the whole "
        "time, so the face the ball leaves is not the face it met "
        "(impact literature; Cheetham dossier).",
    ),
    "cor": _entry(
        "Coefficient of Restitution (COR)",
        "The ratio of separation speed to approach speed along the impact "
        "normal (0 = perfectly plastic, 1 = perfectly elastic). Modern "
        "driver faces are capped near 0.83 by the rules; it scales the "
        "(1 + e) factor in the impulse solve (swing_sim.impact.models).",
    ),
    "coriolis": _entry(
        "Coriolis / Centripetal Terms",
        "The velocity-dependent generalized forces C(θ, ω) in the "
        "pendulum equations of motion, arising from the rotating links: "
        "products like ω1·ω2 and ω1² multiplied by -m2·l1·lc2·sin(θ2) "
        "(swing_sim.reference.coriolis_vector).",
    ),
    "d_plane": _entry(
        "D-Plane",
        "The plane spanned by the club-path vector and the delivered face "
        "normal. The ball launches close to the face normal and spins "
        "about the D-plane's normal, so the face-minus-path difference "
        "sets the spin-axis tilt (Jorgensen, The Physics of Golf; "
        "TrackMan D-plane literature).",
    ),
    "damping": _entry(
        "Damping",
        "Viscous joint torques proportional to angular velocity "
        "(d1·ω1, d2·ω2) that drain energy from the pendulum swing model — "
        "the model's stand-in for soft-tissue and grip losses "
        "(swing_sim.reference.damping_vector).",
    ),
    "dispersion_ellipse": _entry(
        "2σ Dispersion Ellipse",
        "The ellipse covering roughly 95% of simulated landing points, "
        "built from the eigen-decomposition of the carry/lateral "
        "covariance matrix scaled to two standard deviations "
        "(swing_sim.variation.analysis).",
    ),
    "double_pendulum": _entry(
        "Double Pendulum Swing Model",
        "The classic two-link golf swing model — arms and club as two "
        "rigid links in an inclined plane, driven by gravity and released "
        "torques. Its equations of motion come from the Lagrangian: mass "
        "matrix, Coriolis, gravity, and damping terms "
        "(swing_sim / rust swing-core).",
    ),
    "drag": _entry(
        "Drag Coefficient (Cd)",
        "The dimensionless coefficient in the aerodynamic drag force "
        "F = ½ρACd·v², opposing the ball's motion through the air. The "
        "literature flight models differ mainly in how Cd and Cl depend "
        "on spin ratio (swing_sim.flight.models).",
    ),
    "dynamic_loft": _entry(
        "Dynamic Loft",
        "The vertical angle of the delivered face normal at impact — the "
        "club's static loft plus shaft lean, wrist action, and the loft "
        "gained while the face rotates during contact (launch-monitor "
        "delivery parameter; AffineDrift conventions).",
    ),
    "effective_mass": _entry(
        "Effective Mass",
        "The reduced club mass the ball actually feels in an off-center "
        "impact: 1/m_eff = 1/m + (r x n)^T I^-1 (r x n), where r is the "
        "CG-to-contact lever and n the face normal — rotation recoil "
        "eats part of the impulse (swing_sim.impact.models derivation).",
    ),
    "face_angle": _entry(
        "Face Angle",
        "The horizontal direction of the delivered face normal relative "
        "to the target line: positive = open (pointing right of target). "
        "The dominant contributor to launch azimuth (launch-monitor "
        "conventions; AffineDrift 02-parameters).",
    ),
    "flight_time": _entry(
        "Flight Time",
        "Total time aloft, from launch to the terminal ground event of "
        "the flight integration — typically 5-7 s for a driver "
        "(swing_sim.flight metrics).",
    ),
    "friction_spin_cap": _entry(
        "2/7 Friction Spin Cap",
        "The rolling-without-slip limit on the tangential friction "
        "impulse for a uniform solid sphere: J_f = min(μJ, (2/7)·m·v_t). "
        "Beyond it the contact point has stopped sliding, so friction "
        "can add no more spin (Cross 2002, Am. J. Phys. 70, 1093; "
        "swing_sim.impact.models).",
    ),
    "gear_effect": _entry(
        "Gear Effect",
        "Spin created when an off-center impulse makes the head recoil in "
        "rotation and the face surface sweeps under the ball like a gear "
        "tooth: toe hits gain draw-side spin, high hits lose backspin. "
        "Derived from the head's I^-1 (r x J n) recoil and Coulomb "
        "friction (swing_sim.impact.gear_effect).",
    ),
    "geometric_center": _entry(
        "Geometric Center (GC)",
        "The reference point launch monitors track on the clubhead — the "
        "center of the head envelope, within ~6 mm of the CG for a "
        "driver. The ball responds to the impact point, not the GC "
        "(AffineDrift Launch Monitor Technology Review).",
    ),
    "htv": _entry(
        "Horizontal Turning Velocity (HTV)",
        "The clubhead's angular velocity about the shaft axis — the "
        "closing/release component of the swing. Cheetham 2014 tour "
        "driver data: 1,307 ± 304 deg/s (range 652-2,432, n = 94).",
    ),
    "impulse_momentum": _entry(
        "Impulse-Momentum Impact Model",
        "The rigid-body collision model: a normal impulse "
        "J = (1 + e)·m_eff·v_rel exchanged over the ~450 µs contact sets "
        "ball speed, with COR e and effective mass m_eff; friction "
        "supplies the tangential (spin) impulse "
        "(swing_sim.impact.models).",
    ),
    "landing_angle": _entry(
        "Landing Angle",
        "The descent angle below horizontal at the terminal ground event. "
        "Steeper landings stop faster; the driver band is roughly "
        "35-45 deg (swing_sim.flight metrics; launch-monitor norms).",
    ),
    "lateral_offset": _entry(
        "Lateral Landing Offset",
        "The sideways distance from the target line at landing (+ = "
        "right of target): the integrated effect of launch azimuth plus "
        "the curvature from spin-axis tilt — the way launch monitors "
        "report carry offline (swing_sim.flight metrics).",
    ),
    "launch_angle": _entry(
        "Launch Angle",
        "The vertical angle of the ball's initial velocity above the "
        "ground plane — the D-plane compromise between dynamic loft and "
        "attack angle, typically 10-16 deg for a driver "
        "(launch-monitor conventions).",
    ),
    "launch_azimuth": _entry(
        "Launch Azimuth",
        "The horizontal direction of the ball's initial velocity relative "
        "to the target line (+ = right). Dominated by the delivered face "
        "angle with a smaller club-path contribution (D-plane "
        "literature).",
    ),
}
