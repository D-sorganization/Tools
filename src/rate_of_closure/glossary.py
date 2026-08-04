"""Glossary of every technical term used across the app (#4120 V4).

Single source of truth for the Glossary tab (PyQt6) and the Glossary
section of the web clone (``web/src/model/glossary.ts`` mirrors this
dict key-for-key; the vitest parity test pins the key list). Every
explanation panel links here, and fields that map onto one term
pre-select it via :data:`FIELD_TO_TERM`.

Definitions are 1-3 sentences each and carry their source inline —
the AffineDrift Launch Monitor Technology Review, the Cheetham 2014
closure-rate dossier, ``swing_sim`` module derivations (impact /
flight / variation), and the standard golf-physics literature
(Jorgensen, Penner, TrackMan D-plane material).
"""

from __future__ import annotations

from dataclasses import dataclass

from ._contracts import ensure, require

__all__ = ["FIELD_TO_TERM", "GLOSSARY", "GlossaryEntry", "search_terms"]


@dataclass(frozen=True)
class GlossaryEntry:
    """One glossary term.

    Attributes:
        term: Title Case display name.
        definition: 1-3 sentence sourced definition.
    """

    term: str
    definition: str


def _entry(term: str, definition: str) -> GlossaryEntry:
    require(bool(term.strip()), "glossary term must be non-empty", term)
    require(
        len(definition.strip()) >= 60,
        "glossary definitions must be substantive",
        term,
    )
    return GlossaryEntry(term=term, definition=definition)


#: Every term used across the app, keyed snake_case. Sorted by key.
GLOSSARY: dict[str, GlossaryEntry] = {
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
    "lever_arm": _entry(
        "Lever Arm",
        "The vector r from the reference point (GC or CG) to the struck "
        "point. It converts rotation into extra point velocity (ω × r) "
        "in the closure model and impulse into recoil torque (r × Jn) in "
        "the impact model.",
    ),
    "lie_angle": _entry(
        "Lie Angle",
        "The angle between the shaft and the ground plane at impact. It "
        "sets how the shaft-axis (HTV) and swing-plane (SPV) rotation "
        "rates combine into face closure: CCV = HTV sin(lie) + SPV "
        "cos(lie) (Cheetham 2014 reconciliation).",
    ),
    "lift": _entry(
        "Lift Coefficient (Cl)",
        "The dimensionless coefficient of the aerodynamic force "
        "perpendicular to the ball's motion, generated by backspin (the "
        "Magnus effect). Literature models express Cl as a function of "
        "spin ratio, capped at a physical maximum "
        "(swing_sim.flight.models).",
    ),
    "magnus_force": _entry(
        "Magnus Force",
        "The aerodynamic force on a spinning ball, perpendicular to both "
        "the velocity and the spin axis: backspin lifts the ball, a "
        "tilted spin axis curves it sideways. It enters the flight EOM "
        "through the lift term (swing_sim.flight; Penner 2003).",
    ),
    "mass_matrix": _entry(
        "Mass Matrix",
        "The configuration-dependent 2x2 (or 3x3) inertia matrix M(θ) of "
        "the pendulum equations M(θ)·α + C(θ, ω) + G(θ) + D(ω) = 0; its "
        "off-diagonal terms couple the links through the elbow/wrist "
        "angle (swing_sim.reference.mass_matrix).",
    ),
    "moi": _entry(
        "Moment of Inertia (MOI)",
        "A body's resistance to angular acceleration about an axis. The "
        "clubhead MOI (scalar or full 3x3 tensor) sets how much an "
        "off-center impulse twists the head instead of launching the "
        "ball — the club-side term of the effective mass "
        "(swing_sim.impact.models).",
    ),
    "moi_tensor": _entry(
        "MOI Tensor",
        "The full 3x3 inertia tensor I of the clubhead. The exact "
        "off-center effective mass uses the triple-product form "
        "(r x n)^T I^-1 (r x n); a diagonal I·eye(3) reproduces the "
        "scalar-MOI fallback 1/m + |r|²/I exactly "
        "(swing_sim.impact.models derivation).",
    ),
    "monte_carlo": _entry(
        "Monte Carlo Simulation",
        "Running the simulation many times with randomized inputs drawn "
        "from per-variable noise distributions, then reading dispersion "
        "statistics off the output sample. The Variation tab's seeded "
        "engine makes runs exactly reproducible "
        "(swing_sim.variation.engine).",
    ),
    "noise_spec": _entry(
        "Noise Specification (NoiseSpec)",
        "The per-variable description of how an input varies in a "
        "variation study: distribution family (normal, uniform, or "
        "triangular), additive scale, and optional clip truncation "
        "(swing_sim.variation.spec).",
    ),
    "normal_distribution": _entry(
        "Normal Distribution",
        "The bell-curve distribution parameterized by mean and standard "
        "deviation — the default noise family for delivery variables in "
        "variation studies, matching how measurement scatter is usually "
        "reported (swing_sim.variation registry guidance).",
    ),
    "one_at_a_time_sensitivity": _entry(
        "One-at-a-Time (OAT) Sensitivity",
        "A sensitivity method that re-runs the study with only one input "
        "varying at a time, using paired random draws, to attribute "
        "output variance to individual inputs "
        "(swing_sim.variation.analysis).",
    ),
    "pitch": _entry(
        "Screw Pitch",
        "The ratio of translation along the instantaneous screw axis to "
        "rotation about it. A pure rotation has zero pitch; the "
        "clubhead's near-impact motion has a small pitch, which is why "
        "the screw-axis picture works (screw theory; AffineDrift "
        "rotation review).",
    ),
    "plane_inclination": _entry(
        "Swing-Plane Inclination",
        "The orientation of the pendulum swing plane in space — yaw, "
        "side tilt, and forward tilt. Gravity is projected into the "
        "plane (g_inplane = R^T (0, 0, -g)), so steeper planes feel more "
        "in-plane gravity (swing_sim.reference.in_plane_gravity).",
    ),
    "r_isa": _entry(
        "Distance to the Screw Axis (R_ISA)",
        "The distance v/ω from the clubhead to the instantaneous screw "
        "axis. Closure per foot equals 1 / R_ISA, and the path gap "
        "between two reference points separated by d is d / R_ISA — "
        "independent of clubhead speed (AffineDrift closure derivation).",
    ),
    "roll": _entry(
        "Roll",
        "The vertical (crown-sole) curvature of a wood's face. It adds "
        "loft to high strikes and removes it from low ones, partially "
        "compensating the gear effect's backspin change (club-design "
        "literature; rate_of_closure.club face model).",
    ),
    "screw_axis": _entry(
        "Instantaneous Screw Axis (ISA)",
        "The unique line about which a rigid body's motion at one "
        "instant is a rotation plus a slide along that same line "
        "(Chasles' theorem). The clubhead sweeps about it near impact; "
        "its distance is R_ISA (screw theory; AffineDrift rotation "
        "review).",
    ),
    "seed": _entry(
        "Random Seed",
        "The integer that initializes the pseudo-random number generator "
        "of a variation study. The engine derives a per-variable stream "
        "from [seed, crc32(key)], so the same plan and seed always "
        "reproduce the same dataset (swing_sim.variation.engine).",
    ),
    "sensitivity_analysis": _entry(
        "Sensitivity Analysis",
        "Quantifying which inputs drive which outputs. The Variation tab "
        "combines one-at-a-time reruns (local attribution) with Spearman "
        "rank correlation on the full dataset (a cheap global "
        "cross-check) (swing_sim.variation.analysis).",
    ),
    "smash_factor": _entry(
        "Smash Factor",
        "Ball speed divided by clubhead speed — the standard efficiency "
        "measure of an impact. A well-struck driver reaches about "
        "1.48-1.50; off-center hits lose smash through the reduced "
        "effective mass (launch-monitor norms).",
    ),
    "spearman": _entry(
        "Spearman Rank Correlation",
        "A correlation computed on ranks rather than raw values, so it "
        "captures any monotonic input-output relationship without "
        "assuming linearity. Used as the global sensitivity cross-check "
        "in variation studies (swing_sim.variation.analysis).",
    ),
    "spin_axis_tilt": _entry(
        "Spin-Axis Tilt",
        "The tilt of the ball's spin axis away from horizontal (+ = "
        "fade side, right of target). Set by the face-minus-path "
        "difference through the D-plane; it converts backspin into "
        "sideways curvature (TrackMan D-plane literature).",
    ),
    "spin_decay": _entry(
        "Spin Decay",
        "The gradual loss of spin rate during flight from aerodynamic "
        "torque, modeled as an exponential decay in the "
        "MacDonald-Hanzely and constant-coefficient flight models "
        "(swing_sim.flight.models).",
    ),
    "spin_loft": _entry(
        "Spin Loft",
        "The 3-D angle between the delivered face normal and the club "
        "path vector: spin_loft = arccos(v̂ · n̂). It sets how much of "
        "the impact goes into spin instead of speed "
        "(swing_sim.impact.delivery; TrackMan conventions).",
    ),
    "spin_rate": _entry(
        "Spin Rate",
        "The ball's total rotation rate in rpm — the friction impulse of "
        "the impact solve (capped at the 2/7 rolling limit) plus "
        "gear-effect spin. Driver band roughly 2,000-3,500 rpm "
        "(swing_sim.impact; launch-monitor norms).",
    ),
    "spv": _entry(
        "Swing-Plane Velocity (SPV)",
        "The clubhead's angular velocity about the swing-plane normal — "
        "the in-plane rotation of the swing arc. Together with HTV it "
        "assembles the full angular velocity vector (Cheetham 2014 "
        "3-D motion studies).",
    ),
    "time_to_square": _entry(
        "Time to Square",
        "How long before impact the face was one degree open at the "
        "current closure rate — about half a millisecond at tour rates, "
        "the classic framing of release-timing tolerance (closure-rate "
        "dossier).",
    ),
    "triangular_distribution": _entry(
        "Triangular Distribution",
        "A bounded distribution rising linearly to a peak and falling "
        "back — a practical choice in variation studies when only a "
        "min / most-likely / max estimate is available "
        "(swing_sim.variation.spec).",
    ),
    "triple_pendulum": _entry(
        "Triple Pendulum Swing Model",
        "A three-link extension of the double pendulum (torso-arms-club) "
        "solved with the same mass-matrix formalism in a planar frame. "
        "Available as a swing source in the Simulation tab "
        "(rate_of_closure.simulation.sources).",
    ),
    "twist": _entry(
        "Twist",
        "A rigid body's instantaneous motion state — angular velocity "
        "plus the linear velocity of a reference point. The twist "
        "relation v_P = v_ref + ω × r gives every point's velocity from "
        "one twist (screw theory; the core of the closure model).",
    ),
    "uniform_distribution": _entry(
        "Uniform Distribution",
        "A distribution giving every value in a bounded interval equal "
        "probability — used in variation studies when only hard limits, "
        "not a central tendency, are known (swing_sim.variation.spec).",
    ),
}


#: Explanation field -> the glossary term it pre-selects, for every
#: field that maps cleanly onto one term (contract-tested).
FIELD_TO_TERM: dict[str, str] = {
    # RESULT_EXPLANATIONS fields
    "path_deviation_deg": "club_path",
    "aoa_deviation_deg": "attack_angle",
    "tangential_speed_mph": "twist",
    "speed_delta_mph": "twist",
    "closure_rate_dps": "ccv",
    "normalized_closure_deg_per_ft": "r_isa",
    "closure_during_contact_deg": "contact_duration",
    "loft_gain_during_contact_deg": "dynamic_loft",
    # METRIC_EXPLANATIONS fields
    "ccv_dps": "ccv",
    "closure_deg_per_ft": "r_isa",
    "closure_deg_per_inch": "closure_rate",
    "closure_deg_per_ms": "closure_rate",
    "r_isa_m": "r_isa",
    "r_isa_ft": "r_isa",
    "time_to_square_from_1deg_open_ms": "time_to_square",
    "toe_heel_speed_delta_mph": "lever_arm",
    # LAUNCH_EXPLANATIONS fields
    "ball_speed_mph": "smash_factor",
    "launch_angle_deg": "launch_angle",
    "launch_azimuth_deg": "launch_azimuth",
    "spin_rpm": "spin_rate",
    "carry_m": "carry",
    "max_height_m": "apex",
    "flight_time_s": "flight_time",
    "landing_angle_deg": "landing_angle",
    "lateral_m": "lateral_offset",
}


def search_terms(query: str) -> tuple[str, ...]:
    """Glossary keys whose term or definition matches ``query``.

    Case-insensitive substring search over the display term and the
    definition body; an empty query returns every key.

    Args:
        query: Free-text filter.

    Returns:
        Matching keys in glossary (alphabetical) order.
    """
    needle = query.strip().lower()
    keys = tuple(
        key
        for key, entry in GLOSSARY.items()
        if not needle
        or needle in entry.term.lower()
        or needle in entry.definition.lower()
    )
    ensure(not needle or len(keys) <= len(GLOSSARY), "filter cannot grow the set")
    return keys


def _validate() -> None:
    keys = list(GLOSSARY)
    ensure(keys == sorted(keys), "glossary keys must be sorted")
    ensure(len(keys) >= 40, "glossary must cover the app's vocabulary")
    ensure(
        all(target in GLOSSARY for target in FIELD_TO_TERM.values()),
        "every field mapping must point at a real glossary term",
    )


_validate()
