/**
 * Glossary entry data (apex through lie_angle) — see glossary.ts.
 *
 * Split from the glossary module to honor the 500-LOC file budget.
 * Generated from `src/rate_of_closure/glossary.py`.
 */

import type { GlossaryEntry } from "./glossaryTypes";

export const ENTRIES: Record<string, GlossaryEntry> = {
  apex: {
    term: "Apex Height",
    definition:
      "The peak height of the ball's trajectory \u2014 the point where lift and " +
      "gravity momentarily balance vertical motion. Typical driver apex is " +
      "25-40 m (launch-monitor norms; swing_sim.flight metrics).",
  },
  attack_angle: {
    term: "Attack Angle (AoA)",
    definition:
      "The vertical angle of the clubhead's velocity at impact: positive = " +
      "hitting up on the ball. One of the launch-monitor delivery " +
      "parameters (AffineDrift Launch Monitor Technology Review).",
  },
  ball_speed: {
    term: "Ball Speed",
    definition:
      "The speed of the ball immediately after impact, set by the " +
      "effective-mass momentum exchange of the COR impulse model " +
      "(swing_sim.impact). Divided by clubhead speed it gives the smash " +
      "factor.",
  },
  bulge: {
    term: "Bulge",
    definition:
      "The horizontal (heel-toe) curvature of a wood's face. It starts toe " +
      "strikes pointing further right so the gear-effect draw spin curves " +
      "the ball back toward the target (club-design literature; " +
      "rate_of_closure.club face model).",
  },
  carry: {
    term: "Carry Distance",
    definition:
      "The horizontal distance from launch to the first ground contact of " +
      "the integrated trajectory \u2014 no roll-out included (swing_sim.flight " +
      "terminal ground event).",
  },
  ccv: {
    term: "Club Closure Velocity (CCV)",
    definition:
      "The rate the face normal sweeps horizontally (closes), in deg/s: CCV " +
      "= HTV sin(lie) + SPV cos(lie). Cheetham 2014 tour driver data puts " +
      "the mean near 2,100 deg/s.",
  },
  cg_depth: {
    term: "CG Depth",
    definition:
      "How far the clubhead's center of gravity sits behind the face plane. " +
      "A deeper CG lengthens the recoil lever arm of an off-center impulse " +
      "and therefore strengthens gear-effect spin " +
      "(swing_sim.impact.gear_effect derivation).",
  },
  closure_rate: {
    term: "Closure Rate",
    definition:
      "How fast the face angle changes as the club approaches impact \u2014 " +
      "reported as CCV (deg/s), per foot of travel (deg/ft), per inch, or " +
      "per millisecond. The speed-invariant deg/ft form equals 1 / R_ISA " +
      "(AffineDrift closure-rate derivation).",
  },
  club_path: {
    term: "Club Path",
    definition:
      "The horizontal direction of the clubhead's velocity at impact, " +
      "relative to the target line: positive = in-to-out (right of target " +
      "for a right-handed player). Standard launch-monitor sign convention " +
      "(AffineDrift 02-parameters).",
  },
  contact_duration: {
    term: "Contact Duration",
    definition:
      "The time the ball stays compressed on the face \u2014 about 450 " +
      "microseconds for a driver. The face keeps rotating the whole time, " +
      "so the face the ball leaves is not the face it met (impact " +
      "literature; Cheetham dossier).",
  },
  cor: {
    term: "Coefficient of Restitution (COR)",
    definition:
      "The ratio of separation speed to approach speed along the impact " +
      "normal (0 = perfectly plastic, 1 = perfectly elastic). Modern driver " +
      "faces are capped near 0.83 by the rules; it scales the (1 + e) " +
      "factor in the impulse solve (swing_sim.impact.models).",
  },
  coriolis: {
    term: "Coriolis / Centripetal Terms",
    definition:
      "The velocity-dependent generalized forces C(\u03b8, \u03c9) in the pendulum " +
      "equations of motion, arising from the rotating links: products like " +
      "\u03c91\u00b7\u03c92 and \u03c91\u00b2 multiplied by -m2\u00b7l1\u00b7lc2\u00b7sin(\u03b82) " +
      "(swing_sim.reference.coriolis_vector).",
  },
  d_plane: {
    term: "D-Plane",
    definition:
      "The plane spanned by the club-path vector and the delivered face " +
      "normal. The ball launches close to the face normal and spins about " +
      "the D-plane's normal, so the face-minus-path difference sets the " +
      "spin-axis tilt (Jorgensen, The Physics of Golf; TrackMan D-plane " +
      "literature).",
  },
  damping: {
    term: "Damping",
    definition:
      "Viscous joint torques proportional to angular velocity (d1\u00b7\u03c91, " +
      "d2\u00b7\u03c92) that drain energy from the pendulum swing model \u2014 the model's " +
      "stand-in for soft-tissue and grip losses " +
      "(swing_sim.reference.damping_vector).",
  },
  dispersion_ellipse: {
    term: "2\u03c3 Dispersion Ellipse",
    definition:
      "The ellipse covering roughly 95% of simulated landing points, built " +
      "from the eigen-decomposition of the carry/lateral covariance matrix " +
      "scaled to two standard deviations (swing_sim.variation.analysis).",
  },
  double_pendulum: {
    term: "Double Pendulum Swing Model",
    definition:
      "The classic two-link golf swing model \u2014 arms and club as two rigid " +
      "links in an inclined plane, driven by gravity and released torques. " +
      "Its equations of motion come from the Lagrangian: mass matrix, " +
      "Coriolis, gravity, and damping terms (swing_sim / rust swing-core).",
  },
  drag: {
    term: "Drag Coefficient (Cd)",
    definition:
      "The dimensionless coefficient in the aerodynamic drag force F = " +
      "\u00bd\u03c1ACd\u00b7v\u00b2, opposing the ball's motion through the air. The literature " +
      "flight models differ mainly in how Cd and Cl depend on spin ratio " +
      "(swing_sim.flight.models).",
  },
  dynamic_loft: {
    term: "Dynamic Loft",
    definition:
      "The vertical angle of the delivered face normal at impact \u2014 the " +
      "club's static loft plus shaft lean, wrist action, and the loft " +
      "gained while the face rotates during contact (launch-monitor " +
      "delivery parameter; AffineDrift conventions).",
  },
  effective_mass: {
    term: "Effective Mass",
    definition:
      "The reduced club mass the ball actually feels in an off-center " +
      "impact: 1/m_eff = 1/m + (r x n)^T I^-1 (r x n), where r is the " +
      "CG-to-contact lever and n the face normal \u2014 rotation recoil eats " +
      "part of the impulse (swing_sim.impact.models derivation).",
  },
  face_angle: {
    term: "Face Angle",
    definition:
      "The horizontal direction of the delivered face normal relative to " +
      "the target line: positive = open (pointing right of target). The " +
      "dominant contributor to launch azimuth (launch-monitor conventions; " +
      "AffineDrift 02-parameters).",
  },
  flight_time: {
    term: "Flight Time",
    definition:
      "Total time aloft, from launch to the terminal ground event of the " +
      "flight integration \u2014 typically 5-7 s for a driver (swing_sim.flight " +
      "metrics).",
  },
  friction_spin_cap: {
    term: "2/7 Friction Spin Cap",
    definition:
      "The rolling-without-slip limit on the tangential friction impulse " +
      "for a uniform solid sphere: J_f = min(\u03bcJ, (2/7)\u00b7m\u00b7v_t). Beyond it " +
      "the contact point has stopped sliding, so friction can add no more " +
      "spin (Cross 2002, Am. J. Phys. 70, 1093; swing_sim.impact.models).",
  },
  gear_effect: {
    term: "Gear Effect",
    definition:
      "Spin created when an off-center impulse makes the head recoil in " +
      "rotation and the face surface sweeps under the ball like a gear " +
      "tooth: toe hits gain draw-side spin, high hits lose backspin. " +
      "Derived from the head's I^-1 (r x J n) recoil and Coulomb friction " +
      "(swing_sim.impact.gear_effect).",
  },
  geometric_center: {
    term: "Geometric Center (GC)",
    definition:
      "The reference point launch monitors track on the clubhead \u2014 the " +
      "center of the head envelope, within ~6 mm of the CG for a driver. " +
      "The ball responds to the impact point, not the GC (AffineDrift " +
      "Launch Monitor Technology Review).",
  },
  htv: {
    term: "Horizontal Turning Velocity (HTV)",
    definition:
      "The clubhead's angular velocity about the shaft axis \u2014 the " +
      "closing/release component of the swing. Cheetham 2014 tour driver " +
      "data: 1,307 \u00b1 304 deg/s (range 652-2,432, n = 94).",
  },
  impulse_momentum: {
    term: "Impulse-Momentum Impact Model",
    definition:
      "The rigid-body collision model: a normal impulse J = (1 + " +
      "e)\u00b7m_eff\u00b7v_rel exchanged over the ~450 \u00b5s contact sets ball speed, " +
      "with COR e and effective mass m_eff; friction supplies the " +
      "tangential (spin) impulse (swing_sim.impact.models).",
  },
  inverse_dynamics: {
    term: "Inverse Dynamics",
    definition:
      "Computing the joint torques that must have acted, given an observed " +
      "motion: with the pendulum equations M(θ)·α + C(θ, ω) + G(θ) + " +
      "D(ω) = τ, sample θ, ω, α along the swing and solve for τ per " +
      "joint. The passive swing recovers τ ≈ 0, exposing the " +
      "gravity/damping/inertial breakdown " +
      "(rate_of_closure.simulation.kinetics; standard robotics formulation).",
  },
  joint_reaction_force: {
    term: "Joint Reaction Force",
    definition:
      "The force transmitted through a joint — what the proximal segment " +
      "exerts on the distal segment — from Newton-Euler on each segment: " +
      "F = m·(a_com - g), summed up the chain. It is dominated by the " +
      "centripetal cost of swinging mass on an arc " +
      "(rate_of_closure.simulation.kinetics; classical biomechanics " +
      "inverse dynamics).",
  },
  landing_angle: {
    term: "Landing Angle",
    definition:
      "The descent angle below horizontal at the terminal ground event. " +
      "Steeper landings stop faster; the driver band is roughly 35-45 deg " +
      "(swing_sim.flight metrics; launch-monitor norms).",
  },
  lateral_offset: {
    term: "Lateral Landing Offset",
    definition:
      "The sideways distance from the target line at landing (+ = right of " +
      "target): the integrated effect of launch azimuth plus the curvature " +
      "from spin-axis tilt \u2014 the way launch monitors report carry offline " +
      "(swing_sim.flight metrics).",
  },
  launch_angle: {
    term: "Launch Angle",
    definition:
      "The vertical angle of the ball's initial velocity above the ground " +
      "plane \u2014 the D-plane compromise between dynamic loft and attack " +
      "angle, typically 10-16 deg for a driver (launch-monitor " +
      "conventions).",
  },
  launch_azimuth: {
    term: "Launch Azimuth",
    definition:
      "The horizontal direction of the ball's initial velocity relative to " +
      "the target line (+ = right). Dominated by the delivered face angle " +
      "with a smaller club-path contribution (D-plane literature).",
  },
};
