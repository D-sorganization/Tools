/**
 * Glossary of every technical term used across the app (#4120 V4).
 *
 * TypeScript mirror of `src/rate_of_closure/glossary.py` — same keys,
 * same terms, same sourced definitions. The vitest parity test pins the
 * key list so the two glossaries cannot drift apart.
 */

export interface GlossaryEntry {
  /** Title Case display name. */
  term: string;
  /** 1-3 sentence sourced definition. */
  definition: string;
}

/** Every term used across the app, keyed snake_case (Python parity). */
export const GLOSSARY: Record<string, GlossaryEntry> = {
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
  lever_arm: {
    term: "Lever Arm",
    definition:
      "The vector r from the reference point (GC or CG) to the struck " +
      "point. It converts rotation into extra point velocity (\u03c9 \u00d7 r) in the " +
      "closure model and impulse into recoil torque (r \u00d7 Jn) in the impact " +
      "model.",
  },
  lie_angle: {
    term: "Lie Angle",
    definition:
      "The angle between the shaft and the ground plane at impact. It sets " +
      "how the shaft-axis (HTV) and swing-plane (SPV) rotation rates " +
      "combine into face closure: CCV = HTV sin(lie) + SPV cos(lie) " +
      "(Cheetham 2014 reconciliation).",
  },
  lift: {
    term: "Lift Coefficient (Cl)",
    definition:
      "The dimensionless coefficient of the aerodynamic force perpendicular " +
      "to the ball's motion, generated by backspin (the Magnus effect). " +
      "Literature models express Cl as a function of spin ratio, capped at " +
      "a physical maximum (swing_sim.flight.models).",
  },
  magnus_force: {
    term: "Magnus Force",
    definition:
      "The aerodynamic force on a spinning ball, perpendicular to both the " +
      "velocity and the spin axis: backspin lifts the ball, a tilted spin " +
      "axis curves it sideways. It enters the flight EOM through the lift " +
      "term (swing_sim.flight; Penner 2003).",
  },
  mass_matrix: {
    term: "Mass Matrix",
    definition:
      "The configuration-dependent 2x2 (or 3x3) inertia matrix M(\u03b8) of the " +
      "pendulum equations M(\u03b8)\u00b7\u03b1 + C(\u03b8, \u03c9) + G(\u03b8) + D(\u03c9) = 0; its " +
      "off-diagonal terms couple the links through the elbow/wrist angle " +
      "(swing_sim.reference.mass_matrix).",
  },
  moi: {
    term: "Moment of Inertia (MOI)",
    definition:
      "A body's resistance to angular acceleration about an axis. The " +
      "clubhead MOI (scalar or full 3x3 tensor) sets how much an off-center " +
      "impulse twists the head instead of launching the ball \u2014 the " +
      "club-side term of the effective mass (swing_sim.impact.models).",
  },
  moi_tensor: {
    term: "MOI Tensor",
    definition:
      "The full 3x3 inertia tensor I of the clubhead. The exact off-center " +
      "effective mass uses the triple-product form (r x n)^T I^-1 (r x n); " +
      "a diagonal I\u00b7eye(3) reproduces the scalar-MOI fallback 1/m + |r|\u00b2/I " +
      "exactly (swing_sim.impact.models derivation).",
  },
  monte_carlo: {
    term: "Monte Carlo Simulation",
    definition:
      "Running the simulation many times with randomized inputs drawn from " +
      "per-variable noise distributions, then reading dispersion statistics " +
      "off the output sample. The Variation tab's seeded engine makes runs " +
      "exactly reproducible (swing_sim.variation.engine).",
  },
  noise_spec: {
    term: "Noise Specification (NoiseSpec)",
    definition:
      "The per-variable description of how an input varies in a variation " +
      "study: distribution family (normal, uniform, or triangular), " +
      "additive scale, and optional clip truncation " +
      "(swing_sim.variation.spec).",
  },
  normal_distribution: {
    term: "Normal Distribution",
    definition:
      "The bell-curve distribution parameterized by mean and standard " +
      "deviation \u2014 the default noise family for delivery variables in " +
      "variation studies, matching how measurement scatter is usually " +
      "reported (swing_sim.variation registry guidance).",
  },
  one_at_a_time_sensitivity: {
    term: "One-at-a-Time (OAT) Sensitivity",
    definition:
      "A sensitivity method that re-runs the study with only one input " +
      "varying at a time, using paired random draws, to attribute output " +
      "variance to individual inputs (swing_sim.variation.analysis).",
  },
  pitch: {
    term: "Screw Pitch",
    definition:
      "The ratio of translation along the instantaneous screw axis to " +
      "rotation about it. A pure rotation has zero pitch; the clubhead's " +
      "near-impact motion has a small pitch, which is why the screw-axis " +
      "picture works (screw theory; AffineDrift rotation review).",
  },
  plane_inclination: {
    term: "Swing-Plane Inclination",
    definition:
      "The orientation of the pendulum swing plane in space \u2014 yaw, side " +
      "tilt, and forward tilt. Gravity is projected into the plane " +
      "(g_inplane = R^T (0, 0, -g)), so steeper planes feel more in-plane " +
      "gravity (swing_sim.reference.in_plane_gravity).",
  },
  r_isa: {
    term: "Distance to the Screw Axis (R_ISA)",
    definition:
      "The distance v/\u03c9 from the clubhead to the instantaneous screw axis. " +
      "Closure per foot equals 1 / R_ISA, and the path gap between two " +
      "reference points separated by d is d / R_ISA \u2014 independent of " +
      "clubhead speed (AffineDrift closure derivation).",
  },
  roll: {
    term: "Roll",
    definition:
      "The vertical (crown-sole) curvature of a wood's face. It adds loft " +
      "to high strikes and removes it from low ones, partially compensating " +
      "the gear effect's backspin change (club-design literature; " +
      "rate_of_closure.club face model).",
  },
  screw_axis: {
    term: "Instantaneous Screw Axis (ISA)",
    definition:
      "The unique line about which a rigid body's motion at one instant is " +
      "a rotation plus a slide along that same line (Chasles' theorem). The " +
      "clubhead sweeps about it near impact; its distance is R_ISA (screw " +
      "theory; AffineDrift rotation review).",
  },
  seed: {
    term: "Random Seed",
    definition:
      "The integer that initializes the pseudo-random number generator of a " +
      "variation study. The engine derives a per-variable stream from " +
      "[seed, crc32(key)], so the same plan and seed always reproduce the " +
      "same dataset (swing_sim.variation.engine).",
  },
  sensitivity_analysis: {
    term: "Sensitivity Analysis",
    definition:
      "Quantifying which inputs drive which outputs. The Variation tab " +
      "combines one-at-a-time reruns (local attribution) with Spearman rank " +
      "correlation on the full dataset (a cheap global cross-check) " +
      "(swing_sim.variation.analysis).",
  },
  smash_factor: {
    term: "Smash Factor",
    definition:
      "Ball speed divided by clubhead speed \u2014 the standard efficiency " +
      "measure of an impact. A well-struck driver reaches about 1.48-1.50; " +
      "off-center hits lose smash through the reduced effective mass " +
      "(launch-monitor norms).",
  },
  spearman: {
    term: "Spearman Rank Correlation",
    definition:
      "A correlation computed on ranks rather than raw values, so it " +
      "captures any monotonic input-output relationship without assuming " +
      "linearity. Used as the global sensitivity cross-check in variation " +
      "studies (swing_sim.variation.analysis).",
  },
  spin_axis_tilt: {
    term: "Spin-Axis Tilt",
    definition:
      "The tilt of the ball's spin axis away from horizontal (+ = fade " +
      "side, right of target). Set by the face-minus-path difference " +
      "through the D-plane; it converts backspin into sideways curvature " +
      "(TrackMan D-plane literature).",
  },
  spin_decay: {
    term: "Spin Decay",
    definition:
      "The gradual loss of spin rate during flight from aerodynamic torque, " +
      "modeled as an exponential decay in the MacDonald-Hanzely and " +
      "constant-coefficient flight models (swing_sim.flight.models).",
  },
  spin_loft: {
    term: "Spin Loft",
    definition:
      "The 3-D angle between the delivered face normal and the club path " +
      "vector: spin_loft = arccos(v\u0302 \u00b7 n\u0302). It sets how much of the impact " +
      "goes into spin instead of speed (swing_sim.impact.delivery; TrackMan " +
      "conventions).",
  },
  spin_rate: {
    term: "Spin Rate",
    definition:
      "The ball's total rotation rate in rpm \u2014 the friction impulse of the " +
      "impact solve (capped at the 2/7 rolling limit) plus gear-effect " +
      "spin. Driver band roughly 2,000-3,500 rpm (swing_sim.impact; " +
      "launch-monitor norms).",
  },
  spv: {
    term: "Swing-Plane Velocity (SPV)",
    definition:
      "The clubhead's angular velocity about the swing-plane normal \u2014 the " +
      "in-plane rotation of the swing arc. Together with HTV it assembles " +
      "the full angular velocity vector (Cheetham 2014 3-D motion studies).",
  },
  time_to_square: {
    term: "Time to Square",
    definition:
      "How long before impact the face was one degree open at the current " +
      "closure rate \u2014 about half a millisecond at tour rates, the classic " +
      "framing of release-timing tolerance (closure-rate dossier).",
  },
  triangular_distribution: {
    term: "Triangular Distribution",
    definition:
      "A bounded distribution rising linearly to a peak and falling back \u2014 " +
      "a practical choice in variation studies when only a min / " +
      "most-likely / max estimate is available (swing_sim.variation.spec).",
  },
  triple_pendulum: {
    term: "Triple Pendulum Swing Model",
    definition:
      "A three-link extension of the double pendulum (torso-arms-club) " +
      "solved with the same mass-matrix formalism in a planar frame. " +
      "Available as a swing source in the Simulation tab " +
      "(rate_of_closure.simulation.sources).",
  },
  twist: {
    term: "Twist",
    definition:
      "A rigid body's instantaneous motion state \u2014 angular velocity plus " +
      "the linear velocity of a reference point. The twist relation v_P = " +
      "v_ref + \u03c9 \u00d7 r gives every point's velocity from one twist (screw " +
      "theory; the core of the closure model).",
  },
  uniform_distribution: {
    term: "Uniform Distribution",
    definition:
      "A distribution giving every value in a bounded interval equal " +
      "probability \u2014 used in variation studies when only hard limits, not a " +
      "central tendency, are known (swing_sim.variation.spec).",
  },
};

/** Explanation field (camelCase, web keys) -> glossary term. */
export const FIELD_TO_TERM: Record<string, string> = {
  pathDeviationDeg: "club_path",
  aoaDeviationDeg: "attack_angle",
  tangentialSpeedMph: "twist",
  speedDeltaMph: "twist",
  closureRateDps: "ccv",
  normalizedClosureDegPerFt: "r_isa",
  closureDuringContactDeg: "contact_duration",
  loftGainDuringContactDeg: "dynamic_loft",
  ccvDps: "ccv",
  closureDegPerFt: "r_isa",
  closureDegPerInch: "closure_rate",
  closureDegPerMs: "closure_rate",
  rIsaM: "r_isa",
  rIsaFt: "r_isa",
  timeToSquareFrom1DegOpenMs: "time_to_square",
  toeHeelSpeedDeltaMph: "lever_arm",
  ballSpeedMph: "smash_factor",
  launchAngleDeg: "launch_angle",
  launchAzimuthDeg: "launch_azimuth",
  spinRpm: "spin_rate",
  carryM: "carry",
  maxHeightM: "apex",
  flightTimeS: "flight_time",
  landingAngleDeg: "landing_angle",
  lateralM: "lateral_offset",
};

/** Glossary keys whose term or definition matches `query`. */
export function searchTerms(query: string): string[] {
  const needle = query.trim().toLowerCase();
  return Object.keys(GLOSSARY).filter(
    (key) =>
      !needle ||
      GLOSSARY[key].term.toLowerCase().includes(needle) ||
      GLOSSARY[key].definition.toLowerCase().includes(needle),
  );
}
