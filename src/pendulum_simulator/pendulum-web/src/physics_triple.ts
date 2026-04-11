/**
 * Triple Pendulum (3-DOF) Golf Swing Physics Engine — TypeScript.
 *
 * Model: 3-segment pendulum (three arm segments) with clubhead mass at tip.
 *
 * Coordinates:
 *   q = [θ₁, φ₂, φ₃] where:
 *     θ₁ = shoulder angle (absolute)
 *     φ₂ = elbow relative angle (θ₂ = θ₁ + φ₂)
 *     φ₃ = wrist relative angle (θ₃ = θ₁ + φ₂ + φ₃)
 *
 * Design by Contract (DbC):
 *   - Pre-conditions checked via assertFinite / assertPositive helpers.
 *   - Post-conditions asserted on mass matrix symmetry and state finiteness.
 *   - All functions are pure (no side effects).
 *
 * @module physics_triple
 */

// ── Contract helpers ──────────────────────────────────────────────────────────

function assertFinite(v: number, name: string): void {
  if (!isFinite(v))
    throw new RangeError(`[DbC] ${name} must be finite, got ${v}`);
}

function assertPositive(v: number, name: string): void {
  if (!(v > 0)) throw new RangeError(`[DbC] ${name} must be > 0, got ${v}`);
}

function assertNonNeg(v: number, name: string): void {
  if (!(v >= 0)) throw new RangeError(`[DbC] ${name} must be ≥ 0, got ${v}`);
}

// ── Data structures ───────────────────────────────────────────────────────────

export interface TripleParams {
  m1: number; // kg — segment 1 (shoulder to elbow) mass
  m2: number; // kg — segment 2 (elbow to wrist) mass
  m3: number; // kg — segment 3 (wrist to tip) mass
  mClub: number; // kg — clubhead mass (point mass at tip)
  L1: number; // m  — segment 1 length
  L2: number; // m  — segment 2 length
  L3: number; // m  — segment 3 length
  g: number; // m/s²
  b1: number; // N·m·s/rad — viscous damping shoulder
  b2: number; // N·m·s/rad — viscous damping elbow
  b3: number; // N·m·s/rad — viscous damping wrist
}

/** [theta1, phi2, phi3, dtheta1, dphi2, dphi3] */
export type StateTriple = [number, number, number, number, number, number];

export type TorqueFuncTriple = (t: number) => [number, number, number];

/** DbC-validated constructor for TripleParams. */
export function makeTripleParams(p: TripleParams): TripleParams {
  assertPositive(p.m1, "m1");
  assertPositive(p.m2, "m2");
  assertPositive(p.m3, "m3");
  assertNonNeg(p.mClub, "mClub");
  assertPositive(p.L1, "L1");
  assertPositive(p.L2, "L2");
  assertPositive(p.L3, "L3");
  assertNonNeg(p.g, "g");
  assertNonNeg(p.b1, "b1");
  assertNonNeg(p.b2, "b2");
  assertNonNeg(p.b3, "b3");
  return { ...p };
}

// ── Effective mass helper ──────────────────────────────────────────────────────

/** Effective mass of segments 2, 3, and clubhead. */
function m23eff(p: TripleParams): number {
  return p.m2 + p.m3 + p.mClub;
}

/** Effective mass of segments 3 and clubhead. */
function m3eff(p: TripleParams): number {
  return p.m3 + p.mClub;
}

// ── Mass matrix (3×3 inertia matrix) ────────────────────────────────────────

/**
 * 3×3 inertia matrix for triple pendulum.
 * Stored as flat array [M00, M01, M02, M10, M11, M12, M20, M21, M22].
 *
 * M[0][0] = (m1+m2+m3+mClub)*L1² + (m2+m3+mClub)*L2² + (m3+mClub)*L3²
 *           + 2*(m2+m3+mClub)*L1*L2*cos(φ₂)
 *           + 2*(m3+mClub)*L1*L3*cos(φ₂+φ₃)
 *           + 2*(m3+mClub)*L2*L3*cos(φ₃)
 *
 * M[0][1] = (m2+m3+mClub)*L2² + (m3+mClub)*L1*L2*cos(φ₂)
 *           + (m3+mClub)*L3² + (m3+mClub)*L2*L3*cos(φ₃)
 *           + (m3+mClub)*L1*L3*cos(φ₂+φ₃)
 *
 * M[0][2] = (m3+mClub)*L3² + (m3+mClub)*L1*L3*cos(φ₂+φ₃) + (m3+mClub)*L2*L3*cos(φ₃)
 *
 * M[1][1] = (m2+m3+mClub)*L2² + (m3+mClub)*L3² + 2*(m3+mClub)*L2*L3*cos(φ₃)
 *
 * M[1][2] = (m3+mClub)*L3² + (m3+mClub)*L2*L3*cos(φ₃)
 *
 * M[2][2] = (m3+mClub)*L3²
 *
 * Pre:  q[1], q[2] finite.
 * Post: symmetric (M[i][j] === M[j][i]), M[2][2] > 0.
 */
export function massMatrix3(
  q: [number, number, number],
  p: TripleParams,
): number[][] {
  assertFinite(q[1], "phi2");
  assertFinite(q[2], "phi3");

  const phi2 = q[1];
  const phi3 = q[2];
  const phi23 = phi2 + phi3;

  const c2 = Math.cos(phi2);
  const c3 = Math.cos(phi3);
  const c23 = Math.cos(phi23);

  const m23 = m23eff(p);
  const m3 = m3eff(p);

  const M00 =
    (p.m1 + m23) * p.L1 * p.L1 +
    m23 * p.L2 * p.L2 +
    m3 * p.L3 * p.L3 +
    2 * m23 * p.L1 * p.L2 * c2 +
    2 * m3 * p.L1 * p.L3 * c23 +
    2 * m3 * p.L2 * p.L3 * c3;

  const M01 =
    m23 * p.L2 * p.L2 +
    m3 * p.L1 * p.L2 * c2 +
    m3 * p.L3 * p.L3 +
    m3 * p.L2 * p.L3 * c3 +
    m3 * p.L1 * p.L3 * c23;

  const M02 = m3 * p.L3 * p.L3 + m3 * p.L1 * p.L3 * c23 + m3 * p.L2 * p.L3 * c3;

  const M11 = m23 * p.L2 * p.L2 + m3 * p.L3 * p.L3 + 2 * m3 * p.L2 * p.L3 * c3;

  const M12 = m3 * p.L3 * p.L3 + m3 * p.L2 * p.L3 * c3;

  const M22 = m3 * p.L3 * p.L3;

  if (!(M22 > 0)) throw new Error("[DbC post] M22 must be positive");

  return [
    [M00, M01, M02],
    [M01, M11, M12],
    [M02, M12, M22],
  ];
}

// ── Coriolis ──────────────────────────────────────────────────────────────────

function coriolisVector3(
  q: [number, number, number],
  qdot: [number, number, number],
  p: TripleParams,
): [number, number, number] {
  assertFinite(q[1], "phi2");
  assertFinite(q[2], "phi3");
  assertFinite(qdot[0], "dtheta1");
  assertFinite(qdot[1], "dphi2");
  assertFinite(qdot[2], "dphi3");

  const phi2 = q[1];
  const phi3 = q[2];
  const phi23 = phi2 + phi3;

  const dtheta1 = qdot[0];
  const dphi2 = qdot[1];
  const dphi3 = qdot[2];

  const s2 = Math.sin(phi2);
  const s3 = Math.sin(phi3);
  const s23 = Math.sin(phi23);

  const m23 = m23eff(p);
  const m3 = m3eff(p);

  // h12 = -m23*L1*L2*sin(φ₂), h13 = -m3*L1*L3*sin(φ₂+φ₃), h23 = -m3*L2*L3*sin(φ₃)
  const h12 = -m23 * p.L1 * p.L2 * s2;
  const h13 = -m3 * p.L1 * p.L3 * s23;
  const h23 = -m3 * p.L2 * p.L3 * s3;

  const C1 =
    h12 * (2 * dtheta1 * dphi2 + dphi2 * dphi2) +
    h13 * (2 * dtheta1 * (dphi2 + dphi3) + (dphi2 + dphi3) * (dphi2 + dphi3)) +
    h23 * (2 * dphi2 * dphi3 + dphi3 * dphi3);

  const C2 =
    -h12 * dtheta1 * dtheta1 -
    h13 * (dtheta1 + dphi2) * (dtheta1 + dphi2) -
    h23 * (2 * dtheta1 * dphi3 + dphi3 * dphi3);

  const C3 = -h13 * (dtheta1 + dphi2) * (dtheta1 + dphi2) - h23 * dphi2 * dphi2;

  return [C1, C2, C3];
}

// ── Gravity ───────────────────────────────────────────────────────────────────

function gravityVector3(
  q: [number, number, number],
  p: TripleParams,
): [number, number, number] {
  const theta1 = q[0];
  const phi2 = q[1];
  const phi3 = q[2];

  const theta2 = theta1 + phi2;
  const theta3 = theta1 + phi2 + phi3;

  const s1 = Math.sin(theta1);
  const s2 = Math.sin(theta2);
  const s3 = Math.sin(theta3);

  const m23 = m23eff(p);
  const m3 = m3eff(p);

  const G1 =
    (p.m1 + m23) * p.g * p.L1 * s1 +
    m23 * p.g * p.L2 * s2 +
    m3 * p.g * p.L3 * s3;

  const G2 = m23 * p.g * p.L2 * s2 + m3 * p.g * p.L3 * s3;

  const G3 = m3 * p.g * p.L3 * s3;

  return [G1, G2, G3];
}

// ── Friction ──────────────────────────────────────────────────────────────────

export function frictionTorqueVector3(
  qdot: [number, number, number],
  p: TripleParams,
): [number, number, number] {
  assertFinite(qdot[0], "dtheta1");
  assertFinite(qdot[1], "dphi2");
  assertFinite(qdot[2], "dphi3");
  return [-p.b1 * qdot[0], -p.b2 * qdot[1], -p.b3 * qdot[2]];
}

// ── 3×3 linear solve ──────────────────────────────────────────────────────────

function solve3x3(
  M: number[][],
  rhs: [number, number, number],
): [number, number, number] {
  // Compute determinant
  const det =
    M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
    M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
    M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);

  if (Math.abs(det) < 1e-15) throw new Error("Singular mass matrix");

  // Cramer's rule: x_i = det(M_i) / det(M)
  const det1 =
    rhs[0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
    M[0][1] * (rhs[1] * M[2][2] - M[1][2] * rhs[2]) +
    M[0][2] * (rhs[1] * M[2][1] - M[1][1] * rhs[2]);

  const det2 =
    M[0][0] * (rhs[1] * M[2][2] - M[1][2] * rhs[2]) -
    rhs[0] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
    M[0][2] * (M[1][0] * rhs[2] - rhs[1] * M[2][0]);

  const det3 =
    M[0][0] * (M[1][1] * rhs[2] - rhs[1] * M[2][1]) -
    M[0][1] * (M[1][0] * rhs[2] - rhs[1] * M[2][0]) +
    rhs[0] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);

  return [det1 / det, det2 / det, det3 / det];
}

// ── Equations of motion ───────────────────────────────────────────────────────

/**
 * M(q)·q̈ = τ_drive − C − G − F
 *
 * Pre:  state has 6 finite elements.
 * Post: state_dot has 6 finite elements.
 */
export function equationsOfMotion3(
  state: StateTriple,
  t: number,
  p: TripleParams,
  torqueFunc: TorqueFuncTriple,
): StateTriple {
  state.forEach((v, i) => assertFinite(v, `state[${i}]`));

  const q: [number, number, number] = [state[0], state[1], state[2]];
  const qdot: [number, number, number] = [state[3], state[4], state[5]];

  const M = massMatrix3(q, p);
  const C = coriolisVector3(q, qdot, p);
  const G = gravityVector3(q, p);
  const [tf1, tf2, tf3] = frictionTorqueVector3(qdot, p);

  const [tau1, tau2, tau3] = torqueFunc(t);

  const rhs: [number, number, number] = [
    tau1 + tf1 - C[0] - G[0],
    tau2 + tf2 - C[1] - G[1],
    tau3 + tf3 - C[2] - G[2],
  ];

  const [qdd1, qdd2, qdd3] = solve3x3(M, rhs);
  const dot: StateTriple = [qdot[0], qdot[1], qdot[2], qdd1, qdd2, qdd3];
  dot.forEach((v, i) => assertFinite(v, `state_dot[${i}]`));
  return dot;
}

// ── Forward kinematics ────────────────────────────────────────────────────────

export interface Positions3 {
  shoulder: [number, number];
  elbow: [number, number];
  wrist: [number, number];
  tip: [number, number];
}

export function forwardKinematics3(
  q: [number, number, number],
  p: TripleParams,
): Positions3 {
  const theta1 = q[0];
  const theta2 = q[0] + q[1];
  const theta3 = q[0] + q[1] + q[2];

  const shoulder: [number, number] = [0, 0];
  const elbow: [number, number] = [
    p.L1 * Math.sin(theta1),
    -p.L1 * Math.cos(theta1),
  ];
  const wrist: [number, number] = [
    elbow[0] + p.L2 * Math.sin(theta2),
    elbow[1] - p.L2 * Math.cos(theta2),
  ];
  const tip: [number, number] = [
    wrist[0] + p.L3 * Math.sin(theta3),
    wrist[1] - p.L3 * Math.cos(theta3),
  ];

  return { shoulder, elbow, wrist, tip };
}

// ── RK4 integrator ────────────────────────────────────────────────────────────

function rk4Step3(
  state: StateTriple,
  t: number,
  dt: number,
  p: TripleParams,
  tf: TorqueFuncTriple,
): StateTriple {
  const f = (s: StateTriple, ti: number): StateTriple =>
    equationsOfMotion3(s, ti, p, tf);
  const add = (a: StateTriple, b: StateTriple, scale: number): StateTriple =>
    a.map((v, i) => v + b[i] * scale) as StateTriple;

  const k1 = f(state, t);
  const k2 = f(add(state, k1, dt / 2), t + dt / 2);
  const k3 = f(add(state, k2, dt / 2), t + dt / 2);
  const k4 = f(add(state, k3, dt), t + dt);

  return state.map(
    (v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]),
  ) as StateTriple;
}

// ── Simulation ────────────────────────────────────────────────────────────────

export interface SimulationResult3 {
  t: number[];
  states: StateTriple[];
  params: TripleParams;
  torqueFunc: TorqueFuncTriple;
}

/**
 * Integrate equations of motion via fixed-step RK4.
 *
 * Pre:  initialState has 6 finite values, tEnd > 0, dt ∈ (0, tEnd).
 * Post: result.t.length >= 2, all state values finite.
 */
export function runSimulation3(
  params: TripleParams,
  initialState: StateTriple,
  tEnd: number,
  torqueFunc: TorqueFuncTriple,
  dt: number = 0.005,
): SimulationResult3 {
  initialState.forEach((v, i) => assertFinite(v, `initialState[${i}]`));
  if (!(tEnd > 0)) throw new RangeError("[DbC] tEnd must be > 0");
  if (!(dt > 0 && dt < tEnd))
    throw new RangeError("[DbC] dt must be in (0, tEnd)");

  const t: number[] = [];
  const states: StateTriple[] = [];
  let state: StateTriple = [...initialState] as StateTriple;
  let time = 0;

  while (time <= tEnd + 1e-10) {
    t.push(time);
    states.push([...state] as StateTriple);
    state = rk4Step3(state, time, dt, params, torqueFunc);
    time += dt;
  }

  if (t.length < 2)
    throw new Error("[DbC post] Simulation must produce ≥ 2 timesteps");
  return { t, states, params, torqueFunc };
}

// ── Polynomial torque builder ──────────────────────────────────────────────────

export function makePolynomialTorque3(
  coeffsShoulder: number[],
  coeffsElbow: number[],
  coeffsWrist: number[],
): TorqueFuncTriple {
  if (
    coeffsShoulder.length < 1 ||
    coeffsElbow.length < 1 ||
    coeffsWrist.length < 1
  )
    throw new RangeError("[DbC] coefficient arrays must have ≥ 1 element");

  const polyval = (coeffs: number[], t: number): number =>
    coeffs.reduce((acc, c, i) => acc + c * t ** i, 0);

  return (t: number): [number, number, number] => [
    polyval(coeffsShoulder, t),
    polyval(coeffsElbow, t),
    polyval(coeffsWrist, t),
  ];
}
