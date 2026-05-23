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
    if (!isFinite(v)) throw new RangeError(`[DbC] ${name} must be finite, got ${v}`);
}

function assertPositive(v: number, name: string): void {
    if (!(v > 0)) throw new RangeError(`[DbC] ${name} must be > 0, got ${v}`);
}

function assertNonNeg(v: number, name: string): void {
    if (!(v >= 0)) throw new RangeError(`[DbC] ${name} must be ≥ 0, got ${v}`);
}

// ── Data structures ───────────────────────────────────────────────────────────

export interface TripleParams {
    m1: number;        // kg — segment 1 (shoulder to elbow) mass
    m2: number;        // kg — segment 2 (elbow to wrist) mass
    m3: number;        // kg — segment 3 (wrist to tip) mass
    mClub: number;     // kg — clubhead mass (point mass at tip)
    L1: number;        // m  — segment 1 length
    L2: number;        // m  — segment 2 length
    L3: number;        // m  — segment 3 length
    g: number;         // m/s²
    b1: number;        // N·m·s/rad — viscous damping shoulder
    b2: number;        // N·m·s/rad — viscous damping elbow
    b3: number;        // N·m·s/rad — viscous damping wrist
}

/** [theta1, phi2, phi3, dtheta1, dphi2, dphi3] */
export type StateTriple = [number, number, number, number, number, number];

export type TorqueFuncTriple = (t: number) => [number, number, number];

/** DbC-validated constructor for TripleParams. */
export function makeTripleParams(p: TripleParams): TripleParams {
    assertPositive(p.m1, 'm1'); assertPositive(p.m2, 'm2'); assertPositive(p.m3, 'm3');
    assertNonNeg(p.mClub, 'mClub');
    assertPositive(p.L1, 'L1'); assertPositive(p.L2, 'L2'); assertPositive(p.L3, 'L3');
    assertNonNeg(p.g, 'g');
    assertNonNeg(p.b1, 'b1'); assertNonNeg(p.b2, 'b2'); assertNonNeg(p.b3, 'b3');
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
export function massMatrix3(q: [number, number, number], p: TripleParams): number[][] {
    assertFinite(q[1], 'phi2');
    assertFinite(q[2], 'phi3');

    const phi2 = q[1];
    const phi3 = q[2];
    const phi23 = phi2 + phi3;

    const c2 = Math.cos(phi2);
    const c3 = Math.cos(phi3);
    const c23 = Math.cos(phi23);

    const m23 = m23eff(p);
    const m3 = m3eff(p);

    const M00 = (p.m1 + m23) * p.L1 * p.L1
              + m23 * p.L2 * p.L2
              + m3 * p.L3 * p.L3
              + 2 * m23 * p.L1 * p.L2 * c2
              + 2 * m3 * p.L1 * p.L3 * c23
              + 2 * m3 * p.L2 * p.L3 * c3;

    const M01 = m23 * p.L2 * p.L2
              + m3 * p.L1 * p.L2 * c2
              + m3 * p.L3 * p.L3
              + m3 * p.L2 * p.L3 * c3
              + m3 * p.L1 * p.L3 * c23;

    const M02 = m3 * p.L3 * p.L3
              + m3 * p.L1 * p.L3 * c23
              + m3 * p.L2 * p.L3 * c3;

    const M11 = m23 * p.L2 * p.L2
              + m3 * p.L3 * p.L3
              + 2 * m3 * p.L2 * p.L3 * c3;

    const M12 = m3 * p.L3 * p.L3 + m3 * p.L2 * p.L3 * c3;

    const M22 = m3 * p.L3 * p.L3;

    if (!(M22 > 0)) throw new Error('[DbC post] M22 must be positive');

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
    p: TripleParams
): [number, number, number] {
    assertFinite(q[1], 'phi2'); assertFinite(q[2], 'phi3');
    assertFinite(qdot[0], 'dtheta1'); assertFinite(qdot[1], 'dphi2'); assertFinite(qdot[2], 'dphi3');

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

    const C1 = h12 * (2 * dtheta1 * dphi2 + dphi2 * dphi2)
             + h13 * (2 * dtheta1 * (dphi2 + dphi3) + (dphi2 + dphi3) * (dphi2 + dphi3))
             + h23 * (2 * dphi2 * dphi3 + dphi3 * dphi3);

    const C2 = -h12 * dtheta1 * dtheta1
             - h13 * (dtheta1 + dphi2) * (dtheta1 + dphi2)
             - h23 * (2 * dtheta1 * dphi3 + dphi3 * dphi3);

    const C3 = -h13 * (dtheta1 + dphi2) * (dtheta1 + dphi2)
             - h23 * dphi2 * dphi2;

    return [C1, C2, C3];
}

// ── Gravity ───────────────────────────────────────────────────────────────────

function gravityVector3(
    q: [number, number, number],
    p: TripleParams
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

    const G1 = (p.m1 + m23) * p.g * p.L1 * s1
             + m23 * p.g * p.L2 * s2
             + m3 * p.g * p.L3 * s3;

    const G2 = m23 * p.g * p.L2 * s2 + m3 * p.g * p.L3 * s3;

    const G3 = m3 * p.g * p.L3 * s3;

    return [G1, G2, G3];
}

// ── Friction ──────────────────────────────────────────────────────────────────

export function frictionTorqueVector3(
    qdot: [number, number, number],
    p: TripleParams
): [number, number, number] {
    assertFinite(qdot[0], 'dtheta1'); assertFinite(qdot[1], 'dphi2'); assertFinite(qdot[2], 'dphi3');
    return [
        -p.b1 * qdot[0],
        -p.b2 * qdot[1],
        -p.b3 * qdot[2],
    ];
}

// ── 3×3 linear solve ──────────────────────────────────────────────────────────

function solve3x3(M: number[][], rhs: [number, number, number]): [number, number, number] {
    // Compute determinant
    const det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
              - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
              + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);

    if (Math.abs(det) < 1e-15) throw new Error('Singular mass matrix');

    // Cramer's rule: x_i = det(M_i) / det(M)
    const det1 = rhs[0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
               - M[0][1] * (rhs[1] * M[2][2] - M[1][2] * rhs[2])
               + M[0][2] * (rhs[1] * M[2][1] - M[1][1] * rhs[2]);

    const det2 = M[0][0] * (rhs[1] * M[2][2] - M[1][2] * rhs[2])
               - rhs[0] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
               + M[0][2] * (M[1][0] * rhs[2] - rhs[1] * M[2][0]);

    const det3 = M[0][0] * (M[1][1] * rhs[2] - rhs[1] * M[2][1])
               - M[0][1] * (M[1][0] * rhs[2] - rhs[1] * M[2][0])
               + rhs[0] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);

    return [det1 / det, det2 / det, det3 / det];
}

// ── Equations of motion ───────────────────────────────────────────────────────

/**
 * M(q)·q̈ = τ_drive − C − G − F
 *
 * Pre:  state has 6 finite elements.
 * Post: state_dot has 6 finite elements.
 */
export function equationsOfMotion3Mut(
    state: StateTriple,
    t: number,
    p: TripleParams,
    torqueFunc: TorqueFuncTriple,
    outDot: StateTriple,
): void {
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

    outDot[0] = qdot[0];
    outDot[1] = qdot[1];
    outDot[2] = qdot[2];
    outDot[3] = qdd1;
    outDot[4] = qdd2;
    outDot[5] = qdd3;
    outDot.forEach((v, i) => assertFinite(v, `state_dot[${i}]`));
}

export function equationsOfMotion3(
    state: StateTriple,
    t: number,
    p: TripleParams,
    torqueFunc: TorqueFuncTriple,
): StateTriple {
    const dot: StateTriple = [0, 0, 0, 0, 0, 0];
    equationsOfMotion3Mut(state, t, p, torqueFunc, dot);
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
    p: TripleParams
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

function rk4Step3Mut(
    state: StateTriple,
    t: number,
    dt: number,
    p: TripleParams,
    tf: TorqueFuncTriple,
    k1: StateTriple,
    k2: StateTriple,
    k3: StateTriple,
    k4: StateTriple,
    tmp: StateTriple
): void {
    equationsOfMotion3Mut(state, t, p, tf, k1);

    for (let i = 0; i < 6; i++) tmp[i] = state[i] + (dt / 2) * k1[i];
    equationsOfMotion3Mut(tmp, t + dt / 2, p, tf, k2);

    for (let i = 0; i < 6; i++) tmp[i] = state[i] + (dt / 2) * k2[i];
    equationsOfMotion3Mut(tmp, t + dt / 2, p, tf, k3);

    for (let i = 0; i < 6; i++) tmp[i] = state[i] + dt * k3[i];
    equationsOfMotion3Mut(tmp, t + dt, p, tf, k4);

    for (let i = 0; i < 6; i++) {
        state[i] = state[i] + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }
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
    if (!(tEnd > 0)) throw new RangeError('[DbC] tEnd must be > 0');
    if (!(dt > 0 && dt < tEnd)) throw new RangeError('[DbC] dt must be in (0, tEnd)');

    const t: number[] = [];
    const states: StateTriple[] = [];
    const state: StateTriple = [...initialState] as StateTriple;
    let time = 0;

    // ⚡ Bolt Optimization: Pre-allocate buffers for RK4 to eliminate GC pauses
    const k1 = [0, 0, 0, 0, 0, 0] as StateTriple;
    const k2 = [0, 0, 0, 0, 0, 0] as StateTriple;
    const k3 = [0, 0, 0, 0, 0, 0] as StateTriple;
    const k4 = [0, 0, 0, 0, 0, 0] as StateTriple;
    const tmp = [0, 0, 0, 0, 0, 0] as StateTriple;

    while (time <= tEnd + 1e-10) {
        t.push(time);
        states.push([...state] as StateTriple);
        rk4Step3Mut(state, time, dt, params, torqueFunc, k1, k2, k3, k4, tmp);
        time += dt;
    }

    if (t.length < 2) throw new Error('[DbC post] Simulation must produce ≥ 2 timesteps');
    return { t, states, params, torqueFunc };
}

// ── Polynomial torque builder ──────────────────────────────────────────────────

export function makePolynomialTorque3(
    coeffsShoulder: number[],
    coeffsElbow: number[],
    coeffsWrist: number[]
): TorqueFuncTriple {
    if (coeffsShoulder.length < 1 || coeffsElbow.length < 1 || coeffsWrist.length < 1)
        throw new RangeError('[DbC] coefficient arrays must have ≥ 1 element');

    const polyval = (coeffs: number[], t: number): number => {
        // ⚡ Bolt Optimization: Replace .reduce() and t**i with Horner's method
        // to avoid callback overhead and expensive exponentiation in tight integration loop.
        let acc = 0;
        for (let i = coeffs.length - 1; i >= 0; i--) {
            acc = acc * t + coeffs[i];
        }
        return acc;
    };

    return (t: number): [number, number, number] => [
        polyval(coeffsShoulder, t),
        polyval(coeffsElbow, t),
        polyval(coeffsWrist, t),
    ];
}
