/**
 * Double Pendulum Golf Swing Physics Engine — TypeScript.
 *
 * Model: 2-segment pendulum (arms + shaft) with clubhead mass at tip.
 *
 * Design by Contract (DbC):
 *   - Pre-conditions checked via assertFinite / assertPositive helpers.
 *   - Post-conditions asserted on mass matrix symmetry and state finiteness.
 *   - All functions are pure (no side effects).
 *
 * DRY: polynomial torque construction shared; joint-limit penalty reusable;
 *       unit conversions factored into units.ts.
 *
 * @module physics
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

export interface PendulumParams {
    m1: number;       // kg — arms mass
    m2: number;       // kg — shaft mass
    mClub: number;    // kg — clubhead mass (point mass at tip)
    L1: number;       // m  — arms length (shoulder to wrist)
    L2: number;       // m  — shaft length (wrist to clubhead)
    g: number;        // m/s²
    b1: number;       // N·m·s/rad — viscous damping shoulder joint
    b2: number;       // N·m·s/rad — viscous damping wrist joint
    mu1: number;      // N·m — Coulomb friction shoulder
    mu2: number;      // N·m — Coulomb friction wrist
}

export interface JointLimits {
    phiMin: number;   // rad — minimum wrist angle (relative)
    phiMax: number;   // rad — maximum wrist angle (relative)
    stiffness: number; // N·m/rad — penalty spring stiffness
    damping: number;   // N·m·s/rad — penalty damping
}

export interface TorqueClamp {
    maxTorque1: number; // N·m — max absolute shoulder torque
    maxTorque2: number; // N·m — max absolute wrist torque
}

/** Default joint limits: ±90° wrist ROM */
export const DEFAULT_JOINT_LIMITS: JointLimits = {
    phiMin: -Math.PI / 2,
    phiMax: Math.PI / 2,
    stiffness: 500,
    damping: 20,
};

/** Default torque clamp: no clamping */
export const DEFAULT_TORQUE_CLAMP: TorqueClamp = {
    maxTorque1: Infinity,
    maxTorque2: Infinity,
};

/** [theta1, phi, dtheta1, dphi] */
export type State = [number, number, number, number];

export type TorqueFunc = (t: number) => [number, number];

/** DbC-validated constructor for PendulumParams. */
export function makePendulumParams(p: PendulumParams): PendulumParams {
    assertPositive(p.m1, 'm1'); assertPositive(p.m2, 'm2');
    assertNonNeg(p.mClub, 'mClub');
    assertPositive(p.L1, 'L1'); assertPositive(p.L2, 'L2');
    assertNonNeg(p.g, 'g');
    assertNonNeg(p.b1, 'b1'); assertNonNeg(p.b2, 'b2');
    assertNonNeg(p.mu1, 'mu1'); assertNonNeg(p.mu2, 'mu2');
    return { ...p };
}

// ── Effective mass helper (DRY: used throughout) ─────────────────────────────

/** Effective mass of segment 2 = shaft mass + clubhead mass. */
function m2eff(p: PendulumParams): number {
    return p.m2 + p.mClub;
}

// ── Mass matrix ───────────────────────────────────────────────────────────────

/**
 * 2×2 inertia matrix as flat [M11, M12, M21, M22].
 *
 * The clubhead is modeled as a point mass at the tip of segment 2.
 * Shaft is treated as a uniform rod of mass m2, plus point mass mClub at L2.
 *
 * Pre:  phi finite.
 * Post: symmetric (M12 === M21), M22 > 0.
 */
export function massMatrix(phi: number, p: PendulumParams): [number, number, number, number] {
    assertFinite(phi, 'phi');
    const c = Math.cos(phi);
    const me = m2eff(p);
    const M11 = (p.m1 + me) * p.L1 ** 2 + me * p.L2 ** 2 + 2 * me * p.L1 * p.L2 * c;
    const M12 = me * p.L2 ** 2 + me * p.L1 * p.L2 * c;
    const M22 = me * p.L2 ** 2;
    // Post: symmetry trivially satisfied (M12 == M21 by construction)
    if (!(M22 > 0)) throw new Error('[DbC post] M22 must be positive');
    return [M11, M12, M12, M22];
}

export function massMatrixComponents(phi: number, p: PendulumParams) {
    const [M11, M12, , M22] = massMatrix(phi, p);
    return { M11, M12, M21: M12, M22 };
}

// ── Coriolis ──────────────────────────────────────────────────────────────────

function coriolisVector(phi: number, dtheta1: number, dphi: number,
    p: PendulumParams): [number, number] {
    assertFinite(phi, 'phi'); assertFinite(dtheta1, 'dtheta1'); assertFinite(dphi, 'dphi');
    const me = m2eff(p);
    const h = -me * p.L1 * p.L2 * Math.sin(phi);
    return [h * (2 * dtheta1 * dphi + dphi ** 2), -h * dtheta1 ** 2];
}

export interface ForceSourceTerms {
    coriolis: [number, number];
    squaredSpeed: [number, number];
    gravity: [number, number];
    damping: [number, number];
    applied: [number, number];
}

/**
 * Coordinate-explicit generalized-force terms used by the optimization lab.
 *
 * ``coriolis`` is the cross-speed part and ``squaredSpeed`` is the remaining
 * velocity-quadratic part.  Both are source terms on the right-hand side of
 * M(q)qdd, so their signs are the negative of the corresponding C-vector
 * terms.  This split is coordinate dependent and is not a muscle-force model.
 */
export function generalizedForceSources(
    state: State,
    p: PendulumParams,
    applied: [number, number],
): ForceSourceTerms {
    state.forEach((value, index) => assertFinite(value, `state[${index}]`));
    applied.forEach((value, index) => assertFinite(value, `applied[${index}]`));
    const [theta1, phi, dtheta1, dphi] = state;
    const coupling = -m2eff(p) * p.L1 * p.L2 * Math.sin(phi);
    const gravity = gravityVector(theta1, phi, p);
    const damping = frictionTorqueVector(dtheta1, dphi, p);
    return {
        coriolis: [-2 * coupling * dtheta1 * dphi, 0],
        squaredSpeed: [-coupling * dphi ** 2, coupling * dtheta1 ** 2],
        gravity: [-gravity[0], -gravity[1]],
        damping,
        applied: [...applied],
    };
}

// ── Gravity ───────────────────────────────────────────────────────────────────

function gravityVector(theta1: number, phi: number,
    p: PendulumParams): [number, number] {
    const me = m2eff(p);
    const a2 = theta1 + phi;
    const G1 = (p.m1 + me) * p.g * p.L1 * Math.sin(theta1) + me * p.g * p.L2 * Math.sin(a2);
    const G2 = me * p.g * p.L2 * Math.sin(a2);
    return [G1, G2];
}

// ── Friction ──────────────────────────────────────────────────────────────────

export function frictionTorqueVector(dtheta1: number, dphi: number,
    p: PendulumParams): [number, number] {
    assertFinite(dtheta1, 'dtheta1'); assertFinite(dphi, 'dphi');
    const sign = (v: number) => (v > 0 ? 1 : v < 0 ? -1 : 0);
    return [
        -p.b1 * dtheta1 - p.mu1 * sign(dtheta1),
        -p.b2 * dphi - p.mu2 * sign(dphi),
    ];
}

// ── Joint limit penalty torque (smooth barrier) ─────────────────────────────

/**
 * Smooth joint limit penalty using a cubic barrier that activates within
 * a transition zone near the limits. Returns [tau_penalty_1, tau_penalty_2].
 *
 * Pre: limits.stiffness >= 0, limits.damping >= 0.
 * Post: penalty is 0 when phi is within limits.
 */
export function jointLimitTorque(
    phi: number, dphi: number, limits: JointLimits
): [number, number] {
    assertFinite(phi, 'phi');
    assertFinite(dphi, 'dphi');
    let tau2 = 0;

    // Smooth penalty: cubic ramp that grows as phi penetrates beyond limit
    // Transition zone: 0.05 rad (~3 degrees) of smooth onset
    const transitionZone = 0.05;

    if (phi < limits.phiMin) {
        const penetration = limits.phiMin - phi;
        const blend = Math.min(1, penetration / transitionZone);
        const smoothBlend = blend * blend * (3 - 2 * blend); // Hermite smoothstep
        tau2 = smoothBlend * (limits.stiffness * penetration + limits.damping * Math.max(0, -dphi));
    } else if (phi > limits.phiMax) {
        const penetration = phi - limits.phiMax;
        const blend = Math.min(1, penetration / transitionZone);
        const smoothBlend = blend * blend * (3 - 2 * blend);
        tau2 = -smoothBlend * (limits.stiffness * penetration + limits.damping * Math.max(0, dphi));
    }

    // Joint limits only affect the wrist (joint 2); no direct effect on shoulder
    // However, the reaction propagates through the coupled dynamics
    return [0, tau2];
}

// ── Torque clamping (DRY helper) ────────────────────────────────────────────

/**
 * Clamp a torque pair to saturation limits.
 * Pre: clamp values > 0.
 * Post: |result[i]| <= clamp[i].
 */
export function clampTorque(
    tau: [number, number], clamp: TorqueClamp
): [number, number] {
    return [
        Math.max(-clamp.maxTorque1, Math.min(clamp.maxTorque1, tau[0])),
        Math.max(-clamp.maxTorque2, Math.min(clamp.maxTorque2, tau[1])),
    ];
}

// ── 2×2 linear solve ─────────────────────────────────────────────────────────

function solve2x2(M: [number, number, number, number], rhs: [number, number]): [number, number] {
    const [a, b, c, d] = M;
    const det = a * d - b * c;
    if (Math.abs(det) < 1e-15) throw new Error('Singular mass matrix');
    return [(d * rhs[0] - b * rhs[1]) / det, (a * rhs[1] - c * rhs[0]) / det];
}

// ── Equations of motion ───────────────────────────────────────────────────────

/**
 * M(q)·q̈ = τ_drive + τ_friction + τ_joint_limit − C − G
 *
 * Pre:  state has 4 finite elements.
 * Post: state_dot has 4 finite elements.
 */
export function equationsOfMotionMut(
    state: State, t: number, p: PendulumParams,
    torqueFunc: TorqueFunc,
    outDot: State,
    limits?: JointLimits,
    clamp?: TorqueClamp,
): void {
    state.forEach((v, i) => assertFinite(v, `state[${i}]`));
    const [theta1, phi, dtheta1, dphi] = state;
    const M = massMatrix(phi, p);
    const C = coriolisVector(phi, dtheta1, dphi, p);
    const G = gravityVector(theta1, phi, p);
    let [tau1, tau2] = torqueFunc(t);

    if (clamp) [tau1, tau2] = clampTorque([tau1, tau2], clamp);

    const [tf1, tf2] = frictionTorqueVector(dtheta1, dphi, p);

    let jl1 = 0, jl2 = 0;
    if (limits) [jl1, jl2] = jointLimitTorque(phi, dphi, limits);

    const rhs: [number, number] = [
        tau1 + tf1 + jl1 - C[0] - G[0],
        tau2 + tf2 + jl2 - C[1] - G[1],
    ];
    const [qdd1, qdd2] = solve2x2(M, rhs);

    outDot[0] = dtheta1;
    outDot[1] = dphi;
    outDot[2] = qdd1;
    outDot[3] = qdd2;
    outDot.forEach((v, i) => assertFinite(v, `state_dot[${i}]`));
}

export function equationsOfMotion(
    state: State, t: number, p: PendulumParams,
    torqueFunc: TorqueFunc,
    limits?: JointLimits,
    clamp?: TorqueClamp,
): State {
    const dot: State = [0, 0, 0, 0];
    equationsOfMotionMut(state, t, p, torqueFunc, dot, limits, clamp);
    return dot;
}

// ── Forward kinematics ────────────────────────────────────────────────────────

export interface Positions {
    shoulder: [number, number];
    wrist: [number, number];
    tip: [number, number];
}

export function forwardKinematics(theta1: number, phi: number, p: PendulumParams): Positions {
    const a2 = theta1 + phi;
    const wx = p.L1 * Math.sin(theta1);
    const wy = -p.L1 * Math.cos(theta1);
    return {
        shoulder: [0, 0],
        wrist: [wx, wy],
        tip: [wx + p.L2 * Math.sin(a2), wy - p.L2 * Math.cos(a2)],
    };
}

// ── Linear velocities of joints ─────────────────────────────────────────────

export interface JointVelocities {
    shoulderSpeed: number;  // m/s (always 0 for fixed pivot)
    wristSpeed: number;     // m/s
    tipSpeed: number;       // m/s
    wristVel: [number, number]; // [vx, vy] m/s
    tipVel: [number, number];   // [vx, vy] m/s
}

/**
 * Compute linear velocities at each joint via Jacobian.
 * Pre: state finite.
 * Post: speeds >= 0.
 */
export function jointVelocities(state: State, p: PendulumParams): JointVelocities {
    const [theta1, phi, dtheta1, dphi] = state;
    const a2 = theta1 + phi;
    const da2 = dtheta1 + dphi;

    // Wrist velocity: d/dt of (L1*sin(theta1), -L1*cos(theta1))
    const vwx = p.L1 * Math.cos(theta1) * dtheta1;
    const vwy = p.L1 * Math.sin(theta1) * dtheta1;

    // Tip velocity: d/dt of wrist + (L2*sin(a2), -L2*cos(a2))
    const vtx = vwx + p.L2 * Math.cos(a2) * da2;
    const vty = vwy + p.L2 * Math.sin(a2) * da2;

    return {
        shoulderSpeed: 0,
        wristSpeed: Math.sqrt(vwx ** 2 + vwy ** 2),
        tipSpeed: Math.sqrt(vtx ** 2 + vty ** 2),
        wristVel: [vwx, vwy],
        tipVel: [vtx, vty],
    };
}

// ── Base (shoulder) reaction force ──────────────────────────────────────────

export interface BaseForce {
    fx: number; // N — horizontal force at pivot
    fy: number; // N — vertical force at pivot
    magnitude: number; // N
}

/**
 * Compute the reaction force at the base (shoulder pivot).
 * F_base = sum of (m_i * a_cm_i) + sum(m_i * g_vec)
 * Using Newton's second law on the whole system.
 *
 * Pre: state and accelerations finite.
 * Post: magnitude >= 0.
 */
export function baseForce(state: State, qddot: [number, number], p: PendulumParams): BaseForce {
    const [theta1, phi, dtheta1, dphi] = state;
    const [qdd1, qdd2] = qddot;
    const a2 = theta1 + phi;
    const da2 = dtheta1 + dphi;
    const dda2 = qdd1 + qdd2;
    const me = m2eff(p);

    // Center of mass accelerations for arm (at L1/2) and shaft+clubhead (at wrist + L2)
    // Arm COM acceleration (uniform rod, COM at L1/2):
    const ax1 = (p.L1 / 2) * (Math.cos(theta1) * qdd1 - Math.sin(theta1) * dtheta1 ** 2);
    const ay1 = (p.L1 / 2) * (Math.sin(theta1) * qdd1 + Math.cos(theta1) * dtheta1 ** 2);

    // Wrist acceleration:
    const awx = p.L1 * (Math.cos(theta1) * qdd1 - Math.sin(theta1) * dtheta1 ** 2);
    const awy = p.L1 * (Math.sin(theta1) * qdd1 + Math.cos(theta1) * dtheta1 ** 2);

    // Tip acceleration (for clubhead point mass):
    const atx = awx + p.L2 * (Math.cos(a2) * dda2 - Math.sin(a2) * da2 ** 2);
    const aty = awy + p.L2 * (Math.sin(a2) * dda2 + Math.cos(a2) * da2 ** 2);

    // Shaft COM at L2/2 from wrist:
    const asx = awx + (p.L2 / 2) * (Math.cos(a2) * dda2 - Math.sin(a2) * da2 ** 2);
    const asy = awy + (p.L2 / 2) * (Math.sin(a2) * dda2 + Math.cos(a2) * da2 ** 2);

    // F = sum(m_i * a_i) + sum(m_i * [0, g])
    const fx = p.m1 * ax1 + p.m2 * asx + p.mClub * atx;
    const fy = p.m1 * ay1 + p.m2 * asy + p.mClub * aty - (p.m1 + me) * p.g;

    return { fx, fy, magnitude: Math.sqrt(fx ** 2 + fy ** 2) };
}

// ── Zero-torque counterfactual ───────────────────────────────────────────────

/**
 * Compute accelerations under zero driving torque (ZTCF).
 * Only gravity, Coriolis, friction, and joint limits act.
 */
export function ztcfAccelerations(
    state: State, p: PendulumParams, limits?: JointLimits
): [number, number] {
    const [theta1, phi, dtheta1, dphi] = state;
    const M = massMatrix(phi, p);
    const C = coriolisVector(phi, dtheta1, dphi, p);
    const G = gravityVector(theta1, phi, p);
    const [tf1, tf2] = frictionTorqueVector(dtheta1, dphi, p);

    let jl1 = 0, jl2 = 0;
    if (limits) {
        [jl1, jl2] = jointLimitTorque(phi, dphi, limits);
    }

    const rhs: [number, number] = [tf1 + jl1 - C[0] - G[0], tf2 + jl2 - C[1] - G[1]];
    return solve2x2(M, rhs);
}

/**
 * Control vector: difference between actual and zero-torque forces at base.
 * Represents the contribution of active torques to the base reaction force.
 */
export function controlVector(
    state: State, qddotActual: [number, number],
    p: PendulumParams, limits?: JointLimits
): { cvx: number; cvy: number; magnitude: number } {
    const qddotZtcf = ztcfAccelerations(state, p, limits);
    const fActual = baseForce(state, qddotActual, p);
    const fZtcf = baseForce(state, qddotZtcf, p);
    const cvx = fActual.fx - fZtcf.fx;
    const cvy = fActual.fy - fZtcf.fy;
    return { cvx, cvy, magnitude: Math.sqrt(cvx ** 2 + cvy ** 2) };
}

// ── Energy ────────────────────────────────────────────────────────────────────

export function kineticEnergy(state: State, p: PendulumParams): number {
    const [, phi, dtheta1, dphi] = state;
    const [M11, M12, , M22] = massMatrix(phi, p);
    return 0.5 * (M11 * dtheta1 ** 2 + 2 * M12 * dtheta1 * dphi + M22 * dphi ** 2);
}

export function potentialEnergy(state: State, p: PendulumParams): number {
    const [theta1, phi] = state;
    const me = m2eff(p);
    const a2 = theta1 + phi;
    return -(p.m1 + me) * p.g * p.L1 * Math.cos(theta1) - me * p.g * p.L2 * Math.cos(a2);
}

export function totalEnergy(state: State, p: PendulumParams): number {
    return kineticEnergy(state, p) + potentialEnergy(state, p);
}

// ── Polynomial torque builder ─────────────────────────────────────────────────

/** τ(t) = c₀ + c₁t + c₂t² + …
 * Pre: each coefficient array has ≥ 1 element.
 */
export function makePolynomialTorque(
    coeffsShoulder: number[], coeffsWrist: number[]
): TorqueFunc {
    if (coeffsShoulder.length < 1 || coeffsWrist.length < 1)
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

    return (t: number): [number, number] => [
        polyval(coeffsShoulder, t),
        polyval(coeffsWrist, t),
    ];
}

// ── RK4 integrator ────────────────────────────────────────────────────────────

/** Classic 4th-order Runge-Kutta step with out-parameters to prevent GC pauses. */
function rk4StepMut(
    state: State, t: number, dt: number, p: PendulumParams,
    tf: TorqueFunc, limits: JointLimits | undefined, clamp: TorqueClamp | undefined,
    k1: State, k2: State, k3: State, k4: State, tmp: State
): void {
    equationsOfMotionMut(state, t, p, tf, k1, limits, clamp);

    for (let i = 0; i < 4; i++) tmp[i] = state[i] + (dt / 2) * k1[i];
    equationsOfMotionMut(tmp, t + dt / 2, p, tf, k2, limits, clamp);

    for (let i = 0; i < 4; i++) tmp[i] = state[i] + (dt / 2) * k2[i];
    equationsOfMotionMut(tmp, t + dt / 2, p, tf, k3, limits, clamp);

    for (let i = 0; i < 4; i++) tmp[i] = state[i] + dt * k3[i];
    equationsOfMotionMut(tmp, t + dt, p, tf, k4, limits, clamp);

    for (let i = 0; i < 4; i++) {
        state[i] = state[i] + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }
}

// ── Simulation ────────────────────────────────────────────────────────────────

export interface SimulationResult {
    t: number[];
    states: State[];
    params: PendulumParams;
    torqueFunc: TorqueFunc;
    limits?: JointLimits;
    clamp?: TorqueClamp;
}

/**
 * Integrate equations of motion via fixed-step RK4.
 *
 * Pre:  initialState has 4 finite values, tEnd > 0, dt ∈ (0, tEnd).
 * Post: result.t.length >= 2, all state values finite.
 */
export function runSimulation(
    params: PendulumParams,
    initialState: State,
    tEnd: number,
    torqueFunc: TorqueFunc,
    dt: number = 0.005,
    limits?: JointLimits,
    clamp?: TorqueClamp,
): SimulationResult {
    initialState.forEach((v, i) => assertFinite(v, `initialState[${i}]`));
    if (!(tEnd > 0)) throw new RangeError('[DbC] tEnd must be > 0');
    if (!(dt > 0 && dt < tEnd)) throw new RangeError('[DbC] dt must be in (0, tEnd)');

    const t: number[] = [];
    const states: State[] = [];
    // ⚡ Bolt Optimization: Replace spread syntax with pre-allocated array copy to prevent GC pauses
    const state: State = new Array(4) as unknown as State;
    for (let i = 0; i < 4; i++) state[i] = initialState[i];
    let time = 0;

    // ⚡ Bolt Optimization: Pre-allocate buffers for RK4 to eliminate GC pauses
    const k1 = [0, 0, 0, 0] as State;
    const k2 = [0, 0, 0, 0] as State;
    const k3 = [0, 0, 0, 0] as State;
    const k4 = [0, 0, 0, 0] as State;
    const tmp = [0, 0, 0, 0] as State;

    while (time <= tEnd + 1e-10) {
        t.push(time);
        // ⚡ Bolt Optimization: Replace [...state] spread with manual copy in high-frequency RK4 loop
        const s = new Array(4) as unknown as State;
        for (let i = 0; i < 4; i++) s[i] = state[i];
        states.push(s);
        rk4StepMut(state, time, dt, params, torqueFunc, limits, clamp, k1, k2, k3, k4, tmp);
        time += dt;
    }

    if (t.length < 2) throw new Error('[DbC post] Simulation must produce ≥ 2 timesteps');
    return { t, states, params, torqueFunc, limits, clamp };
}

// ── Compute accelerations for a given frame ──────────────────────────────────

/**
 * Compute q̈ at a given simulation frame. Useful for force/control vector.
 */
export function computeAccelerations(
    state: State, t: number, p: PendulumParams,
    torqueFunc: TorqueFunc, limits?: JointLimits, clamp?: TorqueClamp,
): [number, number] {
    const dot = equationsOfMotion(state, t, p, torqueFunc, limits, clamp);
    return [dot[2], dot[3]];
}
