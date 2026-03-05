/**
 * Double Pendulum Physics Engine — TypeScript port.
 *
 * Design by Contract (DbC):
 *   - Pre-conditions checked via assertFinite / assertPositive helpers.
 *   - Post-conditions asserted on mass matrix symmetry and state finiteness.
 *   - All functions are pure (no side effects).
 *
 * DRY: polynomial torque construction shared between presets and editor.
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
    m1: number;  // kg
    m2: number;  // kg
    L1: number;  // m
    L2: number;  // m
    g: number;  // m/s²
    b1: number;  // N·m·s/rad — viscous damping joint 1
    b2: number;  // N·m·s/rad — viscous damping joint 2
    mu1: number; // N·m — Coulomb friction joint 1
    mu2: number; // N·m — Coulomb friction joint 2
}

/** [theta1, phi, dtheta1, dphi] */
export type State = [number, number, number, number];

export type TorqueFunc = (t: number) => [number, number];

/** DbC-validated constructor for PendulumParams. */
export function makePendulumParams(p: PendulumParams): PendulumParams {
    assertPositive(p.m1, 'm1'); assertPositive(p.m2, 'm2');
    assertPositive(p.L1, 'L1'); assertPositive(p.L2, 'L2');
    assertNonNeg(p.g, 'g');
    assertNonNeg(p.b1, 'b1'); assertNonNeg(p.b2, 'b2');
    assertNonNeg(p.mu1, 'mu1'); assertNonNeg(p.mu2, 'mu2');
    return { ...p };
}

// ── Mass matrix ───────────────────────────────────────────────────────────────

/** 2×2 inertia matrix as flat [M11, M12, M21, M22].
 *
 * Pre:  phi finite.
 * Post: symmetric (M12 === M21), M22 > 0.
 */
export function massMatrix(phi: number, p: PendulumParams): [number, number, number, number] {
    assertFinite(phi, 'phi');
    const c = Math.cos(phi);
    const M11 = (p.m1 + p.m2) * p.L1 ** 2 + p.m2 * p.L2 ** 2 + 2 * p.m2 * p.L1 * p.L2 * c;
    const M12 = p.m2 * p.L2 ** 2 + p.m2 * p.L1 * p.L2 * c;
    const M22 = p.m2 * p.L2 ** 2;
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
    const h = -p.m2 * p.L1 * p.L2 * Math.sin(phi);
    return [h * (2 * dtheta1 * dphi + dphi ** 2), -h * dtheta1 ** 2];
}

// ── Gravity ───────────────────────────────────────────────────────────────────

function gravityVector(theta1: number, phi: number,
    p: PendulumParams): [number, number] {
    const a2 = theta1 + phi;
    const G1 = (p.m1 + p.m2) * p.g * p.L1 * Math.sin(theta1) + p.m2 * p.g * p.L2 * Math.sin(a2);
    const G2 = p.m2 * p.g * p.L2 * Math.sin(a2);
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

// ── 2×2 linear solve ─────────────────────────────────────────────────────────

function solve2x2(M: [number, number, number, number], rhs: [number, number]): [number, number] {
    const [a, b, c, d] = M;
    const det = a * d - b * c;
    if (Math.abs(det) < 1e-15) throw new Error('Singular mass matrix');
    return [(d * rhs[0] - b * rhs[1]) / det, (a * rhs[1] - c * rhs[0]) / det];
}

// ── Equations of motion ───────────────────────────────────────────────────────

/** M(q)·q̈ = τ_drive + τ_friction − C − G
 *
 * Pre:  state has 4 finite elements.
 * Post: state_dot has 4 finite elements.
 */
export function equationsOfMotion(state: State, t: number, p: PendulumParams,
    torqueFunc: TorqueFunc): State {
    state.forEach((v, i) => assertFinite(v, `state[${i}]`));
    const [theta1, phi, dtheta1, dphi] = state;
    const M = massMatrix(phi, p);
    const C = coriolisVector(phi, dtheta1, dphi, p);
    const G = gravityVector(theta1, phi, p);
    const [tau1, tau2] = torqueFunc(t);
    const [tf1, tf2] = frictionTorqueVector(dtheta1, dphi, p);
    const rhs: [number, number] = [tau1 + tf1 - C[0] - G[0], tau2 + tf2 - C[1] - G[1]];
    const [qdd1, qdd2] = solve2x2(M, rhs);
    const dot: State = [dtheta1, dphi, qdd1, qdd2];
    dot.forEach((v, i) => assertFinite(v, `state_dot[${i}]`));
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

// ── Energy ────────────────────────────────────────────────────────────────────

export function kineticEnergy(state: State, p: PendulumParams): number {
    const [, phi, dtheta1, dphi] = state;
    const [M11, M12, , M22] = massMatrix(phi, p);
    return 0.5 * (M11 * dtheta1 ** 2 + 2 * M12 * dtheta1 * dphi + M22 * dphi ** 2);
}

export function potentialEnergy(state: State, p: PendulumParams): number {
    const [theta1, phi] = state;
    const a2 = theta1 + phi;
    return -(p.m1 + p.m2) * p.g * p.L1 * Math.cos(theta1) - p.m2 * p.g * p.L2 * Math.cos(a2);
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

    const polyval = (coeffs: number[], t: number): number =>
        coeffs.reduce((acc, c, i) => acc + c * t ** i, 0);

    return (t: number): [number, number] => [
        polyval(coeffsShoulder, t),
        polyval(coeffsWrist, t),
    ];
}

// ── RK4 integrator ────────────────────────────────────────────────────────────

/** Classic 4th-order Runge-Kutta step. */
function rk4Step(state: State, t: number, dt: number, p: PendulumParams,
    tf: TorqueFunc): State {
    const f = (s: State, ti: number): State => equationsOfMotion(s, ti, p, tf);
    const add = (a: State, b: State, scale: number): State =>
        a.map((v, i) => v + b[i] * scale) as State;

    const k1 = f(state, t);
    const k2 = f(add(state, k1, dt / 2), t + dt / 2);
    const k3 = f(add(state, k2, dt / 2), t + dt / 2);
    const k4 = f(add(state, k3, dt), t + dt);

    return state.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])) as State;
}

// ── Simulation ────────────────────────────────────────────────────────────────

export interface SimulationResult {
    t: number[];
    states: State[];
    params: PendulumParams;
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
): SimulationResult {
    initialState.forEach((v, i) => assertFinite(v, `initialState[${i}]`));
    if (!(tEnd > 0)) throw new RangeError('[DbC] tEnd must be > 0');
    if (!(dt > 0 && dt < tEnd)) throw new RangeError('[DbC] dt must be in (0, tEnd)');

    const t: number[] = [];
    const states: State[] = [];
    let state: State = [...initialState] as State;
    let time = 0;

    while (time <= tEnd + 1e-10) {
        t.push(time);
        states.push([...state] as State);
        state = rk4Step(state, time, dt, params, torqueFunc);
        time += dt;
    }

    if (t.length < 2) throw new Error('[DbC post] Simulation must produce ≥ 2 timesteps');
    return { t, states, params };
}
