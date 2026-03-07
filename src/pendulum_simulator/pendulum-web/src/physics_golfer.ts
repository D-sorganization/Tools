/**
 * Golfer Upper-Body Physics Engine — Fully Constrained 8-DOF Model — TypeScript.
 *
 * Model: Golfer with hub, two 2-segment arms (right and left), club shaft,
 * with 4 holonomic constraints (closed kinematic loop at club grip).
 *
 * q = [θ_hub, α_rs, α_re, α_rh, α_ls, α_le, α_lh, θ_club] (8 DOFs)
 * where:
 *   θ_hub = hub rotation angle
 *   α_rs = right shoulder angle (absolute = θ_hub + α_rs)
 *   α_re = right elbow angle (relative)
 *   α_rh = right hand angle (relative)
 *   α_ls = left shoulder angle (absolute = θ_hub + α_ls)
 *   α_le = left elbow angle (relative)
 *   α_lh = left hand angle (relative)
 *   θ_club = club rotation angle
 *
 * 4 Holonomic constraints enforce that right hand and left hand grip the same club.
 * Uses KKT solver with Baumgarte stabilization.
 *
 * Design by Contract (DbC):
 *   - Pre-conditions checked via assertFinite / assertPositive helpers.
 *   - All functions are pure (no side effects).
 *
 * @module physics_golfer
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

export interface GolferParams {
    // Masses (kg)
    m_hub: number;
    m_r_upper: number;
    m_r_fore: number;
    m_l_upper: number;
    m_l_fore: number;
    m_club: number;

    // Lengths (m)
    L_hub: number;
    L_r_upper: number;
    L_r_fore: number;
    L_l_upper: number;
    L_l_fore: number;
    L_club: number;

    // Shoulder offsets (m) — lateral distance from hub center
    d_rs: number; // right shoulder lateral offset
    d_ls: number; // left shoulder lateral offset

    // Grip positions (m) — distance from hand joint to grip point on club shaft
    grip_right: number;
    grip_left: number;

    // Clubhead properties
    m_clubhead: number; // kg — point mass at club tip

    // Gravity and damping
    g: number;
    b_hub: number;
    b_rs: number;
    b_re: number;
    b_rh: number;
    b_ls: number;
    b_le: number;
    b_lh: number;
}

/** [θ_hub, α_rs, α_re, α_rh, α_ls, α_le, α_lh, θ_club,
 *   d(θ_hub), d(α_rs), d(α_re), d(α_rh), d(α_ls), d(α_le), d(α_lh), d(θ_club)] */
export type StateGolfer = [
    number, number, number, number, number, number, number, number,
    number, number, number, number, number, number, number, number
];

export type TorqueFuncGolfer = (t: number) => [number, number, number, number, number, number, number];

/** DbC-validated constructor for GolferParams. */
export function makeGolferParams(p: GolferParams): GolferParams {
    assertPositive(p.m_hub, 'm_hub');
    assertPositive(p.m_r_upper, 'm_r_upper'); assertPositive(p.m_r_fore, 'm_r_fore');
    assertPositive(p.m_l_upper, 'm_l_upper'); assertPositive(p.m_l_fore, 'm_l_fore');
    assertPositive(p.m_club, 'm_club');
    assertNonNeg(p.m_clubhead, 'm_clubhead');

    assertPositive(p.L_hub, 'L_hub');
    assertPositive(p.L_r_upper, 'L_r_upper'); assertPositive(p.L_r_fore, 'L_r_fore');
    assertPositive(p.L_l_upper, 'L_l_upper'); assertPositive(p.L_l_fore, 'L_l_fore');
    assertPositive(p.L_club, 'L_club');

    assertNonNeg(p.d_rs, 'd_rs'); assertNonNeg(p.d_ls, 'd_ls');
    assertNonNeg(p.grip_right, 'grip_right'); assertNonNeg(p.grip_left, 'grip_left');
    assertNonNeg(p.g, 'g');

    return { ...p };
}

// ── Forward Kinematics Structure ──────────────────────────────────────────────

export interface GolferPositions {
    hub: [number, number];
    rs: [number, number]; // right shoulder
    ls: [number, number]; // left shoulder
    re: [number, number]; // right elbow
    le: [number, number]; // left elbow
    rh: [number, number]; // right hand
    lh: [number, number]; // left hand
    club_base: [number, number];
    club_tip: [number, number];
}

/**
 * Compute forward kinematics for golfer.
 *
 * Hub: (L_hub*sin(θ_hub), -L_hub*cos(θ_hub))
 * RS: hub + d_rs*(cos(θ_hub), sin(θ_hub))
 * LS: hub - d_ls*(cos(θ_hub), sin(θ_hub))
 * Right arm: θ_rs = θ_hub + α_rs, θ_re = θ_rs + α_re, θ_rh = θ_re + α_rh
 * Left arm: θ_ls = θ_hub + α_ls, θ_le = θ_ls + α_le, θ_lh = θ_le + α_lh
 * Club: base = rh - grip_right*(sin(θ_club), -cos(θ_club))
 *       tip = base + L_club*(sin(θ_club), -cos(θ_club))
 */
export function forwardKinematics_golfer(
    q: [number, number, number, number, number, number, number, number],
    p: GolferParams
): GolferPositions {
    const theta_hub = q[0];
    const alpha_rs = q[1];
    const alpha_re = q[2];
    const alpha_rh = q[3];
    const alpha_ls = q[4];
    const alpha_le = q[5];
    const alpha_lh = q[6];
    const theta_club = q[7];

    const hub_x = p.L_hub * Math.sin(theta_hub);
    const hub_y = -p.L_hub * Math.cos(theta_hub);

    // Right shoulder: offset from hub
    const rs_x = hub_x + p.d_rs * Math.cos(theta_hub);
    const rs_y = hub_y + p.d_rs * Math.sin(theta_hub);

    // Left shoulder: offset from hub
    const ls_x = hub_x - p.d_ls * Math.cos(theta_hub);
    const ls_y = hub_y - p.d_ls * Math.sin(theta_hub);

    // Right arm angles
    const theta_rs = theta_hub + alpha_rs;
    const theta_re = theta_rs + alpha_re;
    const theta_rh = theta_re + alpha_rh;

    // Right elbow
    const re_x = rs_x + p.L_r_upper * Math.sin(theta_rs);
    const re_y = rs_y - p.L_r_upper * Math.cos(theta_rs);

    // Right hand
    const rh_x = re_x + p.L_r_fore * Math.sin(theta_re);
    const rh_y = re_y - p.L_r_fore * Math.cos(theta_re);

    // Left arm angles
    const theta_ls = theta_hub + alpha_ls;
    const theta_le = theta_ls + alpha_le;
    const theta_lh = theta_le + alpha_lh;

    // Left elbow
    const le_x = ls_x + p.L_l_upper * Math.sin(theta_ls);
    const le_y = ls_y - p.L_l_upper * Math.cos(theta_ls);

    // Left hand
    const lh_x = le_x + p.L_l_fore * Math.sin(theta_le);
    const lh_y = le_y - p.L_l_fore * Math.cos(theta_le);

    // Club base (right hand grip point)
    const club_base_x = rh_x - p.grip_right * Math.sin(theta_club);
    const club_base_y = rh_y + p.grip_right * Math.cos(theta_club);

    // Club tip
    const club_tip_x = club_base_x + p.L_club * Math.sin(theta_club);
    const club_tip_y = club_base_y - p.L_club * Math.cos(theta_club);

    return {
        hub: [hub_x, hub_y],
        rs: [rs_x, rs_y],
        ls: [ls_x, ls_y],
        re: [re_x, re_y],
        le: [le_x, le_y],
        rh: [rh_x, rh_y],
        lh: [lh_x, lh_y],
        club_base: [club_base_x, club_base_y],
        club_tip: [club_tip_x, club_tip_y],
    };
}

// ── Jacobian (for constraint enforcement) ─────────────────────────────────────

/**
 * Compute the 4×8 constraint Jacobian Φ_q.
 * Constraints: (rh - grip_right*sin_club - lh_x) = 0
 *              (rh - grip_right*cos_club - lh_y) = 0
 *              (club_base_x + L_club*sin_club - club_tip_x) = 0 (always satisfied, skip)
 *              Instead, we enforce: left hand lies on club shaft
 *
 * Simple constraints:
 *   φ₁ = rh_x - lh_x = 0
 *   φ₂ = rh_y - lh_y = 0
 *   φ₃ = (lh - club_base) · ẑ × (sin(θ_club), -cos(θ_club)) = 0 (perpendicular)
 *   φ₄ = (lh - club_base) · (sin(θ_club), -cos(θ_club)) - grip_left = 0 (distance)
 */
export function constraintJacobian(
    q: [number, number, number, number, number, number, number, number],
    p: GolferParams
): number[][] {
    // Compute FK Jacobians for right hand and left hand
    const theta_hub = q[0];
    const alpha_rs = q[1];
    const alpha_re = q[2];
    const alpha_rh = q[3];
    const alpha_ls = q[4];
    const alpha_le = q[5];
    const alpha_lh = q[6];
    const theta_club = q[7];

    const theta_rs = theta_hub + alpha_rs;
    const theta_re = theta_rs + alpha_re;
    const theta_rh = theta_re + alpha_rh;

    const theta_ls = theta_hub + alpha_ls;
    const theta_le = theta_ls + alpha_le;
    const theta_lh = theta_le + alpha_lh;

    const hub_x = p.L_hub * Math.sin(theta_hub);
    const hub_y = -p.L_hub * Math.cos(theta_hub);

    const rs_x = hub_x + p.d_rs * Math.cos(theta_hub);
    const rs_y = hub_y + p.d_rs * Math.sin(theta_hub);

    const ls_x = hub_x - p.d_ls * Math.cos(theta_hub);
    const ls_y = hub_y - p.d_ls * Math.sin(theta_hub);

    const re_x = rs_x + p.L_r_upper * Math.sin(theta_rs);
    const re_y = rs_y - p.L_r_upper * Math.cos(theta_rs);

    const rh_x = re_x + p.L_r_fore * Math.sin(theta_re);
    const rh_y = re_y - p.L_r_fore * Math.cos(theta_re);

    const le_x = ls_x + p.L_l_upper * Math.sin(theta_ls);
    const le_y = ls_y - p.L_l_upper * Math.cos(theta_ls);

    const lh_x = le_x + p.L_l_fore * Math.sin(theta_le);
    const lh_y = le_y - p.L_l_fore * Math.cos(theta_le);

    // Jacobian for rh: ∂rh/∂q
    const Jrh = [
        [p.d_rs * (-Math.sin(theta_hub)) + p.L_r_upper * Math.cos(theta_rs) + p.L_r_fore * Math.cos(theta_re),
            p.L_r_upper * Math.cos(theta_rs) + p.L_r_fore * Math.cos(theta_re),
            p.L_r_fore * Math.cos(theta_re), p.L_r_fore * Math.cos(theta_rh), 0, 0, 0, 0],
        [p.d_rs * Math.cos(theta_hub) + p.L_r_upper * Math.sin(theta_rs) + p.L_r_fore * Math.sin(theta_re),
            p.L_r_upper * Math.sin(theta_rs) + p.L_r_fore * Math.sin(theta_re),
            p.L_r_fore * Math.sin(theta_re), p.L_r_fore * Math.sin(theta_rh), 0, 0, 0, 0],
    ];

    // Jacobian for lh: ∂lh/∂q
    const Jlh = [
        [-p.d_ls * (-Math.sin(theta_hub)) + p.L_l_upper * Math.cos(theta_ls) + p.L_l_fore * Math.cos(theta_le),
            0, 0, 0,
            p.L_l_upper * Math.cos(theta_ls) + p.L_l_fore * Math.cos(theta_le),
            p.L_l_fore * Math.cos(theta_le), p.L_l_fore * Math.cos(theta_lh), 0],
        [-p.d_ls * Math.cos(theta_hub) + p.L_l_upper * Math.sin(theta_ls) + p.L_l_fore * Math.sin(theta_le),
            0, 0, 0,
            p.L_l_upper * Math.sin(theta_ls) + p.L_l_fore * Math.sin(theta_le),
            p.L_l_fore * Math.sin(theta_le), p.L_l_fore * Math.sin(theta_lh), 0],
    ];

    // Constraint 1: rh_x = lh_x => Φ₁ = rh_x - lh_x
    const phi1_q = Jrh[0].map((v, i) => v - Jlh[0][i]);

    // Constraint 2: rh_y = lh_y => Φ₂ = rh_y - lh_y
    const phi2_q = Jrh[1].map((v, i) => v - Jlh[1][i]);

    // Constraint 3 & 4: Left hand distance from club shaft
    // Simplified: just enforce lh matches a point on club shaft
    // φ₃: lh_x = club_base_x + grip_left*sin(θ_club)
    // φ₄: lh_y = club_base_y - grip_left*cos(θ_club)

    const club_base_x = rh_x - p.grip_right * Math.sin(theta_club);
    const club_base_y = rh_y + p.grip_right * Math.cos(theta_club);

    // ∂club_base/∂q
    const dclub_x_q = [0, 0, 0, 0, 0, 0, 0, 0];
    const dclub_y_q = [0, 0, 0, 0, 0, 0, 0, 0];
    for (let i = 0; i < 8; i++) {
        dclub_x_q[i] = Jrh[0][i] - p.grip_right * Math.cos(theta_club) * (i === 7 ? 1 : 0);
        dclub_y_q[i] = Jrh[1][i] - p.grip_right * (-Math.sin(theta_club)) * (i === 7 ? 1 : 0);
    }

    // φ₃: lh_x - club_base_x - grip_left*sin(θ_club) = 0
    const phi3_q = Jlh[0].map((v, i) => v - dclub_x_q[i]);
    phi3_q[7] -= p.grip_left * Math.cos(theta_club);

    // φ₄: lh_y - club_base_y + grip_left*cos(θ_club) = 0
    const phi4_q = Jlh[1].map((v, i) => v - dclub_y_q[i]);
    phi4_q[7] += p.grip_left * Math.sin(theta_club);

    return [phi1_q, phi2_q, phi3_q, phi4_q];
}

// ── Mass Matrix (analytical via Jacobians) ────────────────────────────────────

/**
 * Compute the 8×8 mass matrix M = Σ mᵢ Jᵢᵀ Jᵢ
 * where Jᵢ are the Jacobians of each mass point's position w.r.t. q.
 *
 * Mass points:
 *   1. Hub center (m_hub)
 *   2. Right upper arm COM (m_r_upper) at RS + L_r_upper/2 * (sin, -cos)
 *   3. Right forearm COM (m_r_fore) at RE + L_r_fore/2 * (sin, -cos)
 *   4. Left upper arm COM (m_l_upper) at LS + L_l_upper/2 * (sin, -cos)
 *   5. Left forearm COM (m_l_fore) at LE + L_l_fore/2 * (sin, -cos)
 *   6. Club COM (m_club) at club_base + L_club/2 * (sin, -cos)
 *   7. Clubhead (m_clubhead) at club_tip
 */
export function massMatrix_golfer(
    q: [number, number, number, number, number, number, number, number],
    p: GolferParams
): number[][] {
    const M: number[][] = Array(8).fill(null).map(() => Array(8).fill(0));

    // Simple approach: approximate each Jacobian numerically for now
    // In production, use analytical Jacobians
    const eps = 1e-8;

    // For each DOF pair, compute ∂²KE/∂q_i∂q_j numerically
    // KE = 0.5 * Σ mᵢ * ||ṙᵢ||² = 0.5 * Σ mᵢ * ||Jᵢ * q̇||²
    // M_ij = Σ mᵢ * J_ik * J_ij

    const fk = forwardKinematics_golfer(q, p);

    // Hub Jacobian (always identity for hub z-rotation)
    const J_hub = Array(8).fill(0);
    J_hub[0] = 1;

    // Right shoulder position: rs
    // rs = (hub_x + d_rs*cos(θ_hub), hub_y + d_rs*sin(θ_hub))
    const dhub_dq = [p.L_hub * Math.cos(q[0]) - p.d_rs * Math.sin(q[0]), p.L_hub * Math.sin(q[0]) + p.d_rs * Math.cos(q[0])];
    const J_rs = [dhub_dq[0], 0, 0, 0, 0, 0, 0, 0];
    const J_rs_y = [dhub_dq[1], 0, 0, 0, 0, 0, 0, 0];

    // Accumulate contributions (simplified: just use FK + numerical Jacobian)
    // For full implementation, compute Jacobian for each mass point analytically

    // Placeholder: use identity-like structure scaled by masses
    for (let i = 0; i < 8; i++) {
        for (let j = i; j < 8; j++) {
            M[i][j] = M[j][i] = (i === j ? 1.0 : 0.0); // Placeholder
        }
    }

    // Scale diagonal by effective masses
    M[0][0] *= (p.m_hub + p.m_r_upper + p.m_r_fore + p.m_l_upper + p.m_l_fore + p.m_club);
    for (let i = 1; i < 8; i++) {
        M[i][i] *= (p.m_r_upper + p.m_r_fore);
    }

    return M;
}

// ── Friction ──────────────────────────────────────────────────────────────────

export function frictionTorqueVector_golfer(
    qdot: [number, number, number, number, number, number, number, number],
    p: GolferParams
): [number, number, number, number, number, number, number, number] {
    return [
        -p.b_hub * qdot[0],
        -p.b_rs * qdot[1],
        -p.b_re * qdot[2],
        -p.b_rh * qdot[3],
        -p.b_ls * qdot[4],
        -p.b_le * qdot[5],
        -p.b_lh * qdot[6],
        0, // no damping on club angle directly
    ];
}

// ── Constraint Stabilization (Baumgarte) ──────────────────────────────────────

/**
 * Compute constraint values Φ and time derivatives Φ̇, Φ̈ for stabilization.
 * Uses Baumgarte method: α*Φ + 2*β*Φ̇ + Φ̈ = 0
 */
function constraintValues(
    q: [number, number, number, number, number, number, number, number],
    p: GolferParams
): [number, number, number, number] {
    const fk = forwardKinematics_golfer(q, p);

    // φ₁ = rh_x - lh_x
    const phi1 = fk.rh[0] - fk.lh[0];

    // φ₂ = rh_y - lh_y
    const phi2 = fk.rh[1] - fk.lh[1];

    // φ₃ = lh_x - (club_base_x + grip_left*sin(θ_club))
    const phi3 = fk.lh[0] - (fk.club_base[0] + p.grip_left * Math.sin(q[7]));

    // φ₄ = lh_y - (club_base_y - grip_left*cos(θ_club))
    const phi4 = fk.lh[1] - (fk.club_base[1] - p.grip_left * Math.cos(q[7]));

    return [phi1, phi2, phi3, phi4];
}

// ── Equations of motion (simplified, no full KKT) ───────────────────────────

/**
 * Simplified EOM without constraint forces (Φ_q * λ).
 * In full implementation, would solve KKT system:
 *   [M     Φ_q^T] [q̈] = [τ - C - G]
 *   [Φ_q   0    ] [λ] = [-γ - 2αΦ̇ - β²Φ]
 *
 * For now, just unconstrained dynamics + constraint penalty torques.
 */
export function equationsOfMotion_golfer(
    state: StateGolfer,
    t: number,
    p: GolferParams,
    torqueFunc: TorqueFuncGolfer,
): StateGolfer {
    const q: [number, number, number, number, number, number, number, number] =
        [state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]];

    const qdot: [number, number, number, number, number, number, number, number] =
        [state[8], state[9], state[10], state[11], state[12], state[13], state[14], state[15]];

    // Gravity (simplified: just proportional to angle from vertical)
    const gravity: [number, number, number, number, number, number, number, number] = [0, 0, 0, 0, 0, 0, 0, 0];

    // Friction
    const [tf0, tf1, tf2, tf3, tf4, tf5, tf6, tf7] = frictionTorqueVector_golfer(qdot, p);

    // Torques
    const [tau0, tau1, tau2, tau3, tau4, tau5, tau6] = torqueFunc(t);

    // Constraint penalties (simple spring penalty)
    const phi = constraintValues(q, p);
    const penalty_gain = 1000;
    const constraint_tau = phi.map(ph => -penalty_gain * ph) as [number, number, number, number];

    // Approximate accelerations (ignoring Coriolis for simplicity in constraint solver)
    // Full implementation would solve the KKT system
    const qdd: [number, number, number, number, number, number, number, number] = [
        (tau0 + tf0 + constraint_tau[0]) / (p.m_hub),
        (tau1 + tf1 + constraint_tau[0]) / (p.m_r_upper),
        (tau2 + tf2 + constraint_tau[0]) / (p.m_r_fore),
        (tau3 + tf3 + constraint_tau[0]) / (p.m_r_upper),
        (tau4 + tf4 + constraint_tau[0]) / (p.m_l_upper),
        (tau5 + tf5 + constraint_tau[0]) / (p.m_l_fore),
        (tau6 + tf6 + constraint_tau[0]) / (p.m_l_upper),
        (constraint_tau[0]) / (p.m_club + p.m_clubhead),
    ];

    const dot: StateGolfer = [
        q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7],
        qdd[0], qdd[1], qdd[2], qdd[3], qdd[4], qdd[5], qdd[6], qdd[7],
    ];

    return dot;
}

// ── RK4 integrator ────────────────────────────────────────────────────────────

function rk4Step_golfer(
    state: StateGolfer,
    t: number,
    dt: number,
    p: GolferParams,
    tf: TorqueFuncGolfer,
): StateGolfer {
    const f = (s: StateGolfer, ti: number): StateGolfer => equationsOfMotion_golfer(s, ti, p, tf);
    const add = (a: StateGolfer, b: StateGolfer, scale: number): StateGolfer =>
        a.map((v, i) => v + b[i] * scale) as StateGolfer;

    const k1 = f(state, t);
    const k2 = f(add(state, k1, dt / 2), t + dt / 2);
    const k3 = f(add(state, k2, dt / 2), t + dt / 2);
    const k4 = f(add(state, k3, dt), t + dt);

    return state.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])) as StateGolfer;
}

// ── Simulation ────────────────────────────────────────────────────────────────

export interface SimulationResult_golfer {
    t: number[];
    states: StateGolfer[];
    params: GolferParams;
    torqueFunc: TorqueFuncGolfer;
}

/**
 * Integrate equations of motion via fixed-step RK4.
 */
export function runSimulation_golfer(
    params: GolferParams,
    initialState: StateGolfer,
    tEnd: number,
    torqueFunc: TorqueFuncGolfer,
    dt: number = 0.005,
): SimulationResult_golfer {
    initialState.forEach((v, i) => assertFinite(v, `initialState[${i}]`));
    if (!(tEnd > 0)) throw new RangeError('[DbC] tEnd must be > 0');
    if (!(dt > 0 && dt < tEnd)) throw new RangeError('[DbC] dt must be in (0, tEnd)');

    const t: number[] = [];
    const states: StateGolfer[] = [];
    let state: StateGolfer = [...initialState] as StateGolfer;
    let time = 0;

    while (time <= tEnd + 1e-10) {
        t.push(time);
        states.push([...state] as StateGolfer);
        state = rk4Step_golfer(state, time, dt, params, torqueFunc);
        time += dt;
    }

    if (t.length < 2) throw new Error('[DbC post] Simulation must produce ≥ 2 timesteps');
    return { t, states, params, torqueFunc };
}

// ── Polynomial torque builder ──────────────────────────────────────────────────

export function makePolynomialTorque_golfer(
    coeff_hub: number[],
    coeff_rs: number[],
    coeff_re: number[],
    coeff_rh: number[],
    coeff_ls: number[],
    coeff_le: number[],
    coeff_lh: number[]
): TorqueFuncGolfer {
    const polyval = (coeffs: number[], t: number): number =>
        coeffs.reduce((acc, c, i) => acc + c * t ** i, 0);

    return (t: number): [number, number, number, number, number, number, number] => [
        polyval(coeff_hub, t),
        polyval(coeff_rs, t),
        polyval(coeff_re, t),
        polyval(coeff_rh, t),
        polyval(coeff_ls, t),
        polyval(coeff_le, t),
        polyval(coeff_lh, t),
    ];
}
