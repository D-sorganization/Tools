/**
 * TDD tests for the TypeScript physics engine (pendulum-web).
 *
 * Mirrors the Python pytest suite, covering:
 *   1. DbC pre-conditions on makePendulumParams
 *   2. Mass matrix properties (symmetry, positive-definiteness)
 *   3. Equilibrium: zero acceleration at rest
 *   4. Energy conservation (no friction, no driving torque)
 *   5. Friction reduces total energy
 *   6. Friction torques oppose motion
 *   7. Polynomial torque builder
 *   8. Forward kinematics at known positions
 *   9. runSimulation pre-conditions and post-conditions
 */
import { describe, it, expect } from 'vitest';
import {
    makePendulumParams,
    massMatrix,
    massMatrixComponents,
    equationsOfMotion,
    frictionTorqueVector,
    makePolynomialTorque,
    forwardKinematics,
    gripForceAlongHandPath,
    kineticEnergy,
    potentialEnergy,
    totalEnergy,
    runSimulation,
} from '../src/physics';
import type { PendulumParams, State } from '../src/physics';

// ── Fixtures ──────────────────────────────────────────────────────────────────

const stdParams: PendulumParams = makePendulumParams({
    m1: 1.0, m2: 1.0, mClub: 0.0, L1: 1.0, L2: 1.0, g: 9.81,
    b1: 0.0, b2: 0.0, mu1: 0.0, mu2: 0.0,
});

const dampParams: PendulumParams = makePendulumParams({
    m1: 1.0, m2: 1.0, mClub: 0.0, L1: 1.0, L2: 1.0, g: 9.81,
    b1: 0.5, b2: 0.3, mu1: 0.1, mu2: 0.05,
});

const zeroTorque = (_t: number): [number, number] => [0, 0];

// ── DbC: invalid params rejected ──────────────────────────────────────────────

describe('makePendulumParams DbC', () => {
    it('rejects negative mass m1', () => {
        expect(() => makePendulumParams({ ...stdParams, m1: -1.0 })).toThrow();
    });
    it('rejects zero mass m2', () => {
        expect(() => makePendulumParams({ ...stdParams, m2: 0.0 })).toThrow();
    });
    it('rejects zero length L1', () => {
        expect(() => makePendulumParams({ ...stdParams, L1: 0.0 })).toThrow();
    });
    it('rejects negative length L2', () => {
        expect(() => makePendulumParams({ ...stdParams, L2: -0.5 })).toThrow();
    });
    it('rejects negative gravity', () => {
        expect(() => makePendulumParams({ ...stdParams, g: -1.0 })).toThrow();
    });
    it('rejects negative damping b1', () => {
        expect(() => makePendulumParams({ ...stdParams, b1: -0.1 })).toThrow();
    });
    it('accepts zero friction (valid lower bound)', () => {
        expect(() => makePendulumParams({ ...stdParams, mu1: 0.0 })).not.toThrow();
    });
});

// ── Mass matrix ───────────────────────────────────────────────────────────────

describe('massMatrix', () => {
    it('is symmetric', () => {
        const [M11, M12, M21, M22] = massMatrix(0.5, stdParams);
        expect(M12).toBeCloseTo(M21, 12);
    });

    it('M22 is positive (point mass at tip)', () => {
        for (const phi of [-Math.PI, -1, 0, 1, Math.PI]) {
            const [, , , M22] = massMatrix(phi, stdParams);
            expect(M22).toBeGreaterThan(0);
        }
    });

    it('is positive-definite for all phi in [-pi, pi]', () => {
        for (let phi = -Math.PI; phi <= Math.PI; phi += 0.2) {
            const [M11, M12, , M22] = massMatrix(phi, stdParams);
            const det = M11 * M22 - M12 * M12;
            expect(det).toBeGreaterThan(0);     // det > 0 → PD for 2×2 symmetric
            expect(M11).toBeGreaterThan(0);
        }
    });

    it('components dict has matching M12 == M21', () => {
        const c = massMatrixComponents(0.3, stdParams);
        expect(c.M12).toBeCloseTo(c.M21, 12);
    });
});

// ── Equilibrium ───────────────────────────────────────────────────────────────

describe('equationsOfMotion', () => {
    it('equilibrium state → zero angular accelerations', () => {
        const state: State = [0, 0, 0, 0];
        const dot = equationsOfMotion(state, 0, stdParams, zeroTorque);
        expect(Math.abs(dot[2])).toBeLessThan(1e-10);
        expect(Math.abs(dot[3])).toBeLessThan(1e-10);
    });

    it('velocity continuity: first two components equal velocities', () => {
        const state: State = [0.3, 0.1, 0.5, -0.2];
        const dot = equationsOfMotion(state, 0, stdParams, zeroTorque);
        expect(dot[0]).toBeCloseTo(state[2], 12);  // dtheta1 = dtheta1
        expect(dot[1]).toBeCloseTo(state[3], 12);  // dphi    = dphi
    });

    it('rejects non-finite state value', () => {
        const badState: State = [NaN, 0, 0, 0];
        expect(() => equationsOfMotion(badState, 0, stdParams, zeroTorque)).toThrow();
    });
});

// ── Energy conservation ───────────────────────────────────────────────────────

describe('Energy conservation (no friction, no torque)', () => {
    it('|deltaE| < 1e-3 over 2 seconds with RK4', () => {
        const state0: State = [0.3, 0.1, 0.0, 0.0];
        const result = runSimulation(stdParams, state0, 2.0, zeroTorque, 0.005);
        const E0 = totalEnergy(result.states[0], stdParams);
        const Ef = totalEnergy(result.states[result.states.length - 1], stdParams);
        expect(Math.abs(Ef - E0)).toBeLessThan(1e-3);
    });
});

// ── Friction ──────────────────────────────────────────────────────────────────

describe('frictionTorqueVector', () => {
    it('opposes positive velocity (negative torque)', () => {
        const [tf1, tf2] = frictionTorqueVector(1.0, 2.0, dampParams);
        expect(tf1).toBeLessThan(0);
        expect(tf2).toBeLessThan(0);
    });

    it('opposes negative velocity (positive torque)', () => {
        const [tf1, tf2] = frictionTorqueVector(-1.0, -2.0, dampParams);
        expect(tf1).toBeGreaterThan(0);
        expect(tf2).toBeGreaterThan(0);
    });

    it('zero for stationary joint with zero friction params', () => {
        const [tf1, tf2] = frictionTorqueVector(0.0, 0.0, stdParams);
        expect(tf1).toBeCloseTo(0, 12);
        expect(tf2).toBeCloseTo(0, 12);
    });

    it('rejects non-finite velocity', () => {
        expect(() => frictionTorqueVector(Infinity, 0, stdParams)).toThrow();
    });
});

describe('Friction reduces total energy', () => {
    it('energy strictly decreases with damping and no driving torque', () => {
        const state0: State = [0.5, 0.2, 0.5, 0.3];
        const result = runSimulation(dampParams, state0, 3.0, zeroTorque, 0.005);
        const E0 = totalEnergy(result.states[0], dampParams);
        const Ef = totalEnergy(result.states[result.states.length - 1], dampParams);
        expect(Ef).toBeLessThan(E0);
    });
});

// ── Polynomial torque ─────────────────────────────────────────────────────────

describe('makePolynomialTorque', () => {
    it('constant torque: tau(t) == c0 for all t', () => {
        const tf = makePolynomialTorque([5.0], [0.0]);
        for (const t of [0, 0.5, 1.0, 2.0]) {
            const [tau1, tau2] = tf(t);
            expect(tau1).toBeCloseTo(5.0, 10);
            expect(tau2).toBeCloseTo(0.0, 10);
        }
    });

    it('linear torque: tau(t) = c0 + c1*t', () => {
        const tf = makePolynomialTorque([2.0, 3.0], [0.0]);
        expect(tf(0)[0]).toBeCloseTo(2.0, 10);
        expect(tf(1)[0]).toBeCloseTo(5.0, 10);
        expect(tf(2)[0]).toBeCloseTo(8.0, 10);
    });

    it('rejects empty coefficient array', () => {
        expect(() => makePolynomialTorque([], [1.0])).toThrow();
        expect(() => makePolynomialTorque([1.0], [])).toThrow();
    });
});

// ── Forward kinematics ────────────────────────────────────────────────────────

describe('forwardKinematics', () => {
    it('hanging straight down: wrist at (0, -L1), tip at (0, -L1-L2)', () => {
        const pos = forwardKinematics(0, 0, stdParams);
        expect(pos.shoulder).toEqual([0, 0]);
        expect(pos.wrist[0]).toBeCloseTo(0, 10);
        expect(pos.wrist[1]).toBeCloseTo(-1.0, 10);
        expect(pos.tip[0]).toBeCloseTo(0, 10);
        expect(pos.tip[1]).toBeCloseTo(-2.0, 10);
    });
});

describe('gripForceAlongHandPath', () => {
    it('projects the physical grip force onto the direction of hand travel', () => {
        const state: State = [-1.2, -0.8, 8, 12];
        const acceleration = equationsOfMotion(state, 0.1, stdParams, () => [80, 10]);
        const result = gripForceAlongHandPath(
            state,
            [acceleration[2], acceleration[3]],
            stdParams,
        );

        expect(result.handSpeedMps).toBeGreaterThan(0);
        expect(result.tangentForceN).not.toBeNull();
        expect(result.powerW).toBeCloseTo(
            (result.tangentForceN ?? 0) * result.handSpeedMps,
            9,
        );
    });

    it('reports an undefined tangent at zero hand speed', () => {
        const result = gripForceAlongHandPath([0, 0, 0, 0], [0, 0], stdParams);

        expect(result.handSpeedMps).toBe(0);
        expect(result.tangentForceN).toBeNull();
    });
});

// ── runSimulation pre / post conditions ───────────────────────────────────────

describe('runSimulation', () => {
    it('produces >= 2 time steps', () => {
        const result = runSimulation(stdParams, [0.1, 0.0, 0.0, 0.0], 0.1, zeroTorque, 0.005);
        expect(result.t.length).toBeGreaterThanOrEqual(2);
    });

    it('rejects non-finite initial state', () => {
        expect(() =>
            runSimulation(stdParams, [NaN, 0, 0, 0], 1.0, zeroTorque)
        ).toThrow();
    });

    it('rejects tEnd <= 0', () => {
        expect(() =>
            runSimulation(stdParams, [0, 0, 0, 0], 0, zeroTorque)
        ).toThrow();
    });

    it('rejects dt >= tEnd', () => {
        expect(() =>
            runSimulation(stdParams, [0, 0, 0, 0], 1.0, zeroTorque, 2.0)
        ).toThrow();
    });
});
