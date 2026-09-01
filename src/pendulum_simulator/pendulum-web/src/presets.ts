/**
 * Preset configurations — DRY: single source of truth shared by UI, tests, Colab notebook.
 *
 * Realistic defaults:
 *   Arms: m1 = 5.0 kg, L1 = 0.65 m (shoulder to wrist, typical human)
 *   Club equivalent: m2 + mClub = 0.2381186694 kg at L2 = 1.10 m.
 *   This is the tip mass that reproduces a real driver's wrist inertia, not
 *   the real club's physical mass. The 60/40 split matches the Python preset;
 *   this two-point model depends on their sum.
 */
import type { PendulumParams, TorqueFunc } from './physics';
import { makePolynomialTorque, makePendulumParams } from './physics';

export interface Preset {
    name: string;
    params: PendulumParams;
    theta1Deg: number;
    phiDeg: number;
    dtheta1: number;
    dphi: number;
    torqueFunc: TorqueFunc;
    coeffsShoulder: number[];
    coeffsWrist: number[];
    tEnd: number;
    description: string;
}

/** Small factory for building presets (DRY). */
const _preset = (
    name: string,
    m1: number, m2: number, mClub: number,
    L1: number, L2: number,
    b1: number, b2: number, mu1: number, mu2: number,
    theta1Deg: number, phiDeg: number,
    dtheta1: number, dphi: number,
    cShoulder: number[], cWrist: number[],
    tEnd: number,
    description: string,
): Preset => {
    const params = makePendulumParams({
        m1, m2, mClub, L1, L2, g: 9.81, b1, b2, mu1, mu2,
    });
    return {
        name,
        params,
        theta1Deg,
        phiDeg,
        dtheta1,
        dphi,
        torqueFunc: makePolynomialTorque(cShoulder, cWrist),
        coeffsShoulder: cShoulder,
        coeffsWrist: cWrist,
        tEnd,
        description,
    };
};

export const PRESETS: Preset[] = [
    _preset(
        'Golf Swing (passive wrist)',
        5.0, 0.1428712016528925, 0.09524746776859502,
        0.65, 1.10,        // arms 0.65 m, shaft 1.10 m
        0.1, 0.05, 0.02, 0.01,
        -60, 80, 0, 0,
        [-25, 10], [0],
        2.0,
        'Shoulder-driven swing with zero wrist torque — demonstrates passive release via inertial coupling.',
    ),
    _preset(
        'Golf Swing (active wrist)',
        5.0, 0.1428712016528925, 0.09524746776859502,
        0.65, 1.10,
        0.1, 0.05, 0.02, 0.01,
        -60, 80, 0, 0,
        [-25, 10], [5, -3],
        2.0,
        'Adds a small wrist torque for comparison with the passive case.',
    ),
    _preset(
        'Heavy Clubhead',
        5.0, 0.30, 0.35,  // heavier clubhead (350g)
        0.65, 1.10,
        0.1, 0.05, 0.02, 0.01,
        -60, 80, 0, 0,
        [-30, 12], [0],
        2.0,
        'Heavier clubhead (350g) — more momentum transfer, slower release.',
    ),
    _preset(
        'Free Double Pendulum',
        1.0, 1.0, 0.0,    // no clubhead for classic pendulum
        1.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
        90, 0, 0, 0,
        [0], [0],
        5.0,
        'No torques, no clubhead — chaotic dynamics demonstrating energy-conserving Lagrangian mechanics.',
    ),
    _preset(
        'Straight Drop',
        2.0, 1.0, 0.0,
        0.8, 0.8,
        0.0, 0.0, 0.0, 0.0,
        5, 0, 0, 0,
        [0], [0],
        3.0,
        'Near-vertical release with no driving torques.',
    ),
];
