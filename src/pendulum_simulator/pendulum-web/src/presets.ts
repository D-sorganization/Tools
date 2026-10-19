/**
 * Preset configurations — DRY: single source of truth shared by UI, tests, Colab notebook.
 *
 * Realistic defaults:
 *   Arms: m1 = 5.0 kg, L1 = 0.65 m (shoulder to wrist, typical human)
 *   Club (canonical distributed convention, Tools #5431): a uniform shaft of
 *   mass m2 = 0.1078 kg plus a head point mass mClub = 0.2022 kg at L2 = 1.10 m.
 *   The total (0.310 kg) is the real driver's mass and the wrist inertia
 *   Jw = (m2/3 + mClub) L2^2 = 0.2881 kg m^2 matches the real club exactly.
 *   Same split as the Python preset (`distributed_club_split`).
 */
import type { PendulumParams, TorqueFunc } from './physics';
import { makePolynomialTorque, makePendulumParams, STANDARD_GRAVITY } from './physics';

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
        m1, m2, mClub, L1, L2, g: STANDARD_GRAVITY, b1, b2, mu1, mu2,
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
        5.0, 0.10782199586776868, 0.2021780041322313,
        0.65, 1.10,        // arms 0.65 m, shaft 1.10 m
        0.1, 0.05, 0.02, 0.01,
        -60, 80, 0, 0,
        [-25, 10], [0],
        2.0,
        'Shoulder-driven swing with zero wrist torque — demonstrates passive release via inertial coupling.',
    ),
    _preset(
        'Golf Swing (active wrist)',
        5.0, 0.10782199586776868, 0.2021780041322313,
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
        'Heavier clubhead (350g head on a 300g uniform shaft) — more momentum transfer, slower release.',
    ),
    _preset(
        'Free Double Pendulum',
        1.0, 1.0, 0.0,    // uniform 1 kg rod as the second link, no head mass
        1.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
        90, 0, 0, 0,
        [0], [0],
        5.0,
        'No torques, no clubhead — a point-mass arm and a uniform-rod second link (not two point masses); chaotic, energy-conserving Lagrangian dynamics.',
    ),
    _preset(
        'Straight Drop',
        2.0, 1.0, 0.0,
        0.8, 0.8,
        0.0, 0.0, 0.0, 0.0,
        5, 0, 0, 0,
        [0], [0],
        3.0,
        'Near-vertical release with no driving torques; the second link is a uniform rod.',
    ),
];
