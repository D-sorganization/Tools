/**
 * Preset configurations for Triple Pendulum (3-DOF) model.
 *
 * Realistic defaults:
 *   Segment 1 (shoulder–elbow): m1 = 2.5 kg, L1 = 0.30 m
 *   Segment 2 (elbow–wrist): m2 = 1.5 kg, L2 = 0.25 m
 *   Segment 3 (wrist–tip): m3 = 0.5 kg, L3 = 0.15 m
 *   Clubhead: mClub = 0.20 kg
 */

import type { TripleParams, TorqueFuncTriple } from './physics_triple';
import { makeTripleParams, makePolynomialTorque3 } from './physics_triple';

export interface PresetTriple {
    name: string;
    params: TripleParams;
    theta1Deg: number;
    phi2Deg: number;
    phi3Deg: number;
    dtheta1: number;
    dphi2: number;
    dphi3: number;
    torqueFunc: TorqueFuncTriple;
    coeffsShoulder: number[];
    coeffsElbow: number[];
    coeffsWrist: number[];
    tEnd: number;
    description: string;
}

const _preset_triple = (
    name: string,
    m1: number, m2: number, m3: number, mClub: number,
    L1: number, L2: number, L3: number,
    b1: number, b2: number, b3: number,
    theta1Deg: number, phi2Deg: number, phi3Deg: number,
    dtheta1: number, dphi2: number, dphi3: number,
    cShoulder: number[], cElbow: number[], cWrist: number[],
    tEnd: number,
    description: string,
): PresetTriple => {
    const params = makeTripleParams({
        m1, m2, m3, mClub, L1, L2, L3, g: 9.81, b1, b2, b3,
    });
    return {
        name,
        params,
        theta1Deg,
        phi2Deg,
        phi3Deg,
        dtheta1,
        dphi2,
        dphi3,
        torqueFunc: makePolynomialTorque3(cShoulder, cElbow, cWrist),
        coeffsShoulder: cShoulder,
        coeffsElbow: cElbow,
        coeffsWrist: cWrist,
        tEnd,
        description,
    };
};

export const PRESETS_TRIPLE: PresetTriple[] = [
    _preset_triple(
        'Three-Segment Swing (passive)',
        2.5, 1.5, 0.5, 0.20,
        0.30, 0.25, 0.15,
        0.05, 0.04, 0.03,
        -60, 40, 60, 0, 0, 0,
        [-20, 8], [0], [0],
        2.0,
        'Shoulder-driven three-segment swing with passive elbow and wrist release.'
    ),
    _preset_triple(
        'Three-Segment Swing (active)',
        2.5, 1.5, 0.5, 0.20,
        0.30, 0.25, 0.15,
        0.05, 0.04, 0.03,
        -60, 40, 60, 0, 0, 0,
        [-20, 8], [5, -2], [3, -1],
        2.0,
        'Shoulder, elbow, and wrist actively driven for maximum acceleration.'
    ),
    _preset_triple(
        'Free Triple Pendulum',
        1.0, 1.0, 1.0, 0.0,
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
        90, 0, 0, 0, 0, 0,
        [0], [0], [0],
        5.0,
        'No torques, no clubhead — pure Lagrangian dynamics demonstrating coupling.'
    ),
];
