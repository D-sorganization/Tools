import { renderToString } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import App from '../src/App';
import { makePendulumParams, type SimulationResult } from '../src/physics';
import { makeTripleParams, type SimulationResult3 } from '../src/physics_triple';
import { makeGolferParams, type SimulationResult_golfer } from '../src/physics_golfer';
import {
    isDoubleSimulationResult,
    isGolferSimulationResult,
    isTripleSimulationResult,
} from '../src/modelGuards';

describe('App entrypoint', () => {
    it('renders selectors for all pendulum models', () => {
        const html = renderToString(<App />);

        expect(html).toContain('Double Pendulum (2-DOF)');
        expect(html).toContain('Triple Pendulum (3-DOF)');
        expect(html).toContain('Golfer (8-DOF)');
        expect(html).toContain('Force-Source Optimization Lab');
        expect(html).toContain('Optimize selected');
        expect(html).toContain('Optimize all 6');
        expect(html).toContain('id="force-start-arm"');
        expect(html).toContain('id="force-start-wrist"');
        expect(html).toContain('id="force-wrist-step"');
        expect(html).toContain('id="force-fixed-hub"');
        expect(html).toContain('max="30"');
    });
});

describe('model guards', () => {
    const doubleResult: SimulationResult = {
        t: [0, 0.01],
        states: [
            [0, 0, 0, 0],
            [0.1, 0.1, 0, 0],
        ],
        params: makePendulumParams({
            m1: 1,
            m2: 1,
            mClub: 0.2,
            L1: 1,
            L2: 1,
            g: 9.81,
            b1: 0,
            b2: 0,
            mu1: 0,
            mu2: 0,
        }),
        torqueFunc: () => [0, 0],
    };

    const tripleResult: SimulationResult3 = {
        t: [0, 0.01],
        states: [
            [0, 0, 0, 0, 0, 0],
            [0.1, 0.1, 0.1, 0, 0, 0],
        ],
        params: makeTripleParams({
            m1: 1,
            m2: 1,
            m3: 1,
            mClub: 0.2,
            L1: 1,
            L2: 1,
            L3: 1,
            g: 9.81,
            b1: 0,
            b2: 0,
            b3: 0,
        }),
        torqueFunc: () => [0, 0, 0],
    };

    const golferResult: SimulationResult_golfer = {
        t: [0, 0.01],
        states: [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        params: makeGolferParams({
            m_hub: 1,
            m_r_upper: 1,
            m_r_fore: 1,
            m_l_upper: 1,
            m_l_fore: 1,
            m_club: 1,
            L_hub: 0.5,
            L_r_upper: 0.3,
            L_r_fore: 0.3,
            L_l_upper: 0.3,
            L_l_fore: 0.3,
            L_club: 1,
            d_rs: 0.1,
            d_ls: 0.1,
            grip_right: 0.2,
            grip_left: 0.2,
            m_clubhead: 0.2,
            g: 9.81,
            b_hub: 0,
            b_rs: 0,
            b_re: 0,
            b_rh: 0,
            b_ls: 0,
            b_le: 0,
            b_lh: 0,
        }),
        torqueFunc: () => [0, 0, 0, 0, 0, 0, 0],
    };

    it('distinguishes double, triple, and golfer result payloads', () => {
        expect(isDoubleSimulationResult(doubleResult)).toBe(true);
        expect(isTripleSimulationResult(doubleResult)).toBe(false);
        expect(isGolferSimulationResult(doubleResult)).toBe(false);

        expect(isDoubleSimulationResult(tripleResult)).toBe(false);
        expect(isTripleSimulationResult(tripleResult)).toBe(true);
        expect(isGolferSimulationResult(tripleResult)).toBe(false);

        expect(isDoubleSimulationResult(golferResult)).toBe(false);
        expect(isTripleSimulationResult(golferResult)).toBe(false);
        expect(isGolferSimulationResult(golferResult)).toBe(true);
    });
});
