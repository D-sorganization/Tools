import { describe, expect, it } from 'vitest';

import {
    bernsteinTorque,
    buildCandidateSet,
    candidateTorqueFunction,
    DEFAULT_OPTIMIZATION_CONSTRAINTS,
    profileDiagnostics,
    validateBrowserOptimizationConfig,
    type BrowserOptimizationConfig,
    type ForceSourceCandidate,
} from '../src/forceSourceStudy';
import { PRESETS } from '../src/presets';

const params = {
    m1: 5, m2: 0.3, mClub: 0.2, L1: 0.65, L2: 1.1,
    g: 9.81, b1: 0.1, b2: 0.05, mu1: 0.02, mu2: 0.01,
};

function config(): BrowserOptimizationConfig {
    return {
        params,
        initialState: [-2.2, -1.57, 0, 0],
        objective: 'clubhead_speed',
        thoroughness: 'thorough',
        constraints: {
            ...DEFAULT_OPTIMIZATION_CONSTRAINTS,
            candidateBudget: 48,
            robustnessTrials: 1,
        },
    };
}

const candidate: ForceSourceCandidate = {
    basis: 'bernstein_6',
    profile_duration_s: 0.5,
    shoulder_coefficients_nm: [0, 30, 70, 100, 90, 40, 0],
    wrist_coefficients_nm: [0, -10, -8, 0, 12, 20, 0],
};

describe('sixth-order continuous torque profiles', () => {
    it('uses the repository-authoritative inertia-matched driver and hub budget', () => {
        expect(PRESETS[0].params.m2 + PRESETS[0].params.mClub).toBeCloseTo(0.2381186694, 8);
        expect(DEFAULT_OPTIMIZATION_CONSTRAINTS.shoulderTorqueNm).toEqual({ min: -250, max: 250, step: 5 });
        expect(DEFAULT_OPTIMIZATION_CONSTRAINTS.wristTorqueLimitNm).toBe(30);
        expect(DEFAULT_OPTIMIZATION_CONSTRAINTS.targetClubheadSpeedMps).toBe(52.3);
    });

    it('evaluates degree-6 Bernstein control coefficients at the endpoints', () => {
        expect(bernsteinTorque(candidate.shoulder_coefficients_nm, 0)).toBe(0);
        expect(bernsteinTorque(candidate.shoulder_coefficients_nm, 1)).toBe(0);
        expect(bernsteinTorque(candidate.shoulder_coefficients_nm, 0.5)).toBeGreaterThan(50);
    });

    it('stays inside the coefficient bounds for every point on the curve', () => {
        const torque = candidateTorqueFunction(candidate);
        for (let index = 0; index <= 1000; index += 1) {
            const [shoulder, wrist] = torque(0.5 * index / 1000);
            expect(shoulder).toBeGreaterThanOrEqual(0);
            expect(shoulder).toBeLessThanOrEqual(100);
            expect(wrist).toBeGreaterThanOrEqual(-10);
            expect(wrist).toBeLessThanOrEqual(20);
        }
    });

    it('has a continuous, measurable low-torque wrist reversal instead of a jump', () => {
        const diagnostics = profileDiagnostics(candidate, 3);

        expect(diagnostics.wrist_reversal_count).toBe(1);
        expect(diagnostics.wrist_transition_duration_s).toBeGreaterThan(0);
        expect(Number.isFinite(diagnostics.peak_wrist_slew_nm_s)).toBe(true);
    });

    it('builds only bounded, slew-limited, single-reversal polynomial candidates', () => {
        const optimization = config();
        const candidates = buildCandidateSet(optimization);

        expect(candidates).toHaveLength(optimization.constraints.candidateBudget);
        for (const item of candidates) {
            expect(item.basis).toBe('bernstein_6');
            expect(item.shoulder_coefficients_nm).toHaveLength(7);
            expect(item.wrist_coefficients_nm).toHaveLength(7);
            const diagnostics = profileDiagnostics(item, optimization.constraints.transitionTorqueNm);
            expect(diagnostics.peak_shoulder_torque_nm).toBeLessThanOrEqual(
                optimization.constraints.shoulderTorqueNm.max,
            );
            expect(diagnostics.peak_wrist_torque_nm).toBeLessThanOrEqual(
                optimization.constraints.wristTorqueLimitNm,
            );
            expect(diagnostics.peak_shoulder_slew_nm_s).toBeLessThanOrEqual(
                optimization.constraints.maxTorqueSlewNmS + 1e-8,
            );
            expect(diagnostics.peak_wrist_slew_nm_s).toBeLessThanOrEqual(
                optimization.constraints.maxTorqueSlewNmS + 1e-8,
            );
            expect(diagnostics.wrist_reversal_count).toBe(1);
            expect(diagnostics.wrist_transition_duration_s).toBeGreaterThanOrEqual(
                optimization.constraints.minWristTransitionS - 1e-3,
            );
        }
    });

    it('fails closed when the requested endpoint cannot be zero within shoulder bounds', () => {
        const optimization = config();
        optimization.constraints = {
            ...optimization.constraints,
            shoulderTorqueNm: { min: 10, max: 120, step: 1 },
        };

        expect(() => validateBrowserOptimizationConfig(optimization)).toThrow(/zero/i);
    });
});
