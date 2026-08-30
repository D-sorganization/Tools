import { describe, expect, it } from 'vitest';

import {
    candidateGrid,
    FORCE_SOURCE_SCHEMA,
    golfLikeImpactIndex,
    parseForceSourceArtifact,
    type ForceSourceArtifact,
} from '../src/forceSourceStudy';
import {
    interpolateSeries,
    pendulumThumbnailGeometry,
} from '../src/forceSourceView';

function validArtifact(): ForceSourceArtifact {
    const pair = [0, 0.1];
    return {
        schema_version: FORCE_SOURCE_SCHEMA,
        force_attribution_schema: 'force-attribution/v1',
        model: 'test',
        coordinates: ['shoulder_absolute', 'wrist_relative'],
        initial_pose: {
            arm_angle_rad: -2.2,
            wrist_cock_rad: -1.57,
            arm_velocity_rad_s: 0,
            wrist_velocity_rad_s: 0,
        },
        search_profile: { name: 'test' },
        evaluated_count: 2,
        qualified_count: 1,
        interpretation_limits: ['Coordinate dependent.'],
        scenarios: [{
            objective: 'clubhead_speed',
            score: 40,
            candidate: {
                shoulder_torque_nm: 100,
                wrist_drive_nm: 30,
                wrist_restrain_nm: 10,
                onset_s: 0.1,
            },
            impact_time_s: 0.1,
            robustness: {
                sample_count: 9,
                qualified_count: 9,
                qualification_rate: 1,
                median_score: 40,
                worst_score: 38,
                best_score: 41,
                score_spread: 3,
            },
            near_optimal_count: 2,
            boundary_hits: ['wrist_drive_nm:upper'],
            convergence: [39, 40],
            series: {
                time_s: pair,
                arm_angle_rad: pair,
                wrist_cock_rad: pair,
                arm_angular_velocity_rad_s: pair,
                wrist_angular_velocity_rad_s: pair,
                shoulder_torque_nm: pair,
                wrist_torque_nm: pair,
                clubhead_speed_m_s: pair,
                coriolis_tangent_force_n: pair,
                coriolis_power_w: pair,
                squared_speed_tangent_force_n: pair,
                squared_speed_power_w: pair,
            },
        }],
    };
}

describe('force-source artifact contract', () => {
    it('accepts aligned version-1 scenario series', () => {
        expect(parseForceSourceArtifact(validArtifact()).scenarios).toHaveLength(1);
    });

    it('rejects wrist torques above 30 N m', () => {
        const artifact = validArtifact();
        artifact.scenarios[0].candidate.wrist_drive_nm = 31;

        expect(() => parseForceSourceArtifact(artifact)).toThrow(/exceeds/);
    });

    it('rejects misaligned plot and animation series', () => {
        const artifact = validArtifact();
        artifact.scenarios[0].series.clubhead_speed_m_s = [0, 1, 2];

        expect(() => parseForceSourceArtifact(artifact)).toThrow(/lengths must match/);
    });
});

describe('hierarchical search grid', () => {
    it('covers the 30 N m thorough grid at 5 N m increments', () => {
        const candidates = candidateGrid('thorough', 30);

        expect(candidates).toHaveLength(3 * 7 * 7 * 13);
        expect(Math.max(...candidates.map(candidate => candidate.wrist_drive_nm))).toBe(30);
        expect(Math.max(...candidates.map(candidate => candidate.wrist_restrain_nm))).toBe(30);
    });

    it('includes a non-multiple-of-five wrist limit exactly', () => {
        const candidates = candidateGrid('thorough', 28);

        expect(candidates.some(candidate => candidate.wrist_drive_nm === 28)).toBe(true);
        expect(candidates.some(candidate => candidate.wrist_restrain_nm === 28)).toBe(true);
        expect(Math.max(...candidates.map(candidate => candidate.wrist_drive_nm))).toBe(28);
    });

    it('fails closed for limits outside the supported range', () => {
        expect(() => candidateGrid('quick', 31)).toThrow(/\(0, 30\]/);
    });
});

describe('golf-like impact qualification', () => {
    const params = {
        m1: 5, m2: 0.3, mClub: 0.2, L1: 0.65, L2: 1.1,
        g: 9.81, b1: 0.1, b2: 0.05, mu1: 0.02, mu2: 0.01,
    };

    it('accepts a rightward near-horizontal bottom pass', () => {
        const states = [
            [-2.2, -1.57, 0, 0],
            [-0.2, 0.2, 10, 15],
        ] as const;

        expect(golfLikeImpactIndex(states.map(state => [...state]), params)).toBe(1);
    });

    it('rejects an arm loop even when a later crossing is fast', () => {
        const states = [
            [-2.2, -1.57, 0, 0],
            [2 * Math.PI, -2 * Math.PI - 1, 10, 15],
            [-0.2, 0.2, 10, 15],
        ] as const;

        expect(golfLikeImpactIndex(states.map(state => [...state]), params)).toBeNull();
    });
});

describe('comparison animation coordinates', () => {
    const lengths = { L1: 0.65, L2: 1.1 };

    it('draws the hanging links below the shoulder', () => {
        const geometry = pendulumThumbnailGeometry(0, 0, lengths);

        expect(geometry.wristY).toBeGreaterThan(geometry.originY);
        expect(geometry.tipY).toBeGreaterThan(geometry.wristY);
    });

    it('draws the registered top-of-backswing above the shoulder', () => {
        const geometry = pendulumThumbnailGeometry(
            -126.0507 * Math.PI / 180,
            -89.9544 * Math.PI / 180,
            lengths,
        );

        expect(geometry.wristY).toBeLessThan(geometry.originY);
        expect(geometry.tipY).toBeLessThan(geometry.wristY);
    });
});

describe('high-resolution playback interpolation', () => {
    it('interpolates between simulation samples instead of snapping frames', () => {
        expect(interpolateSeries([0, 0.001], [10, 14], 0.00025)).toBeCloseTo(11);
        expect(interpolateSeries([0, 0.001], [10, 14], 0.00075)).toBeCloseTo(13);
    });

    it('clamps cleanly at trajectory boundaries', () => {
        expect(interpolateSeries([0, 0.001], [10, 14], -1)).toBe(10);
        expect(interpolateSeries([0, 0.001], [10, 14], 1)).toBe(14);
    });
});
