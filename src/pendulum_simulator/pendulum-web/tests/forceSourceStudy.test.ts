import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';

import {
    buildCandidateSet,
    buildOptimizationContract,
    artifactWithScenarios,
    DEFAULT_OPTIMIZATION_CONSTRAINTS,
    FORCE_SOURCE_SCHEMA,
    FORCE_SOURCE_OBJECTIVES,
    forceSourceTransferPowers,
    golfLikeImpactIndex,
    optimizeForceSourceComparison,
    parseForceSourceArtifact,
    scoreForceSourceSeries,
    summarizeRobustness,
    validateBrowserOptimizationConfig,
    type BrowserOptimizationConfig,
    type ForceSourceArtifact,
} from '../src/forceSourceStudy';
import {
    interpolateSeries,
    pendulumThumbnailGeometry,
    thumbnailOrigin,
} from '../src/forceSourceView';
import { generalizedForceSources } from '../src/physics';

const params = {
    m1: 5, m2: 0.3, mClub: 0.2, L1: 0.65, L2: 1.1,
    g: 9.81, b1: 0.1, b2: 0.05, mu1: 0.02, mu2: 0.01,
};

function optimizationConfig(): BrowserOptimizationConfig {
    return {
        params,
        initialState: [-2.2, -1.57, 0, 0],
        objective: 'clubhead_speed',
        thoroughness: 'thorough',
        constraints: {
            ...DEFAULT_OPTIMIZATION_CONSTRAINTS,
            shoulderTorqueNm: { min: 60, max: 120, step: 10 },
            wristTorqueLimitNm: 30,
            wristTorqueStepNm: 0.5,
            onsetS: { min: 0.05, max: 0.25, step: 0.01 },
            candidateBudget: 96,
            robustnessTrials: 9,
        },
    };
}

function validArtifact(): ForceSourceArtifact {
    const pair = [0, 0.1];
    const config = optimizationConfig();
    const contract = buildOptimizationContract(config);
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
        parameters: params,
        search_profile: { name: 'test' },
        comparison_contract: contract,
        evaluated_count: 2,
        qualified_count: 1,
        interpretation_limits: ['Coordinate dependent.'],
        scenarios: [{
            objective: 'clubhead_speed',
            score: 0.1,
            candidate: {
                shoulder_torque_nm: 100,
                wrist_drive_nm: 30,
                wrist_restrain_nm: 10,
                onset_s: 0.1,
            },
            impact_time_s: 0.1,
            comparison_contract_id: contract.id,
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
                arm_angle_rad: [-2.2, -2.1],
                wrist_cock_rad: [-1.57, -1.47],
                arm_angular_velocity_rad_s: pair,
                wrist_angular_velocity_rad_s: pair,
                shoulder_torque_nm: pair,
                wrist_torque_nm: pair,
                clubhead_speed_m_s: pair,
                coriolis_tangent_force_n: pair,
                coriolis_power_w: pair,
                squared_speed_tangent_force_n: pair,
                squared_speed_power_w: pair,
                hand_path_tangent_force_n: pair,
            },
        }],
    };
}

describe('force-source artifact contract', () => {
    it('accepts aligned version-2 scenario series', () => {
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

    it('rejects scenarios whose first pose differs from the registered comparison pose', () => {
        const artifact = validArtifact();
        const second = structuredClone(artifact.scenarios[0]);
        second.objective = 'coriolis_impulse';
        second.series.arm_angle_rad[0] = 0.25;
        artifact.scenarios.push(second);

        expect(() => parseForceSourceArtifact(artifact)).toThrow(/shared initial pose/i);
    });

    it('starts a new comparison instead of mixing a changed pose with stale scenarios', () => {
        const existing = validArtifact();
        const replacement = structuredClone(existing.scenarios[0]);
        replacement.objective = 'hand_path_impulse';
        replacement.score = 0.005;
        replacement.series.arm_angle_rad[0] = -1.9;
        replacement.series.wrist_cock_rad[0] = -1.2;
        const config = optimizationConfig();
        config.initialState = [-1.9, -1.2, 0, 0];

        const next = artifactWithScenarios(existing, [replacement], config);

        expect(next.scenarios.map(item => item.objective)).toEqual(['hand_path_impulse']);
    });

    it('starts a new comparison when search settings change under the same pose', () => {
        const existing = validArtifact();
        const replacement = structuredClone(existing.scenarios[0]);
        replacement.objective = 'hand_path_impulse';
        replacement.score = 0.005;
        const config = optimizationConfig();
        config.constraints = { ...config.constraints, candidateBudget: 192 };

        const next = artifactWithScenarios(existing, [replacement], config);

        expect(next.scenarios.map(item => item.objective)).toEqual(['hand_path_impulse']);
        expect(next.comparison_contract.id).not.toBe(existing.comparison_contract.id);
    });

    it('rejects a scenario stamped with a different comparison contract', () => {
        const artifact = validArtifact();
        artifact.scenarios[0].comparison_contract_id = 'force-source-search/v1-mixed';

        expect(() => parseForceSourceArtifact(artifact)).toThrow(/share the registered comparison contract/i);
    });

    it('rejects a speed winner that loses to another displayed scenario', () => {
        const artifact = validArtifact();
        const competitor = structuredClone(artifact.scenarios[0]);
        competitor.objective = 'coriolis_impulse';
        competitor.score = 0;
        competitor.series.clubhead_speed_m_s = [0, 0.2];
        artifact.scenarios.push(competitor);

        expect(() => parseForceSourceArtifact(artifact)).toThrow(/clubhead_speed loses its objective/i);
    });

    it('registers a built-in comparison whose own objective wins every displayed cross-evaluation', () => {
        const artifactUrl = new URL('../public/force-source-comparison.json', import.meta.url);
        const artifact = parseForceSourceArtifact(JSON.parse(readFileSync(artifactUrl, 'utf8')));
        const crossScores = artifact.scenarios.map(item => scoreForceSourceSeries(item.series));

        expect(new Set(artifact.scenarios.map(item => item.comparison_contract_id))).toEqual(
            new Set([artifact.comparison_contract.id]),
        );
        for (const [index, scenario] of artifact.scenarios.entries()) {
            expect(crossScores[index][scenario.objective]).toBeCloseTo(
                Math.max(...crossScores.map(scores => scores[scenario.objective])),
                10,
            );
            expect(scenario.series.arm_angle_rad[0]).toBe(artifact.initial_pose.arm_angle_rad);
            expect(scenario.series.wrist_cock_rad[0]).toBe(artifact.initial_pose.wrist_cock_rad);
        }
    });
});

describe('comparison contract identity', () => {
    it('changes for every setting that can invalidate a comparison, but not for objective choice', () => {
        const baseline = optimizationConfig();
        const baselineId = buildOptimizationContract(baseline).id;
        const changedPose = { ...baseline, initialState: [-2.1, -1.57, 0, 0] as const };
        const changedParams = { ...baseline, params: { ...baseline.params, L2: 1.2 } };
        const changedSearch = { ...baseline, thoroughness: 'research' as const };
        const changedConstraint = {
            ...baseline,
            constraints: { ...baseline.constraints, integrationStepS: 0.001 },
        };
        const changedObjective = { ...baseline, objective: 'hand_path_impulse' as const };

        expect(buildOptimizationContract(changedPose).id).not.toBe(baselineId);
        expect(buildOptimizationContract(changedParams).id).not.toBe(baselineId);
        expect(buildOptimizationContract(changedSearch).id).not.toBe(baselineId);
        expect(buildOptimizationContract(changedConstraint).id).not.toBe(baselineId);
        expect(buildOptimizationContract(changedObjective).id).toBe(baselineId);
    });
});

describe('cross-objective optimizer certification', () => {
    it('does not let another displayed winner beat the speed objective', async () => {
        const config = optimizationConfig();
        config.thoroughness = 'quick';
        config.constraints = {
            ...config.constraints,
            candidateBudget: 160,
            robustnessTrials: 1,
            integrationStepS: 0.002,
        };
        const { objective: _objective, ...baseConfig } = config;

        const scenarios = await optimizeForceSourceComparison(
            baseConfig,
            ['clubhead_speed', 'hand_path_impulse'],
        );
        const crossScores = scenarios.map(item => scoreForceSourceSeries(item.series));
        const speed = scenarios.find(item => item.objective === 'clubhead_speed');

        expect(speed).toBeDefined();
        expect(speed?.score).toBeCloseTo(
            Math.max(...crossScores.map(scores => scores.clubhead_speed)),
            10,
        );
        expect(new Set(scenarios.map(item => item.comparison_contract_id)).size).toBe(1);
        expect(scenarios.every(item => item.provenance?.cross_objective_certified === true)).toBe(true);
    }, 15_000);
});

describe('hierarchical search grid', () => {
    it('honors user bounds, budget, and 0.5 N m wrist granularity', () => {
        const candidates = buildCandidateSet(optimizationConfig());

        expect(candidates).toHaveLength(96);
        expect(Math.max(...candidates.map(candidate => candidate.wrist_drive_nm))).toBe(30);
        expect(Math.max(...candidates.map(candidate => candidate.wrist_restrain_nm))).toBe(30);
        expect(candidates.every(candidate => candidate.wrist_drive_nm * 2 % 1 === 0)).toBe(true);
        expect(candidates.every(candidate => candidate.shoulder_torque_nm >= 60)).toBe(true);
        expect(candidates.every(candidate => candidate.shoulder_torque_nm <= 120)).toBe(true);
    });

    it('fails closed when a start pose violates the chosen joint bounds', () => {
        const config = optimizationConfig();
        config.constraints = {
            ...config.constraints,
            armAngleDeg: { min: -90, max: 45 },
        };

        expect(() => validateBrowserOptimizationConfig(config)).toThrow(/initial arm angle/i);
    });

    it('fails closed for wrist limits above the supported 30 N m', () => {
        const config = optimizationConfig();
        config.constraints = { ...config.constraints, wristTorqueLimitNm: 31 };

        expect(() => validateBrowserOptimizationConfig(config)).toThrow(/30/);
    });

    it('summarizes held-out qualification rather than reporting nominal-only robustness', () => {
        expect(summarizeRobustness([10, null, 12, 8])).toEqual({
            sample_count: 4,
            qualified_count: 3,
            qualification_rate: 0.75,
            median_score: 10,
            worst_score: 8,
            best_score: 12,
            score_spread: 4,
        });
    });
});

describe('golf-like impact qualification', () => {
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

    it('keeps every scenario hub at one shared point in fixed-hub mode', () => {
        const poses = FORCE_SOURCE_OBJECTIVES.map((_, index) => ({
            arm: -0.2 - index * 0.1,
            wrist: 0.2 + index * 0.1,
        }));

        for (const pose of poses) {
            const impact = pendulumThumbnailGeometry(pose.arm, pose.wrist, lengths);
            const origin = thumbnailOrigin('fixed_hub', impact);
            expect(origin).toEqual({ x: 96, y: 88 });
            expect(pendulumThumbnailGeometry(pose.arm, pose.wrist, lengths, origin)).toMatchObject({
                originX: 96,
                originY: 88,
            });
        }
    });

    it('offers impact alignment as an explicit alternative camera frame', () => {
        const impact = pendulumThumbnailGeometry(-0.3, 0.3, lengths);
        const origin = thumbnailOrigin('impact_aligned', impact);
        const aligned = pendulumThumbnailGeometry(-0.3, 0.3, lengths, origin);

        expect(aligned.tipX).toBeCloseTo(150);
        expect(aligned.tipY).toBeCloseTo(148);
    });

    it('rejects an unknown comparison frame instead of silently drifting', () => {
        const impact = pendulumThumbnailGeometry(-0.3, 0.3, lengths);

        expect(() => thumbnailOrigin('unknown' as 'fixed_hub', impact)).toThrow(/Unsupported animation alignment/);
    });
});

describe('objective registry', () => {
    it('includes signed force impulse along the hand path as the sixth objective', () => {
        expect(FORCE_SOURCE_OBJECTIVES).toContain('hand_path_impulse');
        expect(FORCE_SOURCE_OBJECTIVES).toHaveLength(6);
    });
});

describe('energy-transfer sign and interface identity', () => {
    it('reports positive proximal drain and distal delivery with the exact 2:1 identity', () => {
        const state = [-1.2, -0.8, 7, 5] as const;
        const sources = generalizedForceSources([...state], params, [0, 0]);
        const transfer = forceSourceTransferPowers(sources, [...state]);

        expect(transfer.coriolis_to_distal_w).toBeGreaterThan(0);
        expect(transfer.centrifugal_to_distal_w).toBeGreaterThan(0);
        expect(transfer.coriolis_to_distal_w).toBeCloseTo(
            2 * transfer.centrifugal_to_distal_w,
            12,
        );
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
