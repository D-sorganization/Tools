import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

import {
    artifactWithScenarios,
    DEFAULT_FORCE_SOURCE_INITIAL_STATE,
    DEFAULT_OPTIMIZATION_CONSTRAINTS,
    FORCE_SOURCE_OBJECTIVES,
    optimizeForceSourceComparison,
    parseForceSourceArtifact,
} from '../src/forceSourceStudy';
import { PRESETS } from '../src/presets';

const params = PRESETS[0].params;
const baseConfig = {
    params,
    initialState: DEFAULT_FORCE_SOURCE_INITIAL_STATE,
    thoroughness: 'research',
    constraints: {
        ...DEFAULT_OPTIMIZATION_CONSTRAINTS,
        wristTorqueStepNm: 0.5,
        candidateBudget: 2048,
        eliteCandidateCount: 16,
        robustnessTrials: 25,
        integrationStepS: 0.001,
    },
} as const;

const studies = [
    { mode: 'equal_speed', file: 'force-source-comparison.json' },
    { mode: 'equal_effort', file: 'force-source-comparison-equal-effort.json' },
] as const;

// Keep the 1 ms trajectories while avoiding meaningless 17-digit JSON noise.
// Twelve significant digits remain several orders tighter than the integrator
// and artifact-validation tolerances.
const REGISTERED_SETTING_KEYS = new Set([
    'arm_angle_rad', 'wrist_cock_rad', 'arm_velocity_rad_s', 'wrist_velocity_rad_s',
    'm1', 'm2', 'mClub', 'L1', 'L2', 'g', 'b1', 'b2', 'mu1', 'mu2',
]);

function artifactNumberPrecision(_key: string, value: unknown): unknown {
    if (REGISTERED_SETTING_KEYS.has(_key)) return value;
    return typeof value === 'number' && Number.isFinite(value)
        ? Number(value.toPrecision(12))
        : value;
}

for (const study of studies) {
    const config = {
        ...baseConfig,
        constraints: { ...baseConfig.constraints, studyMode: study.mode },
    };
    const scenarios = await optimizeForceSourceComparison(config, FORCE_SOURCE_OBJECTIVES);
    const output = artifactWithScenarios(null, scenarios, {
        ...config,
        objective: 'clubhead_speed',
    });
    parseForceSourceArtifact(output);
    const artifactPath = fileURLToPath(new URL(`../public/${study.file}`, import.meta.url));
    await writeFile(artifactPath, `${JSON.stringify(output, artifactNumberPrecision, 2)}\n`, 'utf8');
}
