import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

import {
    DEFAULT_OPTIMIZATION_CONSTRAINTS,
    optimizeForceSource,
    parseForceSourceArtifact,
} from '../src/forceSourceStudy';

const artifactPath = fileURLToPath(new URL('../public/force-source-comparison.json', import.meta.url));
const existing = parseForceSourceArtifact(JSON.parse(await readFile(artifactPath, 'utf8')));
if (!existing.parameters) throw new Error('The registered artifact must declare model parameters');

const initialState = [
    existing.initial_pose.arm_angle_rad,
    existing.initial_pose.wrist_cock_rad,
    existing.initial_pose.arm_velocity_rad_s,
    existing.initial_pose.wrist_velocity_rad_s,
] as [number, number, number, number];

const scenario = await optimizeForceSource({
    params: existing.parameters,
    initialState,
    objective: 'hand_path_impulse',
    thoroughness: 'research',
    constraints: {
        ...DEFAULT_OPTIMIZATION_CONSTRAINTS,
        wristTorqueStepNm: 0.5,
        candidateBudget: 512,
        robustnessTrials: 25,
        integrationStepS: 0.001,
    },
});

const scenarios = existing.scenarios.filter(item => item.objective !== scenario.objective);
scenarios.push(scenario);
const output = {
    ...existing,
    scenarios,
    provenance: {
        ...existing.provenance,
        hand_path_impulse: scenario.provenance,
    },
};
await writeFile(artifactPath, `${JSON.stringify(output, null, 2)}\n`, 'utf8');
