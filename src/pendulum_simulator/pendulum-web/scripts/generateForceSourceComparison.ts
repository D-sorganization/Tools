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

const artifactPath = fileURLToPath(new URL('../public/force-source-comparison.json', import.meta.url));
const params = PRESETS[0].params;
const baseConfig = {
    params,
    initialState: DEFAULT_FORCE_SOURCE_INITIAL_STATE,
    thoroughness: 'research',
    constraints: {
        ...DEFAULT_OPTIMIZATION_CONSTRAINTS,
        wristTorqueStepNm: 0.5,
        candidateBudget: 512,
        robustnessTrials: 25,
        integrationStepS: 0.001,
    },
} as const;

const scenarios = await optimizeForceSourceComparison(baseConfig, FORCE_SOURCE_OBJECTIVES);
const output = artifactWithScenarios(null, scenarios, {
    ...baseConfig,
    objective: 'clubhead_speed',
});
parseForceSourceArtifact(output);
await writeFile(artifactPath, `${JSON.stringify(output, null, 2)}\n`, 'utf8');
