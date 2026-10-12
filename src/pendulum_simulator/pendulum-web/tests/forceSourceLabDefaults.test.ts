/**
 * The Force Source Lab must produce a qualified winner from its own defaults on
 * the shipped preset (Tools #5431 / #5553). The canonical distributed club
 * changed the inertia the search was tuned for; a Lab that throws "No candidate
 * reached the qualified, non-looping golf impact event" on first click is dead.
 * Qualification criteria are NOT relaxed here: the winner must meet the
 * registered speed window, effort caps and robustness rate exactly.
 */
import { describe, expect, it } from 'vitest';

import { DEFAULT_FORCE_SOURCE_INITIAL_STATE, DEFAULT_OPTIMIZATION_CONSTRAINTS } from '../src/forceSourceTypes';
import { FORCE_SOURCE_OBJECTIVES } from '../src/forceSourceStudy';
import {
    optimizeForceSourceComparison,
    seriesMeetsStudyContract,
} from '../src/forceSourceOptimization';
import { PRESETS } from '../src/presets';

describe('Force Source Lab defaults on the distributed-club preset', () => {
    it('yields a contract-qualified, robust winner for every default objective', async () => {
        const constraints = { ...DEFAULT_OPTIMIZATION_CONSTRAINTS };
        const scenarios = await optimizeForceSourceComparison(
            {
                params: PRESETS[0].params,
                initialState: DEFAULT_FORCE_SOURCE_INITIAL_STATE,
                thoroughness: 'thorough',
                constraints,
            },
            FORCE_SOURCE_OBJECTIVES,
        );
        expect(scenarios.map(item => item.objective)).toEqual([...FORCE_SOURCE_OBJECTIVES]);
        for (const scenario of scenarios) {
            expect(seriesMeetsStudyContract(scenario.series, constraints)).toBe(true);
            expect(scenario.robustness.qualification_rate)
                .toBeGreaterThanOrEqual(constraints.minimumRobustQualificationRate - 1e-12);
        }
    }, 280_000);
});
