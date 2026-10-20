import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import { STANDARD_GRAVITY } from '../src/physics';
import { PRESETS } from '../src/presets';
import { PRESETS_GOLFER } from '../src/presets_golfer';
import { PRESETS_TRIPLE } from '../src/presets_triple';

const CHECKED_IN_FORCE_SOURCE_ARTIFACTS = [
    'force-source-comparison.json',
    'force-source-comparison-equal-effort.json',
];

// Owner decision (Tools #5635): default gravity is standard gravity, 9.80665
// m/s^2, matching UpstreamDrift (previously 9.81).
describe('default gravity', () => {
    it('is standard gravity', () => {
        expect(STANDARD_GRAVITY).toBe(9.80665);
    });

    it('is used by every preset', () => {
        for (const preset of [...PRESETS, ...PRESETS_TRIPLE, ...PRESETS_GOLFER]) {
            expect(preset.params.g).toBe(STANDARD_GRAVITY);
        }
    });

    // The checked-in comparisons are generated from PRESETS[0]; a stale g means
    // the artifact was not regenerated after a gravity change.
    it.each(CHECKED_IN_FORCE_SOURCE_ARTIFACTS)('is recorded in %s', file => {
        const artifact = JSON.parse(readFileSync(new URL(`../public/${file}`, import.meta.url), 'utf8'));
        expect(artifact.parameters.g).toBe(STANDARD_GRAVITY);
    });
});
