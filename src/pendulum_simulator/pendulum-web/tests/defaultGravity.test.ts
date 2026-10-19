import { describe, expect, it } from 'vitest';

import { STANDARD_GRAVITY } from '../src/physics';
import { PRESETS } from '../src/presets';
import { PRESETS_GOLFER } from '../src/presets_golfer';
import { PRESETS_TRIPLE } from '../src/presets_triple';

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
});
