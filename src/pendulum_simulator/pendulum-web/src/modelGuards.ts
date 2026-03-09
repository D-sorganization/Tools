import type { SimulationResult } from './physics';
import type { SimulationResult3 } from './physics_triple';
import type { SimulationResult_golfer } from './physics_golfer';

type AnySimulationResult = SimulationResult | SimulationResult3 | SimulationResult_golfer;

function hasParams(result: unknown): result is AnySimulationResult {
    return (
        typeof result === 'object' &&
        result !== null &&
        'params' in result &&
        typeof result.params === 'object' &&
        result.params !== null
    );
}

export function isGolferSimulationResult(result: unknown): result is SimulationResult_golfer {
    return hasParams(result) && 'm_hub' in result.params;
}

export function isTripleSimulationResult(result: unknown): result is SimulationResult3 {
    return hasParams(result) && 'm3' in result.params;
}

export function isDoubleSimulationResult(result: unknown): result is SimulationResult {
    return (
        hasParams(result) &&
        'm1' in result.params &&
        !('m3' in result.params) &&
        !('m_hub' in result.params)
    );
}
