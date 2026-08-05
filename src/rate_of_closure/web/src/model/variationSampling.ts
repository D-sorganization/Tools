import { covarianceFactor, covarianceMatrix } from "./variationGroups";
import {
  stableSpecId,
  validatePlan,
  type NoiseSpecTs,
  type VariationPlanTs,
} from "./variationSchema";
import { keysForMode, variableDef } from "./variationRegistry";

/** mulberry32: tiny public-domain 32-bit PRNG (Tommy Ettinger). */
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** FNV-1a 32-bit string hash for stable per-spec stream derivation. */
export function fnv1a(text: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

const streamFor = (seed: number, specId: string): (() => number) =>
  mulberry32((seed ^ fnv1a(specId)) >>> 0);

/** Box–Muller standard normal from a uniform source. */
const normalDraw = (rng: () => number): number => {
  let firstUniform = 0;
  while (firstUniform === 0) firstUniform = rng();
  return Math.sqrt(-2 * Math.log(firstUniform)) * Math.cos(2 * Math.PI * rng());
};

/** Inverse-CDF symmetric triangular draw on [-1, 1] with mode 0. */
const triangularDraw = (rng: () => number): number => {
  const uniform = rng();
  return uniform < 0.5
    ? Math.sqrt(2 * uniform) - 1
    : 1 - Math.sqrt(2 * (1 - uniform));
};

export const resolvedBase = (plan: VariationPlanTs): Record<string, number> => {
  const base: Record<string, number> = {};
  for (const key of keysForMode(plan.mode)) {
    base[key] = variableDef(key)!.default;
  }
  return { ...base, ...plan.baseVariables };
};

const clipSample = (value: number, spec: NoiseSpecTs): number => {
  const lower = spec.lower ?? -Infinity;
  const upper = spec.upper ?? Infinity;
  return Math.min(upper, Math.max(lower, value));
};

const sampleIndependent = (
  plan: VariationPlanTs,
  spec: NoiseSpecTs,
  center: number,
): number[] => {
  const rng = streamFor(plan.seed, stableSpecId(spec));
  return Array.from({ length: plan.nRuns }, () => {
    let value: number;
    if (spec.distribution === "normal") {
      value = center + spec.scale * normalDraw(rng);
    } else if (spec.distribution === "uniform") {
      value = center + spec.scale * (2 * rng() - 1);
    } else {
      value = center + spec.scale * triangularDraw(rng);
    }
    return clipSample(value, spec);
  });
};

const sampleGroups = (
  plan: VariationPlanTs,
  base: Record<string, number>,
  specsById: Map<string, NoiseSpecTs>,
): Map<string, number[]> => {
  const sampled = new Map<string, number[]>();
  for (const group of plan.groups ?? []) {
    const specs = group.specIds.map((specId) => specsById.get(specId)!);
    const factor = covarianceFactor(covarianceMatrix(group, specs));
    const independentColumns = specs.map((spec) => {
      const rng = streamFor(plan.seed, stableSpecId(spec));
      return Array.from({ length: plan.nRuns }, () => normalDraw(rng));
    });
    group.specIds.forEach((specId, outputIndex) => {
      const spec = specs[outputIndex];
      const center = base[spec.variableKey];
      sampled.set(
        specId,
        Array.from({ length: plan.nRuns }, (_unused, runIndex) => {
          const deviation = factor[outputIndex].reduce(
            (sum, coefficient, inputIndex) =>
              sum + coefficient * independentColumns[inputIndex][runIndex],
            0,
          );
          return clipSample(center + deviation, spec);
        }),
      );
    });
  }
  return sampled;
};

/** Sample the (nRuns x nSpecs) input matrix, subset-stable per spec ID. */
export function sampleInputs(plan: VariationPlanTs): number[][] {
  validatePlan(plan);
  const base = resolvedBase(plan);
  const specsById = new Map(plan.noise.map((spec) => [stableSpecId(spec), spec]));
  const sampled = sampleGroups(plan, base, specsById);
  for (const spec of plan.noise) {
    const specId = stableSpecId(spec);
    if (!sampled.has(specId)) {
      sampled.set(specId, sampleIndependent(plan, spec, base[spec.variableKey]));
    }
  }

  const columns = plan.noise.map((spec) => sampled.get(stableSpecId(spec))!);
  return Array.from({ length: plan.nRuns }, (_unused, runIndex) =>
    columns.map((column) => column[runIndex]),
  );
}
