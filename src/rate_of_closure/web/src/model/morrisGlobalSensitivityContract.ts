/** Strict, UI-neutral consumer for the versioned Morris screening report. */

export const MORRIS_REPORT_SCHEMA_ID = "swing-sim/morris-global-sensitivity-report" as const;
export const MORRIS_REPORT_SCHEMA_VERSION = 1 as const;
export const MORRIS_METHOD = "morris-elementary-effects" as const;

export type MorrisAvailability = "available" | "constant-output" | "insufficient-data";
export type MorrisSampleAdequacy = "adequate" | "limited" | "insufficient";
export type MorrisTargetKind = "scalar" | "state-point" | "impact" | "shot-outcome";

export interface MorrisDesignProvenance {
  readonly trajectories: number;
  readonly levels: number;
  readonly seed: number;
  readonly totalSamples: number;
  readonly normalizedStep: number;
}
export interface MorrisSourceProvenance {
  readonly specId: string;
  readonly variableKey: string;
  readonly unit: string;
  readonly bounds: readonly [number, number];
  readonly timeWindowS: readonly [number, number] | null;
  readonly pointIds: readonly string[];
}
export interface MorrisTargetProvenance {
  readonly name: string;
  readonly unit: string;
  readonly kind: MorrisTargetKind;
  readonly timeS: number | null;
  readonly pointId: string | null;
  readonly coordinateFrame: string | null;
}
export interface MorrisEffects {
  readonly mu: number | null;
  readonly muStar: number | null;
  readonly muStarStandardError: number | null;
  readonly sigma: number | null;
}
export interface MorrisDenominator {
  readonly totalPairs: number;
  readonly validPairs: number;
  readonly typedNoImpactPairs: number;
  readonly noImpactUnavailablePairs: number;
  readonly failedPairs: number;
  readonly nonfinitePairs: number;
}
export interface MorrisEstimate {
  readonly source: MorrisSourceProvenance;
  readonly target: MorrisTargetProvenance;
  readonly effects: MorrisEffects;
  readonly availability: MorrisAvailability;
  readonly sampleAdequacy: MorrisSampleAdequacy;
  readonly denominator: MorrisDenominator;
}
export interface MorrisReport {
  readonly schemaId: typeof MORRIS_REPORT_SCHEMA_ID;
  readonly schemaVersion: typeof MORRIS_REPORT_SCHEMA_VERSION;
  readonly method: typeof MORRIS_METHOD;
  readonly design: MorrisDesignProvenance;
  readonly assumptions: readonly string[];
  readonly interactionCaveat: string;
  readonly estimates: readonly MorrisEstimate[];
}

const ROOT_FIELDS = ["schema_id", "schema_version", "method", "design", "assumptions", "interaction_caveat", "estimates"] as const;
const DESIGN_FIELDS = ["trajectories", "levels", "seed", "total_samples", "normalized_step"] as const;
const ESTIMATE_FIELDS = ["source", "target", "effects", "availability", "sample_adequacy", "denominator"] as const;
const SOURCE_FIELDS = ["spec_id", "variable_key", "unit", "bounds", "time_window_s", "point_ids"] as const;
const TARGET_FIELDS = ["name", "unit", "kind", "time_s", "point_id", "coordinate_frame"] as const;
const EFFECT_FIELDS = ["mu", "mu_star", "mu_star_standard_error", "sigma"] as const;
const DENOMINATOR_FIELDS = ["total_pairs", "valid_pairs", "typed_no_impact_pairs", "no_impact_unavailable_pairs", "failed_pairs", "nonfinite_pairs"] as const;
const C0_CONTROL_MAX = 0x1f;
const C1_CONTROL_MIN = 0x7f;
const C1_CONTROL_MAX = 0x9f;
// Mirrors the Python producer's metric-level numerical clamp exactly. The
// squared-identity bound below propagates this delta through every square.
const PRODUCER_METRIC_CLAMP_EPSILON_MULTIPLIER = 64;

const asRecord = (value: unknown, name: string): Record<string, unknown> => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new RangeError(`${name} must be a plain object`);
  }
  const prototype = Object.getPrototypeOf(value) as object | null;
  if (prototype !== Object.prototype && prototype !== null) {
    throw new RangeError(`${name} must be a plain object`);
  }
  return value as Record<string, unknown>;
};

const exactFields = (value: Record<string, unknown>, fields: readonly string[], name: string): void => {
  const actual = Object.keys(value).sort();
  const expected = [...fields].sort();
  if (actual.length !== expected.length || actual.some((field, index) => field !== expected[index])) {
    throw new RangeError(`${name} fields do not match the v1 schema`);
  }
};

const containsControlCharacter = (value: string): boolean => Array.from(value).some((character) => {
  const codePoint = character.codePointAt(0) ?? 0;
  return codePoint <= C0_CONTROL_MAX || (codePoint >= C1_CONTROL_MIN && codePoint <= C1_CONTROL_MAX);
});

const stableText = (value: unknown, name: string): string => {
  if (typeof value !== "string" || value === "" || value !== value.trim()) {
    throw new RangeError(`${name} must be a nonempty trimmed string`);
  }
  if (containsControlCharacter(value)) throw new RangeError(`${name} must not contain control characters`);
  return value;
};

const finite = (value: unknown, name: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RangeError(`${name} must be finite`);
  }
  return value;
};

const integerAtLeast = (value: unknown, minimum: number, name: string): number => {
  const parsed = finite(value, name);
  if (!Number.isInteger(parsed) || parsed < minimum) {
    throw new RangeError(`${name} must be an integer >= ${minimum}`);
  }
  return parsed;
};

const nullableFinite = (value: unknown, name: string): number | null => (
  value === null ? null : finite(value, name)
);

const nullableText = (value: unknown, name: string): string | null => (
  value === null ? null : stableText(value, name)
);

const pair = (value: unknown, name: string): readonly [number, number] => {
  if (!Array.isArray(value) || value.length !== 2) throw new RangeError(`${name} must contain two values`);
  const result = [finite(value[0], `${name}[0]`), finite(value[1], `${name}[1]`)] as const;
  if (!(result[0] < result[1])) throw new RangeError(`${name} must satisfy lower < upper`);
  return Object.freeze(result);
};

const optionalPair = (value: unknown, name: string): readonly [number, number] | null => (
  value === null ? null : pair(value, name)
);

const textArray = (value: unknown, name: string): readonly string[] => {
  if (!Array.isArray(value)) throw new RangeError(`${name} must be an array`);
  const parsed = value.map((entry, index) => stableText(entry, `${name}[${index}]`));
  if (new Set(parsed).size !== parsed.length) throw new RangeError(`${name} values must be unique`);
  return Object.freeze(parsed);
};

const vocabulary = <Value extends string>(
  value: unknown, allowed: readonly Value[], name: string,
): Value => {
  const parsed = stableText(value, name);
  if (!allowed.includes(parsed as Value)) throw new RangeError(`${name} is unsupported`);
  return parsed as Value;
};

const parseDesign = (value: unknown): MorrisDesignProvenance => {
  const item = asRecord(value, "Morris design");
  exactFields(item, DESIGN_FIELDS, "Morris design");
  const levels = integerAtLeast(item.levels, 4, "design levels");
  if (levels % 2 !== 0) throw new RangeError("design levels must be even");
  const normalizedStep = finite(item.normalized_step, "design normalized_step");
  const expectedStep = levels / (2 * (levels - 1));
  if (Math.abs(normalizedStep - expectedStep) > Number.EPSILON * 8) {
    throw new RangeError("design normalized_step does not match the Morris grid");
  }
  return Object.freeze({
    trajectories: integerAtLeast(item.trajectories, 1, "design trajectories"),
    levels,
    seed: integerAtLeast(item.seed, 0, "design seed"),
    totalSamples: integerAtLeast(item.total_samples, 1, "design total_samples"),
    normalizedStep,
  });
};

const parseSource = (value: unknown): MorrisSourceProvenance => {
  const item = asRecord(value, "Morris source");
  exactFields(item, SOURCE_FIELDS, "Morris source");
  return Object.freeze({
    specId: stableText(item.spec_id, "source spec_id"),
    variableKey: stableText(item.variable_key, "source variable_key"),
    unit: stableText(item.unit, "source unit"),
    bounds: pair(item.bounds, "source bounds"),
    timeWindowS: optionalPair(item.time_window_s, "source time_window_s"),
    pointIds: textArray(item.point_ids, "source point_ids"),
  });
};

const parseTarget = (value: unknown): MorrisTargetProvenance => {
  const item = asRecord(value, "Morris target");
  exactFields(item, TARGET_FIELDS, "Morris target");
  const kind = vocabulary(item.kind, ["scalar", "state-point", "impact", "shot-outcome"] as const, "target kind");
  const pointId = nullableText(item.point_id, "target point_id");
  const coordinateFrame = nullableText(item.coordinate_frame, "target coordinate_frame");
  if (kind === "state-point" && (pointId === null || coordinateFrame === null)) {
    throw new RangeError("state-point target requires point_id and coordinate_frame");
  }
  return Object.freeze({
    name: stableText(item.name, "target name"),
    unit: stableText(item.unit, "target unit"),
    kind,
    timeS: nullableFinite(item.time_s, "target time_s"),
    pointId,
    coordinateFrame,
  });
};

const parseEffects = (value: unknown): MorrisEffects => {
  const item = asRecord(value, "Morris effects");
  exactFields(item, EFFECT_FIELDS, "Morris effects");
  const effects = {
    mu: nullableFinite(item.mu, "effect mu"),
    muStar: nullableFinite(item.mu_star, "effect mu_star"),
    muStarStandardError: nullableFinite(item.mu_star_standard_error, "effect mu_star_standard_error"),
    sigma: nullableFinite(item.sigma, "effect sigma"),
  };
  for (const [name, metric] of [["mu_star", effects.muStar], ["mu_star_standard_error", effects.muStarStandardError], ["sigma", effects.sigma]] as const) {
    if (metric !== null && metric < 0) throw new RangeError(`effect ${name} must be nonnegative`);
  }
  const nullCount = Object.values(effects).filter((metric) => metric === null).length;
  if (nullCount !== 0 && nullCount !== 4) throw new RangeError("effect estimates must be all finite or all null");
  if (effects.mu !== null && effects.muStar !== null && effects.muStar < Math.abs(effects.mu)) {
    throw new RangeError("effect mu_star must be at least the absolute mean effect");
  }
  return Object.freeze(effects);
};

const parseDenominator = (value: unknown): MorrisDenominator => {
  const item = asRecord(value, "Morris denominator");
  exactFields(item, DENOMINATOR_FIELDS, "Morris denominator");
  return Object.freeze({
    totalPairs: integerAtLeast(item.total_pairs, 0, "denominator total_pairs"),
    validPairs: integerAtLeast(item.valid_pairs, 0, "denominator valid_pairs"),
    typedNoImpactPairs: integerAtLeast(item.typed_no_impact_pairs, 0, "denominator typed_no_impact_pairs"),
    noImpactUnavailablePairs: integerAtLeast(item.no_impact_unavailable_pairs, 0, "denominator no_impact_unavailable_pairs"),
    failedPairs: integerAtLeast(item.failed_pairs, 0, "denominator failed_pairs"),
    nonfinitePairs: integerAtLeast(item.nonfinite_pairs, 0, "denominator nonfinite_pairs"),
  });
};

const metricClampDelta = (muStar: number): number => (
  PRODUCER_METRIC_CLAMP_EPSILON_MULTIPLIER * Number.EPSILON * Math.max(1, muStar)
);

const squaredMetricError = (metric: number, delta: number): number => (
  2 * Math.abs(metric) * delta + delta ** 2
);

const metricIdentityTolerance = (effects: MorrisEffects, sampleCount: number, delta: number): number => {
  const { mu, muStar, muStarStandardError, sigma } = effects;
  if (mu === null || muStar === null || muStarStandardError === null || sigma === null) return 0;
  const sampleCorrection = sampleCount / (sampleCount - 1);
  return squaredMetricError(sigma, delta)
    + sampleCount * squaredMetricError(muStarStandardError, delta)
    + sampleCorrection * (squaredMetricError(muStar, delta) + squaredMetricError(mu, delta));
};

const validateFiniteMetrics = (estimate: MorrisEstimate): void => {
  const { mu, muStar, muStarStandardError, sigma } = estimate.effects;
  if (mu === null || muStar === null || muStarStandardError === null || sigma === null) return;
  if (muStar === 0) {
    if (mu !== 0 || sigma !== 0 || muStarStandardError !== 0
        || estimate.availability !== "constant-output") {
      throw new RangeError("zero mu_star requires zero metrics and constant-output availability");
    }
    return;
  }
  if (estimate.availability === "constant-output") {
    throw new RangeError("constant-output Morris effects must be zero");
  }
  const delta = metricClampDelta(muStar);
  const meanMagnitudeDifference = Math.abs(muStar - Math.abs(mu));
  if (meanMagnitudeDifference <= delta && muStarStandardError <= delta && sigma > delta) {
    throw new RangeError("Morris metric clamp-scale degeneracy is inconsistent");
  }
  if (sigma === 0 && (muStarStandardError !== 0
      || meanMagnitudeDifference > delta)) {
    throw new RangeError("zero-sigma Morris metric relationship failed");
  }
  const sampleCount = estimate.denominator.validPairs;
  const sigmaSquared = sigma ** 2;
  const scaledStandardErrorSquared = sampleCount * muStarStandardError ** 2;
  const meanDifference = sampleCount / (sampleCount - 1) * (muStar ** 2 - mu ** 2);
  const residual = sigmaSquared - scaledStandardErrorSquared - meanDifference;
  const tolerance = metricIdentityTolerance(estimate.effects, sampleCount, delta);
  if (Math.abs(residual) > tolerance) {
    throw new RangeError("Morris metric identity is inconsistent with valid_pairs");
  }
};

const validateEstimate = (estimate: MorrisEstimate, trajectories: number): void => {
  const denominator = estimate.denominator;
  const exclusiveTotal = denominator.validPairs + denominator.noImpactUnavailablePairs
    + denominator.failedPairs + denominator.nonfinitePairs;
  if (denominator.totalPairs !== trajectories || exclusiveTotal !== denominator.totalPairs) {
    throw new RangeError("Morris denominator invariant failed");
  }
  if (denominator.noImpactUnavailablePairs > denominator.typedNoImpactPairs
      || denominator.typedNoImpactPairs > denominator.totalPairs - denominator.failedPairs) {
    throw new RangeError("Morris typed no-impact denominator invariant failed");
  }
  const unavailable = estimate.availability === "insufficient-data";
  const allNull = estimate.effects.mu === null;
  if (unavailable !== allNull || unavailable !== (estimate.sampleAdequacy === "insufficient")) {
    throw new RangeError("Morris availability, adequacy, and null estimates disagree");
  }
  if (estimate.sampleAdequacy === "adequate" && denominator.validPairs < 10) {
    throw new RangeError("adequate Morris estimate requires at least ten valid pairs");
  }
  if (estimate.sampleAdequacy === "limited" && (denominator.validPairs < 2 || denominator.validPairs >= 10)) {
    throw new RangeError("limited Morris estimate requires two through nine valid pairs");
  }
  validateFiniteMetrics(estimate);
};

const parseEstimate = (value: unknown, trajectories: number): MorrisEstimate => {
  const item = asRecord(value, "Morris estimate");
  exactFields(item, ESTIMATE_FIELDS, "Morris estimate");
  const estimate = Object.freeze({
    source: parseSource(item.source),
    target: parseTarget(item.target),
    effects: parseEffects(item.effects),
    availability: vocabulary(item.availability, ["available", "constant-output", "insufficient-data"] as const, "availability"),
    sampleAdequacy: vocabulary(item.sample_adequacy, ["adequate", "limited", "insufficient"] as const, "sample_adequacy"),
    denominator: parseDenominator(item.denominator),
  });
  validateEstimate(estimate, trajectories);
  return estimate;
};

const validateReport = (report: MorrisReport): void => {
  const sources = new Map<string, string>();
  const targets = new Map<string, string>();
  const pairsBySource = new Map<string, Set<string>>();
  for (const estimate of report.estimates) {
    const sourceIdentity = JSON.stringify(estimate.source);
    const previous = sources.get(estimate.source.specId);
    if (previous !== undefined && previous !== sourceIdentity) throw new RangeError("source provenance changes within report");
    sources.set(estimate.source.specId, sourceIdentity);
    const targetIdentity = JSON.stringify(estimate.target);
    const previousTarget = targets.get(estimate.target.name);
    if (previousTarget !== undefined && previousTarget !== targetIdentity) throw new RangeError("target provenance changes within report");
    targets.set(estimate.target.name, targetIdentity);
    const sourceTargets = pairsBySource.get(estimate.source.specId) ?? new Set<string>();
    if (sourceTargets.has(estimate.target.name)) throw new RangeError("source/target estimate pairs must be unique");
    sourceTargets.add(estimate.target.name);
    pairsBySource.set(estimate.source.specId, sourceTargets);
  }
  const expectedSamples = report.design.trajectories * (sources.size + 1);
  if (sources.size === 0 || report.design.totalSamples !== expectedSamples) {
    throw new RangeError("design total_samples does not match trajectories and factor count");
  }
  if (report.estimates.length !== sources.size * targets.size) {
    throw new RangeError("Morris report must contain every source/target estimate pair");
  }
};

export function parseMorrisReport(value: unknown): MorrisReport {
  const item = asRecord(value, "Morris report");
  exactFields(item, ROOT_FIELDS, "Morris report");
  if (item.schema_id !== MORRIS_REPORT_SCHEMA_ID) throw new RangeError("unsupported Morris schema ID");
  if (item.schema_version !== MORRIS_REPORT_SCHEMA_VERSION) throw new RangeError("unsupported Morris schema version");
  if (item.method !== MORRIS_METHOD) throw new RangeError("unsupported Morris method");
  if (!Array.isArray(item.assumptions) || !Array.isArray(item.estimates)) {
    throw new RangeError("Morris assumptions and estimates must be arrays");
  }
  const design = parseDesign(item.design);
  const report = Object.freeze({
    schemaId: MORRIS_REPORT_SCHEMA_ID,
    schemaVersion: MORRIS_REPORT_SCHEMA_VERSION,
    method: MORRIS_METHOD,
    design,
    assumptions: textArray(item.assumptions, "Morris assumptions"),
    interactionCaveat: stableText(item.interaction_caveat, "interaction_caveat"),
    estimates: Object.freeze(item.estimates.map((estimate) => parseEstimate(estimate, design.trajectories))),
  });
  if (report.assumptions.length === 0) throw new RangeError("Morris assumptions must not be empty");
  validateReport(report);
  return report;
}

export function parseMorrisReportJson(source: string): MorrisReport {
  if (typeof source !== "string") throw new TypeError("Morris report JSON source must be a string");
  let value: unknown;
  try {
    value = JSON.parse(source) as unknown;
  } catch (error: unknown) {
    throw new SyntaxError(`Morris report must be valid JSON: ${error instanceof Error ? error.message : "parse failed"}`);
  }
  return parseMorrisReport(value);
}
