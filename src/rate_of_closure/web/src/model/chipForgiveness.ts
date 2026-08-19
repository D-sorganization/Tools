/** All-trial decision statistics for qualified chip-shot robustness studies. */

export const CHIP_COHORTS = [
  "ball_first",
  "ball_only",
  "ground_first",
  "simultaneous_or_grazing",
  "ground_only_miss",
  "no_contact_miss",
  "numerical_failure",
] as const;

export type ChipTrialCohortTs = (typeof CHIP_COHORTS)[number];

export interface ChipTrialRecordTs {
  trialIndex: number;
  cohort: ChipTrialCohortTs;
  loss: number;
  constraintViolated: boolean;
  metrics: Record<string, number | null>;
  diagnostic?: string | null;
  turfContactStatus?: string | null;
}

export interface BinomialEstimateTs {
  count: number;
  probability: number;
  ciLow: number;
  ciHigh: number;
}

export interface MetricDistributionTs {
  name: string;
  supportCount: number;
  unavailableCount: number;
  p05: number | null;
  p50: number | null;
  p95: number | null;
}

export interface ChipStudySummaryOptionsTs {
  seed?: number;
  cvarTailFraction?: number;
  bootstrapSamples?: number;
  turfCalibrationStatus?: "uncalibrated" | "illustrative" | "calibrated";
}

export interface ChipStudySummaryTs {
  sampleCount: number;
  cohorts: Record<ChipTrialCohortTs, BinomialEstimateTs>;
  expectedLoss: number;
  expectedLossCi: [number, number];
  cvarLoss: number;
  cvarTailFraction: number;
  constraintViolationRate: number;
  cleanContactProbability: number;
  metricDistributions: MetricDistributionTs[];
  convergence: Array<{
    sampleCount: number;
    meanLoss: number;
    standardError: number | null;
  }>;
  supportsTurfRankings: boolean;
  rankingScope: string;
}

export interface ChipCandidateScoreTs {
  candidateId: string;
  expectedLoss: number;
  cvarLoss: number;
  cleanProbability: number;
}

const WILSON_Z_95 = 1.959963984540054;
const MIN_BOOTSTRAP_SAMPLES = 64;

const mean = (values: number[]): number =>
  values.reduce((total, value) => total + value, 0) / values.length;

function percentile(values: number[], fraction: number): number {
  const sorted = [...values].sort((first, second) => first - second);
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function wilsonInterval(count: number, sampleCount: number): BinomialEstimateTs {
  const probability = count / sampleCount;
  const squared = WILSON_Z_95 ** 2;
  const denominator = 1 + squared / sampleCount;
  const center = (probability + squared / (2 * sampleCount)) / denominator;
  const spread = WILSON_Z_95 * Math.sqrt(
    probability * (1 - probability) / sampleCount
      + squared / (4 * sampleCount ** 2),
  ) / denominator;
  return {
    count,
    probability,
    ciLow: Math.max(0, center - spread),
    ciHigh: Math.min(1, center + spread),
  };
}

function seededRandom(seed: number): () => number {
  let state = (seed >>> 0) || 0x6d2b79f5;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function bootstrapMeanCi(
  losses: number[], seed: number, sampleCount: number,
): [number, number] {
  const random = seededRandom(seed);
  const means = Array.from({ length: sampleCount }, () => {
    let total = 0;
    for (let index = 0; index < losses.length; index += 1) {
      total += losses[Math.floor(random() * losses.length)];
    }
    return total / losses.length;
  });
  return [percentile(means, 0.025), percentile(means, 0.975)];
}

function convergence(losses: number[]): ChipStudySummaryTs["convergence"] {
  const counts = new Set<number>([losses.length]);
  for (let count = 1; count < losses.length; count *= 2) counts.add(count);
  return [...counts].sort((first, second) => first - second).map((sampleCount) => {
    const prefix = losses.slice(0, sampleCount);
    const meanLoss = mean(prefix);
    const variance = sampleCount < 2 ? null : prefix.reduce(
      (total, value) => total + (value - meanLoss) ** 2, 0,
    ) / (sampleCount - 1);
    return {
      sampleCount,
      meanLoss,
      standardError: variance === null ? null : Math.sqrt(variance / sampleCount),
    };
  });
}

function metricDistributions(records: ChipTrialRecordTs[]): MetricDistributionTs[] {
  const names = [...new Set(records.flatMap((record) => Object.keys(record.metrics)))].sort();
  return names.map((name) => {
    const values = records.flatMap((record) => {
      const value = record.metrics[name];
      return value === null || value === undefined ? [] : [value];
    });
    return {
      name,
      supportCount: values.length,
      unavailableCount: records.length - values.length,
      p05: values.length === 0 ? null : percentile(values, 0.05),
      p50: values.length === 0 ? null : percentile(values, 0.5),
      p95: values.length === 0 ? null : percentile(values, 0.95),
    };
  });
}

function validateRecords(records: ChipTrialRecordTs[]): void {
  if (records.length === 0) throw new RangeError("records must not be empty");
  records.forEach((record, index) => {
    if (record.trialIndex !== index) throw new RangeError("records must be in canonical trial order");
    if (!CHIP_COHORTS.includes(record.cohort)) throw new TypeError("unknown chip cohort");
    if (!Number.isFinite(record.loss) || record.loss < 0) {
      throw new RangeError("loss must be finite and >= 0");
    }
    Object.entries(record.metrics).forEach(([name, value]) => {
      if (name.length === 0) throw new RangeError("metric names must be nonempty");
      if (value !== null && !Number.isFinite(value)) {
        throw new RangeError(`metric ${name} must be finite or null`);
      }
    });
    if (record.turfContactStatus !== null
      && record.turfContactStatus !== undefined
      && record.turfContactStatus.trim().length === 0) {
      throw new RangeError("turfContactStatus must be nonempty or null");
    }
  });
}

/** Summarize every configured trial without silently dropping misses or failures. */
export function summarizeChipTrials(
  records: ChipTrialRecordTs[], options: ChipStudySummaryOptionsTs = {},
): ChipStudySummaryTs {
  validateRecords(records);
  const tailFraction = options.cvarTailFraction ?? 0.1;
  const bootstrapSamples = options.bootstrapSamples ?? 2_000;
  if (!(tailFraction > 0 && tailFraction <= 1)) {
    throw new RangeError("cvarTailFraction must be in (0, 1]");
  }
  if (bootstrapSamples < MIN_BOOTSTRAP_SAMPLES) {
    throw new RangeError(`bootstrapSamples must be >= ${MIN_BOOTSTRAP_SAMPLES}`);
  }
  const losses = records.map((record) => record.loss);
  const tailCount = Math.max(1, Math.ceil(tailFraction * losses.length));
  const worst = [...losses].sort((first, second) => second - first).slice(0, tailCount);
  const status = options.turfCalibrationStatus ?? "uncalibrated";
  const cohorts = Object.fromEntries(CHIP_COHORTS.map((cohort) => [
    cohort,
    wilsonInterval(records.filter((record) => record.cohort === cohort).length, records.length),
  ])) as Record<ChipTrialCohortTs, BinomialEstimateTs>;
  return {
    sampleCount: records.length,
    cohorts,
    expectedLoss: mean(losses),
    expectedLossCi: bootstrapMeanCi(losses, options.seed ?? 0, bootstrapSamples),
    cvarLoss: mean(worst),
    cvarTailFraction: tailFraction,
    constraintViolationRate: records.filter((record) => record.constraintViolated).length / records.length,
    cleanContactProbability: records.filter((record) =>
      record.cohort === "ball_first" || record.cohort === "ball_only").length / records.length,
    metricDistributions: metricDistributions(records),
    convergence: convergence(losses),
    supportsTurfRankings: status === "calibrated",
    rankingScope: status === "calibrated"
      ? "Conditional ranking for the declared noise, calibrated turf, objective, and solver only."
      : `Turf ranking disabled because profile status is ${status}; non-turf comparisons remain conditional.`,
  };
}

const dominates = (first: ChipCandidateScoreTs, second: ChipCandidateScoreTs): boolean =>
  first.expectedLoss <= second.expectedLoss
  && first.cvarLoss <= second.cvarLoss
  && first.cleanProbability >= second.cleanProbability
  && (first.expectedLoss < second.expectedLoss
    || first.cvarLoss < second.cvarLoss
    || first.cleanProbability > second.cleanProbability);

/** Return deterministic nondominated candidates without hiding tradeoffs in weights. */
export function candidateParetoFrontier(
  candidates: ChipCandidateScoreTs[],
): ChipCandidateScoreTs[] {
  if (candidates.length === 0) throw new RangeError("candidates must not be empty");
  if (new Set(candidates.map((candidate) => candidate.candidateId)).size !== candidates.length) {
    throw new RangeError("candidate IDs must be unique");
  }
  return candidates.filter((candidate) => !candidates.some((other) =>
    other !== candidate && dominates(other, candidate),
  )).sort((first, second) =>
    first.expectedLoss - second.expectedLoss || first.candidateId.localeCompare(second.candidateId),
  );
}
