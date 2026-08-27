/**
 * Putt dispersion summary + wire `swing_sim.putt_dispersion/1` —
 * TypeScript mirror of `shared/python/swing_sim/putting/dispersion.py`
 * (#4800 P5).
 *
 * Scope boundary: the Monte-Carlo **execution** is Python-authoritative
 * (the canonical seeded PCG64 sampler in `swing_sim.variation`, which
 * this runtime deliberately does not re-implement — a second sampler
 * would be a second answer). What is twinned is the *outcome*
 * vocabulary: the summary statistics over a cohort of evaluated putts,
 * and the versioned report wire the web consumes.
 *
 * Metrics (derivations in the Python docstring):
 * - **Make percentage** — captured runs as a percentage; capture is the
 *   integrator's decision under the declared model, never a post-hoc
 *   radius test.
 * - **Leave distance** — rest-to-hole distance, `0` when holed;
 *   mean / median / p95 / max, because the tail is what costs strokes.
 * - **Start-line dispersion** — spread of the launch azimuth off the
 *   target line: the stroke's own dispersion, upstream of the green,
 *   and the quantity a putter's MOI acts on.
 *
 * Spread mirrors `finiteSampleStandardDeviation` (the shared Welford
 * sample standard deviation, ddof = 1) and percentiles mirror NumPy's
 * linear interpolation, so a putting study and a full-swing study read
 * the same way. A one-run study has no spread and is refused rather
 * than reported as NaN.
 */

export const PUTT_DISPERSION_FORMAT = "swing_sim.putt_dispersion/1";

/** Two samples are the minimum for a sample standard deviation. */
const MIN_RUNS_FOR_SPREAD = 2;

export interface PuttOutcome {
  holed: boolean;
  /** Launch direction off the target line [deg]; `+` = right. */
  startAzimuthDeg: number;
  /** Rest-to-hole distance [m]; `0` when holed. */
  leaveDistanceM: number;
  totalDistanceM: number;
  /** Lateral offset at rest or capture [m], left positive. */
  breakM: number;
  /** Holmes/Penner effective radius minus the closest approach [m]. */
  captureMarginM: number;
}

export interface PuttVariableDeclaration {
  variableKey: string;
  distribution: string;
  scale: number;
}

export interface PuttDispersionSummary {
  nRuns: number;
  holedCount: number;
  makePercent: number;
  leaveMeanM: number;
  leaveP50M: number;
  leaveP95M: number;
  leaveMaxM: number;
  startLineMeanDeg: number;
  startLineSigmaDeg: number;
  startLineP05Deg: number;
  startLineP95Deg: number;
  totalDistanceMeanM: number;
  totalDistanceSigmaM: number;
}

export interface PuttDispersionReport {
  scenarioId: string;
  seed: number;
  variables: PuttVariableDeclaration[];
  summary: PuttDispersionSummary;
}

const SUMMARY_FIELDS = [
  "n_runs",
  "holed_count",
  "make_percent",
  "leave_mean_m",
  "leave_p50_m",
  "leave_p95_m",
  "leave_max_m",
  "start_line_mean_deg",
  "start_line_sigma_deg",
  "start_line_p05_deg",
  "start_line_p95_deg",
  "total_distance_mean_m",
  "total_distance_sigma_m",
] as const;

const DECLARATION_FIELDS = ["variable_key", "distribution", "scale"] as const;
const REPORT_FIELDS = [
  "format",
  "scenario_id",
  "seed",
  "variables",
  "summary",
] as const;

function canonicalJson(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error("canonical JSON requires finite numbers");
    }
    return JSON.stringify(value);
  }
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    const parts = Object.keys(record)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`);
    return `{${parts.join(",")}}`;
  }
  throw new Error("unsupported canonical JSON value");
}

/** Shared Welford sample spread (ddof = 1); NaN below two samples. */
export function finiteSampleStandardDeviation(values: number[]): number {
  if (values.some((value) => !Number.isFinite(value))) {
    throw new Error("standard-deviation cohort must be finite");
  }
  if (values.length < MIN_RUNS_FOR_SPREAD) return Number.NaN;
  let count = 0;
  let mean = 0;
  let centeredSum = 0;
  for (const value of values) {
    count += 1;
    const delta = value - mean;
    mean += delta / count;
    centeredSum += delta * (value - mean);
  }
  return Math.sqrt(Math.max(0, centeredSum / (count - 1)));
}

/** NumPy's linear-interpolation percentile over a finite cohort. */
export function percentile(values: number[], fraction: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const position = (fraction / 100) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (position - lower) * (sorted[upper] - sorted[lower]);
}

function requireOutcome(outcome: PuttOutcome): PuttOutcome {
  if (typeof outcome.holed !== "boolean") {
    throw new Error("holed must be boolean");
  }
  for (const value of [
    outcome.startAzimuthDeg,
    outcome.leaveDistanceM,
    outcome.totalDistanceM,
    outcome.breakM,
    outcome.captureMarginM,
  ]) {
    if (!Number.isFinite(value)) throw new Error("outcomes must be finite");
  }
  if (outcome.leaveDistanceM < 0 || outcome.totalDistanceM < 0) {
    throw new Error("distances must be non-negative");
  }
  if (outcome.holed && outcome.leaveDistanceM !== 0) {
    throw new Error("a holed putt leaves nothing");
  }
  return outcome;
}

/** Summarize a cohort of sampled putts (module docstring). */
export function summarizePuttOutcomes(
  outcomes: PuttOutcome[],
): PuttDispersionSummary {
  if (outcomes.length < MIN_RUNS_FOR_SPREAD) {
    throw new Error("a dispersion summary needs at least two runs");
  }
  outcomes.forEach(requireOutcome);
  const leaves = outcomes.map((item) => item.leaveDistanceM);
  const starts = outcomes.map((item) => item.startAzimuthDeg);
  const totals = outcomes.map((item) => item.totalDistanceM);
  const holed = outcomes.filter((item) => item.holed).length;
  const mean = (values: number[]): number =>
    values.reduce((total, value) => total + value, 0) / values.length;
  return {
    nRuns: outcomes.length,
    holedCount: holed,
    makePercent: (100 * holed) / outcomes.length,
    leaveMeanM: mean(leaves),
    leaveP50M: percentile(leaves, 50),
    leaveP95M: percentile(leaves, 95),
    leaveMaxM: Math.max(...leaves),
    startLineMeanDeg: mean(starts),
    startLineSigmaDeg: finiteSampleStandardDeviation(starts),
    startLineP05Deg: percentile(starts, 5),
    startLineP95Deg: percentile(starts, 95),
    totalDistanceMeanM: mean(totals),
    totalDistanceSigmaM: finiteSampleStandardDeviation(totals),
  };
}

function requireSummary(summary: PuttDispersionSummary): PuttDispersionSummary {
  for (const name of ["nRuns", "holedCount"] as const) {
    const value = summary[name];
    if (!Number.isInteger(value) || value < 0) {
      throw new Error(`${name} must be a non-negative integer`);
    }
  }
  if (summary.holedCount > summary.nRuns) {
    throw new Error("holedCount cannot exceed nRuns");
  }
  const floats = [
    summary.makePercent,
    summary.leaveMeanM,
    summary.leaveP50M,
    summary.leaveP95M,
    summary.leaveMaxM,
    summary.startLineMeanDeg,
    summary.startLineSigmaDeg,
    summary.startLineP05Deg,
    summary.startLineP95Deg,
    summary.totalDistanceMeanM,
    summary.totalDistanceSigmaM,
  ];
  if (floats.some((value) => !Number.isFinite(value))) {
    throw new Error("summary values must be finite");
  }
  if (summary.makePercent < 0 || summary.makePercent > 100) {
    throw new Error("makePercent must be in [0, 100]");
  }
  return summary;
}

/** Serialize deterministically; identical studies are byte-identical. */
export function puttDispersionToJson(report: PuttDispersionReport): string {
  if (report.scenarioId.trim() === "") {
    throw new Error("scenarioId must be a name");
  }
  if (!Number.isInteger(report.seed) || report.seed < 0) {
    throw new Error("seed must be a non-negative integer");
  }
  requireSummary(report.summary);
  return canonicalJson({
    format: PUTT_DISPERSION_FORMAT,
    scenario_id: report.scenarioId,
    seed: report.seed,
    variables: report.variables.map((item) => ({
      variable_key: item.variableKey,
      distribution: item.distribution,
      scale: item.scale,
    })),
    summary: {
      n_runs: report.summary.nRuns,
      holed_count: report.summary.holedCount,
      make_percent: report.summary.makePercent,
      leave_mean_m: report.summary.leaveMeanM,
      leave_p50_m: report.summary.leaveP50M,
      leave_p95_m: report.summary.leaveP95M,
      leave_max_m: report.summary.leaveMaxM,
      start_line_mean_deg: report.summary.startLineMeanDeg,
      start_line_sigma_deg: report.summary.startLineSigmaDeg,
      start_line_p05_deg: report.summary.startLineP05Deg,
      start_line_p95_deg: report.summary.startLineP95Deg,
      total_distance_mean_m: report.summary.totalDistanceMeanM,
      total_distance_sigma_m: report.summary.totalDistanceSigmaM,
    },
  });
}

function requireExactKeys(
  data: unknown,
  expected: readonly string[],
  what: string,
): Record<string, unknown> {
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw new Error(`${what} must be an object`);
  }
  const record = data as Record<string, unknown>;
  const keys = Object.keys(record).sort();
  const wanted = [...expected].sort();
  if (
    keys.length !== wanted.length ||
    keys.some((key, index) => key !== wanted[index])
  ) {
    throw new Error(`${what} fields must be exactly ${wanted.join(", ")}`);
  }
  return record;
}

function finiteNumber(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number`);
  }
  return value;
}

function integerValue(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new Error(`${name} must be an integer`);
  }
  return value;
}

/** Parse and validate; unknown fields and wrong formats are refused. */
export function puttDispersionFromJson(text: string): PuttDispersionReport {
  if (typeof text !== "string") throw new Error("text must be a string");
  const data: unknown = JSON.parse(text);
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw new Error("putt dispersion must be an object");
  }
  const record = data as Record<string, unknown>;
  if (record.format !== PUTT_DISPERSION_FORMAT) {
    throw new Error(`format must be ${PUTT_DISPERSION_FORMAT}`);
  }
  requireExactKeys(record, REPORT_FIELDS, "putt dispersion");
  if (!Array.isArray(record.variables)) {
    throw new Error("variables must be a list");
  }
  const variables = record.variables.map((item, index) => {
    const section = requireExactKeys(
      item,
      DECLARATION_FIELDS,
      `variables[${index}]`,
    );
    for (const name of ["variable_key", "distribution"]) {
      if (typeof section[name] !== "string" || section[name] === "") {
        throw new Error(`${name} must be a name`);
      }
    }
    const scale = finiteNumber(section.scale, "scale");
    if (!(scale > 0)) throw new Error("scale must be positive");
    return {
      variableKey: section.variable_key as string,
      distribution: section.distribution as string,
      scale,
    };
  });
  const summary = requireExactKeys(record.summary, SUMMARY_FIELDS, "summary");
  if (typeof record.scenario_id !== "string") {
    throw new Error("scenario_id must be a string");
  }
  return {
    scenarioId: record.scenario_id,
    seed: integerValue(record.seed, "seed"),
    variables,
    summary: requireSummary({
      nRuns: integerValue(summary.n_runs, "n_runs"),
      holedCount: integerValue(summary.holed_count, "holed_count"),
      makePercent: finiteNumber(summary.make_percent, "make_percent"),
      leaveMeanM: finiteNumber(summary.leave_mean_m, "leave_mean_m"),
      leaveP50M: finiteNumber(summary.leave_p50_m, "leave_p50_m"),
      leaveP95M: finiteNumber(summary.leave_p95_m, "leave_p95_m"),
      leaveMaxM: finiteNumber(summary.leave_max_m, "leave_max_m"),
      startLineMeanDeg: finiteNumber(
        summary.start_line_mean_deg,
        "start_line_mean_deg",
      ),
      startLineSigmaDeg: finiteNumber(
        summary.start_line_sigma_deg,
        "start_line_sigma_deg",
      ),
      startLineP05Deg: finiteNumber(
        summary.start_line_p05_deg,
        "start_line_p05_deg",
      ),
      startLineP95Deg: finiteNumber(
        summary.start_line_p95_deg,
        "start_line_p95_deg",
      ),
      totalDistanceMeanM: finiteNumber(
        summary.total_distance_mean_m,
        "total_distance_mean_m",
      ),
      totalDistanceSigmaM: finiteNumber(
        summary.total_distance_sigma_m,
        "total_distance_sigma_m",
      ),
    }),
  };
}
