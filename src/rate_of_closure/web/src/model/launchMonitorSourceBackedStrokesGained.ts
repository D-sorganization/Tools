import { finiteLaunchMonitorScalar, type LaunchMonitorRow } from "./launchMonitorAnalysisTypes";
import { parseUniqueJson } from "./strictJson";

const CONTRACT_VERSION = "launch-monitor-strokes-gained-baseline/2.0.0";
const YARDS_PER_METRE = 1.0936132983377078;

export interface BaselineState {
  lie: string; context: string; target: string; distance_yards: number;
  expected_strokes: number; standard_error: number | null;
}

export interface StrokesGainedBaseline {
  baselineId: string; version: string; sourceUrl: string; license: string;
  tableSha256: string; states: BaselineState[];
}

export interface SourceBackedStrokesGainedRequest {
  beforeLieColumn: string; beforeContextColumn: string; beforeTargetColumn: string; beforeDistanceColumn: string;
  afterLieColumn: string; afterContextColumn: string; afterTargetColumn: string; afterDistanceColumn: string;
  beforeDistanceUnit: "yd" | "m"; afterDistanceUnit: "yd" | "m";
  trustedSummary?: {
    playerColumn: string; sessionColumn: string; clubColumn: string;
    orderColumn: string; orderUnit: string; evidence: string;
  };
}

export function buildSourceBackedStrokesGainedPayload(
  rows: LaunchMonitorRow[], baseline: StrokesGainedBaseline, request: SourceBackedStrokesGainedRequest,
): Record<string, unknown> {
  const summaries = request.trustedSummary ? ([
    ["player", request.trustedSummary.playerColumn],
    ["session", request.trustedSummary.sessionColumn],
    ["club", request.trustedSummary.clubColumn],
  ] as const).filter(([, column]) => column).map(([dimension, column]) => ({
    dimension, column, trust_level: "explicit_user_attested",
    evidence: request.trustedSummary?.evidence,
  })) : [];
  const longitudinal = request.trustedSummary?.playerColumn && request.trustedSummary.orderColumn ? {
    order_column: request.trustedSummary.orderColumn, order_unit: request.trustedSummary.orderUnit,
    group_column: request.trustedSummary.playerColumn, group_dimension: "player",
    trust_level: "explicit_user_attested", evidence: request.trustedSummary.evidence, min_samples: 3,
  } : undefined;
  return {
    records: rows,
    baseline: {
      contract_version: CONTRACT_VERSION, baseline_id: baseline.baselineId,
      version: baseline.version, source_url: baseline.sourceUrl, license: baseline.license,
      table_sha256: baseline.tableSha256, states: baseline.states,
    },
    request: {
      start: { lie_column: request.beforeLieColumn, context_column: request.beforeContextColumn,
        target_column: request.beforeTargetColumn, distance_column: request.beforeDistanceColumn,
        distance_unit: request.beforeDistanceUnit },
      finish: { lie_column: request.afterLieColumn, context_column: request.afterContextColumn,
        target_column: request.afterTargetColumn, distance_column: request.afterDistanceColumn,
        distance_unit: request.afterDistanceUnit },
      min_samples: 1, summaries, ...(longitudinal ? { longitudinal } : {}),
    },
  };
}

const canonicalNumber = (value: number) => {
  if (!Number.isFinite(value)) throw new RangeError("Baseline numbers must be finite");
  const normalized = value.toFixed(12).replace(/\.?0+$/, "");
  return normalized === "-0" || normalized === "" ? "0" : normalized;
};

const canonicalStates = (states: BaselineState[]) => JSON.stringify(states
  .map((state) => ({ context: state.context, distance_yards: canonicalNumber(state.distance_yards),
    expected_strokes: canonicalNumber(state.expected_strokes), lie: state.lie,
    standard_error: state.standard_error === null ? null : canonicalNumber(state.standard_error), target: state.target }))
  .sort((left, right) => left.lie.localeCompare(right.lie) || left.context.localeCompare(right.context)
    || left.target.localeCompare(right.target) || Number(left.distance_yards) - Number(right.distance_yards)));

export async function baselineTableHash(states: BaselineState[]): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(canonicalStates(states)));
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

const text = (value: unknown, name: string) => {
  if (typeof value !== "string" || !value.trim()) throw new RangeError(`${name} must be non-empty text`);
  return value.trim();
};

const state = (value: unknown): BaselineState => {
  if (!value || typeof value !== "object") throw new RangeError("Baseline states must be objects");
  const item = value as Record<string, unknown>;
  const keys = Object.keys(item).sort().join(",");
  if (keys !== "context,distance_yards,expected_strokes,lie,standard_error,target") throw new RangeError("Baseline state fields do not match the contract");
  const lie = text(item.lie, "lie").toLowerCase();
  const context = text(item.context, "context").toLowerCase();
  const target = text(item.target, "target").toLowerCase();
  const distance = finiteLaunchMonitorScalar(item.distance_yards as never);
  const expected = finiteLaunchMonitorScalar(item.expected_strokes as never);
  if (distance === null || distance < 0) throw new RangeError("distance_yards must be finite and nonnegative");
  if (expected === null || expected < 0) throw new RangeError("expected_strokes must be finite and nonnegative");
  const standardError = item.standard_error === null ? null : finiteLaunchMonitorScalar(item.standard_error as never);
  if (standardError !== null && standardError < 0) throw new RangeError("standard_error must be null or nonnegative");
  return { lie, context, target, distance_yards: distance, expected_strokes: expected, standard_error: standardError };
};

export async function parseStrokesGainedBaseline(source: string): Promise<StrokesGainedBaseline> {
  if (new TextEncoder().encode(source).length > 10 * 1024 * 1024) throw new RangeError("Baseline exceeds 10 MiB");
  const payload = parseUniqueJson(source, "strokes-gained baseline") as Record<string, unknown>;
  const keys = Object.keys(payload).sort().join(",");
  if (keys !== "baseline_id,contract_version,license,source_url,states,table_sha256,version") {
    throw new RangeError("Baseline artifact fields do not match the contract");
  }
  if (payload.contract_version !== CONTRACT_VERSION) throw new RangeError(`contract_version must be ${CONTRACT_VERSION}`);
  if (!Array.isArray(payload.states) || payload.states.length < 2) throw new RangeError("states needs at least two rows");
  const states = payload.states.map(state);
  const declared = text(payload.table_sha256, "table_sha256").toLowerCase();
  if (declared !== await baselineTableHash(states)) throw new RangeError("Baseline table SHA-256 does not match states");
  const sourceUrl = text(payload.source_url, "source_url");
  const parsedUrl = new URL(sourceUrl);
  if (!(parsedUrl.protocol === "https:" || parsedUrl.protocol === "http:")) throw new RangeError("source_url must be HTTP(S)");
  const identities = new Set(states.map((item) => `${item.lie}\u001f${item.context}\u001f${item.target}\u001f${item.distance_yards}`));
  if (identities.size !== states.length) throw new RangeError("Baseline contains duplicate course states");
  return {
    baselineId: text(payload.baseline_id, "baseline_id"), version: text(payload.version, "version"),
    sourceUrl, license: text(payload.license, "license"), tableSha256: declared, states,
  };
}

/**
 * Error posture — ADR-0048 decision G1-D3 (exclude-and-audit).
 *
 * A malformed shot no longer destroys the session and is never dropped in
 * silence: it is excluded, classified against one of the three canonical
 * reason codes, counted in the returned `exclusions` summary, and reflected in
 * `status`. These names mirror the Python twin
 * (`rate_of_closure.launch_monitor_strokes_gained`) and the canonical layer's
 * `ExcludedRowV1` / `ExclusionSummaryV1`, so all three classify the same
 * malformed row identically.
 */
export type StrokesGainedExclusionReason = "missing_course_state" | "invalid_distance" | "outside_baseline";
export type StrokesGainedResultStatus = "available" | "partial" | "unavailable";
export const STROKES_GAINED_EXCLUSION_REASONS: readonly StrokesGainedExclusionReason[] =
  ["missing_course_state", "invalid_distance", "outside_baseline"];

export interface StrokesGainedExcludedRow {
  sourceIndex: number; reasonCode: StrokesGainedExclusionReason; message: string;
}

export interface StrokesGainedExclusionSummary {
  inputRowCount: number; includedRowCount: number; totalExcluded: number;
  byReason: Record<string, number>;
}

class RowIssue extends Error {
  constructor(readonly reasonCode: StrokesGainedExclusionReason, message: string) {
    super(message);
    this.name = "RowIssue";
  }
}

const courseStateText = (value: unknown, label: string) => {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (!normalized) throw new RowIssue("missing_course_state", `${label} is missing`);
  return normalized;
};

const yards = (value: unknown, unit: "yd" | "m", label: string) => {
  if (typeof value === "boolean") throw new RowIssue("invalid_distance", `${label} must be numeric`);
  const numeric = finiteLaunchMonitorScalar(value as never);
  if (numeric === null) throw new RowIssue("missing_course_state", `${label} is missing`);
  if (numeric < 0) throw new RowIssue("invalid_distance", `${label} must be finite and nonnegative`);
  return numeric * (unit === "m" ? YARDS_PER_METRE : 1);
};

const expected = (baseline: StrokesGainedBaseline, lie: string, context: string, target: string, distance: number) => {
  const matches = baseline.states.filter((item) => item.lie === lie && item.context === context && item.target === target)
    .sort((left, right) => left.distance_yards - right.distance_yards);
  if (!matches.length || distance < matches[0].distance_yards || distance > matches[matches.length - 1].distance_yards) {
    throw new RowIssue("outside_baseline",
      `Course state ${lie}/${context}/${target}/${distance} yd is outside the baseline`);
  }
  const upperIndex = matches.findIndex((item) => item.distance_yards >= distance);
  const upper = matches[upperIndex];
  if (upper.distance_yards === distance || upperIndex === 0) return upper.expected_strokes;
  const lower = matches[upperIndex - 1];
  const fraction = (distance - lower.distance_yards) / (upper.distance_yards - lower.distance_yards);
  return lower.expected_strokes + fraction * (upper.expected_strokes - lower.expected_strokes);
};

const backingRow = (row: LaunchMonitorRow, sourceIndex: number,
  baseline: StrokesGainedBaseline, request: SourceBackedStrokesGainedRequest) => {
  const beforeLie = courseStateText(row[request.beforeLieColumn], "start lie");
  const beforeContext = courseStateText(row[request.beforeContextColumn], "start context");
  const beforeTarget = courseStateText(row[request.beforeTargetColumn], "start target/hole");
  const afterLie = courseStateText(row[request.afterLieColumn], "finish lie");
  const afterContext = courseStateText(row[request.afterContextColumn], "finish context");
  const afterTarget = courseStateText(row[request.afterTargetColumn], "finish target/hole");
  const beforeDistanceYards = yards(row[request.beforeDistanceColumn], request.beforeDistanceUnit, "start distance");
  const afterDistanceYards = yards(row[request.afterDistanceColumn], request.afterDistanceUnit, "finish distance");
  const expectedBefore = expected(baseline, beforeLie, beforeContext, beforeTarget, beforeDistanceYards);
  const expectedAfter = expected(baseline, afterLie, afterContext, afterTarget, afterDistanceYards);
  return { sourceIndex, beforeLie, beforeContext, beforeTarget, beforeDistanceYards,
    afterLie, afterContext, afterTarget, afterDistanceYards,
    expectedBefore, expectedAfter, strokesGained: expectedBefore - 1 - expectedAfter };
};

/**
 * Score every complete shot and audit every incomplete one (ADR-0048 G1-D3).
 *
 * Never throws on row content: a malformed shot is excluded with a
 * `reasonCode`, `status` degrades to `"partial"`, and `mean` is `null` exactly
 * when `status === "unavailable"`. `RangeError` stays reserved for
 * request-level defects the caller declared.
 */
export function calculateSourceBackedStrokesGained(
  rows: LaunchMonitorRow[], baseline: StrokesGainedBaseline, request: SourceBackedStrokesGainedRequest,
) {
  const backingRows: ReturnType<typeof backingRow>[] = [];
  const excludedRows: StrokesGainedExcludedRow[] = [];
  rows.forEach((row, sourceIndex) => {
    try {
      backingRows.push(backingRow(row, sourceIndex, baseline, request));
    } catch (caught) {
      if (!(caught instanceof RowIssue)) throw caught;
      excludedRows.push({ sourceIndex, reasonCode: caught.reasonCode, message: caught.message });
    }
  });
  const values = backingRows.map((row) => row.strokesGained);
  const byReason: Record<string, number> = {};
  for (const excluded of excludedRows) byReason[excluded.reasonCode] = (byReason[excluded.reasonCode] ?? 0) + 1;
  const status: StrokesGainedResultStatus = !backingRows.length ? "unavailable"
    : excludedRows.length ? "partial" : "available";
  return {
    metricName: "source_backed_strokes_gained" as const, unit: "strokes" as const, values,
    mean: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null,
    baselineId: baseline.baselineId, baselineVersion: baseline.version,
    sourceUrl: baseline.sourceUrl, license: baseline.license, tableSha256: baseline.tableSha256,
    backingRows,
    formula: "SG = verified E(before course state) - 1 - verified E(after course state); interpolation stays within an exact lie/context/target stratum.",
    status, excludedRows,
    exclusions: { inputRowCount: rows.length, includedRowCount: backingRows.length,
      totalExcluded: excludedRows.length, byReason } satisfies StrokesGainedExclusionSummary,
  };
}
