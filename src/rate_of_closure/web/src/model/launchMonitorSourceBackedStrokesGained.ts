import { finiteLaunchMonitorScalar, type LaunchMonitorRow } from "./launchMonitorAnalysisTypes";
import { parseUniqueJson } from "./strictJson";

const CONTRACT_VERSION = "launch-monitor-strokes-gained-baseline/1.0.0";
const YARDS_PER_METRE = 1.0936132983377078;

export interface BaselineState {
  lie: string; distance_yards: number; expected_strokes: number;
}

export interface StrokesGainedBaseline {
  baselineId: string; version: string; sourceUrl: string; license: string;
  tableSha256: string; states: BaselineState[];
}

export interface SourceBackedStrokesGainedRequest {
  beforeLieColumn: string; beforeDistanceColumn: string;
  afterLieColumn: string; afterDistanceColumn: string;
  beforeDistanceUnit: "yd" | "m"; afterDistanceUnit: "yd" | "m";
}

const canonicalStates = (states: BaselineState[]) => JSON.stringify(states.map((state) => ({
  distance_yards: state.distance_yards,
  expected_strokes: state.expected_strokes,
  lie: state.lie,
})));

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
  if (keys !== "distance_yards,expected_strokes,lie") throw new RangeError("Baseline state fields do not match the contract");
  const lie = text(item.lie, "lie").toLowerCase();
  const distance = finiteLaunchMonitorScalar(item.distance_yards as never);
  const expected = finiteLaunchMonitorScalar(item.expected_strokes as never);
  if (distance === null || distance < 0) throw new RangeError("distance_yards must be finite and nonnegative");
  if (expected === null || expected < 0) throw new RangeError("expected_strokes must be finite and nonnegative");
  return { lie, distance_yards: distance, expected_strokes: expected };
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
  const identities = new Set(states.map((item) => `${item.lie}\u001f${item.distance_yards}`));
  if (identities.size !== states.length) throw new RangeError("Baseline contains duplicate lie/distance states");
  return {
    baselineId: text(payload.baseline_id, "baseline_id"), version: text(payload.version, "version"),
    sourceUrl, license: text(payload.license, "license"), tableSha256: declared, states,
  };
}

const yards = (value: unknown, unit: "yd" | "m") => {
  const numeric = finiteLaunchMonitorScalar(value as never);
  return numeric === null ? null : numeric * (unit === "m" ? YARDS_PER_METRE : 1);
};

const expected = (baseline: StrokesGainedBaseline, lie: string, distance: number) => {
  const matches = baseline.states.filter((item) => item.lie === lie)
    .sort((left, right) => left.distance_yards - right.distance_yards);
  if (!matches.length || distance < matches[0].distance_yards || distance > matches[matches.length - 1].distance_yards) {
    throw new RangeError(`Course state ${lie}/${distance} yd is outside the baseline`);
  }
  const upperIndex = matches.findIndex((item) => item.distance_yards >= distance);
  const upper = matches[upperIndex];
  if (upper.distance_yards === distance || upperIndex === 0) return upper.expected_strokes;
  const lower = matches[upperIndex - 1];
  const fraction = (distance - lower.distance_yards) / (upper.distance_yards - lower.distance_yards);
  return lower.expected_strokes + fraction * (upper.expected_strokes - lower.expected_strokes);
};

export function calculateSourceBackedStrokesGained(
  rows: LaunchMonitorRow[], baseline: StrokesGainedBaseline, request: SourceBackedStrokesGainedRequest,
) {
  const backingRows = rows.flatMap((row, sourceIndex) => {
    const beforeLie = String(row[request.beforeLieColumn] ?? "").trim().toLowerCase();
    const afterLie = String(row[request.afterLieColumn] ?? "").trim().toLowerCase();
    const beforeDistanceYards = yards(row[request.beforeDistanceColumn], request.beforeDistanceUnit);
    const afterDistanceYards = yards(row[request.afterDistanceColumn], request.afterDistanceUnit);
    if (!beforeLie || !afterLie || beforeDistanceYards === null || afterDistanceYards === null) return [];
    const expectedBefore = expected(baseline, beforeLie, beforeDistanceYards);
    const expectedAfter = expected(baseline, afterLie, afterDistanceYards);
    return [{ sourceIndex, beforeLie, beforeDistanceYards, afterLie, afterDistanceYards,
      expectedBefore, expectedAfter, strokesGained: expectedBefore - 1 - expectedAfter }];
  });
  if (!backingRows.length) throw new RangeError("Source-backed strokes gained requires complete course-state rows");
  const values = backingRows.map((row) => row.strokesGained);
  return {
    metricName: "source_backed_strokes_gained" as const, unit: "strokes" as const, values,
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    baselineId: baseline.baselineId, baselineVersion: baseline.version,
    sourceUrl: baseline.sourceUrl, license: baseline.license, tableSha256: baseline.tableSha256,
    backingRows,
    formula: "SG = verified E(before course state) - 1 - verified E(after course state); interpolation stays within one lie.",
  };
}
