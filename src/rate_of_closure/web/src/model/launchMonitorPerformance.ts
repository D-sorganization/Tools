/** Descriptive performance metrics behind a swappable backend adapter. */

import { finiteLaunchMonitorScalar, type LaunchMonitorRow } from "./launchMonitorAnalysisTypes";

const YARDS_PER_METRE = 1.0936132983377078;
export type DistanceUnit = "yd" | "m";

const toYards = (value: unknown, unit: DistanceUnit): number | null => {
  const numeric = finiteLaunchMonitorScalar(value as never);
  if (numeric === null) return null;
  return unit === "m" ? numeric * YARDS_PER_METRE : numeric;
};

const requireColumns = (rows: LaunchMonitorRow[], columns: string[]) => {
  const available = new Set(rows.flatMap((row) => Object.keys(row)));
  const missing = columns.filter((column) => !available.has(column));
  if (missing.length) throw new RangeError(`Columns are unavailable: ${missing.join(", ")}`);
};

export interface DispersionRequest {
  lateralColumn: string; carryColumn: string;
  lateralUnit: DistanceUnit; carryUnit: DistanceUnit;
}

export function analyzeDispersion(rows: LaunchMonitorRow[], request: DispersionRequest) {
  requireColumns(rows, [request.lateralColumn, request.carryColumn]);
  const points = rows.flatMap((row, sourceIndex) => {
    const lateralYards = toYards(row[request.lateralColumn], request.lateralUnit);
    const carryYards = toYards(row[request.carryColumn], request.carryUnit);
    return lateralYards === null || carryYards === null ? [] : [
      { sourceIndex, lateralYards, carryYards },
    ];
  });
  if (!points.length) throw new RangeError("Dispersion requires finite lateral and carry values");
  const values = points.map(({ lateralYards }) => lateralYards);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.length > 1
    ? values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1) : 0;
  return {
    unit: "yd" as const, points, meanLateralYards: mean,
    standardDeviationYards: Math.sqrt(variance),
    rmsYards: Math.sqrt(values.reduce((sum, value) => sum + value ** 2, 0) / values.length),
    leftCount: values.filter((value) => value < 0).length,
    centerCount: values.filter((value) => value === 0).length,
    rightCount: values.filter((value) => value > 0).length,
    formula: "Lateral sign: negative = yards left, positive = yards right. RMS = sqrt(mean(lateral_yards^2)).",
  };
}

interface StrokesGainedRequest {
  expectedBeforeColumn: string; expectedAfterColumn: string; baselineSourceUrl: string;
}

export function calculateStrokesGained(rows: LaunchMonitorRow[], request: StrokesGainedRequest) {
  let source: URL;
  try { source = new URL(request.baselineSourceUrl); } catch { throw new RangeError("Strokes gained requires an HTTP(S) baseline source URL"); }
  if (!(source.protocol === "http:" || source.protocol === "https:")) {
    throw new RangeError("Strokes gained requires an HTTP(S) baseline source URL");
  }
  requireColumns(rows, [request.expectedBeforeColumn, request.expectedAfterColumn]);
  const values = rows.flatMap((row) => {
    const before = finiteLaunchMonitorScalar(row[request.expectedBeforeColumn]);
    const after = finiteLaunchMonitorScalar(row[request.expectedAfterColumn]);
    return before === null || after === null ? [] : [before - 1 - after];
  });
  if (!values.length) throw new RangeError("Strokes gained requires finite expected-stroke state");
  return {
    metricName: "user_supplied_expected_strokes_sg" as const, unit: "strokes" as const, values,
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    formula: "User-supplied expected-strokes SG = E(before) - 1 - E(after); the app did not reproduce or validate the cited baseline table.",
    sourceUrl: request.baselineSourceUrl,
  };
}

interface TargetErrorRequest {
  carryColumn: string; lateralColumn: string;
  carryUnit: DistanceUnit; lateralUnit: DistanceUnit; targetDistanceYards: number;
}

export function calculateTargetError(rows: LaunchMonitorRow[], request: TargetErrorRequest) {
  if (!(request.targetDistanceYards > 0)) throw new RangeError("Target distance must be positive");
  requireColumns(rows, [request.carryColumn, request.lateralColumn]);
  const values = rows.flatMap((row) => {
    const carry = toYards(row[request.carryColumn], request.carryUnit);
    const lateral = toYards(row[request.lateralColumn], request.lateralUnit);
    return carry === null || lateral === null ? [] : [
      Math.hypot(request.targetDistanceYards - carry, lateral),
    ];
  });
  if (!values.length) throw new RangeError("Target error requires finite carry and lateral values");
  return {
    metricName: "radial_target_error" as const, unit: "yd" as const, values,
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    formula: "radial_target_error = hypot(target_yards - carry_yards, lateral_yards)",
    sourceUrl: null,
  };
}

interface TrendRequest {
  metricColumn: string; sessionColumn: string; sessionOrderColumn: string; playerColumn: string;
  playerIdentityAttested: boolean; sessionIdentityAttested: boolean;
}

export function analyzeSessionTrend(rows: LaunchMonitorRow[], request: TrendRequest) {
  if (!request.playerIdentityAttested || !request.sessionIdentityAttested) {
    throw new RangeError("Player and session identity must both be explicitly attested");
  }
  requireColumns(rows, [request.metricColumn, request.sessionColumn, request.sessionOrderColumn, request.playerColumn]);
  const grouped = new Map<string, { playerId: string; sessionId: string; orders: Set<number>; values: number[] }>();
  rows.forEach((row) => {
    const playerId = String(row[request.playerColumn] ?? "").trim();
    const sessionId = String(row[request.sessionColumn] ?? "").trim();
    const order = finiteLaunchMonitorScalar(row[request.sessionOrderColumn]);
    const value = finiteLaunchMonitorScalar(row[request.metricColumn]);
    if (!playerId || !sessionId || order === null || value === null) return;
    const key = `${playerId}\u001f${sessionId}`;
    const group = grouped.get(key) ?? { playerId, sessionId, orders: new Set<number>(), values: [] };
    group.orders.add(order); group.values.push(value); grouped.set(key, group);
  });
  if (!grouped.size) throw new RangeError("Trend requires finite rows with trusted player/session identity");
  if ([...grouped.values()].some(({ orders }) => orders.size !== 1)) {
    throw new RangeError("Each player session must map to exactly one order value");
  }
  const perPlayer = new Map<string, Array<{ playerId: string; sessionId: string; sessionOrder: number; sampleCount: number; mean: number }>>();
  [...grouped.values()].forEach((group) => {
    const values = perPlayer.get(group.playerId) ?? [];
    values.push({ playerId: group.playerId, sessionId: group.sessionId,
      sessionOrder: [...group.orders][0], sampleCount: group.values.length,
      mean: group.values.reduce((sum, value) => sum + value, 0) / group.values.length });
    perPlayer.set(group.playerId, values);
  });
  const points = [...perPlayer.entries()].sort(([left], [right]) => left.localeCompare(right))
    .flatMap(([, playerPoints]) => {
      let cumulative = 0;
      return playerPoints.sort((left, right) => left.sessionOrder - right.sessionOrder || left.sessionId.localeCompare(right.sessionId))
        .map((point, index) => { cumulative += point.mean; return { ...point, cumulativeMean: cumulative / (index + 1) }; });
    });
  return { metric: request.metricColumn, points,
    formula: "Session mean = mean(metric rows); cumulative mean = equal-weight mean of session means." };
}
