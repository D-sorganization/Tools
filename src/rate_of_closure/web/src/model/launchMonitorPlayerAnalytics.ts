import { finiteLaunchMonitorScalar, type LaunchMonitorRow } from "./launchMonitorAnalysisTypes";

export type MetricUnit = "deg" | "mph" | "m/s" | "rpm" | "yd" | "m" | "%" | "unitless";

export interface DispersionSummary {
  lateralColumn: string;
  carryColumn: string | null;
  sampleCount: number;
  meanLateralYards: number;
  standardDeviationYards: number;
  rmsYards: number;
  leftCount: number;
  rightCount: number;
  centerCount: number;
  points: Array<{ shotId: string; lateralYards: number; carryYards: number | null }>;
}

export interface StrokesGainedShot {
  shotId: string;
  carryYards: number;
  lateralYards: number;
  remainingYards: number;
  expectedBefore: number;
  expectedAfter: number;
  strokesGainedProxy: number;
}

export interface SessionTrendPoint {
  playerId: string;
  sessionId: string;
  order: number;
  sampleCount: number;
  mean: number;
}

export interface PlayerSessionTrend {
  playerId: string;
  points: SessionTrendPoint[];
  slopePerSession: number | null;
  changeFirstToLast: number | null;
}

export interface SessionTrend {
  outcome: string;
  unit: MetricUnit;
  players: PlayerSessionTrend[];
}

const COLUMN_UNITS: Array<[RegExp, MetricUnit]> = [
  [/(angle|path|direction|loft|azimuth|elevation)/i, "deg"],
  [/(spin|rpm)/i, "rpm"],
  [/(speed.*mph|mph)/i, "mph"],
  [/(speed.*mps|speed.*m_s|velocity.*m_s)/i, "m/s"],
  [/(yard|_yd$|carry_distance|total_distance|offline)/i, "yd"],
  [/(_m$|meters?|metres?|observed_lateral_m)/i, "m"],
  [/(percent|percentage|rate$)/i, "%"],
];

export const metricUnit = (column: string): MetricUnit =>
  COLUMN_UNITS.find(([pattern]) => pattern.test(column))?.[1] ?? "unitless";

export const metricLabel = (column: string): string => {
  const unit = metricUnit(column);
  const label = column.replace(/_/g, " ");
  return unit === "unitless" ? label : `${label} (${unit})`;
};

const columnByAliases = (rows: LaunchMonitorRow[], aliases: RegExp[]): string | null => {
  const columns = Object.keys(rows[0] ?? {});
  return columns.find((column) => aliases.some((alias) => alias.test(column))) ?? null;
};

const toYards = (value: number, unit: MetricUnit) => unit === "m" ? value * 1.0936133 : value;

export function dispersionSummary(rows: LaunchMonitorRow[]): DispersionSummary | null {
  const lateralColumn = columnByAliases(rows, [
    /^lateral(_deviation)?$/i, /^offline(_distance)?$/i, /carry.*side/i,
    /observed_lateral/i, /yards?_left_right/i, /^side$/i,
  ]);
  if (!lateralColumn) return null;
  const carryColumn = columnByAliases(rows, [
    /^carry(_distance)?$/i, /observed_carry/i, /^carry_yards?$/i,
  ]);
  const points = rows.flatMap((row, index) => {
    const lateral = finiteLaunchMonitorScalar(row[lateralColumn]);
    if (lateral === null) return [];
    const carry = carryColumn ? finiteLaunchMonitorScalar(row[carryColumn]) : null;
    return [{
      shotId: String(row.shot_id ?? index + 1),
      lateralYards: toYards(lateral, metricUnit(lateralColumn)),
      carryYards: carry === null || !carryColumn ? null : toYards(carry, metricUnit(carryColumn)),
    }];
  });
  if (!points.length) return null;
  const mean = points.reduce((sum, point) => sum + point.lateralYards, 0) / points.length;
  const variance = points.length > 1
    ? points.reduce((sum, point) => sum + (point.lateralYards - mean) ** 2, 0) / (points.length - 1)
    : 0;
  return {
    lateralColumn, carryColumn, sampleCount: points.length, meanLateralYards: mean,
    standardDeviationYards: Math.sqrt(variance),
    rmsYards: Math.sqrt(points.reduce((sum, point) => sum + point.lateralYards ** 2, 0) / points.length),
    leftCount: points.filter((point) => point.lateralYards < 0).length,
    rightCount: points.filter((point) => point.lateralYards > 0).length,
    centerCount: points.filter((point) => point.lateralYards === 0).length,
    points,
  };
}

export const STROKES_GAINED_REFERENCE = [
  { distanceYards: 0, expectedStrokes: 0 },
  { distanceYards: 25, expectedStrokes: 2.45 },
  { distanceYards: 50, expectedStrokes: 2.70 },
  { distanceYards: 100, expectedStrokes: 2.92 },
  { distanceYards: 150, expectedStrokes: 3.10 },
  { distanceYards: 200, expectedStrokes: 3.30 },
  { distanceYards: 250, expectedStrokes: 3.48 },
  { distanceYards: 300, expectedStrokes: 3.65 },
] as const;

export const expectedStrokes = (distanceYards: number): number => {
  const distance = Math.max(0, distanceYards);
  const upper = STROKES_GAINED_REFERENCE.find((point) => point.distanceYards >= distance);
  if (!upper) {
    const last = STROKES_GAINED_REFERENCE[STROKES_GAINED_REFERENCE.length - 1];
    return last.expectedStrokes + (distance - last.distanceYards) * 0.003;
  }
  const index = STROKES_GAINED_REFERENCE.indexOf(upper);
  if (index === 0) return upper.expectedStrokes;
  const lower = STROKES_GAINED_REFERENCE[index - 1];
  const fraction = (distance - lower.distanceYards) / (upper.distanceYards - lower.distanceYards);
  return lower.expectedStrokes + fraction * (upper.expectedStrokes - lower.expectedStrokes);
};

export function strokesGainedProxy(
  rows: LaunchMonitorRow[], targetDistanceYards: number,
): StrokesGainedShot[] {
  const dispersion = dispersionSummary(rows);
  if (!dispersion?.carryColumn || targetDistanceYards <= 0) return [];
  const before = expectedStrokes(targetDistanceYards);
  return dispersion.points.flatMap((point) => {
    if (point.carryYards === null) return [];
    const remaining = Math.hypot(targetDistanceYards - point.carryYards, point.lateralYards);
    const after = expectedStrokes(remaining);
    return [{
      shotId: point.shotId, carryYards: point.carryYards, lateralYards: point.lateralYards,
      remainingYards: remaining, expectedBefore: before, expectedAfter: after,
      strokesGainedProxy: before - 1 - after,
    }];
  });
}

export function sessionTrend(rows: LaunchMonitorRow[], outcome: string): SessionTrend | null {
  const grouped = new Map<string, Map<string, number[]>>();
  rows.forEach((row) => {
    const value = finiteLaunchMonitorScalar(row[outcome]);
    if (value === null) return;
    const playerId = String(row.player_id ?? "unassigned-player");
    const sessionId = String(row.session_id ?? "unassigned-session");
    const sessions = grouped.get(playerId) ?? new Map<string, number[]>();
    sessions.set(sessionId, [...(sessions.get(sessionId) ?? []), value]);
    grouped.set(playerId, sessions);
  });
  const players = [...grouped.entries()].sort(([a], [b]) => a.localeCompare(b)).map(
    ([playerId, sessions]) => {
      const points = [...sessions.entries()].sort(([a], [b]) => a.localeCompare(b)).map(
        ([sessionId, values], order) => ({
          playerId, sessionId, order, sampleCount: values.length,
          mean: values.reduce((sum, value) => sum + value, 0) / values.length,
        }),
      );
      const xMean = (points.length - 1) / 2;
      const yMean = points.reduce((sum, point) => sum + point.mean, 0) / points.length;
      const denominator = points.reduce((sum, point) => sum + (point.order - xMean) ** 2, 0);
      const slope = denominator
        ? points.reduce((sum, point) => sum + (point.order - xMean) * (point.mean - yMean), 0) / denominator
        : null;
      return {
        playerId, points, slopePerSession: slope,
        changeFirstToLast: points.length > 1 ? points[points.length - 1].mean - points[0].mean : null,
      };
    },
  );
  return players.length ? { outcome, unit: metricUnit(outcome), players } : null;
}

export const sessionTrendExportRows = (trend: SessionTrend) => trend.players.flatMap(
  (player) => player.points.map((point) => ({
    ...point,
    playerSlopePerSession: player.slopePerSession,
    playerChangeFirstToLast: player.changeFirstToLast,
    outcome: trend.outcome,
    unit: trend.unit,
  })),
);
