import { finiteLaunchMonitorScalar, type LaunchMonitorRow } from "./launchMonitorAnalysisTypes";
import { normalCdf, normalQuantile, studentQuantile, studentTwoSidedP } from "./launchMonitorAnalysisStatistics";

export interface LongitudinalRequest {
  metricColumn: string; sessionColumn: string; sessionOrderColumn: string; playerColumn: string;
  playerIdentityAttested: boolean; sessionIdentityAttested: boolean; higherIsBetter: boolean;
  confidenceLevel: number; minSessions: number;
}

interface SessionPoint {
  playerId: string; sessionId: string; sessionOrder: number; sampleCount: number;
  mean: number; standardDeviation: number | null; standardError: number | null; cumulativeMean: number;
}

interface PlayerEstimate {
  playerId: string; sessionCount: number; slopePerSession: number | null;
  standardError: number | null; ciLower: number | null; ciUpper: number | null;
  pValue: number | null; rSquared: number | null; firstToLastChange: number | null; status: string;
}

const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
const sampleDeviation = (values: number[]) => values.length < 2 ? null : Math.sqrt(
  values.reduce((sum, value) => sum + (value - mean(values)) ** 2, 0) / (values.length - 1),
);

const validate = (rows: LaunchMonitorRow[], request: LongitudinalRequest) => {
  if (!request.playerIdentityAttested || !request.sessionIdentityAttested) throw new RangeError("Player and session identity must both be explicitly attested");
  if (!Number.isInteger(request.minSessions) || request.minSessions < 3) throw new RangeError("minSessions must be at least three");
  if (!(request.confidenceLevel > 0.5 && request.confidenceLevel < 1)) throw new RangeError("confidenceLevel must be between 0.5 and 1");
  const available = new Set(rows.flatMap((row) => Object.keys(row)));
  const required = [request.metricColumn, request.sessionColumn, request.sessionOrderColumn, request.playerColumn];
  const missing = required.filter((column) => !available.has(column));
  if (missing.length) throw new RangeError(`Columns are unavailable: ${missing.join(", ")}`);
};

const sessions = (rows: LaunchMonitorRow[], request: LongitudinalRequest): SessionPoint[] => {
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
  if (!grouped.size) throw new RangeError("Longitudinal analysis requires complete trusted session rows");
  if ([...grouped.values()].some((group) => group.orders.size !== 1)) throw new RangeError("Each player session must map to exactly one order");
  const duplicateOrders = new Set<string>();
  const points = [...grouped.values()].map((group) => {
    const sessionOrder = [...group.orders][0];
    const orderKey = `${group.playerId}\u001f${sessionOrder}`;
    if (duplicateOrders.has(orderKey)) throw new RangeError("Each player session requires a unique order value");
    duplicateOrders.add(orderKey);
    const standardDeviation = sampleDeviation(group.values);
    return { playerId: group.playerId, sessionId: group.sessionId, sessionOrder,
      sampleCount: group.values.length, mean: mean(group.values), standardDeviation,
      standardError: standardDeviation === null ? null : standardDeviation / Math.sqrt(group.values.length),
      cumulativeMean: 0 };
  }).sort((left, right) => left.playerId.localeCompare(right.playerId) || left.sessionOrder - right.sessionOrder || left.sessionId.localeCompare(right.sessionId));
  const totals = new Map<string, { sum: number; count: number }>();
  return points.map((point) => {
    const prior = totals.get(point.playerId) ?? { sum: 0, count: 0 };
    const next = { sum: prior.sum + point.mean, count: prior.count + 1 };
    totals.set(point.playerId, next); return { ...point, cumulativeMean: next.sum / next.count };
  });
};

const playerEstimate = (playerId: string, points: SessionPoint[], request: LongitudinalRequest): PlayerEstimate => {
  if (points.length < request.minSessions) return { playerId, sessionCount: points.length, slopePerSession: null, standardError: null,
    ciLower: null, ciUpper: null, pValue: null, rSquared: null, firstToLastChange: null, status: "insufficient_sessions" };
  const xMean = mean(points.map((point) => point.sessionOrder));
  const yMean = mean(points.map((point) => point.mean));
  const sxx = points.reduce((sum, point) => sum + (point.sessionOrder - xMean) ** 2, 0);
  const sxy = points.reduce((sum, point) => sum + (point.sessionOrder - xMean) * (point.mean - yMean), 0);
  if (sxx === 0) return { playerId, sessionCount: points.length, slopePerSession: null, standardError: null,
    ciLower: null, ciUpper: null, pValue: null, rSquared: null,
    firstToLastChange: points[points.length - 1].mean - points[0].mean, status: "constant_order" };
  const slope = sxy / sxx; const intercept = yMean - slope * xMean;
  const residualSum = points.reduce((sum, point) => sum + (point.mean - intercept - slope * point.sessionOrder) ** 2, 0);
  const degrees = points.length - 2; const standardError = Math.sqrt(residualSum / degrees / sxx);
  const critical = studentQuantile(0.5 + request.confidenceLevel / 2, degrees);
  const syy = points.reduce((sum, point) => sum + (point.mean - yMean) ** 2, 0);
  const statistic = standardError > 0 ? slope / standardError : Number.POSITIVE_INFINITY;
  return { playerId, sessionCount: points.length, slopePerSession: slope, standardError,
    ciLower: slope - critical * standardError, ciUpper: slope + critical * standardError,
    pValue: studentTwoSidedP(statistic, degrees),
    rSquared: syy > 0 ? sxy ** 2 / sxx / syy : null,
    firstToLastChange: points[points.length - 1].mean - points[0].mean,
    status: syy > 0 ? "ok" : "constant_metric" };
};

const population = (players: PlayerEstimate[], request: LongitudinalRequest) => {
  const eligible = players.filter((player) => player.status === "ok" && player.slopePerSession !== null && player.standardError !== null && player.standardError > 0);
  const empty = { contributorCount: eligible.length, fixedEffectSlope: null, fixedCiLower: null, fixedCiUpper: null,
    randomEffectSlope: null, randomCiLower: null, randomCiUpper: null, tauSquared: null,
    qStatistic: null, iSquaredPct: null, improvementProbability: null };
  if (eligible.length < 2) return empty;
  const variances = eligible.map((player) => player.standardError! ** 2);
  const weights = variances.map((variance) => 1 / variance); const weightSum = weights.reduce((sum, value) => sum + value, 0);
  const fixed = eligible.reduce((sum, player, index) => sum + player.slopePerSession! * weights[index], 0) / weightSum;
  const q = eligible.reduce((sum, player, index) => sum + weights[index] * (player.slopePerSession! - fixed) ** 2, 0);
  const cValue = weightSum - weights.reduce((sum, value) => sum + value ** 2, 0) / weightSum;
  const tauSquared = cValue > 0 ? Math.max(0, (q - (eligible.length - 1)) / cValue) : 0;
  const randomWeights = variances.map((variance) => 1 / (variance + tauSquared));
  const randomWeightSum = randomWeights.reduce((sum, value) => sum + value, 0);
  const random = eligible.reduce((sum, player, index) => sum + player.slopePerSession! * randomWeights[index], 0) / randomWeightSum;
  const critical = normalQuantile(0.5 + request.confidenceLevel / 2);
  const fixedMargin = critical / Math.sqrt(weightSum); const randomSe = 1 / Math.sqrt(randomWeightSum);
  const direction = request.higherIsBetter ? 1 : -1;
  return { contributorCount: eligible.length, fixedEffectSlope: fixed, fixedCiLower: fixed - fixedMargin, fixedCiUpper: fixed + fixedMargin,
    randomEffectSlope: random, randomCiLower: random - critical * randomSe, randomCiUpper: random + critical * randomSe,
    tauSquared, qStatistic: q, iSquaredPct: q > 0 ? Math.max(0, (q - (eligible.length - 1)) / q) * 100 : 0,
    improvementProbability: normalCdf(direction * random / randomSe) };
};

export function analyzeLongitudinalPerformance(rows: LaunchMonitorRow[], request: LongitudinalRequest) {
  validate(rows, request); const sessionPoints = sessions(rows, request);
  const byPlayer = new Map<string, SessionPoint[]>();
  sessionPoints.forEach((point) => byPlayer.set(point.playerId, [...(byPlayer.get(point.playerId) ?? []), point]));
  const players = [...byPlayer.entries()].map(([playerId, points]) => playerEstimate(playerId, points, request));
  return { request, sessionPoints, players, population: population(players, request),
    formula: "Equal-weight session means; player OLS slopes; inverse-variance fixed and DerSimonian-Laird random effects.",
    warnings: ["Observed longitudinal association does not establish causality or isolate practice effects.",
      "Equipment, intent, monitor, environment, selection, fatigue, and regression to the mean can explain change."] };
}
