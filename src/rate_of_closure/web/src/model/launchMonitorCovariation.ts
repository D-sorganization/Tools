import { finiteLaunchMonitorScalar, type LaunchMonitorRow } from "./launchMonitorAnalysisTypes";
import {
  associationValues,
  fisherInterval,
  fisherZ,
  normalQuantile,
} from "./launchMonitorCovariationStatistics";

export interface CovariationRequest {
  xColumn: string;
  yColumn: string;
  playerColumn: string;
  minSamples: number;
  confidenceLevel: number;
}

export interface CovariationUiSettings {
  xColumn: string; yColumn: string; playerColumn: string; selectedPlayer: string;
  method: "pearson" | "spearman"; minSamples: number; confidenceLevel: number;
}

export const defaultCovariationSettings = (rows: LaunchMonitorRow[]): CovariationUiSettings => {
  const columns = Object.keys(rows[0] ?? {});
  const numeric = columns.filter((column) => rows.some(
    (row) => finiteLaunchMonitorScalar(row[column]) !== null,
  ));
  const playerColumn = columns.find((column) => /^player(_id)?$/i.test(column)) ??
    columns.find((column) => /player|golfer|athlete/i.test(column)) ?? "";
  return {
    xColumn: numeric.includes("club_path") ? "club_path" : numeric[0] ?? "",
    yColumn: numeric.includes("face_angle") ? "face_angle" : numeric[1] ?? numeric[0] ?? "",
    playerColumn, selectedPlayer: "", method: "pearson", minSamples: 4, confidenceLevel: 0.95,
  };
};

export interface AssociationSummary {
  sampleCount: number; groupCount: number;
  pearsonR: number | null; spearmanR: number | null;
  slope: number | null; intercept: number | null; rSquared: number | null;
  ciLower: number | null; ciUpper: number | null; status: string;
}

export interface PlayerAssociation extends AssociationSummary {
  playerId: string;
  fixedWeight: number | null;
  randomWeight: number | null;
}

export interface MetaAssociation {
  contributorCount: number; totalSampleCount: number;
  fixedEffectR: number | null; fixedCiLower: number | null; fixedCiUpper: number | null;
  randomEffectR: number | null; randomCiLower: number | null; randomCiUpper: number | null;
  tauSquared: number | null; qStatistic: number | null; iSquaredPct: number | null;
}

export interface CovariationResult {
  request: CovariationRequest;
  completePairCount: number;
  pooledRaw: AssociationSummary;
  withinPlayerCentered: AssociationSummary;
  betweenPlayer: AssociationSummary;
  perPlayer: PlayerAssociation[];
  meta: MetaAssociation;
  backingData: Array<Pair & { centeredX: number; centeredY: number }>;
  warnings: string[];
}

const covariationWarnings = (
  pooled: AssociationSummary, within: AssociationSummary,
): string[] => {
  const warnings = [
    "Associations are descriptive and do not establish causality.",
    "Ranked pair scans are exploratory and require multiple-comparison control or independent confirmation.",
    "Pooling can combine within-player and between-player structure; compare all three summaries.",
  ];
  if (pooled.pearsonR !== null && within.pearsonR !== null &&
    pooled.pearsonR * within.pearsonR < 0) {
    warnings.push(
      "The pooled and within-player Pearson associations have opposite signs; this aggregation reversal can reflect population structure.",
    );
  }
  return warnings;
};

interface Pair { playerId: string; x: number; y: number; sourceIndex: number; shotId: string }
interface CenteredPair extends Pair { sourceX: number; sourceY: number; centeredX: number; centeredY: number }

const validateRequest = (rows: LaunchMonitorRow[], request: CovariationRequest) => {
  if (request.xColumn === request.yColumn) throw new RangeError("X and Y must be different columns");
  if (!(request.confidenceLevel > 0.5 && request.confidenceLevel < 1)) {
    throw new RangeError("Confidence level must be between 0.5 and 1");
  }
  if (!Number.isInteger(request.minSamples) || request.minSamples < 4) {
    throw new RangeError("Minimum samples must be an integer of at least 4");
  }
  if (!rows.some((row) => Object.prototype.hasOwnProperty.call(row, request.playerColumn))) {
    throw new RangeError(`Player column not found: ${request.playerColumn}`);
  }
};

const completePairs = (rows: LaunchMonitorRow[], request: CovariationRequest): Pair[] =>
  rows.flatMap((row, sourceIndex) => {
    const x = finiteLaunchMonitorScalar(row[request.xColumn]);
    const y = finiteLaunchMonitorScalar(row[request.yColumn]);
    const player = row[request.playerColumn];
    const playerId = String(player ?? "").trim();
    if (x === null || y === null || !playerId) return [];
    return [{ playerId, x, y, sourceIndex, shotId: String(row.shot_id ?? sourceIndex + 1) }];
  });

const pairStatus = (pairs: Pair[], minimum: number): string => {
  if (pairs.length < minimum) return "insufficient_samples";
  if (pairs.length === 0) return "ok";
  // ⚡ Bolt Optimization: Replace multiple .map() and Math.max/min spreads with a single-pass loop
  // to eliminate intermediate allocations, avoid call stack limits, and exit early.
  let constantX = true;
  let constantY = true;
  const firstX = pairs[0].x;
  const firstY = pairs[0].y;
  for (let i = 1; i < pairs.length; i++) {
    if (constantX && pairs[i].x !== firstX) constantX = false;
    if (constantY && pairs[i].y !== firstY) constantY = false;
    if (!constantX && !constantY) break;
  }
  if (constantX && constantY) return "constant_both";
  if (constantX) return "constant_x";
  if (constantY) return "constant_y";
  return "ok";
};

const summarize = (
  pairs: Pair[], confidence: number, groupCount: number, minimum: number,
): AssociationSummary => {
  const status = pairStatus(pairs, minimum);
  if (status !== "ok") return {
    sampleCount: pairs.length, groupCount, pearsonR: null, spearmanR: null,
    slope: null, intercept: null, rSquared: null, ciLower: null, ciUpper: null, status,
  };
  const values = associationValues(pairs.map((pair) => pair.x), pairs.map((pair) => pair.y));
  const [ciLower, ciUpper] = fisherInterval(values.pearsonR, pairs.length, confidence);
  return { sampleCount: pairs.length, groupCount, ...values, ciLower, ciUpper, status };
};

const groupPairs = (pairs: Pair[]) => {
  const grouped = new Map<string, Pair[]>();
  for (const pair of pairs) {
    let group = grouped.get(pair.playerId);
    if (!group) {
      group = [];
      grouped.set(pair.playerId, group);
    }
    // Bolt: O(N) grouping instead of O(N^2) immutable spread
    group.push(pair);
  }
  return grouped;
};

const centeredPairs = (grouped: Map<string, Pair[]>): CenteredPair[] => [...grouped.entries()].flatMap(
  ([playerId, pairs]) => {
    const xMean = pairs.reduce((sum, pair) => sum + pair.x, 0) / pairs.length;
    const yMean = pairs.reduce((sum, pair) => sum + pair.y, 0) / pairs.length;
    return pairs.map((pair) => ({
      playerId, sourceX: pair.x, sourceY: pair.y,
      sourceIndex: pair.sourceIndex, shotId: pair.shotId,
      centeredX: pair.x - xMean, centeredY: pair.y - yMean,
      x: pair.x - xMean, y: pair.y - yMean,
    }));
  },
);

const meanPairs = (grouped: Map<string, Pair[]>): Pair[] => [...grouped.entries()].map(
  ([playerId, pairs]) => ({
    playerId,
    x: pairs.reduce((sum, pair) => sum + pair.x, 0) / pairs.length,
    y: pairs.reduce((sum, pair) => sum + pair.y, 0) / pairs.length,
    sourceIndex: -1, shotId: "player-mean",
  }),
);

interface MetaWork { index: number; count: number; z: number; variance: number }

const weightedMean = (work: MetaWork[], weights: number[]) =>
  work.reduce((sum, item, index) => sum + weights[index] * item.z, 0) /
  weights.reduce((sum, weight) => sum + weight, 0);

const pooledInterval = (center: number, weightSum: number, critical: number): [number, number] => {
  const margin = critical / Math.sqrt(weightSum);
  return [Math.tanh(center - margin), Math.tanh(center + margin)];
};

const metaAnalyze = (players: PlayerAssociation[], confidence: number): MetaAssociation => {
  const work = players.flatMap((player, index) => player.status === "ok" && player.pearsonR !== null
    ? [{ index, count: player.sampleCount, z: fisherZ(player.pearsonR), variance: 1 / (player.sampleCount - 3) }]
    : []);
  if (work.length < 2) return {
    contributorCount: work.length,
    totalSampleCount: work.reduce((sum, item) => sum + item.count, 0),
    fixedEffectR: null, fixedCiLower: null,
    fixedCiUpper: null, randomEffectR: null, randomCiLower: null, randomCiUpper: null,
    tauSquared: null, qStatistic: null, iSquaredPct: null,
  };
  const fixedWeights = work.map((item) => 1 / item.variance);
  const fixedCenter = weightedMean(work, fixedWeights);
  const qStatistic = work.reduce(
    (sum, item, index) => sum + fixedWeights[index] * (item.z - fixedCenter) ** 2, 0,
  );
  const weightSum = fixedWeights.reduce((sum, weight) => sum + weight, 0);
  const squaredWeightSum = fixedWeights.reduce((sum, weight) => sum + weight ** 2, 0);
  const degrees = work.length - 1;
  const cValue = weightSum - squaredWeightSum / weightSum;
  const tauSquared = cValue > 0 ? Math.max(0, (qStatistic - degrees) / cValue) : 0;
  const randomWeights = work.map((item) => 1 / (item.variance + tauSquared));
  const randomCenter = weightedMean(work, randomWeights);
  work.forEach((item, index) => {
    players[item.index].fixedWeight = fixedWeights[index] / weightSum;
    players[item.index].randomWeight = randomWeights[index] / randomWeights.reduce((sum, weight) => sum + weight, 0);
  });
  const critical = normalQuantile(0.5 + confidence / 2);
  const fixedInterval = pooledInterval(fixedCenter, weightSum, critical);
  const randomWeightSum = randomWeights.reduce((sum, weight) => sum + weight, 0);
  const randomInterval = pooledInterval(randomCenter, randomWeightSum, critical);
  return {
    contributorCount: work.length,
    totalSampleCount: work.reduce((sum, item) => sum + item.count, 0),
    fixedEffectR: Math.tanh(fixedCenter), fixedCiLower: fixedInterval[0], fixedCiUpper: fixedInterval[1],
    randomEffectR: Math.tanh(randomCenter), randomCiLower: randomInterval[0], randomCiUpper: randomInterval[1],
    tauSquared, qStatistic,
    iSquaredPct: qStatistic > 0 ? Math.max(0, (qStatistic - degrees) / qStatistic) * 100 : 0,
  };
};

export function analyzePlayerCovariation(
  rows: LaunchMonitorRow[], request: CovariationRequest,
): CovariationResult {
  validateRequest(rows, request);
  const pairs = completePairs(rows, request);
  const grouped = groupPairs(pairs);
  const centered = centeredPairs(grouped);
  const withinPlayerCentered = summarize(
    centered, request.confidenceLevel, grouped.size, request.minSamples,
  );
  withinPlayerCentered.ciLower = null;
  withinPlayerCentered.ciUpper = null;
  const pooledRaw = summarize(pairs, request.confidenceLevel, grouped.size, request.minSamples);
  const perPlayer = [...grouped.entries()].sort(([left], [right]) => left.localeCompare(right))
    .map(([playerId, playerPairs]) => ({
      playerId,
      ...summarize(playerPairs, request.confidenceLevel, 1, request.minSamples),
      fixedWeight: null, randomWeight: null,
    }));
  return {
    request, completePairCount: pairs.length,
    pooledRaw,
    withinPlayerCentered,
    betweenPlayer: summarize(meanPairs(grouped), request.confidenceLevel, grouped.size, 2),
    perPlayer, meta: metaAnalyze(perPlayer, request.confidenceLevel),
    backingData: centered.map((pair) => ({
      playerId: pair.playerId, x: pair.sourceX, y: pair.sourceY,
      sourceIndex: pair.sourceIndex, shotId: pair.shotId,
      centeredX: pair.centeredX, centeredY: pair.centeredY,
    })),
    warnings: covariationWarnings(pooledRaw, withinPlayerCentered),
  };
}

export interface PairRanking {
  xColumn: string; yColumn: string; contributorCount: number;
  randomEffectR: number | null; absoluteRandomEffectR: number;
  directionConsistency: number | null;
}

export function rankCovariationPairs(rows: LaunchMonitorRow[], options: {
  columns: string[]; playerColumn: string; minSamples: number; confidenceLevel: number;
}): PairRanking[] {
  const results: PairRanking[] = [];
  options.columns.forEach((xColumn, xIndex) => options.columns.slice(xIndex + 1).forEach((yColumn) => {
    const analysis = analyzePlayerCovariation(rows, { ...options, xColumn, yColumn });
    const included = analysis.perPlayer.filter((player) => player.status === "ok" && player.pearsonR !== null);
    const positive = included.filter((player) => (player.pearsonR ?? 0) > 0).length;
    const negative = included.filter((player) => (player.pearsonR ?? 0) < 0).length;
    const directionConsistency = analysis.meta.randomEffectR === null || !included.length
      ? null : Math.max(positive, negative) / included.length;
    results.push({
      xColumn, yColumn, contributorCount: analysis.meta.contributorCount,
      randomEffectR: analysis.meta.randomEffectR,
      absoluteRandomEffectR: Math.abs(analysis.meta.randomEffectR ?? 0), directionConsistency,
    });
  }));
  return results.sort((left, right) => right.absoluteRandomEffectR - left.absoluteRandomEffectR ||
    right.contributorCount - left.contributorCount || left.xColumn.localeCompare(right.xColumn) ||
    left.yColumn.localeCompare(right.yColumn));
}
