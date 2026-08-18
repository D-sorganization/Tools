/** Deterministic robust optimization over clubs and delivery capabilities. */

import {
  type CapabilityEvaluator,
  type CapabilityObjective,
  type ClubCapability,
  type OptimizationAlternative,
  type OptimizationRequest,
  type OptimizationResult,
  type PlayerCapabilityProfile,
} from "./capabilityContract";
import { covarianceFactor, primeBases, radicalInverse } from "./capabilityMath";
import { contains, type TargetRegionTs } from "./targets";

const FAILURE_PENALTY = 1_000_000;
const BOUNDARY_TOLERANCE = 1e-9;
const PROVENANCE = Object.freeze({
  ensemble: "deterministic-correlated-low-discrepancy/v1",
  flight_metrics: "ball-flight-metrics/v1",
  optimizer: "capability-optimizer/v1",
  target_geometry: "swing_sim.solver.targets/v1",
});

interface Landing { readonly carryM: number; readonly offlineM: number }
interface Counts { readonly completed: number; readonly noImpact: number; readonly failed: number }
interface CandidateSummary { readonly alternative: OptimizationAlternative | null; readonly counts: Counts }
interface RiskMetrics {
  readonly meanCarryM: number; readonly expectedMissM: number;
  readonly holdProbability: number; readonly dispersionRmsM: number;
  readonly cvarMissM: number; readonly downsideCarryM: number;
}

const clubById = (profile: PlayerCapabilityProfile, clubId: string): ClubCapability => {
  const club = profile.clubs.find((item) => item.clubId === clubId);
  if (!club) throw new RangeError(`unknown club_id: ${clubId}`);
  return club;
};

const targetRegion = (request: OptimizationRequest): TargetRegionTs => Object.freeze({
  kind: request.target.kind, distanceM: request.target.distanceM,
  lateralM: request.target.lateralM, radiusM: request.target.radiusM,
  bandHalfLengthM: request.target.bandHalfLengthM, halfWidthM: request.target.halfWidthM,
});

const candidateParameters = (
  club: ClubCapability, candidateIndex: number, seed: number,
): Record<string, number> => {
  if (candidateIndex === 0) return Object.fromEntries(club.parameters.map((item) => [item.parameterId, item.baseline]));
  const sequenceIndex = candidateIndex + seed;
  const bases = primeBases(club.parameters.length);
  return Object.fromEntries(club.parameters.map((item, index) => [
    item.parameterId,
    item.lowerBound + radicalInverse(sequenceIndex, bases[index]) * (item.upperBound - item.lowerBound),
  ]));
};

const dimensionalCovariance = (club: ClubCapability): number[][] => {
  if (club.matrixKind === "covariance") return club.matrix.map((row) => [...row]);
  return club.matrix.map((row, rowIndex) => row.map((entry, columnIndex) =>
    entry * club.parameters[rowIndex].standardDeviation * club.parameters[columnIndex].standardDeviation));
};

const perturbedParameters = (
  club: ClubCapability, nominal: Readonly<Record<string, number>>, sampleIndex: number, seed: number,
): Record<string, number> => {
  const bases = primeBases(club.parameters.length);
  const sequenceIndex = sampleIndex + seed + 1;
  const independent = bases.map((base) => Math.sqrt(3) * (2 * radicalInverse(sequenceIndex, base) - 1));
  const factor = covarianceFactor(dimensionalCovariance(club));
  const correlated = factor.map((row) => row.reduce((sum, value, index) => sum + value * independent[index], 0));
  return Object.fromEntries(club.parameters.map((item, index) => [
    item.parameterId,
    Math.min(item.upperBound, Math.max(item.lowerBound, nominal[item.parameterId] + item.bias + correlated[index])),
  ]));
};

const parseLanding = (evaluation: ReturnType<CapabilityEvaluator>): Landing | null => {
  if (evaluation.status !== "complete") return null;
  const carry = evaluation.metrics.find((item) => item.metricId === "carry_distance")?.value;
  const offline = evaluation.metrics.find((item) => item.metricId === "carry_offline")?.value;
  if (!Number.isFinite(carry) || !Number.isFinite(offline)) return null;
  return Object.freeze({ carryM: carry as number, offlineM: offline as number });
};

const tailMean = (values: readonly number[], alpha: number, reverse: boolean): number => {
  const count = Math.max(1, Math.ceil(values.length * (1 - alpha)));
  const ordered = [...values].sort((left, right) => reverse ? right - left : left - right);
  return ordered.slice(0, count).reduce((sum, value) => sum + value, 0) / count;
};

const limitingConstraints = (
  club: ClubCapability,
  nominal: Readonly<Record<string, number>>,
  successFraction: number,
  extrapolated: boolean,
  request: OptimizationRequest,
): readonly string[] => {
  const limiting = club.parameters.flatMap((item) => {
    if (Math.abs(nominal[item.parameterId] - item.lowerBound) <= BOUNDARY_TOLERANCE) return [`${item.parameterId}:lower_safe_bound`];
    if (Math.abs(nominal[item.parameterId] - item.upperBound) <= BOUNDARY_TOLERANCE) return [`${item.parameterId}:upper_safe_bound`];
    return [];
  });
  if (successFraction < request.minimumSuccessFraction) limiting.push("minimum_success_fraction");
  if (extrapolated) limiting.push("evidence_envelope");
  return Object.freeze(limiting);
};

const objectiveScore = (
  objective: CapabilityObjective,
  risk: RiskMetrics,
  targetDistance: number,
): number => {
  if (objective === "maximize_carry") return -risk.meanCarryM;
  if (objective === "minimize_expected_miss") return risk.expectedMissM;
  if (objective === "maximize_target_hold") return -risk.holdProbability + risk.expectedMissM * 1e-6;
  if (objective === "minimize_variability") return risk.dispersionRmsM;
  if (objective === "minimize_downside") return risk.cvarMissM + risk.downsideCarryM;
  return Math.abs(risk.meanCarryM - targetDistance) + risk.dispersionRmsM;
};

const summarize = (
  club: ClubCapability,
  nominal: Readonly<Record<string, number>>,
  landings: readonly Landing[],
  counts: Counts,
  profile: PlayerCapabilityProfile,
  request: OptimizationRequest,
): OptimizationAlternative | null => {
  if (landings.length === 0) return null;
  const target = targetRegion(request);
  const centerOffline = target.kind === "green" ? target.lateralM : 0;
  const carries = landings.map((item) => item.carryM);
  const offlines = landings.map((item) => item.offlineM);
  const misses = landings.map((item) => Math.hypot(item.carryM - target.distanceM, item.offlineM - centerOffline));
  const meanCarry = carries.reduce((sum, value) => sum + value, 0) / carries.length;
  const meanOffline = offlines.reduce((sum, value) => sum + value, 0) / offlines.length;
  const expectedMiss = misses.reduce((sum, value) => sum + value, 0) / misses.length;
  const dispersion = Math.sqrt(landings.reduce((sum, item) =>
    sum + (item.carryM - meanCarry) ** 2 + (item.offlineM - meanOffline) ** 2, 0) / landings.length);
  const holdProbability = landings.filter((item) => contains(target, item.carryM, item.offlineM)).length / landings.length;
  const cvarMissM = tailMean(misses, request.cvarAlpha, true);
  const downsideCarryM = Math.max(0, target.distanceM - tailMean(carries, request.cvarAlpha, false));
  const risk = { meanCarryM: meanCarry, expectedMissM: expectedMiss, holdProbability, dispersionRmsM: dispersion, cvarMissM, downsideCarryM };
  const successFraction = counts.completed / request.ensembleSize;
  const failureFraction = 1 - successFraction;
  const extrapolated = club.parameters.some((item) =>
    nominal[item.parameterId] < item.evidenceLowerBound || nominal[item.parameterId] > item.evidenceUpperBound);
  const confidence = profile.confidence * club.confidence * successFraction * (extrapolated ? 0.5 : 1);
  let score = objectiveScore(request.objective, risk, target.distanceM);
  if (successFraction < request.minimumSuccessFraction) score += FAILURE_PENALTY * (request.minimumSuccessFraction - successFraction);
  return Object.freeze({
    rank: 1, clubId: club.clubId,
    parameters: Object.freeze(Object.entries(nominal).map(([parameterId, value]) => Object.freeze({ parameterId, value }))),
    score, meanCarryM: meanCarry, expectedMissM: expectedMiss, dispersionRmsM: dispersion,
    targetHoldProbability: holdProbability, cvarMissM, downsideCarryM,
    sampleCount: request.ensembleSize, successfulCount: counts.completed,
    noImpactCount: counts.noImpact, failedCount: counts.failed, failureFraction, confidence,
    limitingConstraints: limitingConstraints(club, nominal, successFraction, extrapolated, request),
    extrapolated, paretoEfficient: false,
  });
};

const evaluateCandidate = (
  club: ClubCapability,
  nominal: Readonly<Record<string, number>>,
  profile: PlayerCapabilityProfile,
  request: OptimizationRequest,
  evaluator: CapabilityEvaluator,
): CandidateSummary => {
  const landings: Landing[] = [];
  let counts: Counts = { completed: 0, noImpact: 0, failed: 0 };
  for (let sampleIndex = 0; sampleIndex < request.ensembleSize; sampleIndex += 1) {
    let evaluation: ReturnType<CapabilityEvaluator>;
    try { evaluation = evaluator(club.clubId, perturbedParameters(club, nominal, sampleIndex, request.seed)); }
    catch { counts = { ...counts, failed: counts.failed + 1 }; continue; }
    const landing = parseLanding(evaluation);
    if (!landing) {
      counts = evaluation.status === "no_impact"
        ? { ...counts, noImpact: counts.noImpact + 1 }
        : { ...counts, failed: counts.failed + 1 };
      continue;
    }
    counts = { ...counts, completed: counts.completed + 1 };
    landings.push(landing);
  }
  return { alternative: summarize(club, nominal, landings, counts, profile, request), counts };
};

const paretoMark = (
  alternatives: readonly OptimizationAlternative[], request: OptimizationRequest,
): readonly OptimizationAlternative[] => {
  if (request.objective !== "distance_control_pareto") return alternatives;
  return alternatives.map((candidate) => {
    const distanceError = Math.abs(candidate.meanCarryM - request.target.distanceM);
    const dominated = alternatives.some((other) => other !== candidate
      && Math.abs(other.meanCarryM - request.target.distanceM) <= distanceError
      && other.dispersionRmsM <= candidate.dispersionRmsM
      && (Math.abs(other.meanCarryM - request.target.distanceM) < distanceError
        || other.dispersionRmsM < candidate.dispersionRmsM));
    return Object.freeze({ ...candidate, paretoEfficient: !dominated });
  });
};

/** Rank robust shot alternatives while delegating all flight physics to `evaluator`. */
export function optimizeCapability(
  profile: PlayerCapabilityProfile,
  request: OptimizationRequest,
  evaluator: CapabilityEvaluator,
): OptimizationResult {
  const clubs = request.clubIds.map((clubId) => clubById(profile, clubId));
  const indices = new Map(clubs.map((club) => [club.clubId, 0]));
  const alternatives: OptimizationAlternative[] = [];
  let aggregate: Counts = { completed: 0, noImpact: 0, failed: 0 };
  for (let evaluationIndex = 0; evaluationIndex < request.candidateBudget; evaluationIndex += 1) {
    const club = clubs[evaluationIndex % clubs.length];
    const candidateIndex = indices.get(club.clubId) as number;
    indices.set(club.clubId, candidateIndex + 1);
    const summary = evaluateCandidate(club, candidateParameters(club, candidateIndex, request.seed), profile, request, evaluator);
    aggregate = { completed: aggregate.completed + summary.counts.completed, noImpact: aggregate.noImpact + summary.counts.noImpact, failed: aggregate.failed + summary.counts.failed };
    if (summary.alternative) alternatives.push(summary.alternative);
  }
  const marked = [...paretoMark(alternatives, request)];
  marked.sort((left, right) => {
    if (request.objective === "distance_control_pareto" && left.paretoEfficient !== right.paretoEfficient) return left.paretoEfficient ? -1 : 1;
    return left.score - right.score || left.clubId.localeCompare(right.clubId)
      || JSON.stringify(left.parameters).localeCompare(JSON.stringify(right.parameters));
  });
  const ranked = Object.freeze(marked.slice(0, request.alternativesCount)
    .map((item, index) => Object.freeze({ ...item, rank: index + 1 })));
  return Object.freeze({
    schemaVersion: "capability-optimization-result/v1", problemId: request.problemId,
    status: ranked.length > 0 ? "solved" : "nonconverged", alternatives: ranked,
    evaluationsAttempted: request.candidateBudget * request.ensembleSize,
    evaluationsCompleted: aggregate.completed, noImpactCount: aggregate.noImpact,
    failedCount: aggregate.failed, provenance: PROVENANCE,
  });
}
