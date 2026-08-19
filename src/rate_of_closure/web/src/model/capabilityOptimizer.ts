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
import { CapabilityOptimizationCancelled } from "./capabilityObservationContract";
import type {
  CapabilityEffectiveStatus,
  CapabilityObservedParameter,
  CapabilityOptimizationOptions,
  CapabilitySampleObservation,
} from "./capabilityObservationContract";
import { FLIGHT_METRIC_IDS } from "./ballFlightMetricContract";
import type { EvaluatedMetric, EvaluationStatus, SolverEvaluation } from "./inverseFlightContract";
import { contains, type TargetRegionTs } from "./targets";

export { CapabilityOptimizationCancelled, capabilitySampleObservationWire } from "./capabilityObservationContract";
export type {
  CapabilityEffectiveStatus, CapabilityObservedParameter, CapabilityOptimizationOptions,
  CapabilitySampleObservation, CapabilitySampleObservationWire,
} from "./capabilityObservationContract";

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
interface CandidateContext { readonly candidateOrdinal: number; readonly clubCandidateOrdinal: number }
interface CandidateEvaluationInput {
  readonly club: ClubCapability; readonly nominal: Readonly<Record<string, number>>;
  readonly profile: PlayerCapabilityProfile; readonly request: OptimizationRequest;
  readonly evaluator: CapabilityEvaluator; readonly context: CandidateContext;
  readonly options: CapabilityOptimizationOptions;
}
interface NormalizedEvaluation {
  readonly sourceStatus: EvaluationStatus | null; readonly effectiveStatus: CapabilityEffectiveStatus;
  readonly reasonCode: string | null; readonly sourceReason: string | null;
  readonly metrics: readonly EvaluatedMetric[]; readonly landing: Landing | null;
}
interface ObservationIdentity {
  readonly attemptOrdinal: number; readonly attemptedCount: number; readonly totalCount: number;
  readonly candidateOrdinal: number; readonly clubCandidateOrdinal: number; readonly sampleOrdinal: number;
}
interface SampleObservationInput {
  readonly club: ClubCapability; readonly nominal: Readonly<Record<string, number>>;
  readonly perturbed: Readonly<Record<string, number>>; readonly evaluation: NormalizedEvaluation;
  readonly request: OptimizationRequest; readonly context: CandidateContext; readonly sampleOrdinal: number;
}
interface RiskMetrics {
  readonly meanCarryM: number; readonly expectedMissM: number;
  readonly holdProbability: number; readonly dispersionRmsM: number;
  readonly cvarMissM: number; readonly downsideCarryM: number;
}
interface ConstraintInput {
  readonly club: ClubCapability; readonly nominal: Readonly<Record<string, number>>;
  readonly successFraction: number; readonly extrapolated: boolean; readonly request: OptimizationRequest;
}
interface SummaryInput {
  readonly club: ClubCapability; readonly nominal: Readonly<Record<string, number>>;
  readonly landings: readonly Landing[]; readonly counts: Counts;
  readonly profile: PlayerCapabilityProfile; readonly request: OptimizationRequest;
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

const parseLanding = (evaluation: SolverEvaluation): Landing | null => {
  if (evaluation.status !== "complete") return null;
  const carry = evaluation.metrics.find((item) => item.metricId === "carry_distance")?.value;
  const offline = evaluation.metrics.find((item) => item.metricId === "carry_offline")?.value;
  if (!Number.isFinite(carry) || !Number.isFinite(offline)) return null;
  return Object.freeze({ carryM: carry as number, offlineM: offline as number });
};

const validEvaluationSemantics = (
  status: EvaluationStatus, metrics: readonly EvaluatedMetric[], reason: string | null,
): boolean => {
  if (new Set(metrics.map(({ metricId }) => metricId)).size !== metrics.length) return false;
  if (status === "complete") return metrics.length > 0 && reason === null;
  return metrics.length === 0 && reason !== null && reason.trim() !== "";
};

const asEvaluation = (value: unknown): SolverEvaluation | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const source = value as Record<string, unknown>;
  const statuses: readonly EvaluationStatus[] = ["complete", "no_impact", "failed", "nonconverged"];
  if (!statuses.includes(source.status as EvaluationStatus) || !Array.isArray(source.metrics)) return null;
  if (source.reason !== null && typeof source.reason !== "string") return null;
  const metrics: EvaluatedMetric[] = [];
  for (const valueMetric of source.metrics) {
    if (!valueMetric || typeof valueMetric !== "object" || Array.isArray(valueMetric)) return null;
    const metric = valueMetric as Record<string, unknown>;
    if (typeof metric.metricId !== "string"
      || !FLIGHT_METRIC_IDS.includes(metric.metricId as EvaluatedMetric["metricId"])
      || typeof metric.value !== "number" || !Number.isFinite(metric.value)
      || typeof metric.provenance !== "string" || metric.provenance.trim() === "") return null;
    metrics.push(Object.freeze({
      metricId: metric.metricId as EvaluatedMetric["metricId"],
      value: metric.value,
      provenance: metric.provenance,
    }));
  }
  const status = source.status as EvaluationStatus;
  const reason = source.reason as string | null;
  if (!validEvaluationSemantics(status, metrics, reason)) return null;
  return Object.freeze({
    status,
    metrics: Object.freeze(metrics),
    reason,
  });
};

const failedEvaluation = (reasonCode: string): NormalizedEvaluation => Object.freeze({
  sourceStatus: null, effectiveStatus: "failed", reasonCode, sourceReason: null,
  metrics: Object.freeze([]), landing: null,
});

const normalizeEvaluation = (evaluator: () => unknown): NormalizedEvaluation => {
  let raw: unknown;
  try { raw = evaluator(); }
  catch { return failedEvaluation("evaluator_exception"); }
  const evaluation = asEvaluation(raw);
  if (!evaluation) return failedEvaluation("invalid_evaluator_result");
  const landing = parseLanding(evaluation);
  if (evaluation.status === "complete") return Object.freeze({
    sourceStatus: evaluation.status,
    effectiveStatus: landing ? "complete" : "failed",
    reasonCode: landing ? null : "missing_required_landing_metrics",
    sourceReason: null, metrics: evaluation.metrics, landing,
  });
  return Object.freeze({
    sourceStatus: evaluation.status,
    effectiveStatus: evaluation.status === "no_impact" ? "no_impact" : "failed",
    reasonCode: evaluation.reason, sourceReason: evaluation.reason,
    metrics: evaluation.metrics, landing: null,
  });
};

const observedParameters = (
  club: ClubCapability,
  nominal: Readonly<Record<string, number>>,
  perturbed: Readonly<Record<string, number>>,
): readonly CapabilityObservedParameter[] => Object.freeze(club.parameters.map((parameter) => Object.freeze({
  parameterId: parameter.parameterId, unit: parameter.unit,
  nominalValue: nominal[parameter.parameterId], perturbedValue: perturbed[parameter.parameterId],
})));

const observationIdentity = (
  request: OptimizationRequest, context: CandidateContext, sampleOrdinal: number,
): ObservationIdentity => {
  const { candidateOrdinal, clubCandidateOrdinal } = context;
  const totalCount = request.candidateBudget * request.ensembleSize;
  const attemptOrdinal = candidateOrdinal * request.ensembleSize + sampleOrdinal;
  const ordinals = [candidateOrdinal, clubCandidateOrdinal, sampleOrdinal, attemptOrdinal, totalCount];
  if (ordinals.some((value) => !Number.isInteger(value) || value < 0)) {
    throw new RangeError("observation identity values must be nonnegative integers");
  }
  if (candidateOrdinal >= request.candidateBudget || sampleOrdinal >= request.ensembleSize
    || attemptOrdinal >= totalCount) throw new RangeError("observation identity exceeds optimization bounds");
  return Object.freeze({
    attemptOrdinal, attemptedCount: attemptOrdinal + 1, totalCount,
    candidateOrdinal, clubCandidateOrdinal, sampleOrdinal,
  });
};

const sampleObservation = (input: SampleObservationInput): CapabilitySampleObservation => {
  const { club, nominal, perturbed, evaluation, request, context, sampleOrdinal } = input;
  const identity = observationIdentity(request, context, sampleOrdinal);
  return Object.freeze({
    schemaVersion: "capability-sample-observation/v1", problemId: request.problemId,
    ...identity, clubId: club.clubId, parameters: observedParameters(club, nominal, perturbed),
    sourceStatus: evaluation.sourceStatus, effectiveStatus: evaluation.effectiveStatus,
    reasonCode: evaluation.reasonCode, sourceReason: evaluation.sourceReason, metrics: evaluation.metrics,
  });
};

const tailMean = (values: readonly number[], alpha: number, reverse: boolean): number => {
  const count = Math.max(1, Math.ceil(values.length * (1 - alpha)));
  const ordered = [...values].sort((left, right) => reverse ? right - left : left - right);
  return ordered.slice(0, count).reduce((sum, value) => sum + value, 0) / count;
};

const limitingConstraints = (input: ConstraintInput): readonly string[] => {
  const { club, nominal, successFraction, extrapolated, request } = input;
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

const summarize = (input: SummaryInput): OptimizationAlternative | null => {
  const { club, nominal, landings, counts, profile, request } = input;
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
    limitingConstraints: limitingConstraints(Object.freeze({
      club, nominal, successFraction, extrapolated, request,
    })),
    extrapolated, paretoEfficient: false,
  });
};

const evaluateCandidate = (input: CandidateEvaluationInput): CandidateSummary => {
  const { club, nominal, profile, request, evaluator, context, options } = input;
  const landings: Landing[] = [];
  let counts: Counts = { completed: 0, noImpact: 0, failed: 0 };
  for (let sampleIndex = 0; sampleIndex < request.ensembleSize; sampleIndex += 1) {
    const attemptOrdinal = context.candidateOrdinal * request.ensembleSize + sampleIndex;
    const totalCount = request.candidateBudget * request.ensembleSize;
    if (options.shouldCancel?.()) throw new CapabilityOptimizationCancelled(attemptOrdinal, totalCount);
    const perturbed = perturbedParameters(club, nominal, sampleIndex, request.seed);
    const evaluation = normalizeEvaluation(() => evaluator(club.clubId, perturbed));
    options.observationSink?.(sampleObservation(Object.freeze({
      club, nominal, perturbed, evaluation, request, context, sampleOrdinal: sampleIndex,
    })));
    if (evaluation.effectiveStatus === "complete") {
      counts = { ...counts, completed: counts.completed + 1 };
      landings.push(evaluation.landing as Landing);
    } else if (evaluation.effectiveStatus === "no_impact") {
      counts = { ...counts, noImpact: counts.noImpact + 1 };
    } else counts = { ...counts, failed: counts.failed + 1 };
  }
  return {
    alternative: summarize(Object.freeze({ club, nominal, landings, counts, profile, request })),
    counts,
  };
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
  options: CapabilityOptimizationOptions = {},
): OptimizationResult {
  if (options.observationSink !== undefined && typeof options.observationSink !== "function") {
    throw new TypeError("observationSink must be a function");
  }
  if (options.shouldCancel !== undefined && typeof options.shouldCancel !== "function") {
    throw new TypeError("shouldCancel must be a function");
  }
  const hooks = Object.freeze({
    observationSink: options.observationSink, shouldCancel: options.shouldCancel,
  });
  const clubs = request.clubIds.map((clubId) => clubById(profile, clubId));
  const indices = new Map(clubs.map((club) => [club.clubId, 0]));
  const alternatives: OptimizationAlternative[] = [];
  let aggregate: Counts = { completed: 0, noImpact: 0, failed: 0 };
  for (let evaluationIndex = 0; evaluationIndex < request.candidateBudget; evaluationIndex += 1) {
    const club = clubs[evaluationIndex % clubs.length];
    const candidateIndex = indices.get(club.clubId) as number;
    indices.set(club.clubId, candidateIndex + 1);
    const summary = evaluateCandidate(Object.freeze({
      club, nominal: candidateParameters(club, candidateIndex, request.seed), profile, request, evaluator,
      context: Object.freeze({ candidateOrdinal: evaluationIndex, clubCandidateOrdinal: candidateIndex }),
      options: hooks,
    }));
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
