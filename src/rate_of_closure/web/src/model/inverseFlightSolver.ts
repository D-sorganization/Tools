/** Deterministic bounded inverse search over an injected forward flight model. */

import {
  type FlightObjective,
  type InverseFlightRequest,
  type InverseFlightResult,
  type ObjectiveResidual,
  type SolutionCandidate,
  type SolverEvaluation,
} from "./inverseFlightContract";
import { FLIGHT_METRIC_IDS } from "./ballFlightMetricContract";

export type ForwardEvaluator = (
  parameters: Readonly<Record<string, number>>,
) => SolverEvaluation;

const CONSTRAINT_PENALTY = 1_000_000;
const PROVENANCE = Object.freeze({
  metric_schema: "ball-flight-metrics/v1",
  sampler: "halton-sequence-with-initial-point",
  solver_id: "deterministic-bounded-search",
  solver_version: "1.0.0",
});

const isPrime = (candidate: number): boolean => {
  for (let divisor = 2; divisor * divisor <= candidate; divisor += 1) {
    if (candidate % divisor === 0) return false;
  }
  return candidate >= 2;
};
const firstPrimes = (count: number): readonly number[] => {
  const primes: number[] = [];
  for (let candidate = 2; primes.length < count; candidate += 1) {
    if (isPrime(candidate)) primes.push(candidate);
  }
  return primes;
};
const radicalInverse = (sourceIndex: number, base: number): number => {
  let result = 0;
  let factor = 1 / base;
  let index = sourceIndex;
  while (index > 0) {
    result += factor * (index % base);
    index = Math.floor(index / base);
    factor /= base;
  }
  return result;
};

const samplePoints = (request: InverseFlightRequest): readonly Readonly<Record<string, number>>[] => {
  const initial = Object.freeze(Object.fromEntries(
    request.variables.map((item) => [item.parameterId, item.initialValue]),
  ));
  if (request.variables.every((item) => item.lowerBound === item.upperBound)) return [initial];
  const points: Readonly<Record<string, number>>[] = [initial];
  const seen = new Set([request.variables.map((item) => initial[item.parameterId]).join("|")]);
  const bases = firstPrimes(request.variables.length);
  const sequenceLimit = request.maxEvaluations * 1024;
  for (let index = 1; points.length < request.maxEvaluations && index <= sequenceLimit; index += 1) {
    const point = Object.freeze(Object.fromEntries(request.variables.map((item, variableIndex) => [
      item.parameterId,
      item.lowerBound + radicalInverse(index, bases[variableIndex]) * (item.upperBound - item.lowerBound),
    ])));
    const key = request.variables.map((item) => point[item.parameterId]).join("|");
    if (!seen.has(key)) { points.push(point); seen.add(key); }
  }
  return Object.freeze(points);
};

const violation = (value: number, objective: FlightObjective): number => {
  if (objective.lowerBound !== null && value < objective.lowerBound) {
    return (objective.lowerBound - value) / objective.tolerance;
  }
  if (objective.upperBound !== null && value > objective.upperBound) {
    return (value - objective.upperBound) / objective.tolerance;
  }
  return 0;
};
const makeResidual = (
  value: number, objective: FlightObjective, provenance: string,
): ObjectiveResidual => {
  const normalizedResidual = objective.mode === "target"
    ? (value - (objective.targetValue as number)) / objective.tolerance
    : objective.mode === "maximize" ? -value / objective.tolerance : value / objective.tolerance;
  return Object.freeze({
    metricId: objective.metricId, unit: objective.unit, mode: objective.mode,
    actualValue: value, targetValue: objective.targetValue, normalizedResidual,
    constraintViolation: violation(value, objective), provenance,
  });
};
const residualFeasible = (residual: ObjectiveResidual, objective: FlightObjective): boolean =>
  (objective.mode !== "target" || Math.abs(residual.normalizedResidual) <= 1)
  && residual.constraintViolation === 0;
const residualLoss = (residual: ObjectiveResidual, objective: FlightObjective): number => {
  const directional = objective.mode === "target"
    ? Math.abs(residual.normalizedResidual) : residual.normalizedResidual;
  return objective.weight * (directional + CONSTRAINT_PENALTY * residual.constraintViolation);
};

const makeCandidate = (
  request: InverseFlightRequest,
  parameters: Readonly<Record<string, number>>,
  evaluation: SolverEvaluation,
  evaluationIndex: number,
): SolutionCandidate | null => {
  const metrics = new Map(evaluation.metrics.map((item) => [item.metricId, item]));
  if (request.objectives.some((objective) => !metrics.has(objective.metricId))) return null;
  const residuals = Object.freeze(request.objectives.map((objective) => {
    const metric = metrics.get(objective.metricId) as typeof evaluation.metrics[number];
    return makeResidual(metric.value, objective, metric.provenance);
  }));
  return Object.freeze({
    rank: 1, evaluationIndex,
    feasible: residuals.every((item, index) => residualFeasible(item, request.objectives[index])),
    score: residuals.reduce((total, item, index) => total + residualLoss(item, request.objectives[index]), 0),
    parameters: Object.freeze(request.variables.map((item) => Object.freeze({
      parameterId: item.parameterId, unit: item.unit, value: parameters[item.parameterId],
    }))),
    residuals,
  });
};

const staticInfeasibility = (request: InverseFlightRequest): string | null => {
  for (const objective of request.objectives) {
    if (objective.targetValue === null) continue;
    if (objective.lowerBound !== null && objective.targetValue < objective.lowerBound) return "target_outside_objective_bounds";
    if (objective.upperBound !== null && objective.targetValue > objective.upperBound) return "target_outside_objective_bounds";
  }
  return null;
};
const emptyResult = (
  request: InverseFlightRequest, status: "infeasible", terminationReason: string,
): InverseFlightResult => Object.freeze({
  schemaVersion: "inverse-flight-result/v1", problemId: request.problemId,
  status, terminationReason, evaluationsAttempted: 0, evaluationsCompleted: 0,
  noImpactCount: 0, failedCount: 0, candidates: Object.freeze([]), provenance: PROVENANCE,
});

const validEvaluation = (evaluation: SolverEvaluation): boolean => {
  if (!evaluation || !["complete", "no_impact", "failed", "nonconverged"].includes(evaluation.status)) return false;
  if (!Array.isArray(evaluation.metrics)) return false;
  if (evaluation.status === "complete") {
    if (evaluation.reason !== null || evaluation.metrics.length === 0) return false;
    const ids = new Set<string>();
    for (const metric of evaluation.metrics) {
      if (!metric || !FLIGHT_METRIC_IDS.includes(metric.metricId) || ids.has(metric.metricId)) return false;
      if (!Number.isFinite(metric.value)
          || typeof metric.provenance !== "string" || metric.provenance.trim() === "") return false;
      ids.add(metric.metricId);
    }
    return true;
  }
  return evaluation.metrics.length === 0
    && typeof evaluation.reason === "string" && evaluation.reason.trim() !== "";
};

/** Search bounded inputs and rank results under explicit objective contracts. */
export function solveInverseFlight(
  request: InverseFlightRequest, evaluator: ForwardEvaluator,
): InverseFlightResult {
  const staticReason = staticInfeasibility(request);
  if (staticReason) return emptyResult(request, "infeasible", staticReason);
  const points = samplePoints(request);
  const candidates: SolutionCandidate[] = [];
  let completed = 0;
  let noImpact = 0;
  let failed = 0;
  points.forEach((parameters, evaluationIndex) => {
    let evaluation: SolverEvaluation;
    try { evaluation = evaluator(parameters); } catch { failed += 1; return; }
    if (!validEvaluation(evaluation)) { failed += 1; return; }
    if (evaluation.status === "no_impact") { noImpact += 1; return; }
    if (evaluation.status !== "complete") { failed += 1; return; }
    const candidate = makeCandidate(request, parameters, evaluation, evaluationIndex);
    if (!candidate) { failed += 1; return; }
    completed += 1;
    candidates.push(candidate);
  });
  candidates.sort((left, right) => {
    if (left.feasible !== right.feasible) return left.feasible ? -1 : 1;
    const leftViolation = left.residuals.reduce((sum, item) => sum + item.constraintViolation, 0);
    const rightViolation = right.residuals.reduce((sum, item) => sum + item.constraintViolation, 0);
    return leftViolation - rightViolation || left.score - right.score || left.evaluationIndex - right.evaluationIndex;
  });
  const ranked = Object.freeze(candidates.slice(0, request.candidateCount)
    .map((item, index) => Object.freeze({ ...item, rank: index + 1 })));
  const attempted = points.length;
  const [status, terminationReason] = ranked[0]?.feasible
    ? ["solved", "objective_tolerances_met"] as const
    : noImpact === attempted ? ["no_impact", "all_evaluations_no_impact"] as const
      : completed > 0 ? ["nonconverged", "evaluation_budget_exhausted"] as const
        : ["nonconverged", "no_complete_evaluations"] as const;
  return Object.freeze({
    schemaVersion: "inverse-flight-result/v1", problemId: request.problemId,
    status, terminationReason, evaluationsAttempted: attempted,
    evaluationsCompleted: completed, noImpactCount: noImpact, failedCount: failed,
    candidates: ranked, provenance: PROVENANCE,
  });
}
