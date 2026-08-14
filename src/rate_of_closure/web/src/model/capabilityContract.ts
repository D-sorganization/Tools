/** Versioned player-capability and robust optimization contracts. */

import type { SolverEvaluation } from "./inverseFlightContract";
export type { SolverEvaluation } from "./inverseFlightContract";
import { covarianceFactor } from "./capabilityMath";

export type CapabilityObjective = "maximize_carry" | "minimize_expected_miss" | "maximize_target_hold" | "minimize_variability" | "minimize_downside" | "distance_control_pareto";
export interface CapabilityParameter {
  readonly parameterId: string; readonly unit: string;
  readonly lowerBound: number; readonly upperBound: number;
  readonly evidenceLowerBound: number; readonly evidenceUpperBound: number;
  readonly baseline: number; readonly bias: number; readonly standardDeviation: number;
}
export interface ClubCapability {
  readonly clubId: string; readonly parameters: readonly CapabilityParameter[];
  readonly matrixKind: "correlation" | "covariance";
  readonly matrix: readonly (readonly number[])[];
  readonly provenance: string; readonly confidence: number;
}
export interface PlayerCapabilityProfile {
  readonly schemaVersion: "player-capability-profile/v1"; readonly profileId: string;
  readonly clubs: readonly ClubCapability[]; readonly provenance: string; readonly confidence: number;
}
export interface TargetDefinition {
  readonly kind: "green" | "fairway"; readonly distanceM: number; readonly lateralM: number;
  readonly radiusM: number; readonly bandHalfLengthM: number; readonly halfWidthM: number;
}
export interface OptimizationRequest {
  readonly schemaVersion: "capability-optimization-request/v1"; readonly problemId: string;
  readonly objective: CapabilityObjective; readonly clubIds: readonly string[];
  readonly target: TargetDefinition; readonly candidateBudget: number; readonly ensembleSize: number;
  readonly alternativesCount: number; readonly seed: number; readonly cvarAlpha: number;
  readonly minimumSuccessFraction: number;
}
export interface OptimizationAlternative {
  readonly rank: number; readonly clubId: string;
  readonly parameters: readonly { readonly parameterId: string; readonly value: number }[];
  readonly score: number; readonly meanCarryM: number; readonly expectedMissM: number;
  readonly dispersionRmsM: number; readonly targetHoldProbability: number; readonly cvarMissM: number;
  readonly downsideCarryM: number; readonly sampleCount: number; readonly successfulCount: number;
  readonly noImpactCount: number; readonly failedCount: number; readonly failureFraction: number;
  readonly confidence: number; readonly limitingConstraints: readonly string[];
  readonly extrapolated: boolean; readonly paretoEfficient: boolean;
}
export interface OptimizationResult {
  readonly schemaVersion: "capability-optimization-result/v1"; readonly problemId: string;
  readonly status: "solved" | "nonconverged"; readonly alternatives: readonly OptimizationAlternative[];
  readonly evaluationsAttempted: number; readonly evaluationsCompleted: number;
  readonly noImpactCount: number; readonly failedCount: number;
  readonly provenance: Readonly<Record<string, string>>;
}
export type CapabilityEvaluator = (
  clubId: string, parameters: Readonly<Record<string, number>>,
) => SolverEvaluation;

export const MAX_CAPABILITY_WIRE_MAGNITUDE = 1e300;

const record = (value: unknown, name: string): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new RangeError(`${name} must be an object`);
  return value as Record<string, unknown>;
};
const exact = (value: Record<string, unknown>, fields: readonly string[], name: string) => {
  if (Object.keys(value).sort().join("|") !== [...fields].sort().join("|")) throw new RangeError(`${name} fields do not match v1 schema`);
};
const text = (value: unknown, name: string): string => {
  if (typeof value !== "string" || value.trim() === "") throw new RangeError(`${name} must be nonempty`);
  return value.trim();
};
const finite = (value: unknown, name: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  if (Math.abs(value) > MAX_CAPABILITY_WIRE_MAGNITUDE) throw new RangeError(`${name} magnitude must not exceed ${MAX_CAPABILITY_WIRE_MAGNITUDE}`);
  return value;
};
const integer = (value: unknown, name: string): number => {
  const parsed = finite(value, name);
  if (!Number.isInteger(parsed)) throw new RangeError(`${name} must be an integer`);
  return parsed;
};
const boolean = (value: unknown, name: string): boolean => {
  if (typeof value !== "boolean") throw new RangeError(`${name} must be a boolean`);
  return value;
};
const probability = (value: unknown, name: string, allowZero = true): number => {
  const parsed = finite(value, name);
  if (parsed > 1 || (allowZero ? parsed < 0 : parsed <= 0)) throw new RangeError(`${name} must lie within ${allowZero ? "[0, 1]" : "(0, 1]"}`);
  return parsed;
};

const parseParameter = (value: unknown): CapabilityParameter => {
  const item = record(value, "capability parameter");
  exact(item, ["baseline", "bias", "evidence_lower_bound", "evidence_upper_bound", "lower_bound", "parameter_id", "standard_deviation", "unit", "upper_bound"], "capability parameter");
  const result = {
    parameterId: text(item.parameter_id, "parameter_id"), unit: text(item.unit, "parameter unit"),
    lowerBound: finite(item.lower_bound, "lower_bound"), upperBound: finite(item.upper_bound, "upper_bound"),
    evidenceLowerBound: finite(item.evidence_lower_bound, "evidence_lower_bound"),
    evidenceUpperBound: finite(item.evidence_upper_bound, "evidence_upper_bound"),
    baseline: finite(item.baseline, "baseline"), bias: finite(item.bias, "bias"),
    standardDeviation: finite(item.standard_deviation, "standard_deviation"),
  };
  if (!(result.lowerBound <= result.evidenceLowerBound && result.evidenceLowerBound <= result.evidenceUpperBound && result.evidenceUpperBound <= result.upperBound)) throw new RangeError("evidence bounds must lie within safe parameter bounds");
  if (result.baseline < result.lowerBound || result.baseline > result.upperBound) throw new RangeError("baseline must lie within safe parameter bounds");
  if (result.standardDeviation < 0) throw new RangeError("standard_deviation must be nonnegative");
  return Object.freeze(result);
};

const parseClub = (value: unknown): ClubCapability => {
  const item = record(value, "club capability");
  exact(item, ["club_id", "confidence", "matrix", "matrix_kind", "parameters", "provenance"], "club capability");
  if (!Array.isArray(item.parameters) || !Array.isArray(item.matrix)) throw new RangeError("club parameters and matrix must be arrays");
  const parameters = Object.freeze(item.parameters.map(parseParameter));
  if (parameters.length === 0 || new Set(parameters.map((entry) => entry.parameterId)).size !== parameters.length) throw new RangeError("club parameter IDs must be nonempty and unique");
  const matrixKind = text(item.matrix_kind, "matrix_kind");
  if (matrixKind !== "correlation" && matrixKind !== "covariance") throw new RangeError("matrix_kind must be correlation or covariance");
  const matrix = Object.freeze(item.matrix.map((sourceRow) => {
    if (!Array.isArray(sourceRow) || sourceRow.length !== parameters.length) throw new RangeError("capability matrix shape must match parameters");
    return Object.freeze(sourceRow.map((entry) => finite(entry, "matrix entry")));
  }));
  if (matrix.length !== parameters.length) throw new RangeError("capability matrix shape must match parameters");
  matrix.forEach((row, rowIndex) => row.forEach((entry, columnIndex) => {
    if (Math.abs(entry - matrix[columnIndex][rowIndex]) > 1e-10) throw new RangeError("capability matrix must be symmetric");
  }));
  covarianceFactor(matrix);
  if (matrixKind === "correlation" && matrix.some((row, index) => Math.abs(row[index] - 1) > 1e-10)) throw new RangeError("correlation matrix must have a unit diagonal");
  return Object.freeze({ clubId: text(item.club_id, "club_id"), parameters, matrixKind, matrix, provenance: text(item.provenance, "club provenance"), confidence: probability(item.confidence, "club confidence") });
};

export function parsePlayerCapabilityProfile(value: unknown): PlayerCapabilityProfile {
  const item = record(value, "player capability profile");
  exact(item, ["clubs", "confidence", "profile_id", "provenance", "schema_version"], "player capability profile");
  if (item.schema_version !== "player-capability-profile/v1" || !Array.isArray(item.clubs)) throw new RangeError("unsupported profile schema or clubs");
  const clubs = Object.freeze(item.clubs.map(parseClub));
  if (clubs.length === 0 || new Set(clubs.map((club) => club.clubId)).size !== clubs.length) throw new RangeError("profile club IDs must be nonempty and unique");
  return Object.freeze({ schemaVersion: "player-capability-profile/v1", profileId: text(item.profile_id, "profile_id"), clubs, provenance: text(item.provenance, "profile provenance"), confidence: probability(item.confidence, "profile confidence") });
}

const parseTarget = (value: unknown): TargetDefinition => {
  const item = record(value, "target definition");
  exact(item, ["band_half_length_m", "distance_m", "half_width_m", "kind", "lateral_m", "radius_m"], "target definition");
  const kind = text(item.kind, "target kind");
  if (kind !== "green" && kind !== "fairway") throw new RangeError("target kind must be green or fairway");
  const result: TargetDefinition = { kind, distanceM: finite(item.distance_m, "distance_m"), lateralM: finite(item.lateral_m, "lateral_m"), radiusM: finite(item.radius_m, "radius_m"), bandHalfLengthM: finite(item.band_half_length_m, "band_half_length_m"), halfWidthM: finite(item.half_width_m, "half_width_m") };
  if (result.distanceM <= 0 || result.radiusM <= 0 || result.bandHalfLengthM <= 0 || result.halfWidthM <= 0) throw new RangeError("target sizes and distance must be positive");
  return Object.freeze(result);
};

export function parseOptimizationRequest(value: unknown): OptimizationRequest {
  const item = record(value, "optimization request");
  exact(item, ["alternatives_count", "candidate_budget", "club_ids", "cvar_alpha", "ensemble_size", "minimum_success_fraction", "objective", "problem_id", "schema_version", "seed", "target"], "optimization request");
  if (item.schema_version !== "capability-optimization-request/v1" || !Array.isArray(item.club_ids)) throw new RangeError("unsupported request schema or club_ids");
  const objective = text(item.objective, "objective") as CapabilityObjective;
  if (!["maximize_carry", "minimize_expected_miss", "maximize_target_hold", "minimize_variability", "minimize_downside", "distance_control_pareto"].includes(objective)) throw new RangeError("unsupported capability objective");
  const clubIds = Object.freeze(item.club_ids.map((club) => text(club, "request club_id")));
  if (clubIds.length === 0 || new Set(clubIds).size !== clubIds.length) throw new RangeError("request club IDs must be nonempty and unique");
  const candidateBudget = integer(item.candidate_budget, "candidate_budget");
  const ensembleSize = integer(item.ensemble_size, "ensemble_size");
  const alternativesCount = integer(item.alternatives_count, "alternatives_count");
  const seed = integer(item.seed, "seed");
  if (candidateBudget < 1 || ensembleSize < 1 || alternativesCount < 1 || alternativesCount > candidateBudget || seed < 0) throw new RangeError("request integer bounds are invalid");
  return Object.freeze({ schemaVersion: "capability-optimization-request/v1", problemId: text(item.problem_id, "problem_id"), objective, clubIds, target: parseTarget(item.target), candidateBudget, ensembleSize, alternativesCount, seed, cvarAlpha: probability(item.cvar_alpha, "cvar_alpha", false), minimumSuccessFraction: probability(item.minimum_success_fraction, "minimum_success_fraction", false) });
}

const parseAlternative = (value: unknown): OptimizationAlternative => {
  const item = record(value, "optimization alternative");
  exact(item, ["rank", "clubId", "parameters", "score", "meanCarryM", "expectedMissM", "dispersionRmsM", "targetHoldProbability", "cvarMissM", "downsideCarryM", "sampleCount", "successfulCount", "noImpactCount", "failedCount", "failureFraction", "confidence", "limitingConstraints", "extrapolated", "paretoEfficient"], "optimization alternative");
  const parameters = Array.isArray(item.parameters) ? item.parameters.map((source) => {
    const parameter = record(source, "alternative parameter");
    return Object.freeze({ parameterId: text(parameter.parameterId, "parameterId"), value: finite(parameter.value, "parameter value") });
  }) : [];
  const constraints = Array.isArray(item.limitingConstraints) ? item.limitingConstraints.map((entry) => text(entry, "limiting constraint")) : [];
  const parsed = { rank: integer(item.rank, "rank"), clubId: text(item.clubId, "clubId"), parameters: Object.freeze(parameters), score: finite(item.score, "score"), meanCarryM: finite(item.meanCarryM, "meanCarryM"), expectedMissM: finite(item.expectedMissM, "expectedMissM"), dispersionRmsM: finite(item.dispersionRmsM, "dispersionRmsM"), targetHoldProbability: probability(item.targetHoldProbability, "targetHoldProbability"), cvarMissM: finite(item.cvarMissM, "cvarMissM"), downsideCarryM: finite(item.downsideCarryM, "downsideCarryM"), sampleCount: integer(item.sampleCount, "sampleCount"), successfulCount: integer(item.successfulCount, "successfulCount"), noImpactCount: integer(item.noImpactCount, "noImpactCount"), failedCount: integer(item.failedCount, "failedCount"), failureFraction: probability(item.failureFraction, "failureFraction"), confidence: probability(item.confidence, "confidence"), limitingConstraints: Object.freeze(constraints), extrapolated: boolean(item.extrapolated, "extrapolated"), paretoEfficient: boolean(item.paretoEfficient, "paretoEfficient") };
  if (parsed.successfulCount + parsed.noImpactCount + parsed.failedCount !== parsed.sampleCount) throw new RangeError("alternative diagnostic counts must sum to sampleCount");
  return Object.freeze(parsed);
};

export function parseOptimizationResult(value: unknown): OptimizationResult {
  const item = record(value, "optimization result");
  exact(item, ["schemaVersion", "problemId", "status", "alternatives", "evaluationsAttempted", "evaluationsCompleted", "noImpactCount", "failedCount", "provenance"], "optimization result");
  if (item.schemaVersion !== "capability-optimization-result/v1" || !Array.isArray(item.alternatives)) throw new RangeError("unsupported result schema or alternatives");
  const status = text(item.status, "status");
  if (status !== "solved" && status !== "nonconverged") throw new RangeError("unsupported result status");
  const provenance = record(item.provenance, "result provenance");
  const alternatives = Object.freeze(item.alternatives.map(parseAlternative));
  if (alternatives.some((alternative, index) => alternative.rank !== index + 1)) throw new RangeError("alternative ranks must be contiguous");
  const evaluationsAttempted = integer(item.evaluationsAttempted, "evaluationsAttempted");
  const evaluationsCompleted = integer(item.evaluationsCompleted, "evaluationsCompleted");
  const noImpactCount = integer(item.noImpactCount, "noImpactCount");
  const failedCount = integer(item.failedCount, "failedCount");
  if (evaluationsCompleted + noImpactCount + failedCount !== evaluationsAttempted) throw new RangeError("result diagnostic counts must sum to evaluationsAttempted");
  return Object.freeze({ schemaVersion: "capability-optimization-result/v1", problemId: text(item.problemId, "problemId"), status, alternatives, evaluationsAttempted, evaluationsCompleted, noImpactCount, failedCount, provenance: Object.freeze(Object.fromEntries(Object.entries(provenance).map(([key, source]) => [key, text(source, key)]))) });
}
