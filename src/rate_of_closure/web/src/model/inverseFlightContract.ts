/** Strict cross-runtime contracts for deterministic desired-flight solving. */

import {
  FLIGHT_METRIC_IDS,
  flightMetricCatalog,
  type FlightMetricId,
} from "./ballFlightMetricContract";

export type ObjectiveMode = "target" | "maximize" | "minimize";
export type EvaluationStatus = "complete" | "no_impact" | "failed" | "nonconverged";
export type SolverStatus = "solved" | "infeasible" | "no_impact" | "nonconverged";

export interface DecisionVariable {
  readonly parameterId: string; readonly unit: string;
  readonly lowerBound: number; readonly upperBound: number; readonly initialValue: number;
}
export interface FlightObjective {
  readonly metricId: FlightMetricId; readonly unit: string; readonly mode: ObjectiveMode;
  readonly targetValue: number | null; readonly lowerBound: number | null;
  readonly upperBound: number | null; readonly tolerance: number; readonly weight: number;
}
export interface InverseFlightRequest {
  readonly schemaVersion: "inverse-flight-request/v1"; readonly problemId: string;
  readonly variables: readonly DecisionVariable[]; readonly objectives: readonly FlightObjective[];
  readonly maxEvaluations: number; readonly candidateCount: number;
}
export interface EvaluatedMetric {
  readonly metricId: FlightMetricId; readonly value: number; readonly provenance: string;
}
export interface SolverEvaluation {
  readonly status: EvaluationStatus; readonly metrics: readonly EvaluatedMetric[];
  readonly reason: string | null;
}
export interface ParameterValue {
  readonly parameterId: string; readonly unit: string; readonly value: number;
}
export interface ObjectiveResidual {
  readonly metricId: FlightMetricId; readonly unit: string; readonly mode: ObjectiveMode;
  readonly actualValue: number; readonly targetValue: number | null;
  readonly normalizedResidual: number; readonly constraintViolation: number;
  readonly provenance: string;
}
export interface SolutionCandidate {
  readonly rank: number; readonly evaluationIndex: number; readonly feasible: boolean;
  readonly score: number; readonly parameters: readonly ParameterValue[];
  readonly residuals: readonly ObjectiveResidual[];
}
export interface InverseFlightResult {
  readonly schemaVersion: "inverse-flight-result/v1"; readonly problemId: string;
  readonly status: SolverStatus; readonly terminationReason: string;
  readonly evaluationsAttempted: number; readonly evaluationsCompleted: number;
  readonly noImpactCount: number; readonly failedCount: number;
  readonly candidates: readonly SolutionCandidate[];
  readonly provenance: Readonly<Record<string, string>>;
}

const record = (value: unknown, name: string): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new RangeError(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
};
const exactKeys = (value: Record<string, unknown>, fields: readonly string[], name: string) => {
  if (Object.keys(value).sort().join("|") !== [...fields].sort().join("|")) {
    throw new RangeError(`${name} fields do not match v1 schema`);
  }
};
const text = (value: unknown, name: string): string => {
  if (typeof value !== "string" || value.trim() === "") throw new RangeError(`${name} must be nonempty`);
  return value.trim();
};
const finite = (value: unknown, name: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return value;
};
const integer = (value: unknown, name: string): number => {
  const parsed = finite(value, name);
  if (!Number.isInteger(parsed)) throw new RangeError(`${name} must be an integer`);
  return parsed;
};
const nullableFinite = (value: unknown, name: string): number | null =>
  value === null ? null : finite(value, name);
const oneOf = <T extends string>(value: unknown, allowed: readonly T[], name: string): T => {
  const candidate = text(value, name);
  if (!allowed.includes(candidate as T)) throw new RangeError(`invalid ${name}: ${candidate}`);
  return candidate as T;
};

const parseVariable = (value: unknown): DecisionVariable => {
  const item = record(value, "decision variable");
  exactKeys(item, ["initial_value", "lower_bound", "parameter_id", "unit", "upper_bound"], "decision variable");
  const variable = {
    parameterId: text(item.parameter_id, "parameter_id"), unit: text(item.unit, "variable unit"),
    lowerBound: finite(item.lower_bound, "variable lower_bound"),
    upperBound: finite(item.upper_bound, "variable upper_bound"),
    initialValue: finite(item.initial_value, "variable initial_value"),
  };
  if (variable.lowerBound > variable.upperBound) throw new RangeError("variable lower_bound must not exceed upper_bound");
  if (variable.initialValue < variable.lowerBound || variable.initialValue > variable.upperBound) {
    throw new RangeError("variable initial_value must lie within bounds");
  }
  return Object.freeze(variable);
};

const parseObjective = (value: unknown): FlightObjective => {
  const item = record(value, "flight objective");
  exactKeys(item, ["lower_bound", "metric_id", "mode", "target_value", "tolerance", "unit", "upper_bound", "weight"], "flight objective");
  const metricId = oneOf(item.metric_id, FLIGHT_METRIC_IDS, "metric_id");
  const definition = flightMetricCatalog().definition(metricId);
  if (!definition.solverObjective) throw new RangeError(`${metricId} is not solver-eligible`);
  const unit = text(item.unit, "objective unit");
  if (unit !== definition.unit) throw new RangeError(`${metricId} canonical unit is ${definition.unit}, not ${unit}`);
  const objective = {
    metricId, unit, mode: oneOf(item.mode, ["target", "maximize", "minimize"] as const, "objective mode"),
    targetValue: nullableFinite(item.target_value, "target_value"),
    lowerBound: nullableFinite(item.lower_bound, "lower_bound"),
    upperBound: nullableFinite(item.upper_bound, "upper_bound"),
    tolerance: finite(item.tolerance, "objective tolerance"),
    weight: finite(item.weight, "objective weight"),
  };
  if (objective.mode === "target" && objective.targetValue === null) throw new RangeError("target objectives require target_value");
  if (objective.mode !== "target" && objective.targetValue !== null) throw new RangeError("maximize/minimize objectives must not define target_value");
  if (objective.lowerBound !== null && objective.upperBound !== null && objective.lowerBound > objective.upperBound) {
    throw new RangeError("objective lower_bound must not exceed upper_bound");
  }
  if (objective.tolerance <= 0 || objective.weight <= 0) throw new RangeError("objective tolerance and weight must be positive");
  return Object.freeze(objective);
};

export function parseInverseFlightRequest(payload: unknown): InverseFlightRequest {
  const root = record(payload, "inverse request");
  exactKeys(root, ["candidate_count", "max_evaluations", "objectives", "problem_id", "schema_version", "variables"], "inverse request");
  if (root.schema_version !== "inverse-flight-request/v1") throw new RangeError("unsupported schema_version");
  if (!Array.isArray(root.variables) || !Array.isArray(root.objectives)) throw new RangeError("variables and objectives must be arrays");
  const variables = Object.freeze(root.variables.map(parseVariable));
  const objectives = Object.freeze(root.objectives.map(parseObjective));
  if (variables.length === 0 || objectives.length === 0) throw new RangeError("inverse solve requires variables and objectives");
  if (new Set(variables.map((item) => item.parameterId)).size !== variables.length) throw new RangeError("decision variable IDs must be unique");
  if (new Set(objectives.map((item) => item.metricId)).size !== objectives.length) throw new RangeError("objective metric IDs must be unique");
  const maxEvaluations = integer(root.max_evaluations, "max_evaluations");
  const candidateCount = integer(root.candidate_count, "candidate_count");
  if (maxEvaluations < 1) throw new RangeError("max_evaluations must be positive");
  if (candidateCount < 1 || candidateCount > maxEvaluations) throw new RangeError("candidate_count must be between one and max_evaluations");
  return Object.freeze({
    schemaVersion: "inverse-flight-request/v1", problemId: text(root.problem_id, "problem_id"),
    variables, objectives, maxEvaluations, candidateCount,
  });
}

const stable = (value: unknown): unknown => Array.isArray(value) ? value.map(stable)
  : value && typeof value === "object" ? Object.fromEntries(Object.entries(value)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, item]) => [key, stable(item)])) : value;
const wireNumber = (value: number): number => Number(value.toFixed(11));
const requestWire = (request: InverseFlightRequest) => ({
  candidate_count: request.candidateCount, max_evaluations: request.maxEvaluations,
  objectives: request.objectives.map((item) => ({
    lower_bound: item.lowerBound, metric_id: item.metricId, mode: item.mode,
    target_value: item.targetValue, tolerance: wireNumber(item.tolerance), unit: item.unit,
    upper_bound: item.upperBound, weight: wireNumber(item.weight),
  })),
  problem_id: request.problemId, schema_version: request.schemaVersion,
  variables: request.variables.map((item) => ({
    initial_value: wireNumber(item.initialValue), lower_bound: wireNumber(item.lowerBound),
    parameter_id: item.parameterId, unit: item.unit, upper_bound: wireNumber(item.upperBound),
  })),
});
export const stableInverseFlightRequestJson = (request: InverseFlightRequest): string =>
  JSON.stringify(stable(requestWire(request)));

const parameterWire = (item: ParameterValue) => ({
  parameter_id: item.parameterId, unit: item.unit, value: wireNumber(item.value),
});
const residualWire = (item: ObjectiveResidual) => ({
  actual_value: wireNumber(item.actualValue), constraint_violation: wireNumber(item.constraintViolation),
  metric_id: item.metricId, mode: item.mode, normalized_residual: wireNumber(item.normalizedResidual),
  provenance: item.provenance, target_value: item.targetValue === null ? null : wireNumber(item.targetValue),
  unit: item.unit,
});
const resultWire = (result: InverseFlightResult) => ({
  candidates: result.candidates.map((item) => ({
    evaluation_index: item.evaluationIndex, feasible: item.feasible,
    parameters: item.parameters.map(parameterWire), rank: item.rank,
    residuals: item.residuals.map(residualWire), score: wireNumber(item.score),
  })),
  evaluations_attempted: result.evaluationsAttempted,
  evaluations_completed: result.evaluationsCompleted,
  failed_count: result.failedCount, no_impact_count: result.noImpactCount,
  problem_id: result.problemId, provenance: result.provenance,
  schema_version: result.schemaVersion, status: result.status,
  termination_reason: result.terminationReason,
});
export const stableInverseFlightResultJson = (result: InverseFlightResult): string =>
  JSON.stringify(stable(resultWire(result)));

const parseParameter = (value: unknown): ParameterValue => {
  const item = record(value, "parameter value");
  exactKeys(item, ["parameter_id", "unit", "value"], "parameter value");
  return Object.freeze({ parameterId: text(item.parameter_id, "parameter_id"), unit: text(item.unit, "unit"), value: finite(item.value, "value") });
};
const parseResidual = (value: unknown): ObjectiveResidual => {
  const item = record(value, "objective residual");
  exactKeys(item, ["actual_value", "constraint_violation", "metric_id", "mode", "normalized_residual", "provenance", "target_value", "unit"], "objective residual");
  return Object.freeze({
    actualValue: finite(item.actual_value, "actual_value"),
    constraintViolation: finite(item.constraint_violation, "constraint_violation"),
    metricId: oneOf(item.metric_id, FLIGHT_METRIC_IDS, "metric_id"),
    mode: oneOf(item.mode, ["target", "maximize", "minimize"] as const, "objective mode"),
    normalizedResidual: finite(item.normalized_residual, "normalized_residual"),
    provenance: text(item.provenance, "provenance"), targetValue: nullableFinite(item.target_value, "target_value"),
    unit: text(item.unit, "unit"),
  });
};
const parseCandidate = (value: unknown): SolutionCandidate => {
  const item = record(value, "solution candidate");
  exactKeys(item, ["evaluation_index", "feasible", "parameters", "rank", "residuals", "score"], "solution candidate");
  if (!Array.isArray(item.parameters) || !Array.isArray(item.residuals) || typeof item.feasible !== "boolean") throw new RangeError("invalid solution candidate collections");
  return Object.freeze({
    evaluationIndex: integer(item.evaluation_index, "evaluation_index"), feasible: item.feasible,
    parameters: Object.freeze(item.parameters.map(parseParameter)), rank: integer(item.rank, "rank"),
    residuals: Object.freeze(item.residuals.map(parseResidual)), score: finite(item.score, "score"),
  });
};
export function parseInverseFlightResult(payload: unknown): InverseFlightResult {
  const root = record(payload, "inverse result");
  exactKeys(root, ["candidates", "evaluations_attempted", "evaluations_completed", "failed_count", "no_impact_count", "problem_id", "provenance", "schema_version", "status", "termination_reason"], "inverse result");
  if (root.schema_version !== "inverse-flight-result/v1" || !Array.isArray(root.candidates)) throw new RangeError("invalid inverse result schema");
  const provenance = record(root.provenance, "provenance");
  if (Object.values(provenance).some((item) => typeof item !== "string")) throw new RangeError("provenance values must be strings");
  return Object.freeze({
    schemaVersion: "inverse-flight-result/v1", problemId: text(root.problem_id, "problem_id"),
    status: oneOf(root.status, ["solved", "infeasible", "no_impact", "nonconverged"] as const, "solver status"),
    terminationReason: text(root.termination_reason, "termination_reason"),
    evaluationsAttempted: integer(root.evaluations_attempted, "evaluations_attempted"),
    evaluationsCompleted: integer(root.evaluations_completed, "evaluations_completed"),
    noImpactCount: integer(root.no_impact_count, "no_impact_count"),
    failedCount: integer(root.failed_count, "failed_count"),
    candidates: Object.freeze(root.candidates.map(parseCandidate)),
    provenance: Object.freeze(provenance as Record<string, string>),
  });
}
