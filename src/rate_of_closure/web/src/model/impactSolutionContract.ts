/** Strict cross-runtime contracts for desired-flight impact solution families. */

import {
  parseInverseFlightRequest,
  stableInverseFlightRequestJson,
  type InverseFlightRequest,
  type ObjectiveResidual,
  type ParameterValue,
} from "./inverseFlightContract";
import { FLIGHT_METRIC_IDS, type FlightMetricId } from "./ballFlightMetricContract";

export type ClubProfileId = "centered_driver" | "centered_iron";
export type ModelAvailability = "available" | "unavailable";
export type ForwardStatus = "complete" | "no_impact" | "model_unavailable" | "failed" | "nonconverged";
export type SolverStatus = "solved" | "infeasible" | "no_impact" | "nonconverged";

export interface ModelManifest {
  readonly impactModelId: string;
  readonly impactStatus: ModelAvailability;
  readonly flightModelId: string;
  readonly flightStatus: ModelAvailability;
  readonly provenance: Readonly<Record<string, string>>;
}
export interface ImpactSolutionRequest {
  readonly schemaVersion: "impact-solution-request/v1";
  readonly inverseRequest: InverseFlightRequest;
  readonly clubProfileId: ClubProfileId;
  readonly flightModelId: string;
  readonly familyCount: number;
  readonly familyRadius: number;
  readonly sensitivityFraction: number;
  readonly impactEventTimeS: number;
  readonly targetFrameId: "target_frame:x_downrange,y_up,z_right";
  readonly deliveryFrameId: "app_frame:x_target,y_up,z_right";
  readonly impactReferencePoint: "ball_center_at_first_contact";
  readonly conventionId: "app_native";
  readonly impactModelId: "rigid_body_centered";
}
export interface MetricValue {
  readonly metricId: FlightMetricId;
  readonly unit: string;
  readonly value: number;
  readonly referenceEvent: string;
  readonly provenance: string;
}
export interface ParameterInterval {
  readonly parameterId: string; readonly unit: string;
  readonly lowerBound: number; readonly upperBound: number;
}
export interface ParameterCorrelation {
  readonly leftParameterId: string; readonly rightParameterId: string;
  readonly coefficient: number; readonly sampleCount: number;
}
export interface MetricSensitivity {
  readonly parameterId: string; readonly parameterUnit: string;
  readonly metricId: FlightMetricId; readonly metricUnit: string;
  readonly derivative: number; readonly method: string;
}
export interface FamilyMember {
  readonly evaluationIndex: number; readonly feasible: boolean; readonly score: number;
  readonly parameters: readonly ParameterValue[]; readonly launchValues: readonly MetricValue[];
  readonly launchResiduals: readonly ObjectiveResidual[]; readonly flightResiduals: readonly ObjectiveResidual[];
}
export interface SolutionFamily {
  readonly familyId: string; readonly rank: number; readonly representativeEvaluationIndex: number;
  readonly members: readonly FamilyMember[]; readonly intervals: readonly ParameterInterval[];
  readonly correlations: readonly ParameterCorrelation[]; readonly sensitivities: readonly MetricSensitivity[];
  readonly launchResiduals: readonly ObjectiveResidual[]; readonly flightResiduals: readonly ObjectiveResidual[];
}
export interface RejectedCandidate {
  readonly evaluationIndex: number; readonly status: ForwardStatus; readonly reason: string;
  readonly parameters: readonly ParameterValue[];
}
export interface ImpactSolutionResult {
  readonly schemaVersion: "impact-solution-result/v1"; readonly problemId: string;
  readonly status: SolverStatus; readonly terminationReason: string; readonly evaluationsAttempted: number;
  readonly families: readonly SolutionFamily[]; readonly rejectedCandidates: readonly RejectedCandidate[];
  readonly modelManifest: ModelManifest; readonly provenance: Readonly<Record<string, string>>;
}

type JsonRecord = Record<string, unknown>;
const record = (value: unknown, name: string): JsonRecord => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new TypeError(`${name} must be an object`);
  return value as JsonRecord;
};
const exact = (value: JsonRecord, keys: readonly string[], name: string): void => {
  const actual = Object.keys(value).sort(); const expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) throw new RangeError(`${name} fields do not match v1 schema`);
};
const text = (value: unknown, name: string): string => {
  if (typeof value !== "string" || value.trim() === "") throw new TypeError(`${name} must be nonempty`);
  return value.trim();
};
const finite = (value: unknown, name: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new TypeError(`${name} must be finite`);
  return value;
};
const integer = (value: unknown, name: string): number => {
  const parsed = finite(value, name); if (!Number.isInteger(parsed)) throw new TypeError(`${name} must be an integer`); return parsed;
};
const oneOf = <T extends string>(value: unknown, values: readonly T[], name: string): T => {
  const parsed = text(value, name); if (!values.includes(parsed as T)) throw new RangeError(`invalid ${name}: ${parsed}`); return parsed as T;
};
const array = (value: unknown, name: string): readonly unknown[] => {
  if (!Array.isArray(value)) throw new TypeError(`${name} must be an array`); return value;
};
const strings = (value: unknown, name: string): Readonly<Record<string, string>> => {
  const parsed = record(value, name);
  if (Object.values(parsed).some((item) => typeof item !== "string" || item.trim() === "")) throw new TypeError(`${name} values must be nonempty strings`);
  return Object.freeze(parsed as Record<string, string>);
};
const metricId = (value: unknown): FlightMetricId => oneOf(value, FLIGHT_METRIC_IDS, "metric_id");

const parseManifest = (value: unknown): ModelManifest => {
  const item = record(value, "model manifest");
  exact(item, ["flight_model_id", "flight_status", "impact_model_id", "impact_status", "provenance"], "model manifest");
  return Object.freeze({
    impactModelId: text(item.impact_model_id, "impact_model_id"),
    impactStatus: oneOf(item.impact_status, ["available", "unavailable"] as const, "impact_status"),
    flightModelId: text(item.flight_model_id, "flight_model_id"),
    flightStatus: oneOf(item.flight_status, ["available", "unavailable"] as const, "flight_status"),
    provenance: strings(item.provenance, "model provenance"),
  });
};

const DELIVERY_UNITS: Readonly<Record<string, string>> = Object.freeze({
  attack_angle_deg: "deg", club_path_deg: "deg", clubhead_speed_mps: "m/s",
  dynamic_loft_deg: "deg", face_angle_deg: "deg",
});
const DELIVERY_LIMITS: Readonly<Record<string, readonly [number, number]>> = Object.freeze({
  attack_angle_deg: [-89, 89], club_path_deg: [-89, 89], clubhead_speed_mps: [1e-9, 100],
  dynamic_loft_deg: [-89, 89], face_angle_deg: [-89, 89],
});

export function parseImpactSolutionRequest(payload: unknown): ImpactSolutionRequest {
  const root = record(payload, "impact solution request");
  exact(root, ["club_profile_id", "convention_id", "delivery_frame_id", "family_count", "family_radius", "flight_model_id", "impact_event_time_s", "impact_model_id", "impact_reference_point", "inverse_request", "schema_version", "sensitivity_fraction", "target_frame_id"], "impact solution request");
  if (root.schema_version !== "impact-solution-request/v1") throw new RangeError("unsupported schema_version");
  if (root.target_frame_id !== "target_frame:x_downrange,y_up,z_right") throw new RangeError("invalid target_frame_id");
  if (root.delivery_frame_id !== "app_frame:x_target,y_up,z_right") throw new RangeError("invalid delivery_frame_id");
  if (root.impact_reference_point !== "ball_center_at_first_contact") throw new RangeError("invalid impact_reference_point");
  if (root.convention_id !== "app_native" || root.impact_model_id !== "rigid_body_centered") throw new RangeError("invalid convention or impact model");
  const inverseRequest = parseInverseFlightRequest(root.inverse_request);
  inverseRequest.variables.forEach((variable) => {
    const unit = DELIVERY_UNITS[variable.parameterId];
    if (!unit) throw new RangeError(`unsupported delivery variable: ${variable.parameterId}`);
    if (variable.unit !== unit) throw new RangeError(`${variable.parameterId} canonical unit is ${unit}, not ${variable.unit}`);
    const [lower, upper] = DELIVERY_LIMITS[variable.parameterId];
    if (variable.lowerBound < lower || variable.upperBound > upper) throw new RangeError(`${variable.parameterId} bounds exceed supported range`);
  });
  const familyCount = integer(root.family_count, "family_count");
  const familyRadius = finite(root.family_radius, "family_radius");
  const sensitivityFraction = finite(root.sensitivity_fraction, "sensitivity_fraction");
  const impactEventTimeS = finite(root.impact_event_time_s, "impact_event_time_s");
  if (familyCount < 1 || familyCount > inverseRequest.candidateCount) throw new RangeError("family_count out of range");
  if (!(familyRadius > 0 && familyRadius <= 1)) throw new RangeError("family_radius out of range");
  if (!(sensitivityFraction > 0 && sensitivityFraction <= 0.25)) throw new RangeError("sensitivity_fraction out of range");
  if (impactEventTimeS < 0) throw new RangeError("impact_event_time_s must be nonnegative");
  return Object.freeze({
    schemaVersion: "impact-solution-request/v1", inverseRequest,
    clubProfileId: oneOf(root.club_profile_id, ["centered_driver", "centered_iron"] as const, "club_profile_id"),
    flightModelId: text(root.flight_model_id, "flight_model_id"), familyCount, familyRadius,
    sensitivityFraction, impactEventTimeS, targetFrameId: root.target_frame_id,
    deliveryFrameId: root.delivery_frame_id, impactReferencePoint: root.impact_reference_point,
    conventionId: root.convention_id, impactModelId: root.impact_model_id,
  });
}

const parseParameter = (value: unknown): ParameterValue => {
  const item = record(value, "parameter value"); exact(item, ["parameter_id", "unit", "value"], "parameter value");
  return Object.freeze({ parameterId: text(item.parameter_id, "parameter_id"), unit: text(item.unit, "unit"), value: finite(item.value, "value") });
};
const parseResidual = (value: unknown): ObjectiveResidual => {
  const item = record(value, "objective residual");
  exact(item, ["actual_value", "constraint_violation", "metric_id", "mode", "normalized_residual", "provenance", "target_value", "unit"], "objective residual");
  const target = item.target_value === null ? null : finite(item.target_value, "target_value");
  return Object.freeze({ actualValue: finite(item.actual_value, "actual_value"), constraintViolation: finite(item.constraint_violation, "constraint_violation"), metricId: metricId(item.metric_id), mode: oneOf(item.mode, ["target", "maximize", "minimize"] as const, "mode"), normalizedResidual: finite(item.normalized_residual, "normalized_residual"), provenance: text(item.provenance, "provenance"), targetValue: target, unit: text(item.unit, "unit") });
};
const parseMetric = (value: unknown): MetricValue => {
  const item = record(value, "metric value"); exact(item, ["metric_id", "provenance", "reference_event", "unit", "value"], "metric value");
  return Object.freeze({ metricId: metricId(item.metric_id), provenance: text(item.provenance, "provenance"), referenceEvent: text(item.reference_event, "reference_event"), unit: text(item.unit, "unit"), value: finite(item.value, "value") });
};
const parseMember = (value: unknown): FamilyMember => {
  const item = record(value, "family member"); exact(item, ["evaluation_index", "feasible", "flight_residuals", "launch_residuals", "launch_values", "parameters", "score"], "family member");
  if (typeof item.feasible !== "boolean") throw new TypeError("feasible must be boolean");
  const evaluationIndex = integer(item.evaluation_index, "evaluation_index");
  if (evaluationIndex < 0) throw new RangeError("evaluation_index must be nonnegative");
  return Object.freeze({ evaluationIndex, feasible: item.feasible, score: finite(item.score, "score"), parameters: Object.freeze(array(item.parameters, "parameters").map(parseParameter)), launchValues: Object.freeze(array(item.launch_values, "launch_values").map(parseMetric)), launchResiduals: Object.freeze(array(item.launch_residuals, "launch_residuals").map(parseResidual)), flightResiduals: Object.freeze(array(item.flight_residuals, "flight_residuals").map(parseResidual)) });
};
const parseInterval = (value: unknown): ParameterInterval => {
  const item = record(value, "parameter interval"); exact(item, ["lower_bound", "parameter_id", "unit", "upper_bound"], "parameter interval");
  const lowerBound = finite(item.lower_bound, "lower_bound"); const upperBound = finite(item.upper_bound, "upper_bound"); if (lowerBound > upperBound) throw new RangeError("interval bounds are unordered");
  return Object.freeze({ lowerBound, parameterId: text(item.parameter_id, "parameter_id"), unit: text(item.unit, "unit"), upperBound });
};
const parseCorrelation = (value: unknown): ParameterCorrelation => {
  const item = record(value, "parameter correlation"); exact(item, ["coefficient", "left_parameter_id", "right_parameter_id", "sample_count"], "parameter correlation");
  const coefficient = finite(item.coefficient, "coefficient"); if (Math.abs(coefficient) > 1) throw new RangeError("correlation coefficient out of range");
  return Object.freeze({ coefficient, leftParameterId: text(item.left_parameter_id, "left_parameter_id"), rightParameterId: text(item.right_parameter_id, "right_parameter_id"), sampleCount: integer(item.sample_count, "sample_count") });
};
const parseSensitivity = (value: unknown): MetricSensitivity => {
  const item = record(value, "metric sensitivity"); exact(item, ["derivative", "method", "metric_id", "metric_unit", "parameter_id", "parameter_unit"], "metric sensitivity");
  return Object.freeze({ derivative: finite(item.derivative, "derivative"), method: text(item.method, "method"), metricId: metricId(item.metric_id), metricUnit: text(item.metric_unit, "metric_unit"), parameterId: text(item.parameter_id, "parameter_id"), parameterUnit: text(item.parameter_unit, "parameter_unit") });
};
const parseFamily = (value: unknown): SolutionFamily => {
  const item = record(value, "solution family"); exact(item, ["correlations", "family_id", "flight_residuals", "intervals", "launch_residuals", "members", "rank", "representative_evaluation_index", "sensitivities"], "solution family");
  const rank = integer(item.rank, "rank");
  const representativeEvaluationIndex = integer(item.representative_evaluation_index, "representative_evaluation_index");
  if (rank < 1 || representativeEvaluationIndex < 0) throw new RangeError("rank and representative_evaluation_index must be nonnegative");
  return Object.freeze({ familyId: text(item.family_id, "family_id"), rank, representativeEvaluationIndex, members: Object.freeze(array(item.members, "members").map(parseMember)), intervals: Object.freeze(array(item.intervals, "intervals").map(parseInterval)), correlations: Object.freeze(array(item.correlations, "correlations").map(parseCorrelation)), sensitivities: Object.freeze(array(item.sensitivities, "sensitivities").map(parseSensitivity)), launchResiduals: Object.freeze(array(item.launch_residuals, "launch_residuals").map(parseResidual)), flightResiduals: Object.freeze(array(item.flight_residuals, "flight_residuals").map(parseResidual)) });
};
const parseRejected = (value: unknown): RejectedCandidate => {
  const item = record(value, "rejected candidate"); exact(item, ["evaluation_index", "parameters", "reason", "status"], "rejected candidate");
  const evaluationIndex = integer(item.evaluation_index, "evaluation_index");
  if (evaluationIndex < 0) throw new RangeError("evaluation_index must be nonnegative");
  return Object.freeze({ evaluationIndex, parameters: Object.freeze(array(item.parameters, "parameters").map(parseParameter)), reason: text(item.reason, "reason"), status: oneOf(item.status, ["complete", "no_impact", "model_unavailable", "failed", "nonconverged"] as const, "status") });
};

export function parseImpactSolutionResult(payload: unknown): ImpactSolutionResult {
  const root = record(payload, "impact solution result"); exact(root, ["evaluations_attempted", "families", "model_manifest", "problem_id", "provenance", "rejected_candidates", "schema_version", "status", "termination_reason"], "impact solution result");
  if (root.schema_version !== "impact-solution-result/v1") throw new RangeError("unsupported result schema_version");
  const families = Object.freeze(array(root.families, "families").map(parseFamily));
  if (families.some((family, index) => family.rank !== index + 1)) throw new RangeError("family ranks must be contiguous");
  const rejectedCandidates = Object.freeze(array(root.rejected_candidates, "rejected_candidates").map(parseRejected));
  const evaluationsAttempted = integer(root.evaluations_attempted, "evaluations_attempted");
  if (families.reduce((sum, family) => sum + family.members.length, 0) + rejectedCandidates.length !== evaluationsAttempted) throw new RangeError("evaluation counts do not reconcile");
  const indices = [...families.flatMap((family) => family.members.map((member) => member.evaluationIndex)), ...rejectedCandidates.map((candidate) => candidate.evaluationIndex)].sort((left, right) => left - right);
  if (indices.some((value, index) => value !== index)) throw new RangeError("each attempted evaluation must appear exactly once");
  return Object.freeze({ schemaVersion: "impact-solution-result/v1", problemId: text(root.problem_id, "problem_id"), status: oneOf(root.status, ["solved", "infeasible", "no_impact", "nonconverged"] as const, "solver status"), terminationReason: text(root.termination_reason, "termination_reason"), evaluationsAttempted, families, rejectedCandidates, modelManifest: parseManifest(root.model_manifest), provenance: strings(root.provenance, "result provenance") });
}

const stable = (value: unknown): unknown => Array.isArray(value) ? value.map(stable) : value && typeof value === "object" ? Object.fromEntries(Object.entries(value).sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => [key, stable(item)])) : value;
const parameterWire = (item: ParameterValue) => ({ parameter_id: item.parameterId, unit: item.unit, value: item.value });
const residualWire = (item: ObjectiveResidual) => ({ actual_value: item.actualValue, constraint_violation: item.constraintViolation, metric_id: item.metricId, mode: item.mode, normalized_residual: item.normalizedResidual, provenance: item.provenance, target_value: item.targetValue, unit: item.unit });
const metricWire = (item: MetricValue) => ({ metric_id: item.metricId, provenance: item.provenance, reference_event: item.referenceEvent, unit: item.unit, value: item.value });
const manifestWire = (item: ModelManifest) => ({ flight_model_id: item.flightModelId, flight_status: item.flightStatus, impact_model_id: item.impactModelId, impact_status: item.impactStatus, provenance: item.provenance });
const requestWire = (item: ImpactSolutionRequest) => ({ club_profile_id: item.clubProfileId, convention_id: item.conventionId, delivery_frame_id: item.deliveryFrameId, family_count: item.familyCount, family_radius: item.familyRadius, flight_model_id: item.flightModelId, impact_event_time_s: item.impactEventTimeS, impact_model_id: item.impactModelId, impact_reference_point: item.impactReferencePoint, inverse_request: JSON.parse(stableInverseFlightRequestJson(item.inverseRequest)), schema_version: item.schemaVersion, sensitivity_fraction: item.sensitivityFraction, target_frame_id: item.targetFrameId });
const memberWire = (item: FamilyMember) => ({ evaluation_index: item.evaluationIndex, feasible: item.feasible, flight_residuals: item.flightResiduals.map(residualWire), launch_residuals: item.launchResiduals.map(residualWire), launch_values: item.launchValues.map(metricWire), parameters: item.parameters.map(parameterWire), score: item.score });
const familyWire = (item: SolutionFamily) => ({ correlations: item.correlations.map((entry) => ({ coefficient: entry.coefficient, left_parameter_id: entry.leftParameterId, right_parameter_id: entry.rightParameterId, sample_count: entry.sampleCount })), family_id: item.familyId, flight_residuals: item.flightResiduals.map(residualWire), intervals: item.intervals.map((entry) => ({ lower_bound: entry.lowerBound, parameter_id: entry.parameterId, unit: entry.unit, upper_bound: entry.upperBound })), launch_residuals: item.launchResiduals.map(residualWire), members: item.members.map(memberWire), rank: item.rank, representative_evaluation_index: item.representativeEvaluationIndex, sensitivities: item.sensitivities.map((entry) => ({ derivative: entry.derivative, method: entry.method, metric_id: entry.metricId, metric_unit: entry.metricUnit, parameter_id: entry.parameterId, parameter_unit: entry.parameterUnit })) });
const resultWire = (item: ImpactSolutionResult) => ({ evaluations_attempted: item.evaluationsAttempted, families: item.families.map(familyWire), model_manifest: manifestWire(item.modelManifest), problem_id: item.problemId, provenance: item.provenance, rejected_candidates: item.rejectedCandidates.map((entry) => ({ evaluation_index: entry.evaluationIndex, parameters: entry.parameters.map(parameterWire), reason: entry.reason, status: entry.status })), schema_version: item.schemaVersion, status: item.status, termination_reason: item.terminationReason });
export const stableImpactSolutionRequestJson = (request: ImpactSolutionRequest): string => JSON.stringify(stable(requestWire(request)));
export const stableImpactSolutionResultJson = (result: ImpactSolutionResult): string => JSON.stringify(stable(resultWire(result)));
