/** Immutable per-sample capability optimizer observation contract. */

import { FLIGHT_METRIC_IDS, type FlightMetricId } from "./ballFlightMetricContract";
import type { EvaluatedMetric, EvaluationStatus } from "./inverseFlightContract";

export type CapabilityEffectiveStatus = "complete" | "no_impact" | "failed";

export interface CapabilityObservedParameter {
  readonly parameterId: string; readonly unit: string;
  readonly nominalValue: number; readonly perturbedValue: number;
}

export interface CapabilitySampleObservation {
  readonly schemaVersion: "capability-sample-observation/v1"; readonly problemId: string;
  readonly attemptOrdinal: number; readonly attemptedCount: number; readonly totalCount: number;
  readonly candidateOrdinal: number; readonly clubCandidateOrdinal: number; readonly sampleOrdinal: number;
  readonly clubId: string; readonly parameters: readonly CapabilityObservedParameter[];
  readonly sourceStatus: EvaluationStatus | null; readonly effectiveStatus: CapabilityEffectiveStatus;
  readonly reasonCode: string | null; readonly sourceReason: string | null;
  readonly metrics: readonly EvaluatedMetric[];
}

export interface CapabilityOptimizationOptions {
  readonly observationSink?: (observation: CapabilitySampleObservation) => void;
  readonly shouldCancel?: () => boolean;
}

export interface CapabilityObservedParameterWire {
  readonly parameter_id: string; readonly unit: string;
  readonly nominal_value: number; readonly perturbed_value: number;
}

export interface CapabilityObservedMetricWire {
  readonly metric_id: string; readonly value: number; readonly provenance: string;
}

export interface CapabilitySampleObservationWire {
  readonly schema_version: "capability-sample-observation/v1"; readonly problem_id: string;
  readonly attempt_ordinal: number; readonly attempted_count: number; readonly total_count: number;
  readonly candidate_ordinal: number; readonly club_candidate_ordinal: number; readonly sample_ordinal: number;
  readonly club_id: string; readonly parameters: readonly CapabilityObservedParameterWire[];
  readonly source_status: EvaluationStatus | null; readonly effective_status: CapabilityEffectiveStatus;
  readonly reason_code: string | null; readonly source_reason: string | null;
  readonly metrics: readonly CapabilityObservedMetricWire[];
}

const OBSERVATION_KEYS = [
  "schemaVersion", "problemId", "attemptOrdinal", "attemptedCount", "totalCount",
  "candidateOrdinal", "clubCandidateOrdinal", "sampleOrdinal", "clubId", "parameters",
  "sourceStatus", "effectiveStatus", "reasonCode", "sourceReason", "metrics",
] as const;
const PARAMETER_KEYS = ["parameterId", "unit", "nominalValue", "perturbedValue"] as const;
const METRIC_KEYS = ["metricId", "value", "provenance"] as const;
const SOURCE_STATUSES = new Set<EvaluationStatus>([
  "complete", "no_impact", "failed", "nonconverged",
]);
const EFFECTIVE_STATUSES = new Set<CapabilityEffectiveStatus>([
  "complete", "no_impact", "failed",
]);
const SOURCELESS_FAILURE_CODES = new Set(["evaluator_exception", "invalid_evaluator_result"]);
const KNOWN_METRIC_IDS = new Set<string>(FLIGHT_METRIC_IDS);

const record = (value: unknown, label: string): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
};

const exactKeys = (
  value: Record<string, unknown>, expected: readonly string[], label: string,
): void => {
  const actual = Object.keys(value).sort().join("|");
  if (actual !== [...expected].sort().join("|")) {
    throw new RangeError(`${label} fields do not match the v1 schema`);
  }
};

const nonempty = (value: unknown, label: string): string => {
  if (typeof value !== "string" || !value.trim()) {
    throw new RangeError(`${label} must be a nonempty string`);
  }
  return value;
};

const nullableText = (value: unknown, label: string): string | null => {
  if (value === null) return null;
  return nonempty(value, label);
};

const finite = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RangeError(`${label} must be finite`);
  }
  return value;
};

const ordinal = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a nonnegative integer`);
  }
  return value;
};

const parseParameter = (value: unknown, index: number): CapabilityObservedParameter => {
  const item = record(value, `parameters[${index}]`);
  exactKeys(item, PARAMETER_KEYS, `parameters[${index}]`);
  return Object.freeze({
    parameterId: nonempty(item.parameterId, `parameters[${index}].parameterId`),
    unit: nonempty(item.unit, `parameters[${index}].unit`),
    nominalValue: finite(item.nominalValue, `parameters[${index}].nominalValue`),
    perturbedValue: finite(item.perturbedValue, `parameters[${index}].perturbedValue`),
  });
};

const parseMetric = (value: unknown, index: number): EvaluatedMetric => {
  const item = record(value, `metrics[${index}]`);
  exactKeys(item, METRIC_KEYS, `metrics[${index}]`);
  const metricId = nonempty(item.metricId, `metrics[${index}].metricId`);
  if (!KNOWN_METRIC_IDS.has(metricId)) throw new RangeError(`unknown metricId: ${metricId}`);
  return Object.freeze({
    metricId: metricId as FlightMetricId,
    value: finite(item.value, `metrics[${index}].value`),
    provenance: nonempty(item.provenance, `metrics[${index}].provenance`),
  });
};

const requireUnique = (values: readonly string[], label: string): void => {
  if (new Set(values).size !== values.length) throw new RangeError(`${label} must be unique`);
};

interface StatusContractFields {
  readonly source: EvaluationStatus | null;
  readonly effective: CapabilityEffectiveStatus;
  readonly code: string | null;
  readonly reason: string | null;
  readonly metrics: readonly EvaluatedMetric[];
}

const requireStatusContract = (fields: StatusContractFields): void => {
  const { source, effective, code, reason, metrics } = fields;
  let valid = false;
  if (effective === "complete") valid = source === "complete" && code === null && reason === null;
  if (effective === "no_impact") valid = source === "no_impact" && code !== null && code === reason;
  if (effective === "failed" && source === null) {
    valid = code !== null && SOURCELESS_FAILURE_CODES.has(code) && reason === null;
  }
  if (effective === "failed" && source === "complete") {
    valid = code === "missing_required_landing_metrics" && reason === null;
  }
  if (effective === "failed" && (source === "failed" || source === "nonconverged")) {
    valid = code !== null && code === reason;
  }
  if (!valid) throw new RangeError("observation status and reason fields are inconsistent");
  const metricIds = new Set(metrics.map(({ metricId }) => metricId));
  const hasLanding = metricIds.has("carry_distance") && metricIds.has("carry_offline");
  if (effective === "complete" && !hasLanding) {
    throw new RangeError("complete observation requires carry_distance and carry_offline metrics");
  }
  if (effective === "failed" && source === "complete" && hasLanding) {
    throw new RangeError("missing_required_landing_metrics requires an incomplete landing pair");
  }
  if ((effective === "no_impact" || source !== "complete") && metrics.length !== 0) {
    throw new RangeError("non-complete source statuses require zero metrics");
  }
};

/** Parse, validate, defensively copy, and deeply freeze one v1 observation. */
export function parseCapabilitySampleObservation(payload: unknown): CapabilitySampleObservation {
  const root = record(payload, "observation");
  exactKeys(root, OBSERVATION_KEYS, "observation");
  if (root.schemaVersion !== "capability-sample-observation/v1") {
    throw new RangeError("unsupported schemaVersion");
  }
  const attemptOrdinal = ordinal(root.attemptOrdinal, "attemptOrdinal");
  const attemptedCount = ordinal(root.attemptedCount, "attemptedCount");
  const totalCount = ordinal(root.totalCount, "totalCount");
  if (attemptedCount !== attemptOrdinal + 1 || totalCount < attemptedCount) {
    throw new RangeError("observation attempt counts are inconsistent");
  }
  if (!Array.isArray(root.parameters) || !root.parameters.length) {
    throw new TypeError("parameters must be a nonempty array");
  }
  if (!Array.isArray(root.metrics)) throw new TypeError("metrics must be an array");
  const parameters = Object.freeze(root.parameters.map(parseParameter));
  const metrics = Object.freeze(root.metrics.map(parseMetric));
  requireUnique(parameters.map(({ parameterId }) => parameterId), "parameter IDs");
  requireUnique(metrics.map(({ metricId }) => metricId), "metric IDs");
  const sourceStatus = root.sourceStatus as EvaluationStatus | null;
  const effectiveStatus = root.effectiveStatus as CapabilityEffectiveStatus;
  if (sourceStatus !== null && !SOURCE_STATUSES.has(sourceStatus)) {
    throw new RangeError("sourceStatus is not supported");
  }
  if (!EFFECTIVE_STATUSES.has(effectiveStatus)) throw new RangeError("effectiveStatus is not supported");
  const reasonCode = nullableText(root.reasonCode, "reasonCode");
  const sourceReason = nullableText(root.sourceReason, "sourceReason");
  requireStatusContract({
    source: sourceStatus, effective: effectiveStatus,
    code: reasonCode, reason: sourceReason, metrics,
  });
  return Object.freeze({
    schemaVersion: "capability-sample-observation/v1",
    problemId: nonempty(root.problemId, "problemId"), attemptOrdinal, attemptedCount, totalCount,
    candidateOrdinal: ordinal(root.candidateOrdinal, "candidateOrdinal"),
    clubCandidateOrdinal: ordinal(root.clubCandidateOrdinal, "clubCandidateOrdinal"),
    sampleOrdinal: ordinal(root.sampleOrdinal, "sampleOrdinal"),
    clubId: nonempty(root.clubId, "clubId"), parameters, sourceStatus, effectiveStatus,
    reasonCode, sourceReason, metrics,
  });
}

export class CapabilityOptimizationCancelled extends Error {
  readonly attemptedCount: number;
  readonly totalCount: number;

  constructor(attemptedCount: number, totalCount: number) {
    if (!Number.isInteger(attemptedCount) || attemptedCount < 0) {
      throw new RangeError("attemptedCount must be a nonnegative integer");
    }
    if (!Number.isInteger(totalCount) || totalCount < attemptedCount) {
      throw new RangeError("totalCount must be an integer not less than attemptedCount");
    }
    super(`capability optimization cancelled after ${attemptedCount} of ${totalCount} evaluations`);
    this.name = "CapabilityOptimizationCancelled";
    this.attemptedCount = attemptedCount;
    this.totalCount = totalCount;
  }
}

/** Serialize one observation to the exact cross-runtime snake_case wire schema. */
export const capabilitySampleObservationWire = (
  observation: CapabilitySampleObservation,
): CapabilitySampleObservationWire => Object.freeze({
  schema_version: observation.schemaVersion, problem_id: observation.problemId,
  attempt_ordinal: observation.attemptOrdinal, attempted_count: observation.attemptedCount,
  total_count: observation.totalCount, candidate_ordinal: observation.candidateOrdinal,
  club_candidate_ordinal: observation.clubCandidateOrdinal, sample_ordinal: observation.sampleOrdinal,
  club_id: observation.clubId,
  parameters: Object.freeze(observation.parameters.map((item) => Object.freeze({
    parameter_id: item.parameterId, unit: item.unit,
    nominal_value: item.nominalValue, perturbed_value: item.perturbedValue,
  }))),
  source_status: observation.sourceStatus, effective_status: observation.effectiveStatus,
  reason_code: observation.reasonCode, source_reason: observation.sourceReason,
  metrics: Object.freeze(observation.metrics.map((item) => Object.freeze({
    metric_id: item.metricId, value: item.value, provenance: item.provenance,
  }))),
});
