/** Bounded capability-observation adapter for scalar-ensemble/v1. */

import type { TargetDefinition } from "./capabilityContract";
import {
  parseCapabilitySampleObservation,
  type CapabilityObservedParameter,
  type CapabilitySampleObservation,
} from "./capabilityObservationContract";
import { flightMetricCatalog, type FlightMetricId } from "./ballFlightMetricContract";
import {
  createScalarEnsemble,
  scalarEnsembleRowId,
  type ScalarEnsembleResult,
  type ScalarEnsembleRow,
  type ScalarVariableDefinition,
} from "./scalarEnsembleContract";
import { contains, residualM, signedDistance, type TargetRegionTs } from "./targets";

export const MAX_CAPABILITY_OBSERVATION_ROWS = 100_000;
const ADAPTER_ID = "capability-sample-observation/scalar-ensemble/v1";
const SOURCE_SCHEMA = "capability-sample-observation/v1";
export type CapabilityCohort = "complete" | "no_impact" | "failed";

export interface CapabilityObservationEnsembleInput {
  readonly observations: Iterable<CapabilitySampleObservation>;
  readonly target: TargetDefinition;
  readonly maxRows: number;
  readonly sourceProvenance: string;
}

const TARGET_VARIABLES: readonly ScalarVariableDefinition[] = Object.freeze([
  { key: "target_downrange_residual", label: "Target Downrange Residual", unit: "m", stage_key: "target", category_key: "target" },
  { key: "target_lateral_residual", label: "Target Lateral Residual", unit: "m", stage_key: "target", category_key: "target" },
  { key: "target_residual", label: "Target Center Miss Distance", unit: "m", stage_key: "target", category_key: "target" },
  { key: "target_signed_distance", label: "Target Signed Distance", unit: "m", stage_key: "target", category_key: "target" },
  { key: "target_solver_residual", label: "Target Solver Residual", unit: "m", stage_key: "target", category_key: "target" },
  { key: "target_contains", label: "Inside Target", unit: "1", stage_key: "target", category_key: "target" },
]);

const orderedObservations = (
  observations: readonly CapabilitySampleObservation[],
): readonly CapabilitySampleObservation[] => {
  if (!observations.length) throw new RangeError("capability observations must be nonempty");
  const ordered = [...observations].sort((left, right) => left.attemptOrdinal - right.attemptOrdinal);
  if (new Set(ordered.map(({ attemptOrdinal }) => attemptOrdinal)).size !== ordered.length) {
    throw new RangeError("attemptOrdinal values must be unique");
  }
  if (ordered.some(({ attemptOrdinal }, index) => attemptOrdinal !== index)) {
    throw new RangeError("attemptOrdinal values must form a contiguous prefix from zero");
  }
  if (new Set(ordered.map(({ problemId }) => problemId)).size !== 1) {
    throw new RangeError("observations must share one problemId");
  }
  if (new Set(ordered.map(({ totalCount }) => totalCount)).size !== 1) {
    throw new RangeError("observations must share one totalCount");
  }
  const scalarIds = new Set(metricIds());
  if (ordered.some(({ metrics }) => metrics.some(({ metricId }) => !scalarIds.has(metricId)))) {
    throw new RangeError("capability observations may contain only scalar flight metrics");
  }
  return ordered;
};

const declarations = (
  observations: readonly CapabilitySampleObservation[],
): readonly CapabilityObservedParameter[] => {
  const byClub = new Map<string, readonly CapabilityObservedParameter[]>();
  const declared = new Map<string, CapabilityObservedParameter>();
  observations.forEach((observation) => {
    const previous = byClub.get(observation.clubId);
    const same = previous?.length === observation.parameters.length
      && previous.every((item, index) => {
        const current = observation.parameters[index];
        return item.parameterId === current.parameterId && item.unit === current.unit;
      });
    if (previous !== undefined && !same) {
      throw new RangeError("parameter declarations changed within a club");
    }
    byClub.set(observation.clubId, observation.parameters);
    observation.parameters.forEach((parameter) => {
      const prior = declared.get(parameter.parameterId);
      if (prior && prior.unit !== parameter.unit) {
        throw new RangeError("parameter declarations use conflicting units");
      }
      if (!prior) declared.set(parameter.parameterId, parameter);
    });
  });
  return [...declared.values()];
};

const metricIds = (): readonly FlightMetricId[] => flightMetricCatalog().definitions
  .filter(({ signRule }) => signRule !== "vector_components")
  .map(({ metricId }) => metricId);

const variables = (
  parameters: readonly CapabilityObservedParameter[], metrics: readonly FlightMetricId[],
): readonly ScalarVariableDefinition[] => {
  const definitions: ScalarVariableDefinition[] = [];
  parameters.forEach((parameter) => {
    const label = parameter.parameterId.split("_").map((part) => {
      const initial = part.charCodeAt(0);
      return initial >= 97 && initial <= 122
        ? String.fromCharCode(initial - 32) + part.slice(1)
        : part;
    }).join(" ");
    definitions.push(
      { key: `nominal.${parameter.parameterId}`, label: `Nominal ${label}`, unit: parameter.unit, stage_key: "nominal", category_key: "parameter" },
      { key: `perturbed.${parameter.parameterId}`, label: `Perturbed ${label}`, unit: parameter.unit, stage_key: "perturbed", category_key: "parameter" },
    );
  });
  const catalog = flightMetricCatalog();
  metrics.forEach((metricId) => {
    const definition = catalog.definition(metricId);
    definitions.push({ key: `metric.${metricId}`, label: definition.label, unit: definition.unit, stage_key: "evaluation", category_key: "metric" });
  });
  return [...definitions, ...TARGET_VARIABLES];
};

const targetRegion = (target: TargetDefinition): TargetRegionTs => ({
  kind: target.kind, distanceM: target.distanceM, lateralM: target.lateralM,
  radiusM: target.radiusM, bandHalfLengthM: target.bandHalfLengthM,
  halfWidthM: target.halfWidthM,
});

const targetValues = (
  observation: CapabilitySampleObservation, target: TargetRegionTs,
): Readonly<Record<string, number>> => {
  const metrics = new Map(observation.metrics.map(({ metricId, value }) => [metricId, value]));
  const carry = metrics.get("carry_distance");
  const offline = metrics.get("carry_offline");
  if (carry === undefined || offline === undefined) {
    throw new RangeError("complete observation requires carry and offline metrics");
  }
  const centerOffline = target.kind === "green" ? target.lateralM : 0;
  const downrangeResidual = carry - target.distanceM;
  const lateralResidual = offline - centerOffline;
  return {
    target_downrange_residual: downrangeResidual,
    target_lateral_residual: lateralResidual,
    target_residual: Math.hypot(downrangeResidual, lateralResidual),
    target_signed_distance: signedDistance(target, carry, offline),
    target_solver_residual: residualM(target, carry, offline),
    target_contains: Number(contains(target, carry, offline)),
  };
};

const rowValues = (
  observation: CapabilitySampleObservation,
  variableKeys: readonly string[],
  target: TargetRegionTs,
): Readonly<Record<string, number | null>> => {
  const values: Record<string, number | null> = Object.fromEntries(
    variableKeys.map((key) => [key, null]),
  );
  observation.parameters.forEach((parameter) => {
    values[`nominal.${parameter.parameterId}`] = parameter.nominalValue;
    values[`perturbed.${parameter.parameterId}`] = parameter.perturbedValue;
  });
  if (observation.effectiveStatus === "complete") {
    observation.metrics.forEach(({ metricId, value }) => { values[`metric.${metricId}`] = value; });
    Object.assign(values, targetValues(observation, target));
  }
  return values;
};

const rowAttributes = (
  observation: CapabilitySampleObservation, metrics: readonly FlightMetricId[],
): Readonly<Record<string, string | null>> => {
  const provenance = new Map(observation.metrics.map(({ metricId, provenance: source }) =>
    [metricId, source]));
  return {
    club_id: observation.clubId, attempt_ordinal: String(observation.attemptOrdinal),
    attempted_count: String(observation.attemptedCount), total_count: String(observation.totalCount),
    candidate_ordinal: String(observation.candidateOrdinal),
    club_candidate_ordinal: String(observation.clubCandidateOrdinal),
    sample_ordinal: String(observation.sampleOrdinal), source_status: observation.sourceStatus,
    effective_status: observation.effectiveStatus, reason_code: observation.reasonCode,
    source_reason: observation.sourceReason,
    ...Object.fromEntries(metrics.map((metricId) =>
      [`metric.${metricId}.provenance`, provenance.get(metricId) ?? null])),
  };
};

const rows = (
  observations: readonly CapabilitySampleObservation[],
  definitions: readonly ScalarVariableDefinition[],
  metrics: readonly FlightMetricId[], target: TargetRegionTs,
): readonly ScalarEnsembleRow<CapabilityCohort>[] => {
  const variableKeys = definitions.map(({ key }) => key);
  return observations.map((observation) => {
    const seriesId = `candidate:${observation.candidateOrdinal}/club:${observation.clubId}`;
    return {
      row_id: scalarEnsembleRowId(observation.sampleOrdinal, seriesId),
      trial_index: observation.sampleOrdinal, series_id: seriesId,
      cohort: observation.effectiveStatus,
      values: rowValues(observation, variableKeys, target),
      attributes: rowAttributes(observation, metrics),
    };
  });
};

export type CapabilityObservationEnsembleBuilderInput = Omit<
  CapabilityObservationEnsembleInput, "observations"
>;

const buildAccepted = (
  accepted: readonly CapabilitySampleObservation[],
  input: CapabilityObservationEnsembleBuilderInput,
): ScalarEnsembleResult<CapabilityCohort> => {
  const observations = orderedObservations(accepted);
  const metrics = metricIds();
  const definitions = variables(declarations(observations), metrics);
  return createScalarEnsemble({
    result_id: observations[0].problemId,
    provenance: { adapter_id: ADAPTER_ID, source_schema_version: SOURCE_SCHEMA, source_provenance: input.sourceProvenance },
    stages: [
      { key: "nominal", label: "Nominal Parameters" },
      { key: "perturbed", label: "Perturbed Parameters" },
      { key: "evaluation", label: "Evaluator Metrics" },
      { key: "target", label: "Target Diagnostics" },
    ],
    categories: [
      { key: "parameter", label: "Capability Parameters" },
      { key: "metric", label: "Evaluator Metrics" },
      { key: "target", label: "Target Diagnostics" },
    ],
    variables: definitions,
    cohorts: [
      { key: "complete", label: "Complete" },
      { key: "no_impact", label: "No Impact" },
      { key: "failed", label: "Failed" },
    ],
    rows: rows(observations, definitions, metrics, targetRegion(input.target)),
  });
};

/** Bounded observation sink that rejects before retaining row maxRows + 1. */
export class CapabilityObservationEnsembleBuilder {
  private readonly input: CapabilityObservationEnsembleBuilderInput;
  private readonly observations: CapabilitySampleObservation[] = [];

  constructor(input: CapabilityObservationEnsembleBuilderInput) {
    if (!Number.isInteger(input.maxRows) || input.maxRows < 1 ||
        input.maxRows > MAX_CAPABILITY_OBSERVATION_ROWS) {
      throw new RangeError(`maxRows must be an integer within [1, ${MAX_CAPABILITY_OBSERVATION_ROWS}]`);
    }
    if (!input.sourceProvenance.trim()) throw new RangeError("sourceProvenance must be nonempty");
    this.input = Object.freeze({
      ...input,
      target: Object.freeze({ ...input.target }),
    });
  }

  get retainedCount(): number { return this.observations.length; }

  accept(observation: CapabilitySampleObservation): void {
    if (this.retainedCount >= this.input.maxRows) {
      throw new RangeError(`observation row exceeds maxRows ${this.input.maxRows}`);
    }
    this.observations.push(parseCapabilitySampleObservation(observation));
  }

  build(): ScalarEnsembleResult<CapabilityCohort> {
    return buildAccepted(this.observations, this.input);
  }
}

/** Stream observations through the bounded builder; overflow is never truncated. */
export function buildCapabilityObservationEnsemble(
  input: CapabilityObservationEnsembleInput,
): ScalarEnsembleResult<CapabilityCohort> {
  const builder = new CapabilityObservationEnsembleBuilder({
    target: input.target, maxRows: input.maxRows,
    sourceProvenance: input.sourceProvenance,
  });
  for (const observation of input.observations) builder.accept(observation);
  return builder.build();
}

const codePointCompare = (left: string, right: string): number => {
  const leftPoints = [...left].map((item) => item.codePointAt(0) ?? 0);
  const rightPoints = [...right].map((item) => item.codePointAt(0) ?? 0);
  for (let index = 0; index < Math.min(leftPoints.length, rightPoints.length); index += 1) {
    const difference = leftPoints[index] - rightPoints[index];
    if (difference !== 0) return difference;
  }
  return leftPoints.length - rightPoints.length;
};

const canonicalNumber = (value: number): string => {
  if (!Number.isFinite(value)) throw new RangeError("stable JSON numbers must be finite");
  if (value === 0) return "0";
  if (Number.isInteger(value)) return BigInt(value).toString();
  const fixed = value.toFixed(11);
  const trimmed = fixed.includes(".") ? fixed.replace(/0+$/, "").replace(/\.$/, "") : fixed;
  return trimmed === "-0" ? "0" : trimmed;
};

const canonicalJson = (value: unknown): string => {
  if (value === null) return "null";
  if (typeof value === "number") return canonicalNumber(value);
  if (typeof value === "string" || typeof value === "boolean") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (typeof value === "object") {
    const entries = Object.entries(value).sort(([left], [right]) => codePointCompare(left, right));
    return `{${entries.map(([key, item]) =>
      `${JSON.stringify(key)}:${canonicalJson(item)}`).join(",")}}`;
  }
  throw new TypeError("stable JSON supports only JSON-compatible values");
};

/** Serialize with stable key order and the shared 11-decimal numeric policy. */
export const stableCapabilityObservationEnsembleJson = (
  ensemble: ScalarEnsembleResult<CapabilityCohort>,
): string => canonicalJson(ensemble);
