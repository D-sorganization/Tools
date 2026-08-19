/** Strict import-only adapter for Python-owned regional scalar ensembles. */

import { exact, record } from "./flightGroundValidation";
import {
  type ScalarEnsembleResult,
  type ScalarEnsembleRow,
  type ScalarVariableDefinition,
} from "./scalarEnsembleContract";
import { parseScalarEnsembleWire } from "./scalarEnsembleWire";
import { parseUniqueJson } from "./strictJson";
import {
  readBoundedUtf8File,
  type BoundedUtf8File,
} from "./boundedUtf8File";
import type {
  RegionalGroundExecutionFailureReason,
  RegionalGroundExecutionStatus,
} from "./groundRegionalExecution";

export const MAX_REGIONAL_GROUND_RESULT_BYTES = 8_388_608;
export const MAX_REGIONAL_GROUND_RESULT_ROWS = 100_000;

const STUDY_ADAPTER = "flight-regional-ground/scalar-ensemble/v1";
const VARIATION_ADAPTER = "regional-ground-variation/scalar-ensemble/v1";
const PIPELINE_SCHEMA = "flight-regional-ground-pipeline/v1";
const VARIATION_SCHEMA = "variation-plan/v2+flight-regional-ground-pipeline/v1";
const BASE_ATTRIBUTE_FIELDS = [
  "source_kind", "endpoint_qualification", "transfer_field_id", "transfer_reason",
  "bounce_termination", "regional_status", "regional_failure_reason", "ground_status",
  "ground_termination", "ground_request_sha256", "bounce_execution_input_sha256",
  "regional_plan_sha256", "ground_model_id", "ground_model_version",
] as const;
const VARIATION_ATTRIBUTE_FIELDS = [
  ...BASE_ATTRIBUTE_FIELDS, "variation_seed", "variation_trial_index",
  "variation_input_sha256", "variation_regional_plan_sha256",
] as const;
const OUTPUT_KEYS = [
  "metric.carry_distance", "ground.bounce_air_distance", "ground.skid_distance",
  "metric.roll_distance", "ground.surface_path_distance", "metric.total_distance",
  "ground.final_downrange", "metric.final_offline", "metric.bounce_count",
] as const;
const COHORTS = [
  { key: "complete", label: "Complete at Rest" },
  { key: "partial", label: "Partial or Censored" },
  { key: "cancelled", label: "Cancelled" },
  { key: "failed", label: "Failed" },
  { key: "unavailable", label: "Unavailable" },
] as const;
const BASE_VARIABLES = [
  ["metric.carry_distance", "Carry Distance", "m", "flight_metric"],
  ["ground.bounce_air_distance", "Bounce Air Distance", "m", "ground_detail"],
  ["ground.skid_distance", "Skid Distance", "m", "ground_detail"],
  ["metric.roll_distance", "Roll Distance", "m", "flight_metric"],
  ["ground.surface_path_distance", "Surface Path Distance", "m", "ground_detail"],
  ["metric.total_distance", "Total Distance", "m", "flight_metric"],
  ["ground.final_downrange", "Final Downrange", "m", "ground_detail"],
  ["metric.final_offline", "Final Offline", "m", "flight_metric"],
  ["metric.bounce_count", "Bounce Count", "count", "flight_metric"],
] as const;
const INPUT_VARIABLES = [
  ["input.ground.base.rolling_resistance", "Rolling Resistance"],
  ["input.ground.base.normal_restitution", "Restitution"],
] as const;
const EXECUTION_STATUSES = [
  "complete", "partial", "cancelled", "failed",
] as const satisfies readonly RegionalGroundExecutionStatus[];
const EXECUTION_FAILURE_REASONS = [
  "cancelled", "step_limit", "surface_transition_limit", "unsupported_surface",
  "numerical_failure", "composition_failure",
] as const satisfies readonly RegionalGroundExecutionFailureReason[];

export type RegionalGroundCohort =
  | "complete" | "partial" | "cancelled" | "failed" | "unavailable";
export type RegionalGroundResult = ScalarEnsembleResult<RegionalGroundCohort>;
export type RegionalGroundResultFile = BoundedUtf8File;

const sameJson = (actual: unknown, expected: unknown, name: string): void => {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new RangeError(name + " does not match the Python-owned regional contract");
  }
};

const digest = (value: unknown, name: string): string => {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) {
    throw new RangeError(name + " sha256 must be 64 lowercase hexadecimal characters");
  }
  return value;
};

const oneOf = (value: unknown, allowed: readonly string[], name: string): string => {
  if (typeof value !== "string" || !allowed.includes(value)) {
    throw new RangeError(name + " is not a supported regional evidence value");
  }
  return value;
};

const nullableOneOf = (
  value: unknown,
  allowed: readonly string[],
  name: string,
): string | null => value === null ? null : oneOf(value, allowed, name);

const expectedBaseVariables = (): readonly ScalarVariableDefinition[] =>
  BASE_VARIABLES.map(([key, label, unit, category_key]) => ({
    key, label, unit, stage_key: "ground_stop", category_key,
  }));

const expectedInputVariable = (
  key: string,
): ScalarVariableDefinition | undefined => {
  const match = INPUT_VARIABLES.find(([candidate]) => candidate === key);
  return match === undefined ? undefined : {
    key: match[0], label: match[1], unit: "1",
    stage_key: "ground_input", category_key: "ground_parameter",
  };
};

const validateDefinitions = (result: ScalarEnsembleResult<string>): void => {
  const variation = result.provenance.adapter_id === VARIATION_ADAPTER;
  sameJson(result.cohorts, COHORTS, "cohorts");
  sameJson(result.stages, variation ? [
    { key: "ground_input", label: "Ground Material Input" },
    { key: "ground_stop", label: "Ground Stop" },
  ] : [{ key: "ground_stop", label: "Ground Stop" }], "stages");
  sameJson(result.categories, variation ? [
    { key: "ground_parameter", label: "Ground Parameter" },
    { key: "flight_metric", label: "Canonical Flight Metrics" },
    { key: "ground_detail", label: "Ground Phase Detail" },
  ] : [
    { key: "flight_metric", label: "Canonical Flight Metrics" },
    { key: "ground_detail", label: "Ground Phase Detail" },
  ], "categories");
  validateVariables(result.variables, variation);
};

const validateVariables = (
  variables: readonly ScalarVariableDefinition[],
  variation: boolean,
): void => {
  const base = expectedBaseVariables();
  const inputs = variables.slice(0, variables.length - base.length);
  sameJson(variables.slice(-base.length), base, "ground output variables");
  if (!variation && inputs.length !== 0) {
    throw new RangeError("study result cannot declare variation input variables");
  }
  if (variation && (inputs.length < 1 || inputs.length > INPUT_VARIABLES.length)) {
    throw new RangeError("variation result input-variable count is invalid");
  }
  inputs.forEach((item) => {
    const expected = expectedInputVariable(item.key);
    if (expected === undefined) throw new RangeError("unsupported regional input variable");
    sameJson(item, expected, "regional input variable");
  });
  if (new Set(inputs.map(({ key }) => key)).size !== inputs.length) {
    throw new RangeError("regional input variables must be unique");
  }
};

const validateProvenance = (result: ScalarEnsembleResult<string>): boolean => {
  const { adapter_id: adapter, source_schema_version: schema } = result.provenance;
  if (adapter === STUDY_ADAPTER && schema === PIPELINE_SCHEMA) return false;
  if (adapter === VARIATION_ADAPTER && schema === VARIATION_SCHEMA) return true;
  throw new RangeError("unsupported regional scalar ensemble provenance");
};

const validateQualification = (
  row: ScalarEnsembleRow<string>,
  evidence: Record<string, unknown>,
): void => {
  const allowed: Readonly<Record<string, readonly string[]>> = {
    complete: ["complete_rest"],
    partial: ["censored"],
    cancelled: ["cancelled", "censored"],
    failed: ["failed", "censored"],
    unavailable: ["unavailable", "summary_unavailable"],
  };
  oneOf(evidence.endpoint_qualification, allowed[row.cohort] ?? [], "endpoint qualification");
  const complete = row.cohort === "complete";
  OUTPUT_KEYS.forEach((key) => {
    if ((row.values[key] !== null) !== complete) {
      throw new RangeError("unqualified regional outcomes must retain typed null values");
    }
  });
};

const validatePipelineEvidence = (evidence: Record<string, unknown>): void => {
  if (evidence.transfer_field_id !== null || evidence.transfer_reason !== null) {
    throw new RangeError("pipeline evidence cannot declare transfer failure fields");
  }
  digest(evidence.ground_request_sha256, "ground_request_sha256");
  digest(evidence.bounce_execution_input_sha256, "bounce_execution_input_sha256");
  digest(evidence.regional_plan_sha256, "regional_plan_sha256");
  oneOf(evidence.bounce_termination, [
    "settled_to_skid", "cancelled", "time_limit", "event_limit",
    "no_recontact", "numerical_failure",
  ], "bounce_termination");
  nullableOneOf(evidence.regional_status, EXECUTION_STATUSES, "regional_status");
  nullableOneOf(
    evidence.regional_failure_reason,
    EXECUTION_FAILURE_REASONS,
    "regional_failure_reason",
  );
  nullableOneOf(evidence.ground_status, [
    "complete", "partial", "failed", "unavailable",
  ], "ground_status");
  nullableOneOf(evidence.ground_termination, [
    "rest", "time_limit", "event_limit", "left_surface",
    "numerical_failure", "unavailable_input",
  ], "ground_termination");
  if ((evidence.ground_model_id === null) !== (evidence.ground_model_version === null)) {
    throw new RangeError("ground model identity and version must be jointly nullable");
  }
  if (evidence.endpoint_qualification === "complete_rest" &&
      (evidence.ground_model_id === null || evidence.regional_status !== "complete" ||
       evidence.ground_status !== "complete" || evidence.ground_termination !== "rest")) {
    throw new RangeError("complete-rest evidence requires complete statuses and model identity");
  }
};

const validateTransferEvidence = (
  row: ScalarEnsembleRow<string>,
  evidence: Record<string, unknown>,
): void => {
  if (row.cohort !== "unavailable") {
    throw new RangeError("transfer failure must remain in the unavailable cohort");
  }
  if (evidence.endpoint_qualification !== "unavailable") {
    throw new RangeError("transfer failure endpoint qualification must be unavailable");
  }
  oneOf(evidence.transfer_field_id, [
    "terminal_angular_velocity_rad_s", "physical_contact_bracket", "surface_profile",
  ], "transfer_field_id");
  oneOf(evidence.transfer_reason, [
    "source_does_not_propagate", "no_physical_contact", "unsupported_surface",
    "source_out_of_bounds",
  ], "transfer_reason");
  [
    "bounce_termination", "regional_status", "regional_failure_reason", "ground_status",
    "ground_termination", "ground_request_sha256", "bounce_execution_input_sha256",
    "regional_plan_sha256", "ground_model_id", "ground_model_version",
  ].forEach((key) => {
    if (evidence[key] !== null) throw new RangeError("transfer failure phase evidence must be null");
  });
};

const validateVariationEvidence = (
  row: ScalarEnsembleRow<string>,
  evidence: Record<string, unknown>,
): void => {
  if (!/^(0|[1-9]\d*)$/.test(String(evidence.variation_seed))) {
    throw new RangeError("variation_seed must be a nonnegative integer string");
  }
  if (evidence.variation_trial_index !== String(row.trial_index)) {
    throw new RangeError("variation_trial_index must match trial ordering");
  }
  digest(evidence.variation_input_sha256, "variation_input_sha256");
  const planDigest = digest(
    evidence.variation_regional_plan_sha256,
    "variation_regional_plan_sha256",
  );
  if (evidence.source_kind === "pipeline" && evidence.regional_plan_sha256 !== planDigest) {
    throw new RangeError("variation regional-plan digest must match pipeline evidence");
  }
  Object.keys(row.values).filter((key) => key.startsWith("input.ground."))
    .forEach((key) => {
      if (row.values[key] === null) throw new RangeError("variation inputs cannot be null");
    });
};

const validateRows = (result: ScalarEnsembleResult<string>, variation: boolean): void => {
  if (result.rows.length === 0) throw new RangeError("regional result rows must be nonempty");
  const series = result.rows[0].series_id;
  result.rows.forEach((row, index) => {
    if (row.trial_index !== index || row.series_id !== series) {
      throw new RangeError("regional result must preserve exact trial ordering and series identity");
    }
    const evidence = record(row.attributes, `regional row[${index}] attributes`);
    exact(
      evidence,
      variation ? VARIATION_ATTRIBUTE_FIELDS : BASE_ATTRIBUTE_FIELDS,
      `regional row[${index}] attributes`,
    );
    validateQualification(row, evidence);
    const source = oneOf(evidence.source_kind, ["pipeline", "transfer_failure"], "source_kind");
    if (source === "pipeline") validatePipelineEvidence(evidence);
    else validateTransferEvidence(row, evidence);
    if (variation) validateVariationEvidence(row, evidence);
  });
};

/** Parse a bounded JSON string; this function imports evidence and never runs physics. */
export const regionalGroundResultFromJson = (text: string): RegionalGroundResult => {
  if (typeof text !== "string") throw new TypeError("regional ground result JSON must be text");
  if (new TextEncoder().encode(text).byteLength > MAX_REGIONAL_GROUND_RESULT_BYTES) {
    throw new RangeError("regional ground result exceeds maximum wire size");
  }
  const result = parseScalarEnsembleWire(
    parseUniqueJson(text),
    MAX_REGIONAL_GROUND_RESULT_ROWS,
  );
  const variation = validateProvenance(result);
  validateDefinitions(result);
  validateRows(result, variation);
  return result as RegionalGroundResult;
};

/** Read, fatally decode, and completely validate one browser-selected result. */
export const readRegionalGroundResultFile = async (
  file: RegionalGroundResultFile,
): Promise<RegionalGroundResult> => regionalGroundResultFromJson(
  await readBoundedUtf8File(
    file,
    MAX_REGIONAL_GROUND_RESULT_BYTES,
    "regional ground result",
  ),
);
