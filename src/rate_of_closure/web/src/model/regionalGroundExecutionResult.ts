/** Strict job-bound envelope for complete regional-ground scalar results. */

import { canonicalGroundJson } from "./flightGroundContract";
import { exact, record, text } from "./flightGroundValidation";
import type { RegionalGroundExecutionJob } from "./regionalGroundExecutionJob";
import {
  parseScalarEnsembleWire,
} from "./scalarEnsembleWire";
import type { ScalarEnsembleResult } from "./scalarEnsembleContract";
import { sha256Text } from "./sha256";
import { parseUniqueJson } from "./strictJson";

export const REGIONAL_GROUND_EXECUTION_RESULT_SCHEMA_VERSION =
  "rate-of-closure/regional-ground-execution-result/v1" as const;
export const MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES = 8_388_608;
const MAX_REGIONAL_GROUND_EXECUTION_RESULT_ROWS = 100_000;
const ROOT_FIELDS = [
  "schema_version",
  "job_id",
  "job_sha256",
  "input_sha256",
  "dataset_sha256",
  "dataset",
] as const;

export interface RegionalGroundExecutionResult {
  readonly schema_version: typeof REGIONAL_GROUND_EXECUTION_RESULT_SCHEMA_VERSION;
  readonly job_id: string;
  readonly job_sha256: string;
  readonly input_sha256: string;
  readonly dataset_sha256: string;
  readonly dataset: ScalarEnsembleResult<string>;
}

const digest = (value: unknown, name: string): string => {
  const parsed = text(value, name);
  if (!/^[0-9a-f]{64}$/.test(parsed)) {
    throw new RangeError(name + " must be 64 lowercase hexadecimal characters");
  }
  return parsed;
};

const stableId = (value: unknown, name: string): string => {
  const parsed = text(value, name);
  if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(parsed)) {
    throw new RangeError(name + " must be a stable identifier");
  }
  return parsed;
};

const resultWire = (result: RegionalGroundExecutionResult): Readonly<Record<string, unknown>> => ({
  schema_version: result.schema_version,
  job_id: result.job_id,
  job_sha256: result.job_sha256,
  input_sha256: result.input_sha256,
  dataset_sha256: result.dataset_sha256,
  dataset: result.dataset,
});

/** Parse and freeze one exact result envelope without executing physics. */
export const parseRegionalGroundExecutionResult = (
  value: unknown,
): RegionalGroundExecutionResult => {
  const item = record(value, "regional-ground execution result");
  exact(item, ROOT_FIELDS, "regional-ground execution result");
  if (item.schema_version !== REGIONAL_GROUND_EXECUTION_RESULT_SCHEMA_VERSION) {
    throw new RangeError("unsupported regional-ground execution result schema_version");
  }
  const dataset = parseScalarEnsembleWire(
    item.dataset,
    MAX_REGIONAL_GROUND_EXECUTION_RESULT_ROWS,
  );
  const datasetSha = digest(item.dataset_sha256, "dataset_sha256");
  if (sha256Text(canonicalGroundJson(dataset)) !== datasetSha) {
    throw new RangeError("dataset_sha256 must match the complete dataset authority");
  }
  return Object.freeze({
    schema_version: REGIONAL_GROUND_EXECUTION_RESULT_SCHEMA_VERSION,
    job_id: stableId(item.job_id, "job_id"),
    job_sha256: digest(item.job_sha256, "job_sha256"),
    input_sha256: digest(item.input_sha256, "input_sha256"),
    dataset_sha256: datasetSha,
    dataset,
  });
};

/** Fail unless a parsed result carries every identity of one validated job. */
export const assertRegionalGroundExecutionResultMatchesJob = (
  result: RegionalGroundExecutionResult,
  job: RegionalGroundExecutionJob,
): void => {
  const identities = ["job_id", "job_sha256", "input_sha256"] as const;
  identities.forEach((field) => {
    if (result[field] !== job[field]) {
      throw new RangeError(field + " must match the expected execution job");
    }
  });
  const request = record(job.variation_request, "variation request");
  if (result.dataset.result_id !== request.result_id) {
    throw new RangeError("dataset result_id must match the expected execution job");
  }
  if (result.dataset.rows.length !== job.execution_options.max_trials) {
    throw new RangeError("dataset trial count must match the expected execution job");
  }
  if (result.dataset.rows.some((row, index) => row.trial_index !== index)) {
    throw new RangeError("dataset trial ordering must match the expected execution job");
  }
  if (result.dataset.rows.some((row) => row.series_id !== request.series_id)) {
    throw new RangeError("dataset series_id must match the expected execution job");
  }
};

/** Parse bounded strict UTF-8 JSON with duplicate-field rejection. */
export const regionalGroundExecutionResultFromJson = (
  textValue: string,
): RegionalGroundExecutionResult => {
  if (typeof textValue !== "string") {
    throw new TypeError("regional-ground execution result JSON must be text");
  }
  if (new TextEncoder().encode(textValue).byteLength >
      MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES) {
    throw new RangeError("regional-ground execution result exceeds maximum wire size");
  }
  return parseRegionalGroundExecutionResult(parseUniqueJson(textValue));
};

/** Serialize one fully revalidated envelope with canonical numeric JSON. */
export const stableRegionalGroundExecutionResultJson = (
  value: unknown,
): string => {
  const canonical = canonicalGroundJson(resultWire(
    parseRegionalGroundExecutionResult(value),
  ));
  if (new TextEncoder().encode(canonical).byteLength >
      MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES) {
    throw new RangeError("regional-ground execution result exceeds maximum wire size");
  }
  return canonical;
};
