/** Strict editor-snapshot wire for Python-authoritative ground-job preparation. */

import { canonicalGroundJson } from "./flightGroundContract";
import { exact, oneOf, record, text } from "./flightGroundValidation";
import {
  parseRegionalGroundExecutionLaunch,
  type ExecutionJobLaunch,
} from "./regionalGroundExecutionJob";
import {
  regionalGroundVariationRequestFromJson,
  stableRegionalGroundVariationRequestJson,
} from "./regionalGroundVariationRequestWire";
import { parseUniqueJson } from "./strictJson";

export const REGIONAL_GROUND_JOB_PREPARATION_REQUEST_SCHEMA =
  "rate-of-closure/regional-ground-job-preparation-request/v1" as const;
export const MAX_REGIONAL_GROUND_JOB_PREPARATION_REQUEST_BYTES = 1_048_576;
type WireObject = Readonly<Record<string, unknown>>;

export interface RegionalGroundJobPreparationRequest {
  readonly schema_version: typeof REGIONAL_GROUND_JOB_PREPARATION_REQUEST_SCHEMA;
  readonly unit_system: "SI";
  readonly job_id: string;
  readonly launch: ExecutionJobLaunch;
  readonly variation_request: WireObject;
}

const ROOT_FIELDS = [
  "schema_version", "unit_system", "job_id", "launch", "variation_request",
] as const;

const deepFreeze = <T>(value: T): T => {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.values(value as Record<string, unknown>).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
};

const stableId = (value: unknown): string => {
  const parsed = text(value, "job_id");
  if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(parsed)) {
    throw new RangeError("job_id must be a stable identifier");
  }
  return parsed;
};

const parseVariationRequest = (value: unknown): WireObject => {
  const parsed = regionalGroundVariationRequestFromJson(canonicalGroundJson(value));
  return deepFreeze(
    JSON.parse(stableRegionalGroundVariationRequestJson(parsed)) as WireObject,
  );
};

const payload = (request: RegionalGroundJobPreparationRequest): WireObject => ({
  schema_version: request.schema_version,
  unit_system: request.unit_system,
  job_id: request.job_id,
  launch: request.launch,
  variation_request: request.variation_request,
});

const validateJsonBytes = (source: string): void => {
  if (typeof source !== "string") {
    throw new TypeError("regional-ground job preparation request JSON must be text");
  }
  if (new TextEncoder().encode(source).byteLength >
      MAX_REGIONAL_GROUND_JOB_PREPARATION_REQUEST_BYTES) {
    throw new RangeError(
      "regional-ground job preparation request exceeds maximum wire size",
    );
  }
};

/** Parse one exact editor snapshot without preparing or running a study. */
export const parseRegionalGroundJobPreparationRequest = (
  value: unknown,
): RegionalGroundJobPreparationRequest => {
  const item = record(value, "regional-ground job preparation request");
  exact(item, ROOT_FIELDS, "regional-ground job preparation request");
  return deepFreeze({
    schema_version: oneOf(
      item.schema_version,
      [REGIONAL_GROUND_JOB_PREPARATION_REQUEST_SCHEMA] as const,
      "schema_version",
    ),
    unit_system: oneOf(item.unit_system, ["SI"] as const, "unit_system"),
    job_id: stableId(item.job_id),
    launch: parseRegionalGroundExecutionLaunch(item.launch),
    variation_request: parseVariationRequest(item.variation_request),
  });
};

/** Parse bounded duplicate-safe UTF-8 JSON into one exact editor snapshot. */
export const regionalGroundJobPreparationRequestFromJson = (
  source: string,
): RegionalGroundJobPreparationRequest => {
  validateJsonBytes(source);
  return parseRegionalGroundJobPreparationRequest(parseUniqueJson(source));
};

/** Serialize one validated editor snapshot with the shared canonical policy. */
export const stableRegionalGroundJobPreparationRequestJson = (
  value: unknown,
): string => {
  const source = canonicalGroundJson(payload(
    parseRegionalGroundJobPreparationRequest(value),
  ));
  validateJsonBytes(source);
  return source;
};
