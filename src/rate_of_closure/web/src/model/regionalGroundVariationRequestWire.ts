/** Strict, deterministic browser wire contract for combined regional-ground studies. */

import { canonicalGroundJson } from "./flightGroundContract";
import {
  array,
  exact,
  finiteRaw,
  record,
} from "./flightGroundValidation";
import { parseGroundRegionalMaterialPlanRequest } from "./groundRegionalPlan";
import {
  validateRegionalGroundVariationRequest,
  type RegionalGroundVariationRequestTs,
} from "./regionalGroundVariationWorkspace";
import { parseUniqueJson } from "./strictJson";
import {
  SCHEMA_VERSION,
  planFromJson,
  planToJson,
  type VariationPlanTs,
} from "./variation";
import { variationPlanSha256 } from "./variationExecutionMetadata";

export const REGIONAL_GROUND_VARIATION_REQUEST_SCHEMA =
  "rate-of-closure/regional-ground-variation-request/v2" as const;
export const REGIONAL_GROUND_VARIATION_REQUEST_SCHEMA_VERSION = 2 as const;
export const MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES = 1_048_576;

const ROOT_FIELDS = [
  "schema", "schema_version", "variation_plan", "variation_plan_sha256", "regional_plan",
  "result_id", "source_provenance", "max_rows", "series_id",
] as const;
const VARIATION_FIELDS = [
  "schema_version", "mode", "base_variables", "noise", "n_runs", "seed",
  "flight_model", "groups",
] as const;
const NOISE_FIELDS = [
  "variable_key", "distribution", "scale", "lower", "upper", "spec_id",
  "time_window_s", "point_ids",
] as const;
const GROUP_FIELDS = ["group_id", "spec_ids", "matrix_kind", "matrix"] as const;

const nonblankText = (value: unknown, name: string): string => {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new TypeError(name + " must be nonblank text");
  }
  return value;
};

const nullableNumber = (value: unknown, name: string): number | null =>
  value === null ? null : finiteRaw(value, name);

const exactInteger = (value: unknown, name: string, minimum?: number): number => {
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new TypeError(name + " must be an integer");
  }
  if (!Number.isSafeInteger(value)) throw new RangeError(name + " must lie within the safe range");
  if (minimum !== undefined && value < minimum) {
    throw new RangeError(name + " must be at least " + minimum);
  }
  return value;
};

const validateNoiseShape = (value: unknown, index: number): void => {
  const name = `variation_plan noise[${index}]`;
  const item = record(value, name);
  exact(item, NOISE_FIELDS, name);
  nonblankText(item.variable_key, name + " variable_key");
  nonblankText(item.distribution, name + " distribution");
  nonblankText(item.spec_id, name + " spec_id");
  finiteRaw(item.scale, name + " scale");
  nullableNumber(item.lower, name + " lower");
  nullableNumber(item.upper, name + " upper");
  const window = item.time_window_s;
  if (window !== null) {
    const values = array(window, name + " time_window_s");
    if (values.length !== 2) throw new RangeError(name + " time_window_s must contain two numbers");
    values.forEach((entry) => finiteRaw(entry, name + " time_window_s"));
  }
  array(item.point_ids, name + " point_ids")
    .forEach((point) => nonblankText(point, name + " point_id"));
};

const validateGroupShape = (value: unknown, index: number): void => {
  const name = `variation_plan groups[${index}]`;
  const item = record(value, name);
  exact(item, GROUP_FIELDS, name);
  nonblankText(item.group_id, name + " group_id");
  nonblankText(item.matrix_kind, name + " matrix_kind");
  array(item.spec_ids, name + " spec_ids")
    .forEach((specId) => nonblankText(specId, name + " spec_id"));
  array(item.matrix, name + " matrix").forEach((row, rowIndex) => {
    array(row, `${name} matrix[${rowIndex}]`)
      .forEach((entry) => finiteRaw(entry, name + " matrix value"));
  });
};

const validateBaseShape = (value: unknown): void => {
  const base = record(value, "variation_plan base_variables");
  Object.entries(base).forEach(([key, entry]) => {
    nonblankText(key, "variation_plan base_variables key");
    finiteRaw(entry, `variation_plan base_variables[${JSON.stringify(key)}]`);
  });
};

const parseVariationPlan = (value: unknown): VariationPlanTs => {
  const item = record(value, "variation_plan");
  exact(item, VARIATION_FIELDS, "variation_plan");
  const version = exactInteger(item.schema_version, "variation_plan schema_version");
  if (version !== SCHEMA_VERSION) {
    throw new RangeError("unsupported variation_plan schema_version " + version);
  }
  nonblankText(item.mode, "variation_plan mode");
  nonblankText(item.flight_model, "variation_plan flight_model");
  validateBaseShape(item.base_variables);
  exactInteger(item.n_runs, "variation_plan n_runs");
  exactInteger(item.seed, "variation_plan seed", 0);
  array(item.noise, "variation_plan noise").forEach(validateNoiseShape);
  array(item.groups, "variation_plan groups").forEach(validateGroupShape);
  return planFromJson(JSON.stringify(item));
};

const requestPayload = (request: RegionalGroundVariationRequestTs): unknown => ({
  schema: REGIONAL_GROUND_VARIATION_REQUEST_SCHEMA,
  schema_version: REGIONAL_GROUND_VARIATION_REQUEST_SCHEMA_VERSION,
  variation_plan: JSON.parse(planToJson(request.plan)) as unknown,
  variation_plan_sha256: variationPlanSha256(request.plan),
  regional_plan: request.regionalPlan,
  result_id: request.resultId,
  source_provenance: request.sourceProvenance,
  max_rows: request.maxRows,
  series_id: request.seriesId,
});

/** Parse one exact v2 document without applying it or executing physics. */
export const regionalGroundVariationRequestFromJson = (
  text: string,
): RegionalGroundVariationRequestTs => {
  if (typeof text !== "string") throw new TypeError("regional-ground variation request JSON must be text");
  if (new TextEncoder().encode(text).byteLength > MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES) {
    throw new RangeError("regional-ground variation request exceeds maximum wire size");
  }
  const payload = parseUniqueJson(text);
  canonicalGroundJson(payload);
  const item = record(payload, "regional-ground variation request");
  exact(item, ROOT_FIELDS, "regional-ground variation request");
  if (item.schema !== REGIONAL_GROUND_VARIATION_REQUEST_SCHEMA) {
    throw new RangeError("unsupported regional-ground variation request schema");
  }
  const version = exactInteger(item.schema_version, "schema_version");
  if (version !== REGIONAL_GROUND_VARIATION_REQUEST_SCHEMA_VERSION) {
    throw new RangeError("unsupported schema_version " + version);
  }
  const seriesId = item.series_id === null
    ? null
    : nonblankText(item.series_id, "series_id");
  const variationPlan = parseVariationPlan(item.variation_plan);
  if (item.variation_plan_sha256 !== variationPlanSha256(variationPlan)) {
    throw new RangeError("regional-ground variation plan digest mismatch");
  }
  const request: RegionalGroundVariationRequestTs = {
    plan: variationPlan,
    regionalPlan: parseGroundRegionalMaterialPlanRequest(item.regional_plan),
    resultId: nonblankText(item.result_id, "result_id"),
    sourceProvenance: nonblankText(item.source_provenance, "source_provenance"),
    maxRows: exactInteger(item.max_rows, "max_rows", 1),
    seriesId,
  };
  validateRegionalGroundVariationRequest(request);
  return request;
};

/** Serialize one validated request with the shared 11-decimal canonical policy. */
export const stableRegionalGroundVariationRequestJson = (
  request: RegionalGroundVariationRequestTs,
): string => {
  validateRegionalGroundVariationRequest(request);
  return canonicalGroundJson(requestPayload(request));
};
