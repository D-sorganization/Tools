/** Strict fail-closed client for the local Python regional-ground authority. */

import { exact, oneOf, record, text } from "./flightGroundValidation";
import { parseUniqueJson } from "./strictJson";

export const REGIONAL_GROUND_AUTHORITY_CAPABILITY_SCHEMA =
  "rate-of-closure/regional-ground-authority-capability/v1" as const;
export const REGIONAL_GROUND_AUTHORITY_CAPABILITY_PATH =
  "/api/rate-of-closure/v1/capabilities" as const;
const AUTHORITY_ID = "rate-of-closure-python-authority";
const AUTHORITY_VERSION = "1";
const MAX_CAPABILITY_BYTES = 4_096;
const FIELDS = [
  "schema_version", "authority_id", "authority_version", "available",
  "regional_ground_execution", "reason_code", "detail",
] as const;
const PYTHON_REASONS = [
  "execution_profile_unqualified", "runner_not_started",
] as const;

export type RegionalGroundAuthorityReason =
  | typeof PYTHON_REASONS[number]
  | "authority_unreachable"
  | "authority_invalid_response";

export interface RegionalGroundAuthorityCapability {
  readonly schema_version: typeof REGIONAL_GROUND_AUTHORITY_CAPABILITY_SCHEMA;
  readonly authority_id: typeof AUTHORITY_ID;
  readonly authority_version: typeof AUTHORITY_VERSION;
  readonly available: false;
  readonly regional_ground_execution: false;
  readonly reason_code: RegionalGroundAuthorityReason;
  readonly detail: string;
}

type Fetcher = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

export const unavailableRegionalGroundAuthorityCapability = (
  reason_code: RegionalGroundAuthorityReason,
  detail: string,
): RegionalGroundAuthorityCapability => Object.freeze({
  schema_version: REGIONAL_GROUND_AUTHORITY_CAPABILITY_SCHEMA,
  authority_id: AUTHORITY_ID,
  authority_version: AUTHORITY_VERSION,
  available: false,
  regional_ground_execution: false,
  reason_code,
  detail,
});

/** Parse the exact v1 capability. Execution remains disabled in this slice. */
export const parseRegionalGroundAuthorityCapability = (
  value: unknown,
): RegionalGroundAuthorityCapability => {
  const item = record(value, "regional-ground authority capability");
  exact(item, FIELDS, "regional-ground authority capability");
  if (item.schema_version !== REGIONAL_GROUND_AUTHORITY_CAPABILITY_SCHEMA) {
    throw new RangeError("unsupported regional-ground authority capability schema");
  }
  if (item.authority_id !== AUTHORITY_ID || item.authority_version !== AUTHORITY_VERSION) {
    throw new RangeError("unsupported regional-ground authority identity");
  }
  if (item.available !== false) throw new RangeError("authority availability is not qualified");
  if (item.regional_ground_execution !== false) {
    throw new RangeError("regional-ground execution is not qualified");
  }
  return unavailableRegionalGroundAuthorityCapability(
    oneOf(item.reason_code, PYTHON_REASONS, "authority reason"),
    text(item.detail, "authority detail"),
  );
};

const readCapabilityResponse = async (response: Response): Promise<unknown> => {
  if (!response.ok) throw new Error("authority response was not successful");
  const declaredLength = Number(response.headers.get("content-length") ?? "0");
  if (declaredLength > MAX_CAPABILITY_BYTES) throw new RangeError("capability response exceeds byte limit");
  const source = await response.text();
  if (new TextEncoder().encode(source).byteLength > MAX_CAPABILITY_BYTES) {
    throw new RangeError("capability response exceeds byte limit");
  }
  return parseUniqueJson(source);
};

/** Query the same-origin proxy and convert every failure into typed unavailability. */
export const fetchRegionalGroundAuthorityCapability = async (
  fetcher: Fetcher = fetch,
  signal?: AbortSignal,
): Promise<RegionalGroundAuthorityCapability> => {
  let response: Response;
  try {
    response = await fetcher(REGIONAL_GROUND_AUTHORITY_CAPABILITY_PATH, {
      cache: "no-store",
      credentials: "omit",
      headers: { Accept: "application/json" },
      signal,
    });
  } catch {
    return unavailableRegionalGroundAuthorityCapability(
      "authority_unreachable",
      "Local Python execution authority is unreachable.",
    );
  }
  try {
    return parseRegionalGroundAuthorityCapability(await readCapabilityResponse(response));
  } catch {
    return unavailableRegionalGroundAuthorityCapability(
      "authority_invalid_response",
      "Local Python execution authority returned invalid capability evidence.",
    );
  }
};
