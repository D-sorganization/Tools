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
const MAX_CAPABILITY_DETAIL_LENGTH = 240;
const FIELDS = [
  "schema_version", "authority_id", "authority_version", "available",
  "regional_ground_execution", "reason_code", "detail",
] as const;
const PYTHON_REASONS = [
  "qualified_execution_profile", "execution_profile_unqualified", "runner_not_started",
] as const;

export type RegionalGroundAuthorityReason =
  | typeof PYTHON_REASONS[number]
  | "authority_unreachable"
  | "authority_invalid_response";

export interface RegionalGroundAuthorityCapability {
  readonly schema_version: typeof REGIONAL_GROUND_AUTHORITY_CAPABILITY_SCHEMA;
  readonly authority_id: typeof AUTHORITY_ID;
  readonly authority_version: typeof AUTHORITY_VERSION;
  readonly available: boolean;
  readonly regional_ground_execution: boolean;
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

export const qualifiedRegionalGroundAuthorityCapability = (
): RegionalGroundAuthorityCapability => Object.freeze({
  schema_version: REGIONAL_GROUND_AUTHORITY_CAPABILITY_SCHEMA,
  authority_id: AUTHORITY_ID,
  authority_version: AUTHORITY_VERSION,
  available: true,
  regional_ground_execution: true,
  reason_code: "qualified_execution_profile",
  detail: "Qualified Python regional-ground execution is available.",
});

/** Parse the exact discriminated v1 service-level capability. */
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
  if (typeof item.available !== "boolean" || typeof item.regional_ground_execution !== "boolean") {
    throw new TypeError("authority capability flags must be booleans");
  }
  if (item.available !== item.regional_ground_execution) {
    throw new RangeError("authority capability flags must be consistent");
  }
  const reason = oneOf(item.reason_code, PYTHON_REASONS, "authority reason");
  const detail = text(item.detail, "authority detail");
  if (detail.length > MAX_CAPABILITY_DETAIL_LENGTH) {
    throw new RangeError("authority detail exceeds the v1 length bound");
  }
  const qualified = reason === "qualified_execution_profile";
  if (item.available !== qualified) {
    throw new RangeError("qualified authority reason and flags must agree");
  }
  return qualified
    ? Object.freeze({ ...qualifiedRegionalGroundAuthorityCapability(), detail })
    : unavailableRegionalGroundAuthorityCapability(reason, detail);
};

const readCapabilityResponse = async (response: Response): Promise<unknown> => {
  if (!response.ok) throw new Error("authority response was not successful");
  const mediaType = response.headers.get("content-type")?.split(";", 1)[0].trim().toLowerCase();
  if (mediaType !== "application/json") {
    throw new TypeError("authority response must use application/json");
  }
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
