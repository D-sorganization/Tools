/** Strict discovery contract for the local Rate Morris authority. */

import {
  MORRIS_AUTHORITY_SCHEMA_VERSION,
  MORRIS_JOB_SCHEMA_ID,
} from "./morrisAuthorityContract";

export const MORRIS_CAPABILITY_SCHEMA_ID = "rate-of-closure/morris-authority-capability" as const;
export const MORRIS_REQUEST_SCHEMA_ID = "rate-of-closure/morris-request" as const;
export const MORRIS_AUTHORITY_API_PREFIX = "/api/rate-of-closure/v1" as const;

export interface MorrisAuthorityCapability {
  readonly schemaId: typeof MORRIS_CAPABILITY_SCHEMA_ID;
  readonly schemaVersion: typeof MORRIS_AUTHORITY_SCHEMA_VERSION;
  readonly available: boolean;
  readonly apiPrefix: string;
  readonly requestSchemaId: typeof MORRIS_REQUEST_SCHEMA_ID;
  readonly jobSchemaId: typeof MORRIS_JOB_SCHEMA_ID;
}

const CAPABILITY_FIELDS = [
  "schema_id", "schema_version", "available", "api_prefix",
  "request_schema_id", "job_schema_id",
] as const;

const plainRecord = (value: unknown): Record<string, unknown> => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new RangeError("Morris capability must be a plain object");
  }
  const prototype = Object.getPrototypeOf(value) as object | null;
  if (prototype !== Object.prototype && prototype !== null) {
    throw new RangeError("Morris capability must be a plain object");
  }
  return value as Record<string, unknown>;
};

const exactFields = (item: Record<string, unknown>): void => {
  const actual = Object.keys(item).sort();
  const expected = [...CAPABILITY_FIELDS].sort();
  if (actual.length !== expected.length || actual.some((field, index) => field !== expected[index])) {
    throw new RangeError("Morris capability fields do not match the v1 schema");
  }
};

export function parseMorrisAuthorityCapability(value: unknown): MorrisAuthorityCapability {
  const item = plainRecord(value);
  exactFields(item);
  if (item.schema_id !== MORRIS_CAPABILITY_SCHEMA_ID) throw new RangeError("unsupported Morris capability schema ID");
  if (item.schema_version !== MORRIS_AUTHORITY_SCHEMA_VERSION) throw new RangeError("unsupported Morris capability version");
  if (typeof item.available !== "boolean") throw new RangeError("Morris capability available must be boolean");
  if (item.api_prefix !== MORRIS_AUTHORITY_API_PREFIX) throw new RangeError("unsupported Morris authority API prefix");
  if (item.request_schema_id !== MORRIS_REQUEST_SCHEMA_ID) throw new RangeError("unsupported Morris request schema ID");
  if (item.job_schema_id !== MORRIS_JOB_SCHEMA_ID) throw new RangeError("unsupported Morris job schema ID");
  return Object.freeze({
    schemaId: MORRIS_CAPABILITY_SCHEMA_ID,
    schemaVersion: MORRIS_AUTHORITY_SCHEMA_VERSION,
    available: item.available,
    apiPrefix: MORRIS_AUTHORITY_API_PREFIX,
    requestSchemaId: MORRIS_REQUEST_SCHEMA_ID,
    jobSchemaId: MORRIS_JOB_SCHEMA_ID,
  });
}
