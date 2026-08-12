/** Strict wire parser and injected transport for the Rate Morris authority. */

import { parseMorrisReport, type MorrisReport } from "./morrisGlobalSensitivityContract";

export const MORRIS_JOB_SCHEMA_ID = "rate-of-closure/morris-job" as const;
export const MORRIS_AUTHORITY_SCHEMA_VERSION = 1 as const;

export type MorrisJobStatus = "queued" | "running" | "completed" | "cancelled" | "failed";

export interface MorrisJobError {
  readonly code: string;
  readonly message: string;
}

export interface MorrisJobEnvelope {
  readonly schemaId: typeof MORRIS_JOB_SCHEMA_ID;
  readonly schemaVersion: typeof MORRIS_AUTHORITY_SCHEMA_VERSION;
  readonly jobId: string;
  readonly requestId: string;
  readonly status: MorrisJobStatus;
  readonly completedSamples: number;
  readonly totalSamples: number;
  readonly cancelRequested: boolean;
  readonly report: MorrisReport | null;
  readonly error: MorrisJobError | null;
}

const JOB_FIELDS = [
  "schema_id", "schema_version", "job_id", "request_id", "status",
  "completed_samples", "total_samples", "cancel_requested", "report", "error",
] as const;
const ERROR_FIELDS = ["code", "message"] as const;
const STATUSES = ["queued", "running", "completed", "cancelled", "failed"] as const;

const record = (value: unknown, name: string): Record<string, unknown> => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new RangeError(`${name} must be a plain object`);
  }
  const prototype = Object.getPrototypeOf(value) as object | null;
  if (prototype !== Object.prototype && prototype !== null) {
    throw new RangeError(`${name} must be a plain object`);
  }
  return value as Record<string, unknown>;
};

const exact = (value: Record<string, unknown>, fields: readonly string[], name: string): void => {
  const actual = Object.keys(value).sort();
  const expected = [...fields].sort();
  if (actual.length !== expected.length || actual.some((field, index) => field !== expected[index])) {
    throw new RangeError(`${name} fields do not match the v1 schema`);
  }
};

const text = (value: unknown, name: string): string => {
  if (typeof value !== "string" || value === "" || value !== value.trim()) {
    throw new RangeError(`${name} must be a nonempty trimmed string`);
  }
  return value;
};

const count = (value: unknown, name: string): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a nonnegative safe integer`);
  }
  return value;
};

const parseError = (value: unknown): MorrisJobError | null => {
  if (value === null) return null;
  const item = record(value, "Morris job error");
  exact(item, ERROR_FIELDS, "Morris job error");
  return Object.freeze({ code: text(item.code, "error code"), message: text(item.message, "error message") });
};

const statusValue = (value: unknown): MorrisJobStatus => {
  const parsed = text(value, "job status");
  if (!STATUSES.includes(parsed as MorrisJobStatus)) throw new RangeError("job status is unsupported");
  return parsed as MorrisJobStatus;
};

export function parseMorrisJobEnvelope(value: unknown): MorrisJobEnvelope {
  const item = record(value, "Morris job envelope");
  exact(item, JOB_FIELDS, "Morris job envelope");
  if (item.schema_id !== MORRIS_JOB_SCHEMA_ID) throw new RangeError("unsupported Morris job schema ID");
  if (item.schema_version !== MORRIS_AUTHORITY_SCHEMA_VERSION) throw new RangeError("unsupported Morris job schema version");
  const status = statusValue(item.status);
  const completedSamples = count(item.completed_samples, "completed_samples");
  const totalSamples = count(item.total_samples, "total_samples");
  if (totalSamples < 1 || completedSamples > totalSamples) throw new RangeError("job progress invariant failed");
  if (typeof item.cancel_requested !== "boolean") throw new RangeError("cancel_requested must be boolean");
  const report = item.report === null ? null : parseMorrisReport(item.report);
  const error = parseError(item.error);
  if ((status === "completed") !== (report !== null) || (status === "completed" && completedSamples !== totalSamples)) {
    throw new RangeError("only a fully completed job may carry a report");
  }
  if ((status === "failed") !== (error !== null)) throw new RangeError("only a failed job must carry an error");
  return Object.freeze({
    schemaId: MORRIS_JOB_SCHEMA_ID,
    schemaVersion: MORRIS_AUTHORITY_SCHEMA_VERSION,
    jobId: text(item.job_id, "job_id"),
    requestId: text(item.request_id, "request_id"),
    status,
    completedSamples,
    totalSamples,
    cancelRequested: item.cancel_requested,
    report,
    error,
  });
}
