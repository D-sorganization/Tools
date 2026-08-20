/** Strict future-facing REST client for the local Python ground authority. */

import {
  fetchRegionalGroundAuthorityCapability,
  type RegionalGroundAuthorityCapability,
} from "./regionalGroundAuthority";
import {
  assertRegionalGroundExecutionResultMatchesJob,
  MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES,
  regionalGroundExecutionResultFromJson,
  type RegionalGroundExecutionResult,
} from "./regionalGroundExecutionResult";
import {
  MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES,
  parseRegionalGroundExecutionJob,
  regionalGroundExecutionJobFromJson,
  stableRegionalGroundExecutionJobJson,
  type RegionalGroundExecutionJob,
} from "./regionalGroundExecutionJob";
import {
  parseRegionalGroundJobPreparationRequest,
  stableRegionalGroundJobPreparationRequestJson,
  type RegionalGroundJobPreparationRequest,
} from "./regionalGroundJobPreparationRequest";
import {
  boolean,
  exact,
  integer,
  oneOf,
  record,
  text,
} from "./flightGroundValidation";
import { canonicalGroundJson } from "./flightGroundContract";
import { parseUniqueJson } from "./strictJson";

export const REGIONAL_GROUND_AUTHORITY_JOB_STATUS_SCHEMA =
  "rate-of-closure/regional-ground-authority-job-status/v1" as const;
export const REGIONAL_GROUND_AUTHORITY_JOBS_PATH =
  "/api/rate-of-closure/v1/regional-ground/jobs" as const;
export const REGIONAL_GROUND_JOB_PREPARATIONS_PATH =
  "/api/rate-of-closure/v1/regional-ground/job-preparations" as const;
export const MAX_REGIONAL_GROUND_AUTHORITY_STATUS_BYTES = 4_096;

const STATUS_FIELDS = [
  "schema_version", "job_id", "job_sha256", "status", "completed", "total",
  "result_available", "failure",
] as const;
const JOB_STATES = [
  "queued", "running", "cancel_requested", "succeeded", "failed", "cancelled",
] as const;
const FAILURE_CODES = ["execution_failed", "result_rejected"] as const;
const FAILURE_STAGES = [
  "authority_restart",
  "cancellation_callback", "preflight", "executor", "validation", "progress_callback",
  "publication", "runner", "result_validation",
] as const;
const REQUEST_ERROR_CODES = [
  "invalid_job", "body_too_large", "unsupported_media_type", "execution_unavailable",
  "job_conflict", "job_not_found", "result_unavailable", "invalid_preparation",
  "preparation_unavailable", "preparation_failed",
] as const;
type Fetcher = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

export type RegionalGroundAuthorityJobState = typeof JOB_STATES[number];
export type RegionalGroundAuthorityFailureCode = typeof FAILURE_CODES[number];
export type RegionalGroundAuthorityFailureStage = typeof FAILURE_STAGES[number];
export type RegionalGroundAuthorityRequestErrorCode =
  | typeof REQUEST_ERROR_CODES[number]
  | "authentication_required"
  | "authority_error";

export interface RegionalGroundAuthorityJobFailure {
  readonly code: RegionalGroundAuthorityFailureCode;
  readonly stage: RegionalGroundAuthorityFailureStage;
}

export interface RegionalGroundAuthorityJobStatus {
  readonly schema_version: typeof REGIONAL_GROUND_AUTHORITY_JOB_STATUS_SCHEMA;
  readonly job_id: string;
  readonly job_sha256: string;
  readonly status: RegionalGroundAuthorityJobState;
  readonly completed: number;
  readonly total: number;
  readonly result_available: boolean;
  readonly failure: RegionalGroundAuthorityJobFailure | null;
}

/** Stable client-side identity for a non-successful authority response. */
export class RegionalGroundAuthorityRequestError extends Error {
  readonly httpStatus: number;
  readonly code: RegionalGroundAuthorityRequestErrorCode;

  constructor(httpStatus: number, code: RegionalGroundAuthorityRequestErrorCode) {
    super(`regional-ground authority request failed (${code})`);
    this.name = "RegionalGroundAuthorityRequestError";
    this.httpStatus = httpStatus;
    this.code = code;
  }
}

export interface RegionalGroundAuthorityClient {
  capability: (signal?: AbortSignal) => Promise<RegionalGroundAuthorityCapability>;
  submit: (
    job: RegionalGroundExecutionJob,
    signal?: AbortSignal,
  ) => Promise<RegionalGroundAuthorityJobStatus>;
  status: (
    job: RegionalGroundExecutionJob,
    signal?: AbortSignal,
  ) => Promise<RegionalGroundAuthorityJobStatus>;
  cancel: (
    job: RegionalGroundExecutionJob,
    signal?: AbortSignal,
  ) => Promise<RegionalGroundAuthorityJobStatus>;
  result: (
    job: RegionalGroundExecutionJob,
    signal?: AbortSignal,
  ) => Promise<RegionalGroundExecutionResult>;
}

/** Complete v1 authority surface, including non-executing job preparation. */
export interface RegionalGroundPreparationAuthorityClient
  extends RegionalGroundAuthorityClient {
  prepare: (
    request: RegionalGroundJobPreparationRequest,
    signal?: AbortSignal,
  ) => Promise<RegionalGroundExecutionJob>;
}

const digest = (value: unknown, name: string): string => {
  const parsed = text(value, name);
  if (!/^[0-9a-f]{64}$/.test(parsed)) {
    throw new RangeError(name + " must be 64 lowercase hexadecimal characters");
  }
  return parsed;
};

const failure = (value: unknown): RegionalGroundAuthorityJobFailure | null => {
  if (value === null) return null;
  const item = record(value, "failure");
  exact(item, ["code", "stage"], "failure");
  return Object.freeze({
    code: oneOf(item.code, FAILURE_CODES, "failure code"),
    stage: oneOf(item.stage, FAILURE_STAGES, "failure stage"),
  });
};

const validateStatusSemantics = (status: RegionalGroundAuthorityJobStatus): void => {
  if (status.completed > status.total) {
    throw new RangeError("completed must not exceed total");
  }
  if (status.status === "queued" && status.completed !== 0) {
    throw new RangeError("queued status must have zero completed");
  }
  const succeeded = status.status === "succeeded";
  if (status.result_available !== succeeded ||
      (succeeded && status.completed !== status.total)) {
    throw new RangeError("succeeded status must expose a complete result");
  }
  if ((status.status === "failed") !== (status.failure !== null)) {
    throw new RangeError("failure is required only for failed status");
  }
};

/** Parse one exact bounded status and bind it to its validated source job. */
export const parseRegionalGroundAuthorityJobStatus = (
  value: unknown,
  expectedJob: RegionalGroundExecutionJob,
): RegionalGroundAuthorityJobStatus => {
  const job = parseRegionalGroundExecutionJob(expectedJob);
  const item = record(value, "regional-ground authority job status");
  exact(item, STATUS_FIELDS, "regional-ground authority job status");
  const parsed = Object.freeze({
    schema_version: oneOf(
      item.schema_version,
      [REGIONAL_GROUND_AUTHORITY_JOB_STATUS_SCHEMA] as const,
      "job status schema_version",
    ),
    job_id: text(item.job_id, "job_id"),
    job_sha256: digest(item.job_sha256, "job_sha256"),
    status: oneOf(item.status, JOB_STATES, "job status"),
    completed: integer(item.completed, "completed"),
    total: integer(item.total, "total", 1),
    result_available: boolean(item.result_available, "result_available"),
    failure: failure(item.failure),
  });
  (["job_id", "job_sha256"] as const).forEach((field) => {
    if (parsed[field] !== job[field]) {
      throw new RangeError(field + " must match the expected execution job");
    }
  });
  if (parsed.total !== job.execution_options.max_trials) {
    throw new RangeError("total must match the expected execution job");
  }
  validateStatusSemantics(parsed);
  return parsed;
};

const validateStatusJsonBytes = (source: string): void => {
  if (typeof source !== "string") {
    throw new TypeError("regional-ground authority job status JSON must be text");
  }
  if (new TextEncoder().encode(source).byteLength >
      MAX_REGIONAL_GROUND_AUTHORITY_STATUS_BYTES) {
    throw new RangeError("regional-ground authority job status exceeds byte limit");
  }
};

/** Parse bounded duplicate-safe status JSON against its exact source job. */
export const regionalGroundAuthorityJobStatusFromJson = (
  source: string,
  expectedJob: RegionalGroundExecutionJob,
): RegionalGroundAuthorityJobStatus => {
  validateStatusJsonBytes(source);
  return parseRegionalGroundAuthorityJobStatus(parseUniqueJson(source), expectedJob);
};

/** Serialize a validated job-bound status with the shared canonical JSON policy. */
export const stableRegionalGroundAuthorityJobStatusJson = (
  value: unknown,
  expectedJob: RegionalGroundExecutionJob,
): string => {
  const source = canonicalGroundJson(
    parseRegionalGroundAuthorityJobStatus(value, expectedJob),
  );
  validateStatusJsonBytes(source);
  return source;
};

const boundedResponseText = async (response: Response, maximum: number): Promise<string> => {
  const declared = response.headers.get("content-length");
  if (declared !== null && (!/^\d+$/.test(declared) || Number(declared) > maximum)) {
    throw new RangeError("regional-ground authority response exceeds byte limit");
  }
  const source = await response.text();
  if (new TextEncoder().encode(source).byteLength > maximum) {
    throw new RangeError("regional-ground authority response exceeds byte limit");
  }
  return source;
};

const requestError = (response: Response, source: string): RegionalGroundAuthorityRequestError => {
  if (response.status === 401) {
    return new RegionalGroundAuthorityRequestError(401, "authentication_required");
  }
  try {
    const item = record(parseUniqueJson(source), "authority error");
    exact(item, ["code", "detail"], "authority error");
    const code = oneOf(item.code, REQUEST_ERROR_CODES, "authority error code");
    text(item.detail, "authority error detail");
    const validStatus: Readonly<Record<typeof REQUEST_ERROR_CODES[number], number>> = {
      invalid_job: 400,
      body_too_large: 413,
      unsupported_media_type: 415,
      execution_unavailable: 503,
      job_conflict: 409,
      job_not_found: 404,
      result_unavailable: 409,
      invalid_preparation: 400,
      preparation_unavailable: 503,
      preparation_failed: 422,
    };
    if (validStatus[code] === response.status) {
      return new RegionalGroundAuthorityRequestError(response.status, code);
    }
  } catch {
    // Unknown or malformed server errors collapse to one non-sensitive identity.
  }
  return new RegionalGroundAuthorityRequestError(response.status, "authority_error");
};

const readBoundedText = async (response: Response, maximum: number): Promise<string> => {
  const source = await boundedResponseText(response, maximum);
  if (!response.ok) throw requestError(response, source);
  return source;
};

const requestInit = (
  method: "GET" | "POST",
  signal: AbortSignal | undefined,
  body?: string,
): RequestInit => ({
  method,
  cache: "no-store",
  credentials: "omit",
  headers: body === undefined
    ? { Accept: "application/json" }
    : { Accept: "application/json", "Content-Type": "application/json" },
  signal,
  ...(body === undefined ? {} : { body }),
});

/** Create an injectable same-origin client without claiming server availability. */
export const createRegionalGroundAuthorityClient = (
  fetcher: Fetcher = fetch,
): RegionalGroundPreparationAuthorityClient => {
  const jobPath = (job: RegionalGroundExecutionJob): string =>
    `${REGIONAL_GROUND_AUTHORITY_JOBS_PATH}/${encodeURIComponent(job.job_id)}`;
  const readStatus = async (
    job: RegionalGroundExecutionJob,
    path: string,
    init: RequestInit,
  ): Promise<RegionalGroundAuthorityJobStatus> => {
    const source = await readBoundedText(
      await fetcher(path, init),
      MAX_REGIONAL_GROUND_AUTHORITY_STATUS_BYTES,
    );
    return regionalGroundAuthorityJobStatusFromJson(source, job);
  };
  return Object.freeze({
    capability: (signal?: AbortSignal) =>
      fetchRegionalGroundAuthorityCapability(fetcher, signal),
    prepare: async (
      source: RegionalGroundJobPreparationRequest,
      signal?: AbortSignal,
    ) => {
      const request = parseRegionalGroundJobPreparationRequest(source);
      const response = await readBoundedText(
        await fetcher(REGIONAL_GROUND_JOB_PREPARATIONS_PATH, requestInit(
          "POST", signal, stableRegionalGroundJobPreparationRequestJson(request),
        )),
        MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES,
      );
      const job = regionalGroundExecutionJobFromJson(response);
      if (job.job_id !== request.job_id ||
          canonicalGroundJson(job.launch) !== canonicalGroundJson(request.launch) ||
          canonicalGroundJson(job.variation_request) !==
            canonicalGroundJson(request.variation_request)) {
        throw new RangeError(
          "prepared job must match the requested job_id, launch, and variation_request",
        );
      }
      return job;
    },
    submit: async (source: RegionalGroundExecutionJob, signal?: AbortSignal) => {
      const job = parseRegionalGroundExecutionJob(source);
      return readStatus(job, REGIONAL_GROUND_AUTHORITY_JOBS_PATH, requestInit(
        "POST", signal, stableRegionalGroundExecutionJobJson(job),
      ));
    },
    status: async (source: RegionalGroundExecutionJob, signal?: AbortSignal) => {
      const job = parseRegionalGroundExecutionJob(source);
      return readStatus(job, jobPath(job), requestInit("GET", signal));
    },
    cancel: async (source: RegionalGroundExecutionJob, signal?: AbortSignal) => {
      const job = parseRegionalGroundExecutionJob(source);
      return readStatus(job, `${jobPath(job)}/cancel`, requestInit("POST", signal));
    },
    result: async (source: RegionalGroundExecutionJob, signal?: AbortSignal) => {
      const job = parseRegionalGroundExecutionJob(source);
      const payload = await readBoundedText(
        await fetcher(`${jobPath(job)}/result`, requestInit("GET", signal)),
        MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES,
      );
      const result = regionalGroundExecutionResultFromJson(payload);
      assertRegionalGroundExecutionResultMatchesJob(result, job);
      return result;
    },
  });
};
