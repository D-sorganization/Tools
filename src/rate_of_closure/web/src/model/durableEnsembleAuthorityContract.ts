/** Strict request, capability, and job contracts for durable ensemble transport. */

import {
  exact,
  integer,
  oneOf,
  record,
  text,
} from "./flightGroundValidation";
import {
  parseDurableEnsembleEvidence,
  type DurableEnsembleEvidence,
} from "./durableEnsembleEvidence";
import {
  serializeMorrisAuthorityBase,
  type MorrisAuthorityBase,
} from "./morrisAuthorityRequest";
import { morrisStableId } from "./morrisStableId";
import { planToJson, type VariationPlanTs } from "./variation";
import { variationPlanSha256 } from "./variationExecutionMetadata";

export const DURABLE_REQUEST_SCHEMA_ID = "rate-of-closure/durable-ensemble-request";
export const DURABLE_JOB_SCHEMA_ID = "rate-of-closure/durable-ensemble-job";
export const DURABLE_SCOPE = "passive-double-pendulum-global-perturbations/v1";
export const DURABLE_AUTHORITY_VERSION = 1;
export const DURABLE_REQUEST_VERSION = 2;

export type DurableJobStatus = "queued" | "running" | "completed" | "cancelled" | "failed";

export interface DurableEnsembleJob {
  readonly schemaId: typeof DURABLE_JOB_SCHEMA_ID;
  readonly jobId: string;
  readonly requestId: string;
  readonly archiveId: string;
  readonly status: DurableJobStatus;
  readonly completedTrials: number;
  readonly totalTrials: number;
  readonly cancelRequested: boolean;
  readonly evidence: DurableEnsembleEvidence | null;
  readonly error: string | null;
}

export interface DurableEnsembleCapability {
  readonly available: boolean;
  readonly apiPrefix: "/api/rate-of-closure/v1";
}

export interface DurableEnsembleRequestInput {
  readonly requestId: string;
  readonly archiveId: string;
  readonly plan: VariationPlanTs;
  readonly base: MorrisAuthorityBase;
  readonly chunkSize: number;
  readonly clubheadSpeedMph?: number;
  readonly contactMode?: "delivery_inspection" | "fixed_ball_contact";
}

export interface DurableEnsembleRequestDocument {
  readonly schema_id: typeof DURABLE_REQUEST_SCHEMA_ID;
  readonly schema_version: typeof DURABLE_REQUEST_VERSION;
  readonly scope: typeof DURABLE_SCOPE;
  readonly request_id: string;
  readonly archive_id: string;
  readonly base: Readonly<Record<string, string | number>>;
  readonly plan: Readonly<Record<string, unknown>>;
  readonly plan_sha256: string;
  readonly chunk_size: number;
}

const JOB_FIELDS = [
  "schema_id", "schema_version", "job_id", "request_id", "archive_id",
  "status", "completed_trials", "total_trials", "cancel_requested", "evidence", "error",
] as const;
const CAPABILITY_FIELDS = [
  "schema_id", "schema_version", "available", "api_prefix", "scope",
  "request_schema_id", "job_schema_id",
] as const;
const STATUSES = ["queued", "running", "completed", "cancelled", "failed"] as const;

export function parseDurableEnsembleCapability(value: unknown): DurableEnsembleCapability {
  const item = record(value, "durable ensemble capability");
  exact(item, CAPABILITY_FIELDS, "durable ensemble capability");
  if (item.schema_id !== "rate-of-closure/durable-ensemble-authority-capability"
      || item.schema_version !== DURABLE_AUTHORITY_VERSION
      || item.api_prefix !== "/api/rate-of-closure/v1"
      || item.scope !== DURABLE_SCOPE
      || item.request_schema_id !== DURABLE_REQUEST_SCHEMA_ID
      || item.job_schema_id !== DURABLE_JOB_SCHEMA_ID) {
    throw new RangeError("durable ensemble capability is incompatible");
  }
  if (typeof item.available !== "boolean") {
    throw new TypeError("durable ensemble capability available must be boolean");
  }
  return Object.freeze({
    available: item.available,
    apiPrefix: "/api/rate-of-closure/v1" as const,
  });
}

export function parseDurableEnsembleJob(value: unknown): DurableEnsembleJob {
  const item = record(value, "durable ensemble job");
  exact(item, JOB_FIELDS, "durable ensemble job");
  if (item.schema_id !== DURABLE_JOB_SCHEMA_ID
      || item.schema_version !== DURABLE_AUTHORITY_VERSION) {
    throw new RangeError("durable ensemble job schema is unsupported");
  }
  const status = oneOf(item.status, STATUSES, "durable ensemble job status");
  const completed = integer(item.completed_trials, "completed_trials");
  const total = integer(item.total_trials, "total_trials", 1);
  if (completed > total) throw new RangeError("durable ensemble job progress is invalid");
  if (typeof item.cancel_requested !== "boolean") {
    throw new TypeError("cancel_requested must be boolean");
  }
  const evidence = item.evidence === null ? null : parseDurableEnsembleEvidence(item.evidence);
  const error = item.error === null ? null : text(item.error, "durable ensemble job error");
  if ((status === "failed") !== (error !== null)
      || (status === "completed" && completed !== total)
      || (evidence !== null && evidence.archive.analyzedTrialCount !== completed)) {
    throw new RangeError("durable ensemble job lifecycle fields are inconsistent");
  }
  return Object.freeze({
    schemaId: DURABLE_JOB_SCHEMA_ID,
    jobId: morrisStableId(item.job_id, "job_id"),
    requestId: morrisStableId(item.request_id, "request_id"),
    archiveId: morrisStableId(item.archive_id, "archive_id"),
    status,
    completedTrials: completed,
    totalTrials: total,
    cancelRequested: item.cancel_requested,
    evidence,
    error,
  });
}

export function serializeDurableEnsembleRequest(
  input: DurableEnsembleRequestInput,
): DurableEnsembleRequestDocument {
  if (input.plan.mode !== "swing") throw new RangeError("durable ensemble plan mode must be swing");
  if (input.plan.nRuns < 1 || input.plan.nRuns > 100_000) {
    throw new RangeError("durable ensemble nRuns must be within [1, 100000]");
  }
  if (input.plan.noise.some((spec) => spec.timeWindowS !== null
      || (spec.pointIds?.length ?? 0) > 0)) {
    throw new RangeError("durable ensemble scope supports only global perturbations");
  }
  const plan = JSON.parse(planToJson(input.plan)) as Record<string, unknown>;
  delete plan.ball_setup;
  const speed = input.clubheadSpeedMph ?? 113;
  if (!Number.isFinite(speed) || speed <= 0) throw new RangeError("clubhead speed must be positive");
  const mode = input.contactMode ?? "fixed_ball_contact";
  const chunkSize = integer(input.chunkSize, "chunkSize", 1);
  if (chunkSize > 4096) throw new RangeError("chunkSize must not exceed 4096");
  const base = Object.freeze({
    ...serializeMorrisAuthorityBase(input.base),
    clubhead_speed_mph: speed,
    contact_mode: mode,
  });
  return Object.freeze({
    schema_id: DURABLE_REQUEST_SCHEMA_ID,
    schema_version: DURABLE_REQUEST_VERSION,
    scope: DURABLE_SCOPE,
    request_id: morrisStableId(input.requestId, "requestId"),
    archive_id: morrisStableId(input.archiveId, "archiveId"),
    base,
    plan: Object.freeze(plan),
    plan_sha256: variationPlanSha256({ ...input.plan, ballSetup: undefined }),
    chunk_size: chunkSize,
  });
}
