/** UI-neutral matched regional-ground execution presentation state. */

import type { RegionalGroundAuthorityCapability } from "./regionalGroundAuthority";
import type {
  RegionalGroundAuthorityFailureStage,
  RegionalGroundAuthorityJobStatus,
} from "./regionalGroundAuthorityClient";
import {
  parseRegionalGroundExecutionJob,
  type RegionalGroundExecutionJob,
} from "./regionalGroundExecutionJob";
import {
  assertRegionalGroundExecutionResultMatchesJob,
  type RegionalGroundExecutionResult,
} from "./regionalGroundExecutionResult";

export type RegionalGroundPresentationStatus =
  | "idle"
  | "running"
  | "cancel_requested"
  | "cancelled"
  | "failed"
  | "succeeded";

export interface RegionalGroundExecutionSummary {
  readonly schema_version: string;
  readonly model_id: string;
  readonly model_version: string;
  readonly producer: string;
  readonly producer_version: string;
  readonly source_revision: string;
  readonly input_sha256: string;
}

export interface RegionalGroundExecutionPresentationView {
  readonly summary: RegionalGroundExecutionSummary;
  readonly job_sha256: string;
  readonly execution_enabled: false;
  readonly disabled_reason_code: string;
  readonly disabled_detail: string;
  readonly status: RegionalGroundPresentationStatus;
  readonly completed: number;
  readonly total: number;
  readonly failure_stage: RegionalGroundAuthorityFailureStage | null;
  readonly result_schema_version: string | null;
  readonly result_sha256: string | null;
}

const counts = (
  view: RegionalGroundExecutionPresentationView,
  completed: number,
  total: number,
): void => {
  if (!Number.isSafeInteger(completed) || !Number.isSafeInteger(total)) {
    throw new TypeError("execution counts must be safe integers");
  }
  if (total !== view.total || completed < view.completed || completed > total) {
    throw new RangeError("execution counts must be monotonic and match the job");
  }
};

/** Project exact job evidence and the authoritative false capability. */
export const initialRegionalGroundExecutionPresentation = (
  value: RegionalGroundExecutionJob,
  capability: RegionalGroundAuthorityCapability,
): RegionalGroundExecutionPresentationView => {
  const job = parseRegionalGroundExecutionJob(value);
  if (capability.regional_ground_execution !== false) {
    throw new RangeError("presentation requires the false execution capability");
  }
  return Object.freeze({
    summary: Object.freeze({
      schema_version: job.schema_version,
      model_id: job.flight.model_id,
      model_version: job.flight.model_version,
      producer: job.provenance.producer,
      producer_version: job.provenance.producer_version,
      source_revision: job.provenance.source_revision,
      input_sha256: job.input_sha256,
    }),
    job_sha256: job.job_sha256,
    execution_enabled: capability.regional_ground_execution,
    disabled_reason_code: capability.reason_code,
    disabled_detail: capability.detail,
    status: "idle",
    completed: 0,
    total: job.execution_options.max_trials,
    failure_stage: null,
    result_schema_version: null,
    result_sha256: null,
  });
};

export const presentRegionalGroundProgress = (
  view: RegionalGroundExecutionPresentationView,
  completed: number,
  total: number,
): RegionalGroundExecutionPresentationView => {
  counts(view, completed, total);
  return Object.freeze({ ...view, status: "running", completed });
};

export const presentRegionalGroundCancelRequested = (
  view: RegionalGroundExecutionPresentationView,
): RegionalGroundExecutionPresentationView => Object.freeze({
  ...view,
  status: "cancel_requested",
});

export const presentRegionalGroundCancelled = (
  view: RegionalGroundExecutionPresentationView,
  completed: number,
  total: number,
): RegionalGroundExecutionPresentationView => {
  counts(view, completed, total);
  return Object.freeze({ ...view, status: "cancelled", completed });
};

export const presentRegionalGroundFailure = (
  view: RegionalGroundExecutionPresentationView,
  stage: RegionalGroundAuthorityFailureStage,
  completed: number,
  total: number,
): RegionalGroundExecutionPresentationView => {
  counts(view, completed, total);
  return Object.freeze({
    ...view,
    status: "failed",
    completed,
    failure_stage: stage,
  });
};

/** Project one exact controller status without initiating or cancelling work. */
export const presentRegionalGroundAuthorityStatus = (
  view: RegionalGroundExecutionPresentationView,
  status: RegionalGroundAuthorityJobStatus,
): RegionalGroundExecutionPresentationView => {
  if (status.job_sha256 !== view.job_sha256) {
    throw new RangeError("status job_sha256 must match the presentation");
  }
  counts(view, status.completed, status.total);
  const mapped = status.status === "queued" ? "idle" : status.status;
  return Object.freeze({
    ...view,
    status: mapped,
    completed: status.completed,
    failure_stage: status.failure?.stage ?? null,
  });
};

export const presentRegionalGroundResult = (
  view: RegionalGroundExecutionPresentationView,
  result: RegionalGroundExecutionResult,
  job: RegionalGroundExecutionJob,
): RegionalGroundExecutionPresentationView => {
  assertRegionalGroundExecutionResultMatchesJob(result, job);
  if (result.job_sha256 !== view.job_sha256) {
    throw new RangeError("result job_sha256 must match the presentation");
  }
  return Object.freeze({
    ...view,
    status: "succeeded",
    completed: view.total,
    result_schema_version: result.schema_version,
    result_sha256: result.dataset_sha256,
  });
};
