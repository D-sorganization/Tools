/** Strict path-free durable ensemble evidence consumed by both client surfaces. */

import {
  array,
  exact,
  finiteRaw,
  integer,
  oneOf,
  record,
  text,
} from "./flightGroundValidation";

export const DURABLE_ENSEMBLE_EVIDENCE_SCHEMA = "rate/durable-ensemble-evidence/v1";
export const DURABLE_ENSEMBLE_ANALYSIS_METHOD = "incremental-welford-sample-moments/v1";
export const DURABLE_ENSEMBLE_FRAME = "app_frame:x_target,y_up,z_right";
export const DURABLE_ENSEMBLE_LIMITATIONS = [
  "Model-scenario output is not human evidence or a coaching recommendation.",
  "Incremental moments do not retain quantiles, correlations, or trial rows.",
  "An in-progress archive describes only its verified contiguous prefix.",
] as const;
export const DURABLE_OUTPUTS = [
  ["candidate_time_s", "s"], ["closest_approach_m", "m"],
  ["contact_margin_m", "m"], ["impact_time_s", "s"],
  ["clubhead_speed_mps", "m/s"], ["spin_loft_deg", "deg"],
  ["face_to_path_deg", "deg"], ["spin_axis_tilt_deg", "deg"],
  ["ball_speed_mph", "mph"], ["launch_angle_deg", "deg"],
  ["launch_azimuth_deg", "deg"], ["spin_rpm", "rpm"],
  ["carry_m", "m"], ["lateral_m", "m"], ["max_height_m", "m"],
  ["flight_time_s", "s"], ["landing_angle_deg", "deg"],
] as const;

export type DurableArchiveStatus = "in_progress" | "complete";

export interface DurableArchiveEvidence {
  readonly headerSha256: string;
  readonly status: DurableArchiveStatus;
  readonly trialCount: number;
  readonly analyzedTrialCount: number;
  readonly failedCount: number;
  readonly chunkCount: number;
  readonly elapsedS: number | null;
}

export interface DurableAnalysisEvidence {
  readonly methodId: typeof DURABLE_ENSEMBLE_ANALYSIS_METHOD;
  readonly sampleCount: number;
  readonly pointIds: readonly string[];
  readonly coordinateFrame: typeof DURABLE_ENSEMBLE_FRAME;
}

export interface DurableOutputMoments {
  readonly name: string;
  readonly unit: string;
  readonly availableCount: number;
  readonly mean: number | null;
  readonly sampleStd: number | null;
}

export interface DurableEnsembleEvidence {
  readonly schemaVersion: typeof DURABLE_ENSEMBLE_EVIDENCE_SCHEMA;
  readonly archive: DurableArchiveEvidence;
  readonly analysis: DurableAnalysisEvidence;
  readonly statusCounts: Readonly<Record<string, number>>;
  readonly failureTypeCounts: Readonly<Record<string, number>>;
  readonly outputMoments: readonly DurableOutputMoments[];
  readonly limitations: typeof DURABLE_ENSEMBLE_LIMITATIONS;
}

const ROOT_FIELDS = [
  "schema_version", "archive", "analysis", "status_counts",
  "failure_type_counts", "output_moments", "limitations",
] as const;
const ARCHIVE_FIELDS = [
  "header_sha256", "status", "trial_count", "analyzed_trial_count",
  "failed_count", "chunk_count", "elapsed_s",
] as const;
const ANALYSIS_FIELDS = [
  "method_id", "sample_count", "point_ids", "coordinate_frame",
] as const;
const MOMENT_FIELDS = ["name", "unit", "available_count", "mean", "sample_std"] as const;
const STATUS_FIELDS = ["evaluated_hit", "evaluated_no_impact", "numerical_failure"] as const;

const nullableFinite = (value: unknown, name: string): number | null =>
  value === null ? null : finiteRaw(value, name);

const sha256 = (value: unknown): string => {
  const parsed = text(value, "header_sha256");
  if (!/^[0-9a-f]{64}$/.test(parsed)) throw new RangeError("header_sha256 is invalid");
  return parsed;
};

function parseArchive(value: unknown): DurableArchiveEvidence {
  const item = record(value, "durable archive");
  exact(item, ARCHIVE_FIELDS, "durable archive");
  const result = {
    headerSha256: sha256(item.header_sha256),
    status: oneOf(item.status, ["in_progress", "complete"] as const, "archive status"),
    trialCount: integer(item.trial_count, "trial_count", 1),
    analyzedTrialCount: integer(item.analyzed_trial_count, "analyzed_trial_count"),
    failedCount: integer(item.failed_count, "failed_count"),
    chunkCount: integer(item.chunk_count, "chunk_count"),
    elapsedS: nullableFinite(item.elapsed_s, "elapsed_s"),
  };
  if (result.analyzedTrialCount > result.trialCount
      || result.failedCount > result.analyzedTrialCount
      || result.chunkCount > result.analyzedTrialCount) {
    throw new RangeError("durable archive prefix counts are inconsistent");
  }
  if ((result.status === "complete") !== (result.elapsedS !== null)
      || (result.status === "complete" && result.analyzedTrialCount !== result.trialCount)) {
    throw new RangeError("durable archive status and completion fields are inconsistent");
  }
  if (result.elapsedS !== null && result.elapsedS < 0) {
    throw new RangeError("elapsed_s must be nonnegative");
  }
  return Object.freeze(result);
}

function parseAnalysis(value: unknown): DurableAnalysisEvidence {
  const item = record(value, "durable analysis");
  exact(item, ANALYSIS_FIELDS, "durable analysis");
  if (item.method_id !== DURABLE_ENSEMBLE_ANALYSIS_METHOD) {
    throw new RangeError("durable analysis method is unsupported");
  }
  if (item.coordinate_frame !== DURABLE_ENSEMBLE_FRAME) {
    throw new RangeError("durable analysis coordinate frame is unsupported");
  }
  const pointIds = array(item.point_ids, "point_ids").map((value) => text(value, "point_id"));
  if (!pointIds.length || new Set(pointIds).size !== pointIds.length) {
    throw new RangeError("point_ids must be nonempty and unique");
  }
  return Object.freeze({
    methodId: DURABLE_ENSEMBLE_ANALYSIS_METHOD,
    sampleCount: integer(item.sample_count, "sample_count", 1),
    pointIds: Object.freeze(pointIds),
    coordinateFrame: DURABLE_ENSEMBLE_FRAME,
  });
}

function parseCounts(value: unknown, fields: readonly string[], name: string): Record<string, number> {
  const item = record(value, name);
  exact(item, fields, name);
  return Object.fromEntries(fields.map((field) => [field, integer(item[field], `${name} ${field}`)]));
}

function parseFailureCounts(value: unknown): Record<string, number> {
  const item = record(value, "failure type counts");
  if (Object.keys(item).length > 256) throw new RangeError("failure type counts exceed 256 entries");
  return Object.fromEntries(Object.entries(item).map(([name, value]) => [
    text(name, "failure type"), integer(value, `failure type ${name}`, 1),
  ]));
}

function parseMoments(value: unknown, prefixCount: number): readonly DurableOutputMoments[] {
  const items = array(value, "output moments");
  if (items.length !== DURABLE_OUTPUTS.length) throw new RangeError("output moments are not canonical");
  return Object.freeze(items.map((value, index) => {
    const item = record(value, `output moment ${index}`);
    exact(item, MOMENT_FIELDS, `output moment ${index}`);
    const [expectedName, expectedUnit] = DURABLE_OUTPUTS[index];
    if (item.name !== expectedName || item.unit !== expectedUnit) {
      throw new RangeError("output moment name or unit is not canonical");
    }
    const count = integer(item.available_count, "available_count");
    const mean = nullableFinite(item.mean, "mean");
    const sampleStd = nullableFinite(item.sample_std, "sample_std");
    if (count > prefixCount || (mean === null) !== (count === 0)
        || (sampleStd === null) !== (count < 2) || (sampleStd !== null && sampleStd < 0)) {
      throw new RangeError("output moment availability is inconsistent");
    }
    return Object.freeze({ name: expectedName, unit: expectedUnit, availableCount: count, mean, sampleStd });
  }));
}

function parseLimitations(value: unknown): typeof DURABLE_ENSEMBLE_LIMITATIONS {
  const items = array(value, "limitations");
  if (items.length !== DURABLE_ENSEMBLE_LIMITATIONS.length
      || items.some((item, index) => item !== DURABLE_ENSEMBLE_LIMITATIONS[index])) {
    throw new RangeError("limitations do not match the registered scientific boundary");
  }
  return DURABLE_ENSEMBLE_LIMITATIONS;
}

/** Parse, validate, and freeze path-free evidence received across a Worker boundary. */
export function parseDurableEnsembleEvidence(value: unknown): DurableEnsembleEvidence {
  const item = record(value, "durable ensemble evidence");
  exact(item, ROOT_FIELDS, "durable ensemble evidence");
  if (item.schema_version !== DURABLE_ENSEMBLE_EVIDENCE_SCHEMA) {
    throw new RangeError("durable ensemble evidence schema is unsupported");
  }
  const archive = parseArchive(item.archive);
  const statuses = parseCounts(item.status_counts, STATUS_FIELDS, "status counts");
  const failures = parseFailureCounts(item.failure_type_counts);
  if (Object.values(statuses).reduce((total, count) => total + count, 0)
      !== archive.analyzedTrialCount) throw new RangeError("status counts do not cover the prefix");
  if (statuses.numerical_failure !== archive.failedCount
      || Object.values(failures).reduce((total, count) => total + count, 0)
      !== archive.failedCount) throw new RangeError("failure counts do not cover numerical failures");
  return Object.freeze({
    schemaVersion: DURABLE_ENSEMBLE_EVIDENCE_SCHEMA,
    archive,
    analysis: parseAnalysis(item.analysis),
    statusCounts: Object.freeze(statuses),
    failureTypeCounts: Object.freeze(failures),
    outputMoments: parseMoments(item.output_moments, archive.analyzedTrialCount),
    limitations: parseLimitations(item.limitations),
  });
}
