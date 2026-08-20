/** Pure view models for Morris authority progress and target-scoped results. */

import type { MorrisJobEnvelope, MorrisJobStatus } from "./morrisAuthorityContract";
import type {
  MorrisAvailability,
  MorrisReport,
  MorrisSampleAdequacy,
  MorrisTargetProvenance,
} from "./morrisGlobalSensitivityContract";
import { RATE_MORRIS_VARIABLE_KEYS } from "./morrisAuthorityRequest";
import { variableLabel } from "./variationRegistry";

export interface MorrisJobPresentation {
  readonly status: MorrisJobStatus;
  readonly terminal: boolean;
  readonly completedSamples: number;
  readonly totalSamples: number;
  readonly progressFraction: number | null;
  readonly cancelRequested: boolean;
  readonly canCancel: boolean;
  readonly canPresentResults: boolean;
  readonly message: string;
  readonly errorCode: string | null;
  readonly errorMessage: string | null;
}

export interface MorrisPresentationRow {
  readonly rank: number | null;
  readonly specId: string;
  readonly variableKey: string;
  readonly label: string;
  readonly sourceUnit: string;
  readonly sourceLower: number;
  readonly sourceUpper: number;
  readonly mu: number | null;
  readonly muStar: number | null;
  readonly muStarStandardError: number | null;
  readonly sigma: number | null;
  readonly availability: MorrisAvailability;
  readonly sampleAdequacy: MorrisSampleAdequacy;
  readonly totalPairs: number;
  readonly validPairs: number;
  readonly typedNoImpactPairs: number;
  readonly noImpactUnavailablePairs: number;
  readonly failedPairs: number;
  readonly nonfinitePairs: number;
}

export interface MorrisReportPresentation {
  readonly target: MorrisTargetPresentation;
  readonly rows: readonly MorrisPresentationRow[];
}

export interface MorrisTargetPresentation extends MorrisTargetProvenance {
  readonly label: string;
}

const TERMINAL = new Set<MorrisJobStatus>(["completed", "cancelled", "failed"]);

const messageForJob = (job: MorrisJobEnvelope): string => {
  const message = job.status === "queued" ? "Morris study queued"
    : job.status === "running" ? `Morris study running: ${job.completedSamples}/${job.totalSamples}`
      : job.status === "completed" ? "Morris study completed"
        : job.status === "cancelled" ? "Morris study cancelled"
          : "Morris study failed";
  return job.cancelRequested && !TERMINAL.has(job.status)
    ? `${message}; cancellation requested`
    : message;
};

export function presentMorrisJob(job: MorrisJobEnvelope): MorrisJobPresentation {
  const terminal = TERMINAL.has(job.status);
  return Object.freeze({
    status: job.status,
    terminal,
    completedSamples: job.completedSamples,
    totalSamples: job.totalSamples,
    progressFraction: job.totalSamples === 0 ? null : job.completedSamples / job.totalSamples,
    cancelRequested: job.cancelRequested,
    canCancel: !terminal && !job.cancelRequested,
    canPresentResults: job.status === "completed" && job.report !== null,
    message: messageForJob(job),
    errorCode: job.error?.code ?? null,
    errorMessage: job.error?.message ?? null,
  });
}

const canonicalFactorIndex = (variableKey: string): number => {
  const index = RATE_MORRIS_VARIABLE_KEYS.indexOf(variableKey as typeof RATE_MORRIS_VARIABLE_KEYS[number]);
  return index < 0 ? Number.MAX_SAFE_INTEGER : index;
};

const rowOrder = (left: MorrisPresentationRow, right: MorrisPresentationRow): number => {
  if (left.muStar === null) return right.muStar === null ? sourceOrder(left, right) : 1;
  if (right.muStar === null) return -1;
  return right.muStar - left.muStar || sourceOrder(left, right);
};

const sourceOrder = (left: MorrisPresentationRow, right: MorrisPresentationRow): number => (
  canonicalFactorIndex(left.variableKey) - canonicalFactorIndex(right.variableKey)
  || left.specId.localeCompare(right.specId)
);

const withRank = (row: MorrisPresentationRow, rank: number | null): MorrisPresentationRow => Object.freeze({
  ...row, rank,
});

const OUTPUT_LABELS: Readonly<Record<string, string>> = Object.freeze({
  candidate_time_s: "Candidate Contact Time", closest_approach_m: "Closest Approach",
  contact_margin_m: "Contact Margin", impact_time_s: "Impact Time",
  clubhead_speed_mps: "Clubhead Speed", spin_loft_deg: "Spin Loft",
  face_to_path_deg: "Face to Path", spin_axis_tilt_deg: "Spin-Axis Tilt",
  ball_speed_mph: "Ball Speed", launch_angle_deg: "Launch Angle",
  launch_azimuth_deg: "Launch Direction", spin_rpm: "Spin Rate", carry_m: "Carry",
  lateral_m: "Lateral Landing Position", max_height_m: "Maximum Height",
  flight_time_s: "Flight Time", landing_angle_deg: "Landing Angle",
});

const titleFromName = (name: string): string => name.split("_")
  .filter((part) => part !== "")
  .map((part) => part[0]?.toUpperCase() + part.slice(1))
  .join(" ");

export function presentMorrisReport(report: MorrisReport, targetName: string): MorrisReportPresentation {
  if (typeof targetName !== "string" || targetName === "" || targetName !== targetName.trim()) {
    throw new RangeError("Morris target name must be a nonempty trimmed string");
  }
  const estimates = report.estimates.filter((estimate) => estimate.target.name === targetName);
  if (estimates.length === 0) throw new RangeError("Morris target is not present in the report");
  const rows = estimates.map((estimate): MorrisPresentationRow => Object.freeze({
    rank: null,
    specId: estimate.source.specId,
    variableKey: estimate.source.variableKey,
    label: variableLabel(estimate.source.variableKey),
    sourceUnit: estimate.source.unit,
    sourceLower: estimate.source.bounds[0],
    sourceUpper: estimate.source.bounds[1],
    mu: estimate.effects.mu,
    muStar: estimate.effects.muStar,
    muStarStandardError: estimate.effects.muStarStandardError,
    sigma: estimate.effects.sigma,
    availability: estimate.availability,
    sampleAdequacy: estimate.sampleAdequacy,
    totalPairs: estimate.denominator.totalPairs,
    validPairs: estimate.denominator.validPairs,
    typedNoImpactPairs: estimate.denominator.typedNoImpactPairs,
    noImpactUnavailablePairs: estimate.denominator.noImpactUnavailablePairs,
    failedPairs: estimate.denominator.failedPairs,
    nonfinitePairs: estimate.denominator.nonfinitePairs,
  })).sort(rowOrder).map((row, index) => withRank(row, row.muStar === null ? null : index + 1));
  const target = Object.freeze({
    ...estimates[0].target,
    label: OUTPUT_LABELS[targetName] ?? titleFromName(targetName),
  });
  return Object.freeze({ target, rows: Object.freeze(rows) });
}
