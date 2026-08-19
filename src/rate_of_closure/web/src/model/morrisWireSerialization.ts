/** Exact inverse serialization for already-validated Morris authority values. */

import type { MorrisJobEnvelope } from "./morrisAuthorityContract";
import type { MorrisEstimate, MorrisReport } from "./morrisGlobalSensitivityContract";

const sourceDocument = (estimate: MorrisEstimate) => ({
  spec_id: estimate.source.specId,
  variable_key: estimate.source.variableKey,
  unit: estimate.source.unit,
  bounds: [...estimate.source.bounds],
  time_window_s: estimate.source.timeWindowS === null ? null : [...estimate.source.timeWindowS],
  point_ids: [...estimate.source.pointIds],
});

const targetDocument = (estimate: MorrisEstimate) => ({
  name: estimate.target.name,
  unit: estimate.target.unit,
  kind: estimate.target.kind,
  time_s: estimate.target.timeS,
  point_id: estimate.target.pointId,
  coordinate_frame: estimate.target.coordinateFrame,
});

const estimateDocument = (estimate: MorrisEstimate) => ({
  source: sourceDocument(estimate),
  target: targetDocument(estimate),
  effects: {
    mu: estimate.effects.mu,
    mu_star: estimate.effects.muStar,
    mu_star_standard_error: estimate.effects.muStarStandardError,
    sigma: estimate.effects.sigma,
  },
  availability: estimate.availability,
  sample_adequacy: estimate.sampleAdequacy,
  denominator: {
    total_pairs: estimate.denominator.totalPairs,
    valid_pairs: estimate.denominator.validPairs,
    typed_no_impact_pairs: estimate.denominator.typedNoImpactPairs,
    no_impact_unavailable_pairs: estimate.denominator.noImpactUnavailablePairs,
    failed_pairs: estimate.denominator.failedPairs,
    nonfinite_pairs: estimate.denominator.nonfinitePairs,
  },
});

export const morrisReportToDocument = (report: MorrisReport) => ({
  schema_id: report.schemaId,
  schema_version: report.schemaVersion,
  method: report.method,
  design: {
    trajectories: report.design.trajectories,
    levels: report.design.levels,
    seed: report.design.seed,
    total_samples: report.design.totalSamples,
    normalized_step: report.design.normalizedStep,
  },
  assumptions: [...report.assumptions],
  interaction_caveat: report.interactionCaveat,
  estimates: report.estimates.map(estimateDocument),
});

export const morrisJobToDocument = (job: MorrisJobEnvelope) => ({
  schema_id: job.schemaId,
  schema_version: job.schemaVersion,
  job_id: job.jobId,
  request_id: job.requestId,
  status: job.status,
  completed_samples: job.completedSamples,
  total_samples: job.totalSamples,
  cancel_requested: job.cancelRequested,
  report: job.report === null ? null : morrisReportToDocument(job.report),
  error: job.error === null ? null : { code: job.error.code, message: job.error.message },
});
