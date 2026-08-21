import rawRunCatalog from "../../../shared/python/swing_sim/rotating_base/resources/rotating_base_registered_runs_v1.json";

import {
  ROTATING_BASE_BOUNDARIES,
  ROTATING_BASE_MODEL_TIER,
  ROTATING_BASE_SOURCE_REVISION,
  ROTATING_BASE_STUDY,
  ROTATING_BASE_STUDY_SHA256,
  type MatchingRule,
  type RotatingBaseCase,
  type RotatingBaseCaseMetrics,
  type TorsoProfile,
} from "./rotatingBaseStudy";

export const ROTATING_BASE_RUN_CATALOG_SHA256 =
  "66493b833955c6492a00eae4a600df795df60a6f473f9a11c403084b58e51678";

export interface RotatingBaseRunTrace {
  time_s: number[];
  torso_rate_rad_s: number[];
  club_rate_rad_s: number[];
  clubhead_speed_m_s: number[];
  contact_power_on_club_w: number[];
  force_generated_couple_nm: number[];
  force_on_club_n: number[][][];
  distal_segment_kinetic_energy_j: number[];
}

export interface RotatingBaseRun {
  schema_id: "swing-sim/rotating-base-run-result";
  schema_version: 1;
  source_revision: string;
  model_tier: string;
  boundaries: typeof ROTATING_BASE_BOUNDARIES;
  request: {
    case_index: number;
    torso_profile: TorsoProfile;
    matching_rule: MatchingRule;
    initial_torso_rate_rad_s: number;
  };
  case: {
    case_index: number;
    torso_profile: TorsoProfile;
    matching_rule: MatchingRule;
    initial_torso_rate_rad_s: number;
    valid: boolean;
    exclusion_reasons: string[];
    metrics: RotatingBaseCaseMetrics;
  };
  trace: RotatingBaseRunTrace;
}

export interface RotatingBaseRunCatalog {
  schema_id: "swing-sim/rotating-base-run-catalog";
  schema_version: 1;
  source_revision: string;
  study_sha256: string;
  model_tier: string;
  attempted_run_count: 18;
  runs: RotatingBaseRun[];
}

const METRIC_KEYS: Array<keyof RotatingBaseCaseMetrics> = [
  "initial_club_rate_rad_s",
  "final_torso_rate_rad_s",
  "impact_speed_m_s",
  "clubhead_speed_gain_m_s",
  "contact_work_on_club_j",
  "braking_grip_work_j",
  "force_couple_work_j",
  "negative_along_path_impulse_ns",
  "bilateral_wrist_work_j",
  "total_control_work_j",
  "distal_energy_gain_j",
  "peak_grip_force_n",
  "maximum_constraint_residual_m",
  "maximum_velocity_constraint_residual_m_s",
  "maximum_contact_power_identity_residual_w",
  "work_energy_closure_j",
];

const TRACE_KEYS: Array<
  Exclude<keyof RotatingBaseRunTrace, "force_on_club_n">
> = [
  "time_s",
  "torso_rate_rad_s",
  "club_rate_rad_s",
  "clubhead_speed_m_s",
  "contact_power_on_club_w",
  "force_generated_couple_nm",
  "distal_segment_kinetic_energy_j",
];

function record(value: unknown, name: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
}

function finite(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${name} must be finite`);
  }
  return value;
}

function finiteArray(value: unknown, name: string, count?: number): number[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new TypeError(`${name} must be a nonempty array`);
  }
  if (count !== undefined && value.length !== count) {
    throw new RangeError(`${name} must share the trace length`);
  }
  return value.map((item, index) => finite(item, `${name}[${index}]`));
}

function validateIdentity(
  value: Record<string, unknown>,
  authority: RotatingBaseCase,
  name: string,
): void {
  if (
    value.case_index !== authority.case_index ||
    value.torso_profile !== authority.torso_profile ||
    value.matching_rule !== authority.matching_rule ||
    value.initial_torso_rate_rad_s !== authority.initial_torso_rate_rad_s
  ) {
    throw new RangeError(`${name} does not match the qualified study row`);
  }
}

function validateTrace(value: unknown, runIndex: number): RotatingBaseRunTrace {
  const trace = record(value, `run ${runIndex} trace`);
  const time = finiteArray(trace.time_s, `run ${runIndex} time`);
  if (time.length < 2 || time.some((item, index) => index > 0 && item <= time[index - 1])) {
    throw new RangeError(`run ${runIndex} time must be strictly increasing`);
  }
  for (const key of TRACE_KEYS.slice(1)) {
    finiteArray(trace[key], `run ${runIndex} ${key}`, time.length);
  }
  if (!Array.isArray(trace.force_on_club_n) || trace.force_on_club_n.length !== time.length) {
    throw new RangeError(`run ${runIndex} grip forces must share the trace length`);
  }
  trace.force_on_club_n.forEach((sample, sampleIndex) => {
    if (!Array.isArray(sample) || sample.length !== 2) {
      throw new RangeError(`run ${runIndex} grip sample ${sampleIndex} must retain two hands`);
    }
    sample.forEach((hand, handIndex) => {
      if (!Array.isArray(hand) || hand.length !== 2) {
        throw new RangeError(`run ${runIndex} hand ${handIndex} must retain two planar components`);
      }
      hand.forEach((component, componentIndex) =>
        finite(component, `run ${runIndex} grip ${sampleIndex}/${handIndex}/${componentIndex}`),
      );
    });
  });
  return trace as unknown as RotatingBaseRunTrace;
}

function validateRun(value: unknown, index: number): RotatingBaseRun {
  const run = record(value, `run ${index}`);
  if (run.schema_id !== "swing-sim/rotating-base-run-result" || run.schema_version !== 1) {
    throw new RangeError(`run ${index} schema is unqualified`);
  }
  if (run.source_revision !== ROTATING_BASE_SOURCE_REVISION || run.model_tier !== ROTATING_BASE_MODEL_TIER) {
    throw new RangeError(`run ${index} authority is unqualified`);
  }
  const boundaries = record(run.boundaries, `run ${index} boundaries`);
  if (
    Object.keys(boundaries).sort().join(",") !==
      "coaching_recommendation,coordinate_semantics,human_validation" ||
    boundaries.coordinate_semantics !== ROTATING_BASE_BOUNDARIES.coordinate_semantics ||
    boundaries.human_validation !== ROTATING_BASE_BOUNDARIES.human_validation ||
    boundaries.coaching_recommendation !== ROTATING_BASE_BOUNDARIES.coaching_recommendation
  ) {
    throw new RangeError(`run ${index} scientific boundaries changed`);
  }
  const authority = ROTATING_BASE_STUDY.cases[index];
  const request = record(run.request, `run ${index} request`);
  const retainedCase = record(run.case, `run ${index} case`);
  validateIdentity(request, authority, `run ${index} request`);
  validateIdentity(retainedCase, authority, `run ${index} case`);
  if (
    retainedCase.valid !== authority.valid ||
    JSON.stringify(retainedCase.exclusion_reasons) !== JSON.stringify(authority.exclusion_reasons)
  ) {
    throw new RangeError(`run ${index} adverse-case status changed`);
  }
  const metrics = record(retainedCase.metrics, `run ${index} metrics`);
  for (const key of METRIC_KEYS) {
    const actual = finite(metrics[key], `run ${index} ${key}`);
    if (Math.abs(actual - authority[key]) > 1e-10) {
      throw new RangeError(`run ${index} ${key} does not match the qualified study`);
    }
  }
  validateTrace(run.trace, index);
  return run as unknown as RotatingBaseRun;
}

export function validateRotatingBaseRunCatalog(value: unknown): RotatingBaseRunCatalog {
  const catalog = record(value, "rotating-base run catalog");
  if (catalog.schema_id !== "swing-sim/rotating-base-run-catalog" || catalog.schema_version !== 1) {
    throw new RangeError("rotating-base run catalog schema is unqualified");
  }
  if (
    catalog.source_revision !== ROTATING_BASE_SOURCE_REVISION ||
    catalog.study_sha256 !== ROTATING_BASE_STUDY_SHA256 ||
    catalog.model_tier !== ROTATING_BASE_MODEL_TIER
  ) {
    throw new RangeError("rotating-base run catalog authority is unqualified");
  }
  if (catalog.attempted_run_count !== 18 || !Array.isArray(catalog.runs) || catalog.runs.length !== 18) {
    throw new RangeError("rotating-base run catalog must retain all 18 runs");
  }
  return {
    ...(catalog as unknown as RotatingBaseRunCatalog),
    runs: catalog.runs.map(validateRun),
  };
}

export function registeredRun(
  catalog: RotatingBaseRunCatalog,
  profile: TorsoProfile,
  matchingRule: MatchingRule,
  torsoRateRadS: number,
): RotatingBaseRun {
  const selected = catalog.runs.find(
    (run) =>
      run.request.torso_profile === profile &&
      run.request.matching_rule === matchingRule &&
      run.request.initial_torso_rate_rad_s === torsoRateRadS,
  );
  if (!selected) throw new RangeError("selection is outside the registered run catalog");
  return selected;
}

export const ROTATING_BASE_RUN_CATALOG = validateRotatingBaseRunCatalog(rawRunCatalog);
