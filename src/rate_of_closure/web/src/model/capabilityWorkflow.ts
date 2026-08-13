/** Versioned authoring and persistence facade for capability optimization. */

import {
  parseOptimizationRequest,
  parsePlayerCapabilityProfile,
  type CapabilityObjective,
  type CapabilityParameter,
  type OptimizationRequest,
  type PlayerCapabilityProfile,
} from "./capabilityContract";
import type {
  CapabilityFlightEvaluatorConfig,
  CapabilitySpinDefault,
} from "./capabilityFlightEvaluator";

export const CAPABILITY_WORKFLOW_SCHEMA_VERSION =
  "capability-optimization-workflow/v1" as const;
export const MAX_WORKFLOW_OBSERVATIONS = 100_000;
const PROVENANCE = "rate-of-closure/capability-workflow/user-authored/v1";
const AUTHORING_BOUNDS: ReadonlyArray<readonly [keyof CapabilityWorkflowInputs, number, number]> = [
  ["ballSpeedMps", 1, 100], ["ballSpeedStdMps", 0, 30],
  ["launchAngleDeg", -10, 45], ["launchAngleStdDeg", 0, 30],
  ["launchDirectionDeg", -30, 30], ["launchDirectionStdDeg", 0, 30],
  ["totalSpinRpm", 0, 20_000], ["spinAxisTiltDeg", -90, 90],
  ["targetDistanceM", 0.1, 1_000], ["targetLateralM", -500, 500],
  ["targetRadiusM", 0.1, 500], ["maxTimeS", 0.001, 120],
  ["trajectorySampleIntervalS", 0.001, 0.1],
  ["candidateBudget", 1, MAX_WORKFLOW_OBSERVATIONS],
  ["ensembleSize", 1, MAX_WORKFLOW_OBSERVATIONS],
  ["alternativesCount", 1, MAX_WORKFLOW_OBSERVATIONS],
  ["seed", 0, 2 ** 31 - 1],
];

export interface CapabilityWorkflowInputs {
  readonly profileId: string; readonly clubId: string;
  readonly ballSpeedMps: number; readonly ballSpeedStdMps: number;
  readonly launchAngleDeg: number; readonly launchAngleStdDeg: number;
  readonly launchDirectionDeg: number; readonly launchDirectionStdDeg: number;
  readonly totalSpinRpm: number; readonly spinAxisTiltDeg: number;
  readonly targetDistanceM: number; readonly targetLateralM: number;
  readonly targetRadiusM: number; readonly objective: CapabilityObjective;
  readonly candidateBudget: number; readonly ensembleSize: number;
  readonly alternativesCount: number; readonly seed: number;
  readonly maxTimeS: number; readonly trajectorySampleIntervalS: number;
}

export interface CapabilityWorkflowDocument {
  readonly schemaVersion: typeof CAPABILITY_WORKFLOW_SCHEMA_VERSION;
  readonly profile: PlayerCapabilityProfile;
  readonly request: OptimizationRequest;
  readonly evaluatorConfig: CapabilityFlightEvaluatorConfig;
}

export const defaultCapabilityWorkflowInputs = (): CapabilityWorkflowInputs => ({
  profileId: "representative-driver-profile", clubId: "driver",
  ballSpeedMps: 67, ballSpeedStdMps: 1.5,
  launchAngleDeg: 12.5, launchAngleStdDeg: 1,
  launchDirectionDeg: 0, launchDirectionStdDeg: 1.5,
  totalSpinRpm: 2600, spinAxisTiltDeg: 0,
  targetDistanceM: 230, targetLateralM: 0, targetRadiusM: 12,
  objective: "maximize_target_hold", candidateBudget: 8,
  ensembleSize: 12, alternativesCount: 3, seed: 4197,
  maxTimeS: 10, trajectorySampleIntervalS: 0.01,
});

const finite = (value: unknown, name: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RangeError(`${name} must be finite`);
  }
  return value;
};

const text = (value: unknown, name: string): string => {
  if (typeof value !== "string" || !value.trim()) {
    throw new RangeError(`${name} must be nonempty text`);
  }
  return value.trim();
};

const validateAuthoringInputs = (input: CapabilityWorkflowInputs): void => {
  AUTHORING_BOUNDS.forEach(([key, minimum, maximum]) => {
    const value = input[key];
    if (typeof value !== "number" || !Number.isFinite(value) ||
        value < minimum || value > maximum) {
      throw new RangeError(`${key} must lie within [${minimum}, ${maximum}]`);
    }
  });
};

const parameter = (
  parameterId: string, unit: string, baseline: number, standardDeviation: number,
): CapabilityParameter => {
  const domains: Readonly<Record<string, readonly [number, number, number, number]>> = {
    ball_speed: [1, 100, 20, 90], launch_angle: [-10, 45, 0, 35],
    launch_direction: [-30, 30, -15, 15],
  };
  const [lowerBound, upperBound, evidenceLowerBound, evidenceUpperBound] =
    domains[parameterId];
  return Object.freeze({
    parameterId, unit, lowerBound, upperBound, evidenceLowerBound,
    evidenceUpperBound, baseline: finite(baseline, `${parameterId} baseline`),
    bias: 0, standardDeviation: finite(
      standardDeviation, `${parameterId} standard deviation`),
  });
};

const profile = (input: CapabilityWorkflowInputs): PlayerCapabilityProfile => {
  const parameters = Object.freeze([
    parameter("ball_speed", "m/s", input.ballSpeedMps, input.ballSpeedStdMps),
    parameter("launch_angle", "deg", input.launchAngleDeg, input.launchAngleStdDeg),
    parameter("launch_direction", "deg", input.launchDirectionDeg, input.launchDirectionStdDeg),
  ]);
  return parsePlayerCapabilityProfile({
    schema_version: "player-capability-profile/v1",
    profile_id: text(input.profileId, "profileId"), provenance: PROVENANCE,
    confidence: 0.8, clubs: [{ club_id: text(input.clubId, "clubId"),
      parameters: parameters.map(parameterWire), matrix_kind: "correlation",
      matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      provenance: PROVENANCE, confidence: 0.8 }],
  });
};

const parameterWire = (item: CapabilityParameter): Record<string, unknown> => ({
  parameter_id: item.parameterId, unit: item.unit,
  lower_bound: item.lowerBound, upper_bound: item.upperBound,
  evidence_lower_bound: item.evidenceLowerBound,
  evidence_upper_bound: item.evidenceUpperBound, baseline: item.baseline,
  bias: item.bias, standard_deviation: item.standardDeviation,
});

const request = (
  input: CapabilityWorkflowInputs, clubId: string,
): OptimizationRequest => parseOptimizationRequest({
  schema_version: "capability-optimization-request/v1",
  problem_id: `capability-${text(input.profileId, "profileId")}`,
  objective: input.objective, club_ids: [clubId],
  target: { kind: "green", distance_m: finite(input.targetDistanceM, "targetDistanceM"),
    lateral_m: finite(input.targetLateralM, "targetLateralM"),
    radius_m: finite(input.targetRadiusM, "targetRadiusM"),
    band_half_length_m: finite(input.targetRadiusM, "targetRadiusM"),
    half_width_m: finite(input.targetRadiusM, "targetRadiusM") },
  candidate_budget: input.candidateBudget, ensemble_size: input.ensembleSize,
  alternatives_count: input.alternativesCount, seed: input.seed,
  cvar_alpha: 0.9, minimum_success_fraction: 0.8,
});

const evaluatorConfig = (
  input: CapabilityWorkflowInputs, clubId: string,
): CapabilityFlightEvaluatorConfig => Object.freeze({
  maxTimeS: finite(input.maxTimeS, "maxTimeS"),
  trajectorySampleIntervalS: finite(
    input.trajectorySampleIntervalS, "trajectorySampleIntervalS"),
  spinDefaults: Object.freeze([Object.freeze({ clubId,
    totalSpinRpm: finite(input.totalSpinRpm, "totalSpinRpm"),
    spinAxisTiltDeg: finite(input.spinAxisTiltDeg, "spinAxisTiltDeg"),
    provenance: PROVENANCE })]),
});

const validateDocument = (
  document: CapabilityWorkflowDocument,
): CapabilityWorkflowDocument => {
  const available = new Set(document.profile.clubs.map(({ clubId }) => clubId));
  if (document.request.clubIds.some((clubId) => !available.has(clubId))) {
    throw new RangeError("request clubIds must exist in the profile");
  }
  const requested = new Set(document.request.clubIds);
  const spinIds = document.evaluatorConfig.spinDefaults.map(({ clubId }) => clubId);
  if (new Set(spinIds).size !== spinIds.length) {
    throw new RangeError("spin default clubIds must be unique");
  }
  const spinClubs = new Set(
    spinIds,
  );
  if (spinClubs.size !== requested.size ||
      [...spinClubs].some((clubId) => !requested.has(clubId))) {
    throw new RangeError("spin default clubIds must exactly match request clubIds");
  }
  const attempts = document.request.candidateBudget * document.request.ensembleSize;
  if (attempts > MAX_WORKFLOW_OBSERVATIONS) {
    throw new RangeError(`workflow may not exceed ${MAX_WORKFLOW_OBSERVATIONS} observations`);
  }
  if (document.evaluatorConfig.maxTimeS <= 0) {
    throw new RangeError("maxTimeS must be > 0");
  }
  const interval = document.evaluatorConfig.trajectorySampleIntervalS;
  if (interval < 0.001 || interval > 0.1) {
    throw new RangeError("trajectorySampleIntervalS must lie within [0.001, 0.1]");
  }
  if (Math.abs(interval / 0.001 - Math.round(interval / 0.001)) > 1e-9) {
    throw new RangeError("trajectorySampleIntervalS must align to the 0.001 s step");
  }
  if (document.evaluatorConfig.spinDefaults.some(({ totalSpinRpm }) =>
    totalSpinRpm < 0)) {
    throw new RangeError("totalSpinRpm must lie within [0, Infinity]");
  }
  if (document.evaluatorConfig.spinDefaults.some(({ spinAxisTiltDeg }) =>
    spinAxisTiltDeg < -90 || spinAxisTiltDeg > 90)) {
    throw new RangeError("spinAxisTiltDeg must lie within [-90, 90]");
  }
  if (document.request.alternativesCount > document.request.candidateBudget) {
    throw new RangeError("alternativesCount must not exceed candidateBudget");
  }
  validateAuthoringInputs(capabilityWorkflowInputs(document));
  return Object.freeze(document);
};

export function buildCapabilityWorkflow(
  input: CapabilityWorkflowInputs,
): CapabilityWorkflowDocument {
  if (input.ballSpeedMps <= 0) {
    throw new RangeError("ballSpeedMps must be greater than zero");
  }
  if (input.alternativesCount > input.candidateBudget) {
    throw new RangeError("alternativesCount must not exceed candidateBudget");
  }
  const parsedProfile = profile(input);
  const clubId = parsedProfile.clubs[0].clubId;
  return validateDocument({ schemaVersion: CAPABILITY_WORKFLOW_SCHEMA_VERSION,
    profile: parsedProfile, request: request(input, clubId),
    evaluatorConfig: evaluatorConfig(input, clubId) });
}

const profileWire = (value: PlayerCapabilityProfile): Record<string, unknown> => ({
  schema_version: value.schemaVersion, profile_id: value.profileId,
  provenance: value.provenance, confidence: value.confidence,
  clubs: value.clubs.map((club) => ({ club_id: club.clubId,
    parameters: club.parameters.map(parameterWire), matrix_kind: club.matrixKind,
    matrix: club.matrix, provenance: club.provenance, confidence: club.confidence })),
});

const requestWire = (value: OptimizationRequest): Record<string, unknown> => ({
  schema_version: value.schemaVersion, problem_id: value.problemId,
  objective: value.objective, club_ids: value.clubIds,
  target: { kind: value.target.kind, distance_m: value.target.distanceM,
    lateral_m: value.target.lateralM, radius_m: value.target.radiusM,
    band_half_length_m: value.target.bandHalfLengthM,
    half_width_m: value.target.halfWidthM },
  candidate_budget: value.candidateBudget, ensemble_size: value.ensembleSize,
  alternatives_count: value.alternativesCount, seed: value.seed,
  cvar_alpha: value.cvarAlpha,
  minimum_success_fraction: value.minimumSuccessFraction,
});

const configWire = (value: CapabilityFlightEvaluatorConfig): Record<string, unknown> => ({
  max_time_s: value.maxTimeS,
  trajectory_sample_interval_s: value.trajectorySampleIntervalS,
  spin_defaults: value.spinDefaults.map((item) => ({ club_id: item.clubId,
    total_spin_rpm: item.totalSpinRpm, spin_axis_tilt_deg: item.spinAxisTiltDeg,
    provenance: item.provenance })),
});

const exact = (value: Record<string, unknown>, fields: readonly string[], name: string): void => {
  if (Object.keys(value).sort().join("|") !== [...fields].sort().join("|")) {
    throw new RangeError(`${name} fields do not match v1 schema`);
  }
};

const record = (value: unknown, name: string): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new RangeError(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
};

const parseConfig = (value: unknown): CapabilityFlightEvaluatorConfig => {
  const source = record(value, "evaluator_config");
  exact(source, ["max_time_s", "spin_defaults", "trajectory_sample_interval_s"], "evaluator_config");
  if (!Array.isArray(source.spin_defaults)) throw new RangeError("spin_defaults must be an array");
  const spinDefaults = source.spin_defaults.map((item): CapabilitySpinDefault => {
    const spin = record(item, "spin default");
    exact(spin, ["club_id", "provenance", "spin_axis_tilt_deg", "total_spin_rpm"], "spin default");
    return Object.freeze({ clubId: text(spin.club_id, "clubId"),
      totalSpinRpm: finite(spin.total_spin_rpm, "totalSpinRpm"),
      spinAxisTiltDeg: finite(spin.spin_axis_tilt_deg, "spinAxisTiltDeg"),
      provenance: text(spin.provenance, "provenance") });
  });
  return Object.freeze({ maxTimeS: finite(source.max_time_s, "maxTimeS"),
    trajectorySampleIntervalS: finite(
      source.trajectory_sample_interval_s, "trajectorySampleIntervalS"),
    spinDefaults: Object.freeze(spinDefaults) });
};

export const capabilityWorkflowToJson = (
  document: CapabilityWorkflowDocument,
): string => JSON.stringify({ evaluator_config: configWire(document.evaluatorConfig),
  profile: profileWire(document.profile), request: requestWire(document.request),
  schema_version: document.schemaVersion });

export function capabilityWorkflowFromJson(source: string): CapabilityWorkflowDocument {
  const value: unknown = JSON.parse(source);
  const payload = record(value, "capability workflow");
  exact(payload, ["evaluator_config", "profile", "request", "schema_version"], "capability workflow");
  if (payload.schema_version !== CAPABILITY_WORKFLOW_SCHEMA_VERSION) {
    throw new RangeError("unsupported capability workflow schema_version");
  }
  return validateDocument({ schemaVersion: CAPABILITY_WORKFLOW_SCHEMA_VERSION,
    profile: parsePlayerCapabilityProfile(payload.profile),
    request: parseOptimizationRequest(payload.request),
    evaluatorConfig: parseConfig(payload.evaluator_config) });
}

/** Project a strict persisted single-club document back into editable inputs. */
export function capabilityWorkflowInputs(
  document: CapabilityWorkflowDocument,
): CapabilityWorkflowInputs {
  if (document.profile.clubs.length !== 1 ||
      document.evaluatorConfig.spinDefaults.length !== 1) {
    throw new RangeError("interactive workflow supports exactly one club and spin default");
  }
  const club = document.profile.clubs[0];
  const parameters = new Map(club.parameters.map((item) => [item.parameterId, item]));
  const ballSpeed = parameters.get("ball_speed");
  const launchAngle = parameters.get("launch_angle");
  const launchDirection = parameters.get("launch_direction");
  if (!ballSpeed || !launchAngle || !launchDirection || parameters.size !== 3) {
    throw new RangeError("interactive workflow requires the three launch parameters");
  }
  const spin = document.evaluatorConfig.spinDefaults[0];
  const target = document.request.target;
  return Object.freeze({ profileId: document.profile.profileId, clubId: club.clubId,
    ballSpeedMps: ballSpeed.baseline, ballSpeedStdMps: ballSpeed.standardDeviation,
    launchAngleDeg: launchAngle.baseline, launchAngleStdDeg: launchAngle.standardDeviation,
    launchDirectionDeg: launchDirection.baseline,
    launchDirectionStdDeg: launchDirection.standardDeviation,
    totalSpinRpm: spin.totalSpinRpm, spinAxisTiltDeg: spin.spinAxisTiltDeg,
    targetDistanceM: target.distanceM, targetLateralM: target.lateralM,
    targetRadiusM: target.radiusM, objective: document.request.objective,
    candidateBudget: document.request.candidateBudget,
    ensembleSize: document.request.ensembleSize,
    alternativesCount: document.request.alternativesCount, seed: document.request.seed,
    maxTimeS: document.evaluatorConfig.maxTimeS,
    trajectorySampleIntervalS: document.evaluatorConfig.trajectorySampleIntervalS });
}
