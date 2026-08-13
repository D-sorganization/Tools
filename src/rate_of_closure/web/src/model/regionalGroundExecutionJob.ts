/** Strict UI-neutral orchestration contract for seeded regional-ground jobs. */
import { ballSetupFromJson, ballSetupToJson } from "./ballSetup";
import { canonicalGroundJson } from "./flightGroundContract";
import type {
  GroundCalibration,
  GroundProvenance,
  GroundSurfaceProfile,
  GroundVec3,
} from "./flightGroundTypes";
import {
  exact,
  finiteRaw,
  integer,
  oneOf,
  parseCalibration,
  parseProvenance,
  parseSurface,
  positive,
  record,
  text,
  vector,
} from "./flightGroundValidation";
import {
  regionalGroundVariationRequestFromJson,
  stableRegionalGroundVariationRequestJson,
} from "./regionalGroundVariationRequestWire";
import {
  parseQualifiedExecutionAuthority,
  type RegionalExecutionOptionsWire,
} from "./regionalGroundExecutionQualification";
import type { GroundRegionalMaterialPlanRequest } from "./groundRegionalPlan";
import { sha256Text } from "./sha256";
import { parseUniqueJson } from "./strictJson";

export const REGIONAL_GROUND_EXECUTION_JOB_SCHEMA_VERSION =
  "rate-of-closure/regional-ground-execution-job/v1" as const;
export const MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES = 1_048_576;
const FLIGHT_FRAME = "flight_frame:x_forward,y_left,z_up" as const;
const MAX_CAPTURE_SPEED_M_S = 100;
type WireObject = Readonly<Record<string, unknown>>;

export interface ExecutionJobLaunch {
  readonly frame: typeof FLIGHT_FRAME;
  readonly ball_speed_m_s: number;
  readonly launch_angle_rad: number;
  readonly azimuth_angle_rad: number;
  readonly spin_rate_rpm: number;
  readonly spin_axis_unit: GroundVec3;
  readonly ball_mass_kg: number;
  readonly ball_radius_m: number;
  readonly air_density_kg_m3: number;
  readonly gravity_m_s2: number;
  readonly wind_speed_m_s: number;
  readonly wind_direction_rad: number;
  readonly ball_setup: WireObject;
}

export interface ExecutionJobFlight {
  readonly model_id: string;
  readonly model_version: string;
  readonly settings: Readonly<Record<string, number>>;
  readonly trajectory_sha256: string;
  readonly result_sha256: string;
}

export interface ExecutionJobTransfer {
  readonly request_id: string;
  readonly surface: GroundSurfaceProfile;
  readonly calibration: GroundCalibration;
  readonly provenance: GroundProvenance;
  readonly max_time_s: number;
  readonly output_interval_s: number;
  readonly max_events: number;
  readonly rotational_inertia_factor: number;
  readonly surface_sha256: string;
  readonly settings_sha256: string;
}

export interface GroundExecutionOptions {
  readonly max_trials: number;
}

export interface RegionalGroundExecutionJob {
  readonly schema_version: typeof REGIONAL_GROUND_EXECUTION_JOB_SCHEMA_VERSION;
  readonly unit_system: "SI";
  readonly job_id: string;
  readonly launch: ExecutionJobLaunch;
  readonly flight: ExecutionJobFlight;
  readonly transfer: ExecutionJobTransfer;
  readonly capture_speed_m_s: number;
  readonly execution_options: GroundExecutionOptions;
  readonly regional_execution_options: RegionalExecutionOptionsWire;
  readonly qualified_regional_plan: GroundRegionalMaterialPlanRequest;
  readonly qualified_plan_sha256: string;
  readonly variation_request: WireObject;
  readonly input_sha256: string;
  readonly provenance: GroundProvenance;
  readonly job_sha256: string;
}

const ROOT_FIELDS = [
  "schema_version", "unit_system", "job_id", "launch", "flight", "transfer",
  "capture_speed_m_s", "execution_options", "variation_request",
  "regional_execution_options", "qualified_regional_plan",
  "qualified_plan_sha256",
  "input_sha256", "provenance", "job_sha256",
] as const;
const LAUNCH_FIELDS = [
  "frame", "ball_speed_m_s", "launch_angle_rad", "azimuth_angle_rad",
  "spin_rate_rpm", "spin_axis_unit", "ball_mass_kg", "ball_radius_m",
  "air_density_kg_m3", "gravity_m_s2", "wind_speed_m_s",
  "wind_direction_rad", "ball_setup",
] as const;
const BALL_SETUP_FIELDS = [
  "support_mode", "tee_height_m", "height_reference", "ball_center_m",
] as const;
const FLIGHT_FIELDS = [
  "model_id", "model_version", "settings", "trajectory_sha256", "result_sha256",
] as const;
const TRANSFER_FIELDS = [
  "request_id", "surface", "calibration", "provenance", "max_time_s",
  "output_interval_s", "max_events", "rotational_inertia_factor",
  "surface_sha256", "settings_sha256",
] as const;
const EXECUTION_FIELDS = ["max_trials"] as const;

const digest = (value: unknown, name: string): string => {
  const parsed = text(value, name);
  if (!/^[0-9a-f]{64}$/.test(parsed)) {
    throw new RangeError(name + " must be 64 lowercase hexadecimal characters");
  }
  return parsed;
};

const boundedPositive = (value: unknown, name: string, maximum: number): number => {
  const parsed = positive(value, name);
  if (parsed > maximum) throw new RangeError(name + " exceeds " + maximum);
  return parsed;
};

const stableId = (value: unknown, name: string): string => {
  const parsed = text(value, name);
  if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(parsed)) {
    throw new RangeError(name + " must be a stable identifier");
  }
  return parsed;
};

const deepFreeze = <T>(value: T): T => {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.values(value as Record<string, unknown>).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
};

const parseBallSetupWire = (value: unknown): WireObject => {
  const item = record(value, "ball_setup");
  exact(item, BALL_SETUP_FIELDS, "ball_setup");
  oneOf(item.support_mode, ["ground", "tee"] as const, "support_mode");
  finiteRaw(item.tee_height_m, "tee_height_m");
  vector(item.ball_center_m, "ball_center_m");
  const parsed = ballSetupFromJson(item);
  return deepFreeze(ballSetupToJson(parsed));
};

const parseLaunch = (value: unknown): ExecutionJobLaunch => {
  const item = record(value, "launch");
  exact(item, LAUNCH_FIELDS, "launch");
  const spinAxis = vector(item.spin_axis_unit, "spin_axis_unit");
  if (Math.abs(Math.hypot(...spinAxis) - 1) > 1e-6) {
    throw new RangeError("spin_axis_unit must be a unit vector");
  }
  const launchAngle = finiteRaw(item.launch_angle_rad, "launch_angle_rad");
  if (Math.abs(launchAngle) > Math.PI / 2) {
    throw new RangeError("launch_angle_rad must lie within [-pi/2, pi/2]");
  }
  const spinRate = finiteRaw(item.spin_rate_rpm, "spin_rate_rpm");
  const windSpeed = finiteRaw(item.wind_speed_m_s, "wind_speed_m_s");
  const ballSpeed = finiteRaw(item.ball_speed_m_s, "ball_speed_m_s");
  if (ballSpeed < 0 || spinRate < 0 || windSpeed < 0) {
    throw new RangeError("ball speed, spin rate, and wind speed must be nonnegative");
  }
  return deepFreeze({
    frame: oneOf(item.frame, [FLIGHT_FRAME] as const, "launch frame"),
    ball_speed_m_s: ballSpeed,
    launch_angle_rad: launchAngle,
    azimuth_angle_rad: finiteRaw(item.azimuth_angle_rad, "azimuth_angle_rad"),
    spin_rate_rpm: spinRate,
    spin_axis_unit: spinAxis,
    ball_mass_kg: positive(item.ball_mass_kg, "ball_mass_kg"),
    ball_radius_m: positive(item.ball_radius_m, "ball_radius_m"),
    air_density_kg_m3: positive(item.air_density_kg_m3, "air_density_kg_m3"),
    gravity_m_s2: positive(item.gravity_m_s2, "gravity_m_s2"),
    wind_speed_m_s: windSpeed,
    wind_direction_rad: finiteRaw(item.wind_direction_rad, "wind_direction_rad"),
    ball_setup: parseBallSetupWire(item.ball_setup),
  });
};

const parseFlight = (value: unknown): ExecutionJobFlight => {
  const item = record(value, "flight");
  exact(item, FLIGHT_FIELDS, "flight");
  const rawSettings = record(item.settings, "flight settings");
  const entries = Object.entries(rawSettings);
  if (entries.length < 1 || entries.length > 64) {
    throw new RangeError("flight settings must contain between 1 and 64 values");
  }
  const settings = Object.fromEntries(entries.map(([key, setting]) => [
    stableId(key, "flight setting id"),
    finiteRaw(setting, "flight setting " + key),
  ]));
  return deepFreeze({
    model_id: text(item.model_id, "model_id"),
    model_version: text(item.model_version, "model_version"),
    settings,
    trajectory_sha256: digest(item.trajectory_sha256, "trajectory_sha256"),
    result_sha256: digest(item.result_sha256, "result_sha256"),
  });
};

const transferAuthority = (transfer: Omit<ExecutionJobTransfer,
  "surface_sha256" | "settings_sha256">): WireObject => transfer;

const parseTransfer = (value: unknown): ExecutionJobTransfer => {
  const item = record(value, "transfer");
  exact(item, TRANSFER_FIELDS, "transfer");
  const maxTime = boundedPositive(item.max_time_s, "transfer max_time_s", 3_600);
  const interval = boundedPositive(
    item.output_interval_s,
    "transfer output_interval_s",
    maxTime,
  );
  const inertia = boundedPositive(
    item.rotational_inertia_factor,
    "rotational_inertia_factor",
    1,
  );
  const authority = deepFreeze({
    request_id: text(item.request_id, "transfer request_id"),
    surface: parseSurface(item.surface),
    calibration: parseCalibration(item.calibration),
    provenance: parseProvenance(item.provenance),
    max_time_s: maxTime,
    output_interval_s: interval,
    max_events: integer(item.max_events, "transfer max_events", 1),
    rotational_inertia_factor: inertia,
  });
  const surfaceSha = sha256Text(canonicalGroundJson(authority.surface));
  if (digest(item.surface_sha256, "surface_sha256") !== surfaceSha) {
    throw new RangeError("surface_sha256 must match the embedded surface authority");
  }
  const settingsSha = sha256Text(canonicalGroundJson(transferAuthority(authority)));
  if (digest(item.settings_sha256, "settings_sha256") !== settingsSha) {
    throw new RangeError("settings_sha256 must match the transfer settings authority");
  }
  return deepFreeze({
    ...authority,
    surface_sha256: surfaceSha,
    settings_sha256: settingsSha,
  });
};

const parseExecutionOptions = (value: unknown): GroundExecutionOptions => {
  const item = record(value, "execution_options");
  exact(item, EXECUTION_FIELDS, "execution_options");
  return Object.freeze({
    max_trials: integer(item.max_trials, "max_trials", 1),
  });
};

const parseVariationWire = (value: unknown): WireObject => {
  const parsed = regionalGroundVariationRequestFromJson(canonicalGroundJson(value));
  return deepFreeze(JSON.parse(stableRegionalGroundVariationRequestJson(parsed)) as WireObject);
};

const inputPayload = (job: RegionalGroundExecutionJob): WireObject => ({
  launch: job.launch,
  flight: job.flight,
  transfer: job.transfer,
  capture_speed_m_s: job.capture_speed_m_s,
  execution_options: job.execution_options,
  regional_execution_options: job.regional_execution_options,
  qualified_regional_plan: job.qualified_regional_plan,
  qualified_plan_sha256: job.qualified_plan_sha256,
  variation_request: job.variation_request,
});

const jobPayload = (job: RegionalGroundExecutionJob, includeDigest: boolean): WireObject => ({
  schema_version: job.schema_version,
  unit_system: job.unit_system,
  job_id: job.job_id,
  ...inputPayload(job),
  input_sha256: job.input_sha256,
  provenance: job.provenance,
  ...(includeDigest ? { job_sha256: job.job_sha256 } : {}),
});

const validateCrossContractAuthority = (job: RegionalGroundExecutionJob): void => {
  const variationPlan = record(
    job.variation_request.variation_plan,
    "variation_plan",
  );
  if (variationPlan.flight_model !== job.flight.model_id) {
    throw new RangeError("flight model_id must match variation request flight_model");
  }
};

/** Parse and deeply freeze one exact job without executing any physics. */
export const parseRegionalGroundExecutionJob = (
  value: unknown,
): RegionalGroundExecutionJob => {
  const item = record(value, "regional-ground execution job");
  exact(item, ROOT_FIELDS, "regional-ground execution job");
  const options = parseExecutionOptions(item.execution_options);
  const variation = parseVariationWire(item.variation_request);
  const variationPlan = record(variation.variation_plan, "variation_plan");
  if (options.max_trials !== integer(variationPlan.n_runs, "variation n_runs", 1)) {
    throw new RangeError("max_trials must equal variation request n_runs");
  }
  const launch = parseLaunch(item.launch);
  const transfer = parseTransfer(item.transfer);
  const sourcePlan = record(variation.regional_plan, "regional_plan");
  const qualified = parseQualifiedExecutionAuthority(
    item.regional_execution_options,
    item.qualified_regional_plan,
    item.qualified_plan_sha256,
    sourcePlan as unknown as GroundRegionalMaterialPlanRequest,
    launch,
    transfer.surface,
  );
  const parsed = {
    schema_version: oneOf(
      item.schema_version,
      [REGIONAL_GROUND_EXECUTION_JOB_SCHEMA_VERSION] as const,
      "schema_version",
    ),
    unit_system: oneOf(item.unit_system, ["SI"] as const, "unit_system"),
    job_id: stableId(item.job_id, "job_id"),
    launch,
    flight: parseFlight(item.flight),
    transfer,
    capture_speed_m_s: boundedPositive(
      item.capture_speed_m_s,
      "capture_speed_m_s",
      MAX_CAPTURE_SPEED_M_S,
    ),
    execution_options: options,
    regional_execution_options: qualified.options,
    qualified_regional_plan: qualified.plan,
    qualified_plan_sha256: qualified.planSha256,
    variation_request: variation,
    input_sha256: digest(item.input_sha256, "input_sha256"),
    provenance: parseProvenance(item.provenance),
    job_sha256: digest(item.job_sha256, "job_sha256"),
  } satisfies RegionalGroundExecutionJob;
  validateCrossContractAuthority(parsed);
  const expectedInput = sha256Text(canonicalGroundJson(inputPayload(parsed)));
  if (parsed.input_sha256 !== expectedInput ||
      parsed.provenance.input_sha256 !== expectedInput) {
    throw new RangeError("input_sha256 must match the embedded input authority");
  }
  const expectedJob = sha256Text(canonicalGroundJson(jobPayload(parsed, false)));
  if (parsed.job_sha256 !== expectedJob) {
    throw new RangeError("job_sha256 must match the complete job authority");
  }
  return deepFreeze(parsed);
};

/** Parse bounded strict UTF-8 JSON with duplicate-field rejection. */
export const regionalGroundExecutionJobFromJson = (
  textValue: string,
): RegionalGroundExecutionJob => {
  if (typeof textValue !== "string") {
    throw new TypeError("regional-ground execution job JSON must be text");
  }
  if (new TextEncoder().encode(textValue).byteLength >
      MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES) {
    throw new RangeError("regional-ground execution job exceeds maximum wire size");
  }
  return parseRegionalGroundExecutionJob(parseUniqueJson(textValue));
};

/** Serialize one validated job using the shared canonical numeric policy. */
export const stableRegionalGroundExecutionJobJson = (value: unknown): string => {
  const textValue = canonicalGroundJson(jobPayload(
    parseRegionalGroundExecutionJob(value),
    true,
  ));
  if (new TextEncoder().encode(textValue).byteLength >
      MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES) {
    throw new RangeError("regional-ground execution job exceeds maximum wire size");
  }
  return textValue;
};
