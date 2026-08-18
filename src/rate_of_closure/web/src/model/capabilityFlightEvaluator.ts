/** Model-backed capability evaluator for robust launch optimization. */

import { deriveFlightMetricResult, type FlightMetricInputs } from "./ballFlightMetrics";
import type {
  CapabilityEvaluator,
  ClubCapability,
  OptimizationRequest,
  PlayerCapabilityProfile,
} from "./capabilityContract";
import { simulateFlight, type Launch } from "./flight";
import { fromFlightFrame, scale, type Vec3 } from "./impactPhysics";
import type { EvaluatedMetric, SolverEvaluation } from "./inverseFlightContract";

export interface CapabilityFlightEvaluatorConfig {
  readonly maxTimeS: number;
  readonly trajectorySampleIntervalS: number;
  readonly spinDefaults: readonly CapabilitySpinDefault[];
}

export interface CapabilitySpinDefault {
  readonly clubId: string;
  readonly totalSpinRpm: number;
  readonly spinAxisTiltDeg: number;
  readonly provenance: string;
}

const MODEL_ID = "waterloo_penner";
const MODEL_VERSION = "waterloo-penner-coefficients/v1";
const PROVENANCE = `${MODEL_ID}:${MODEL_VERSION}:typescript-rk4`;
const INTEGRATION_STEP_S = 0.001;
const REQUIRED = Object.freeze(["ball_speed", "launch_angle", "launch_direction"]);
const OPTIONAL = Object.freeze(["total_spin", "spin_axis_tilt"]);
const UNITS: Readonly<Record<string, string>> = Object.freeze({
  ball_speed: "m/s", launch_angle: "deg", launch_direction: "deg",
  total_spin: "rpm", spin_axis_tilt: "deg",
});
const PHYSICAL_DOMAINS: Readonly<Record<string, readonly [number, number, boolean]>> =
  Object.freeze({
    ball_speed: [0, Number.POSITIVE_INFINITY, false],
    launch_angle: [-90, 90, true],
    launch_direction: [-180, 180, true],
    total_spin: [0, Number.POSITIVE_INFINITY, true],
    spin_axis_tilt: [-90, 90, true],
  });
const DEFAULT_CONFIG: CapabilityFlightEvaluatorConfig = Object.freeze({
  maxTimeS: 10, trajectorySampleIntervalS: 0.01,
  spinDefaults: Object.freeze([]),
});

interface Binding {
  readonly clubs: ReadonlyMap<string, ClubCapability>;
  readonly spinDefaults: ReadonlyMap<string, CapabilitySpinDefault>;
  readonly targetPositionM: Vec3;
  readonly config: CapabilityFlightEvaluatorConfig;
}

interface BoundLaunch {
  readonly value: Launch;
  readonly spinSource: string;
}

const finite = (value: unknown, name: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RangeError(`${name} must be finite`);
  }
  return value;
};

const physical = (parameterId: string, value: number): void => {
  const [lower, upper, inclusiveLower] = PHYSICAL_DOMAINS[parameterId];
  const lowerValid = inclusiveLower ? value >= lower : value > lower;
  if (lowerValid && value <= upper) return;
  if (parameterId === "ball_speed") {
    throw new RangeError("ball_speed must be greater than zero");
  }
  throw new RangeError(parameterId + " must lie within [" + lower + ", " + upper + "]");
};

const config = (
  overrides: Partial<CapabilityFlightEvaluatorConfig>,
): CapabilityFlightEvaluatorConfig => {
  const merged = { ...DEFAULT_CONFIG, ...overrides };
  const parsed = {
    maxTimeS: finite(merged.maxTimeS, "maxTimeS"),
    trajectorySampleIntervalS: finite(
      merged.trajectorySampleIntervalS, "trajectorySampleIntervalS"),
    spinDefaults: Object.freeze((merged.spinDefaults ?? []).map((item) => {
      if (!item.clubId.trim() || !item.provenance.trim()) {
        throw new RangeError("spin default clubId and provenance must be nonempty");
      }
      const parsedDefault = Object.freeze({
        clubId: item.clubId.trim(),
        totalSpinRpm: finite(item.totalSpinRpm, "totalSpinRpm"),
        spinAxisTiltDeg: finite(item.spinAxisTiltDeg, "spinAxisTiltDeg"),
        provenance: item.provenance.trim(),
      });
      physical("total_spin", parsedDefault.totalSpinRpm);
      physical("spin_axis_tilt", parsedDefault.spinAxisTiltDeg);
      return parsedDefault;
    })),
  };
  if (parsed.maxTimeS <= 0) throw new RangeError("maxTimeS must be > 0");
  if (parsed.trajectorySampleIntervalS < INTEGRATION_STEP_S ||
      parsed.trajectorySampleIntervalS > 0.1) {
    throw new RangeError("trajectorySampleIntervalS must lie within [0.001, 0.1]");
  }
  const ratio = parsed.trajectorySampleIntervalS / INTEGRATION_STEP_S;
  if (Math.abs(ratio - Math.round(ratio)) > 1e-9) {
    throw new RangeError("trajectorySampleIntervalS must align to the integration step");
  }
  const defaultIds = parsed.spinDefaults.map((item) => item.clubId);
  if (new Set(defaultIds).size !== defaultIds.length) {
    throw new RangeError("spin default clubIds must be unique");
  }
  return Object.freeze(parsed);
};

const validateClub = (
  club: ClubCapability,
  spinDefaults: ReadonlyMap<string, CapabilitySpinDefault>,
): void => {
  const declared = new Set(club.parameters.map((item) => item.parameterId));
  const missing = REQUIRED.filter((parameterId) => !declared.has(parameterId));
  if (missing.length > 0) throw new RangeError(`${club.clubId} is missing: ${missing.join(",")}`);
  if ([...declared].some((parameterId) => !REQUIRED.includes(parameterId) &&
    !OPTIONAL.includes(parameterId))) {
    throw new RangeError(`${club.clubId} declares unsupported capability parameters`);
  }
  const declaredSpin = OPTIONAL.filter((parameterId) => declared.has(parameterId));
  if (declaredSpin.length === 1) {
    throw new RangeError("total_spin and spin_axis_tilt must be declared together");
  }
  if (declaredSpin.length === 0 && !spinDefaults.has(club.clubId)) {
    throw new RangeError(club.clubId + " requires an explicit spin default");
  }
  club.parameters.forEach((item) => {
    if (item.unit !== UNITS[item.parameterId]) {
      throw new RangeError(`${item.parameterId} must use ${UNITS[item.parameterId]}, not ${item.unit}`);
    }
    physical(item.parameterId, item.lowerBound);
    physical(item.parameterId, item.upperBound);
  });
};

const binding = (
  profile: PlayerCapabilityProfile,
  request: OptimizationRequest,
  activeConfig: CapabilityFlightEvaluatorConfig,
): Binding => {
  const spinDefaults = new Map(
    activeConfig.spinDefaults.map((item) => [item.clubId, item] as const),
  );
  const clubs = new Map(request.clubIds.map((clubId) => {
    const club = profile.clubs.find((item) => item.clubId === clubId);
    if (!club) throw new RangeError(`unknown clubId: ${clubId}`);
    validateClub(club, spinDefaults);
    return [clubId, club] as const;
  }));
  return Object.freeze({
    clubs,
    spinDefaults,
    config: activeConfig,
    targetPositionM: [request.target.distanceM, 0, request.target.lateralM] as Vec3,
  });
};

const sample = (
  club: ClubCapability,
  values: Readonly<Record<string, number>>,
): Readonly<Record<string, number>> => {
  const parameters = new Map(club.parameters.map((item) => [item.parameterId, item]));
  if (Object.keys(values).sort().join("|") !== [...parameters.keys()].sort().join("|")) {
    throw new RangeError("capability sample fields do not match the club profile");
  }
  const parsed = Object.fromEntries(Object.entries(values).map(([key, value]) =>
    [key, finite(value, key)]));
  Object.entries(parsed).forEach(([parameterId, value]) => {
    const parameter = parameters.get(parameterId)!;
    physical(parameterId, value);
    if (value < parameter.lowerBound || value > parameter.upperBound) {
      throw new RangeError(`${parameterId} lies outside declared safe bounds`);
    }
  });
  return Object.freeze(parsed);
};

const launch = (
  values: Readonly<Record<string, number>>,
  spinDefault: CapabilitySpinDefault | undefined,
): BoundLaunch => {
  const direction = -values.launch_direction * Math.PI / 180;
  const usesDefault = values.total_spin === undefined;
  const tiltDeg = values.spin_axis_tilt ?? spinDefault!.spinAxisTiltDeg;
  const tilt = tiltDeg * Math.PI / 180;
  const value = Object.freeze({
    ballSpeedMps: values.ball_speed,
    launchAngleRad: values.launch_angle * Math.PI / 180,
    azimuthRad: direction,
    spinRpm: values.total_spin ?? spinDefault!.totalSpinRpm,
    // Positive target-frame tilt is toward fade/right: (0, -sin, cos).
    spinAxis: [0, -Math.cos(tilt), -Math.sin(tilt)] as Vec3,
  });
  const spinSource = usesDefault
    ? "fixed_club_default:" + spinDefault!.provenance
    : "sampled_profile";
  return Object.freeze({ value, spinSource });
};

const inputs = (launchValue: Launch, activeBinding: Binding): FlightMetricInputs => {
  const activeConfig = activeBinding.config;
  const sampleEvery = Math.round(activeConfig.trajectorySampleIntervalS / INTEGRATION_STEP_S);
  const flight = simulateFlight(launchValue, activeConfig.maxTimeS, INTEGRATION_STEP_S, sampleEvery);
  return Object.freeze({
    trajectory: Object.freeze(flight.trajectory.map((point) => Object.freeze({
      timeS: point.time, positionM: fromFlightFrame(point.position),
      velocityMps: fromFlightFrame(point.velocity),
    }))),
    spinVectorRpm: scale(fromFlightFrame(launchValue.spinAxis), launchValue.spinRpm),
    targetPositionM: activeBinding.targetPositionM,
  });
};

const crossedGround = (metricInputs: FlightMetricInputs): boolean =>
  metricInputs.trajectory.some((point, index) => index > 0 &&
    metricInputs.trajectory[index - 1].positionM[1] > 0 && point.positionM[1] <= 0);

const evaluation = (
  metricInputs: FlightMetricInputs,
  spinSource: string,
): SolverEvaluation => {
  const crossed = crossedGround(metricInputs);
  const result = deriveFlightMetricResult(metricInputs, {
    modelId: MODEL_ID, modelVersion: MODEL_VERSION,
    integrationStatus: crossed ? "complete" : "nonconverged",
    terminationReason: crossed ? "ground_crossing" : "max_time_reached",
    environment: { air_model: "standard", integrator: "typescript-rk4" },
    wind: { model: "still_air" }, uncertaintyStatus: "deterministic",
    frameId: "target_frame:x_downrange,y_up,z_right",
  });
  const unavailable = ["carry_distance", "carry_offline"].some((metricId) =>
    result.value(metricId).status === "unavailable");
  if (!crossed || unavailable) return Object.freeze({
    status: "nonconverged", metrics: Object.freeze([]),
    reason: "no_ground_crossing_before_max_time",
  });
  const metrics = result.values.filter((value) => typeof value.numeric === "number")
    .map((value): EvaluatedMetric => Object.freeze({
      metricId: value.metricId, value: value.numeric as number,
      provenance: `ball-flight-result/v1|${PROVENANCE}|spin:${spinSource}|${value.provenance}`,
    }));
  return Object.freeze({ status: "complete", metrics: Object.freeze(metrics), reason: null });
};

/** Bind a profile/request to an optimizer-compatible full-flight evaluator. */
export function makeCapabilityFlightEvaluator(
  profile: PlayerCapabilityProfile,
  request: OptimizationRequest,
  overrides: Partial<CapabilityFlightEvaluatorConfig> = {},
): CapabilityEvaluator {
  const activeBinding = binding(profile, request, config(overrides));
  return (clubId, parameters) => {
    const club = activeBinding.clubs.get(clubId);
    if (!club) throw new RangeError(`unknown requested clubId: ${clubId}`);
    const boundLaunch = launch(
      sample(club, parameters),
      activeBinding.spinDefaults.get(clubId),
    );
    return evaluation(
      inputs(boundLaunch.value, activeBinding),
      boundLaunch.spinSource,
    );
  };
}
