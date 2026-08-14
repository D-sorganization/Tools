/** Pure trajectory-to-result derivation for the canonical flight catalog. */

import type { Vec3 } from "./impactPhysics";
import {
  FLIGHT_METRIC_IDS,
  flightMetricCatalog,
  type FlightMetricId,
  type ValueStatus,
} from "./ballFlightMetricContract";

export type AvailabilityReason = "insufficient_trajectory" | "no_ground_crossing" |
  "zero_horizontal_speed" | "zero_spin" | "target_not_configured" |
  "ground_model_required";

export interface MetricTrajectoryPoint {
  readonly timeS: number;
  readonly positionM: Vec3;
  readonly velocityMps: Vec3;
}
export interface GroundModelResult {
  readonly modelId: string;
  readonly totalDistanceM: number;
  readonly rollDistanceM: number;
  readonly bounceCount: number;
  readonly finalOfflineM: number;
}
export interface FlightMetricInputs {
  readonly trajectory: readonly MetricTrajectoryPoint[];
  readonly spinVectorRpm: Vec3;
  readonly targetPositionM?: Vec3;
  readonly groundResult?: GroundModelResult;
}
export interface FlightRunManifest {
  readonly modelId: string;
  readonly modelVersion: string;
  readonly integrationStatus: string;
  readonly terminationReason: string;
  readonly environment: Readonly<Record<string, string>>;
  readonly wind: Readonly<Record<string, string>>;
  readonly uncertaintyStatus: string;
  readonly frameId: "target_frame:x_downrange,y_up,z_right";
}
export interface FlightMetricValue {
  readonly metricId: FlightMetricId;
  readonly status: ValueStatus;
  readonly numeric: number | Vec3 | null;
  readonly reason: AvailabilityReason | null;
  readonly provenance: string;
}
export interface FlightMetricResult {
  readonly schemaVersion: "ball-flight-result/v1";
  readonly manifest: FlightRunManifest;
  readonly values: readonly FlightMetricValue[];
  value(metricId: string): FlightMetricValue;
  scalar(metricId: string): number;
  vector(metricId: string): Vec3;
}

const MIN_SPEED = 1e-12;
const finite = (value: number, name: string): number => {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return value;
};
const vector = (value: Vec3, name: string): Vec3 => {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${name} must contain three finite components`);
  }
  return [value[0], value[1], value[2]];
};
const norm = (value: Vec3): number => Math.hypot(...value);
const degrees = (radians: number): number => radians * 180 / Math.PI;
const lerpVector = (first: Vec3, second: Vec3, fraction: number): Vec3 => [
  first[0] + fraction * (second[0] - first[0]),
  first[1] + fraction * (second[1] - first[1]),
  first[2] + fraction * (second[2] - first[2]),
];

const validateManifest = (manifest: FlightRunManifest): FlightRunManifest => {
  const required = [manifest.modelId, manifest.modelVersion, manifest.integrationStatus,
    manifest.terminationReason, manifest.uncertaintyStatus, manifest.frameId];
  if (required.some((value) => value.trim() === "")) throw new RangeError("manifest fields must be nonempty");
  for (const [section, values] of [["environment", manifest.environment], ["wind", manifest.wind]] as const) {
    if (Object.entries(values).some(([key, value]) => key.trim() === "" || value.trim() === "")) {
      throw new RangeError(`${section} keys and values must be nonempty`);
    }
  }
  return Object.freeze({
    ...manifest,
    environment: Object.freeze(Object.fromEntries(Object.entries(manifest.environment).sort())),
    wind: Object.freeze(Object.fromEntries(Object.entries(manifest.wind).sort())),
  });
};

const validateGround = (ground: GroundModelResult | undefined): GroundModelResult | undefined => {
  if (!ground) return undefined;
  if (ground.modelId.trim() === "") throw new RangeError("ground modelId must be nonempty");
  const values = [ground.totalDistanceM, ground.rollDistanceM, ground.bounceCount, ground.finalOfflineM];
  if (values.some((value) => !Number.isFinite(value))) throw new RangeError("ground outputs must be finite");
  if (!Number.isInteger(ground.bounceCount)) throw new RangeError("bounceCount must be an integer");
  if (ground.totalDistanceM < 0 || ground.rollDistanceM < 0 || ground.bounceCount < 0) {
    throw new RangeError("ground distances and bounceCount must be nonnegative");
  }
  return Object.freeze({ ...ground });
};

const validateInputs = (inputs: FlightMetricInputs): FlightMetricInputs => {
  const trajectory = inputs.trajectory.map((point, index) => {
    finite(point.timeS, "timeS");
    if (point.timeS < 0) throw new RangeError("timeS must be nonnegative");
    if (index > 0 && point.timeS <= inputs.trajectory[index - 1].timeS) {
      throw new RangeError("trajectory times must be strictly increasing");
    }
    return Object.freeze({
      timeS: point.timeS,
      positionM: vector(point.positionM, "positionM"),
      velocityMps: vector(point.velocityMps, "velocityMps"),
    });
  });
  return Object.freeze({
    trajectory: Object.freeze(trajectory), spinVectorRpm: vector(inputs.spinVectorRpm, "spinVectorRpm"),
    targetPositionM: inputs.targetPositionM ? vector(inputs.targetPositionM, "targetPositionM") : undefined,
    groundResult: validateGround(inputs.groundResult),
  });
};

const available = (
  metricId: FlightMetricId, numeric: number | Vec3, provenance: string,
): FlightMetricValue => Object.freeze({
  metricId, status: flightMetricCatalog().definition(metricId).defaultStatus,
  numeric, reason: null, provenance,
});
const unavailable = (
  metricId: FlightMetricId, reason: AvailabilityReason,
): FlightMetricValue => Object.freeze({
  metricId, status: "unavailable", numeric: null, reason,
  provenance: flightMetricCatalog().definition(metricId).provenance,
});

interface LandingState { point: MetricTrajectoryPoint; segmentEnd: number }
const landingState = (points: readonly MetricTrajectoryPoint[]): LandingState | null => {
  let wasAirborne = false;
  for (let index = 1; index < points.length; index += 1) {
    const first = points[index - 1];
    const second = points[index];
    wasAirborne ||= first.positionM[1] > 0;
    if (!wasAirborne || second.positionM[1] > 0) continue;
    const denominator = first.positionM[1] - second.positionM[1];
    const fraction = denominator === 0 ? 1 : first.positionM[1] / denominator;
    return {
      segmentEnd: index,
      point: {
        timeS: first.timeS + fraction * (second.timeS - first.timeS),
        positionM: lerpVector(first.positionM, second.positionM, fraction),
        velocityMps: lerpVector(first.velocityMps, second.velocityMps, fraction),
      },
    };
  }
  return null;
};

const launchValues = (inputs: FlightMetricInputs): Map<FlightMetricId, FlightMetricValue> => {
  const values = new Map<FlightMetricId, FlightMetricValue>();
  const initial = inputs.trajectory[0].velocityMps;
  const horizontal = Math.hypot(initial[0], initial[2]);
  const spinMagnitude = norm(inputs.spinVectorRpm);
  values.set("initial_velocity", available("initial_velocity", initial, "trajectory.initial_velocity"));
  values.set("ball_speed", available("ball_speed", norm(initial), "derived.initial_velocity"));
  values.set("vertical_launch_angle", available("vertical_launch_angle",
    degrees(Math.atan2(initial[1], horizontal)), "derived.initial_velocity"));
  values.set("spin_vector", available("spin_vector", inputs.spinVectorRpm, "impact.spin_vector_rpm"));
  values.set("total_spin", available("total_spin", spinMagnitude, "derived.spin_vector_rpm"));
  values.set("launch_direction", horizontal > MIN_SPEED
    ? available("launch_direction", degrees(Math.atan2(initial[2], initial[0])), "derived.initial_velocity")
    : unavailable("launch_direction", "zero_horizontal_speed"));
  values.set("spin_axis_tilt", spinMagnitude > MIN_SPEED
    ? available("spin_axis_tilt", degrees(Math.atan2(inputs.spinVectorRpm[1], inputs.spinVectorRpm[2])), "derived.spin_vector_rpm")
    : unavailable("spin_axis_tilt", "zero_spin"));
  return values;
};

const curve = (points: readonly MetricTrajectoryPoint[], heading: number): number => {
  const origin = points[0].positionM;
  const lateral = points.map((sample) => {
    const deltaX = sample.positionM[0] - origin[0];
    const deltaZ = sample.positionM[2] - origin[2];
    return -Math.sin(heading) * deltaX + Math.cos(heading) * deltaZ;
  });
  return lateral.reduce((selected, value) => Math.abs(value) > Math.abs(selected) ? value : selected, 0);
};

const landingValues = (
  inputs: FlightMetricInputs, landing: LandingState,
): Map<FlightMetricId, FlightMetricValue> => {
  const values = new Map<FlightMetricId, FlightMetricValue>();
  const first = inputs.trajectory[0];
  const point = landing.point;
  const deltaX = point.positionM[0] - first.positionM[0];
  const deltaZ = point.positionM[2] - first.positionM[2];
  const horizontal = Math.hypot(point.velocityMps[0], point.velocityMps[2]);
  const launchHorizontal = Math.hypot(first.velocityMps[0], first.velocityMps[2]);
  const airborne = [...inputs.trajectory.slice(0, landing.segmentEnd), point];
  values.set("landing_position", available("landing_position", point.positionM, "derived.linear_ground_interpolation"));
  values.set("landing_velocity", available("landing_velocity", point.velocityMps, "derived.linear_ground_interpolation"));
  values.set("carry_distance", available("carry_distance", Math.hypot(deltaX, deltaZ), "derived.landing_position"));
  values.set("carry_offline", available("carry_offline", deltaZ, "derived.landing_position"));
  values.set("apex_height", available("apex_height", Math.max(...airborne.map((sample) => sample.positionM[1])), "derived.trajectory_samples"));
  values.set("flight_time", available("flight_time", point.timeS - first.timeS, "derived.landing_time"));
  values.set("terminal_speed", available("terminal_speed", norm(point.velocityMps), "derived.landing_velocity"));
  values.set("landing_angle", horizontal > MIN_SPEED
    ? available("landing_angle", degrees(Math.atan2(-point.velocityMps[1], horizontal)), "derived.landing_velocity")
    : unavailable("landing_angle", "zero_horizontal_speed"));
  values.set("terminal_direction", horizontal > MIN_SPEED
    ? available("terminal_direction", degrees(Math.atan2(point.velocityMps[2], point.velocityMps[0])), "derived.landing_velocity")
    : unavailable("terminal_direction", "zero_horizontal_speed"));
  values.set("curve", launchHorizontal > MIN_SPEED
    ? available("curve", curve(airborne, Math.atan2(first.velocityMps[2], first.velocityMps[0])), "derived.initial_vertical_plane")
    : unavailable("curve", "zero_horizontal_speed"));
  return values;
};

const targetValues = (
  inputs: FlightMetricInputs, landing: LandingState,
): Map<FlightMetricId, FlightMetricValue> => {
  const ids: FlightMetricId[] = ["target_residual", "target_downrange_residual", "target_lateral_residual"];
  if (!inputs.targetPositionM) return new Map(ids.map((id) => [id, unavailable(id, "target_not_configured")]));
  const residual: Vec3 = [
    landing.point.positionM[0] - inputs.targetPositionM[0],
    landing.point.positionM[1] - inputs.targetPositionM[1],
    landing.point.positionM[2] - inputs.targetPositionM[2],
  ];
  return new Map([
    ["target_residual", available("target_residual", norm(residual), "derived.target_residual")],
    ["target_downrange_residual", available("target_downrange_residual", residual[0], "derived.target_residual")],
    ["target_lateral_residual", available("target_lateral_residual", residual[2], "derived.target_residual")],
  ]);
};

const groundValues = (inputs: FlightMetricInputs): Map<FlightMetricId, FlightMetricValue> => {
  const ids: FlightMetricId[] = ["total_distance", "roll_distance", "bounce_count", "final_offline"];
  const ground = inputs.groundResult;
  if (!ground) return new Map(ids.map((id) => [id, unavailable(id, "ground_model_required")]));
  return new Map([
    ["total_distance", available("total_distance", ground.totalDistanceM, ground.modelId)],
    ["roll_distance", available("roll_distance", ground.rollDistanceM, ground.modelId)],
    ["bounce_count", available("bounce_count", ground.bounceCount, ground.modelId)],
    ["final_offline", available("final_offline", ground.finalOfflineM, ground.modelId)],
  ]);
};

const makeResult = (
  manifest: FlightRunManifest, values: Map<FlightMetricId, FlightMetricValue>,
): FlightMetricResult => {
  const ordered = Object.freeze(FLIGHT_METRIC_IDS.map((metricId) => {
    const value = values.get(metricId);
    if (!value) throw new Error(`internal metric omission: ${metricId}`);
    return value;
  }).sort((left, right) => left.metricId.localeCompare(right.metricId)));
  return Object.freeze({
    schemaVersion: "ball-flight-result/v1" as const, manifest, values: ordered,
    value: (metricId: string) => {
      const value = ordered.find((item) => item.metricId === metricId);
      if (!value) throw new RangeError(`unknown ball-flight metric: ${metricId}`);
      return value;
    },
    scalar(metricId: string): number {
      const numeric = this.value(metricId).numeric;
      if (typeof numeric !== "number") throw new RangeError(`${metricId} is not an available scalar`);
      return numeric;
    },
    vector(metricId: string): Vec3 {
      const numeric = this.value(metricId).numeric;
      if (!Array.isArray(numeric)) throw new RangeError(`${metricId} is not an available vector`);
      return numeric as Vec3;
    },
  });
};

export function deriveFlightMetricResult(
  rawInputs: FlightMetricInputs, rawManifest: FlightRunManifest,
): FlightMetricResult {
  const inputs = validateInputs(rawInputs);
  const manifest = validateManifest(rawManifest);
  if (inputs.trajectory.length === 0) {
    return makeResult(manifest, new Map(FLIGHT_METRIC_IDS.map((id) =>
      [id, unavailable(id, "insufficient_trajectory")])));
  }
  const values = launchValues(inputs);
  const landing = landingState(inputs.trajectory);
  const groundIds = new Set<FlightMetricId>(["total_distance", "roll_distance", "bounce_count", "final_offline"]);
  const remaining = FLIGHT_METRIC_IDS.filter((id) => !values.has(id) && !groundIds.has(id));
  if (landing) {
    landingValues(inputs, landing).forEach((value, id) => values.set(id, value));
    targetValues(inputs, landing).forEach((value, id) => values.set(id, value));
  } else {
    const reason = inputs.trajectory.length < 2 ? "insufficient_trajectory" : "no_ground_crossing";
    remaining.forEach((id) => values.set(id, unavailable(id, reason)));
    if (!inputs.targetPositionM) {
      ["target_residual", "target_downrange_residual", "target_lateral_residual"]
        .forEach((id) => values.set(id as FlightMetricId, unavailable(id as FlightMetricId, "target_not_configured")));
    }
  }
  groundValues(inputs).forEach((value, id) => values.set(id, value));
  return makeResult(manifest, values);
}

const stable = (value: unknown): unknown => Array.isArray(value) ? value.map(stable)
  : value && typeof value === "object" ? Object.fromEntries(Object.entries(value)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, item]) => [key, stable(item)])) : value;
const wireNumber = (value: number): number => Number(value.toFixed(11));
const wireNumeric = (value: number | Vec3 | null): number | Vec3 | null =>
  Array.isArray(value) ? value.map(wireNumber) as Vec3
    : typeof value === "number" ? wireNumber(value) : null;

/** Serialize with the same sorted-key wire contract as Python exports. */
export const stableFlightMetricResultJson = (result: FlightMetricResult): string =>
  JSON.stringify(stable({
    manifest: {
      environment: result.manifest.environment,
      frame_id: result.manifest.frameId,
      integration_status: result.manifest.integrationStatus,
      model_id: result.manifest.modelId,
      model_version: result.manifest.modelVersion,
      termination_reason: result.manifest.terminationReason,
      uncertainty_status: result.manifest.uncertaintyStatus,
      wind: result.manifest.wind,
    },
    schema_version: result.schemaVersion,
    values: result.values.map((value) => ({
      metric_id: value.metricId, numeric: wireNumeric(value.numeric),
      provenance: value.provenance, reason: value.reason, status: value.status,
    })),
  }));
