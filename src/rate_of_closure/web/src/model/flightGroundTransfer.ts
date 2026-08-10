/** Flight-frame to strict ground-request transfer with physical contact bracketing. */

import { resolveBallSetup, type BallSetup } from "./ballSetup";
import {
  simulateFlightWithOptions,
  type AngularFlightPoint,
  type FlightResult,
  type FlightSimulationOptions,
  type Launch,
} from "./flight";
import { parseFlightToGroundRequest } from "./flightGroundContract";
import {
  FLIGHT_TO_GROUND_REQUEST_VERSION,
  GROUND_TARGET_FRAME,
  type FlightToGroundRequest,
  type GroundCalibration,
  type GroundContactState,
  type GroundProvenance,
  type GroundSurfaceProfile,
  type GroundVec3,
} from "./flightGroundTypes";
import {
  dot,
  groundSignedGapM,
  nonnegative,
  parseSurface,
  positive,
  relativeNormalSpeedMps,
  vector,
} from "./flightGroundValidation";
import type { Vec3 } from "./impactPhysics";

const CONTACT_SPEED_TOLERANCE_M_S = 1e-12;
const GRAZING_GAP_TOLERANCE_M = 1e-9;

export interface FlightToGroundTransferConfig {
  readonly requestId: string;
  /** Source-flight coordinate of the launch/tee ball centre. */
  readonly flightOrigin: GroundVec3;
  /** Plane coordinates already translated to the launch/tee ball-centre origin. */
  readonly launchRelativeSurface: GroundSurfaceProfile;
  readonly surfaceOrigin: "launch_ball_center";
  readonly ballRadiusM: number;
  readonly ballMassKg: number;
  readonly rotationalInertiaFactor: number;
  readonly maxTimeS: number;
  readonly outputIntervalS: number;
  readonly maxEvents: number;
  readonly calibration: GroundCalibration;
  readonly provenance: GroundProvenance;
}

export type FlightToGroundUnavailableReason =
  | "missing_terminal_angular_velocity"
  | "invalid_trajectory"
  | "no_physical_contact"
  | "grazing_contact";

export type FlightToGroundTransferOutcome =
  | { readonly status: "available"; readonly request: FlightToGroundRequest }
  | {
    readonly status: "unavailable";
    readonly reason: FlightToGroundUnavailableReason;
    readonly provenance: string;
  };

export type GroundFlightSimulationOptions = Omit<FlightSimulationOptions, "terminalGapM">;

/** Ground height in the launch-centre target frame for an existing ball setup. */
export function launchRelativeSurfaceHeightM(
  ballRadiusM: number,
  ballSetup: BallSetup,
): number {
  const radius = positive(ballRadiusM, "ballRadiusM");
  const setup = resolveBallSetup(ballSetup);
  return -(radius + setup.teeHeightM);
}

/** Convert a flight-frame vector into target x-downrange/y-up/z-right axes. */
export const flightVectorToTarget = (value: readonly number[]): GroundVec3 => {
  const parsed = vector(value, "flight vector");
  return Object.freeze([parsed[0], parsed[2], -parsed[1]]);
};

const pointToTargetState = (
  point: AngularFlightPoint,
  flightOrigin: GroundVec3,
): GroundContactState => {
  if (!point.angularVelocityRadS) {
    throw new RangeError("terminal angular velocity is unavailable");
  }
  const relativePosition: GroundVec3 = [
    point.position[0] - flightOrigin[0],
    point.position[1] - flightOrigin[1],
    point.position[2] - flightOrigin[2],
  ];
  return Object.freeze({
    time_s: nonnegative(point.time, "flight point time"),
    frame: GROUND_TARGET_FRAME,
    position_m: flightVectorToTarget(relativePosition),
    velocity_m_s: flightVectorToTarget(point.velocity),
    angular_velocity_rad_s: flightVectorToTarget(point.angularVelocityRadS),
  });
};

const unavailable = (
  reason: FlightToGroundUnavailableReason,
  provenance: string,
): FlightToGroundTransferOutcome => Object.freeze({ status: "unavailable", reason, provenance });

const validateTimes = (points: readonly AngularFlightPoint[]): boolean =>
  points.length >= 2 && points.every((point, index) =>
    Number.isFinite(point.time) && point.time >= 0
    && (index === 0 || point.time > points[index - 1].time)
  );

const startsAtDeclaredOrigin = (
  points: readonly AngularFlightPoint[],
  origin: GroundVec3,
): boolean => points[0]?.time === 0 && points[0].position.every(
  (component, index) => component === origin[index],
);

const findBracket = (
  states: readonly GroundContactState[],
  surface: GroundSurfaceProfile,
  radius: number,
): readonly [GroundContactState, GroundContactState] | null => {
  for (let index = 1; index < states.length; index += 1) {
    const separated = states[index - 1];
    const penetrating = states[index];
    const gaps = [
      groundSignedGapM(separated, surface, radius),
      groundSignedGapM(penetrating, surface, radius),
    ];
    const speeds = [
      relativeNormalSpeedMps(separated, surface),
      relativeNormalSpeedMps(penetrating, surface),
    ];
    if (gaps[0] > 0 && gaps[1] <= 0
      && speeds.every((speed) => speed < -CONTACT_SPEED_TOLERANCE_M_S)) {
      return [separated, penetrating];
    }
  }
  return null;
};

const observedGrazingContact = (
  states: readonly GroundContactState[],
  surface: GroundSurfaceProfile,
  radius: number,
): boolean => {
  let observedSeparation = false;
  for (const state of states) {
    const gap = groundSignedGapM(state, surface, radius);
    if (gap > GRAZING_GAP_TOLERANCE_M) {
      observedSeparation = true;
    } else if (observedSeparation && Math.abs(gap) <= GRAZING_GAP_TOLERANCE_M
      && relativeNormalSpeedMps(state, surface) >= -CONTACT_SPEED_TOLERANCE_M_S) {
      return true;
    }
  }
  return false;
};

const requestFromBracket = (
  bracket: readonly [GroundContactState, GroundContactState],
  config: FlightToGroundTransferConfig,
  surface: GroundSurfaceProfile,
): FlightToGroundRequest => parseFlightToGroundRequest({
  schema_version: FLIGHT_TO_GROUND_REQUEST_VERSION,
  request_id: config.requestId,
  unit_system: "SI",
  surface,
  last_separated_state: bracket[0],
  first_penetrating_state: bracket[1],
  ball_radius_m: config.ballRadiusM,
  ball_mass_kg: config.ballMassKg,
  rotational_inertia_factor: config.rotationalInertiaFactor,
  max_time_s: config.maxTimeS,
  output_interval_s: config.outputIntervalS,
  max_events: config.maxEvents,
  calibration: config.calibration,
  provenance: config.provenance,
});

/** Build a request or return typed evidence that no qualified handoff exists. */
export function buildFlightToGroundRequest(
  points: readonly AngularFlightPoint[],
  config: FlightToGroundTransferConfig,
): FlightToGroundTransferOutcome {
  if (config.surfaceOrigin !== "launch_ball_center") {
    throw new RangeError("surfaceOrigin must be launch_ball_center");
  }
  if (!validateTimes(points)) return unavailable("invalid_trajectory", "flight trajectory");
  if (points.some((point) => !point.angularVelocityRadS)) {
    return unavailable("missing_terminal_angular_velocity", "flight trajectory");
  }
  let states: readonly GroundContactState[];
  try {
    const origin = vector(config.flightOrigin, "flightOrigin");
    if (!startsAtDeclaredOrigin(points, origin)) {
      return unavailable("invalid_trajectory", "missing canonical launch-origin datum");
    }
    states = points.map((point) => pointToTargetState(point, origin));
  } catch {
    return unavailable("invalid_trajectory", "flight trajectory");
  }
  const surface = parseSurface(config.launchRelativeSurface);
  const radius = positive(config.ballRadiusM, "ballRadiusM");
  const bracket = findBracket(states, surface, radius);
  if (bracket) {
    return Object.freeze({ status: "available", request: requestFromBracket(bracket, config, surface) });
  }
  const reason = observedGrazingContact(states, surface, radius)
    ? "grazing_contact" : "no_physical_contact";
  return unavailable(reason, surface.provider_id + "@" + surface.provider_version);
}

const targetGapForPosition = (
  positionFlightM: Vec3,
  surface: GroundSurfaceProfile,
  radiusM: number,
): number => {
  const target = flightVectorToTarget(positionFlightM);
  return dot([
    target[0],
    target[1] - surface.height_m,
    target[2],
  ], surface.normal_unit) - radiusM;
};

/** Run the existing web integrator through the configured physical plane. */
export function simulateFlightForGroundTransfer(
  launch: Launch,
  config: FlightToGroundTransferConfig,
  options: GroundFlightSimulationOptions = {},
): FlightResult {
  if (config.surfaceOrigin !== "launch_ball_center") {
    throw new RangeError("surfaceOrigin must be launch_ball_center");
  }
  const surface = parseSurface(config.launchRelativeSurface);
  const radius = positive(config.ballRadiusM, "ballRadiusM");
  const result = simulateFlightWithOptions(launch, {
    ...options,
    terminalGapM: (position) => targetGapForPosition(position, surface, radius),
  });
  const origin = vector(config.flightOrigin, "flightOrigin");
  return {
    ...result,
    trajectory: result.trajectory.map((point) => ({
      ...point,
      position: [
        point.position[0] + origin[0],
        point.position[1] + origin[1],
        point.position[2] + origin[2],
      ],
    })),
  };
}
