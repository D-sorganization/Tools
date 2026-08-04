/**
 * Standalone flight-explorer logic for the web clone (epic #4120, V2).
 *
 * TypeScript twin of `rate_of_closure/simulation/flight_explorer.py`,
 * parity-banded by `flightExplorer.test.ts` against the pytest pinned
 * case: build flight-frame launch conditions from launch-monitor ball
 * numbers (app signs: azimuth and lateral + = right of target, spin
 * axis tilt + = fade side) and integrate with the Waterloo/Penner
 * model. The full 7-model picker stays Python-side until the P7 WASM
 * kernels land.
 */

import { simulateFlight, type FlightPoint, type Launch } from "./flight";
import {
  BALL_POSITION,
  MPH_PER_MPS,
  add,
  fromFlightFrame,
  type Vec3,
} from "./simulation";

const rad = (d: number): number => (d * Math.PI) / 180.0;
const deg = (r: number): number => (r * 180.0) / Math.PI;

export interface DirectLaunchInput {
  ballSpeedMph: number;
  launchAngleDeg: number;
  azimuthDeg: number; // + = right of target (app convention)
  spinRpm: number;
  spinAxisTiltDeg: number; // + = fade side (curves right)
}

/** Twin of `launch_from_direct` (app signs -> flight frame). */
export function directLaunch(input: DirectLaunchInput): Launch {
  if (!(input.ballSpeedMph > 0)) {
    throw new Error("ballSpeedMph must be > 0");
  }
  // App azimuth + = right; flight-frame azimuth + = left: flip. The
  // fade-side tilt (+) needs a downward (-z flight) sidespin component,
  // so the legacy spin-axis-angle decomposition gets the flipped angle
  // too (same derivation as the Python twin).
  const azimuthRad = -rad(input.azimuthDeg);
  const axisAngle = -rad(input.spinAxisTiltDeg);
  const backspin = Math.cos(axisAngle);
  const sidespin = Math.sin(axisAngle);
  const spinAxis: Vec3 = [
    sidespin * Math.sin(azimuthRad),
    -backspin,
    sidespin * Math.cos(azimuthRad),
  ];
  return {
    ballSpeedMps: input.ballSpeedMph / MPH_PER_MPS,
    launchAngleRad: rad(input.launchAngleDeg),
    azimuthRad,
    spinRpm: input.spinRpm,
    spinAxis,
  };
}

export interface FlightExplorationTs {
  /** App-frame trajectory from the tee (x target, y up, z right). */
  points: FlightPoint[];
  metrics: {
    ballSpeedMph: number;
    launchAngleDeg: number;
    launchAzimuthDeg: number; // + = right of target
    spinRpm: number;
    carryM: number;
    maxHeightM: number;
    flightTimeS: number;
    landingAngleDeg: number;
    lateralM: number; // + = right of target
  };
}

/** Twin of `explore_flight` (Waterloo/Penner only on web). */
export function exploreFlight(launch: Launch): FlightExplorationTs {
  const result = simulateFlight(launch);
  const points = result.trajectory.map((point) => ({
    ...point,
    position: add(fromFlightFrame(point.position), BALL_POSITION),
    velocity: fromFlightFrame(point.velocity),
  }));
  return {
    points,
    metrics: {
      ballSpeedMph: launch.ballSpeedMps * MPH_PER_MPS,
      launchAngleDeg: deg(launch.launchAngleRad),
      // Flight azimuth + = left; app convention + = right of target.
      launchAzimuthDeg: -deg(launch.azimuthRad),
      spinRpm: launch.spinRpm,
      carryM: result.carryM,
      maxHeightM: result.maxHeightM,
      flightTimeS: result.flightTimeS,
      landingAngleDeg: result.landingAngleDeg,
      // Flight lateral + = left; app lateral + = right.
      lateralM: -result.lateralM,
    },
  };
}
