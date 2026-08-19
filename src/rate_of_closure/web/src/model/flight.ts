/**
 * Launch derivation + Waterloo/Penner flight model for the web clone
 * (epic #4103). Ports swing_sim/flight/launch.py and the Waterloo/Penner
 * model of swing_sim/flight/models.py with fixed-step RK4 (Python uses
 * scipy RK45; the parity tests band the difference).
 *
 * NOTE (P7): replaced by the tools-core ball_flight WASM kernel.
 */

import {
  norm,
  scale,
  type Vec3,
} from "./impactPhysics";
import { integrateFlight } from "./flightIntegrator";
import { type WindScenario } from "./wind";

const RPM_TO_RAD_S = (2.0 * Math.PI) / 60.0;

// --- Launch derivation + Waterloo/Penner flight --------------------------

export interface Launch {
  ballSpeedMps: number;
  launchAngleRad: number;
  azimuthRad: number;
  spinRpm: number;
  spinAxis: Vec3; // flight frame, unit
  windScenario?: WindScenario;
}
/** Port of swing_sim/flight/launch.py (flight-frame inputs). */
export function deriveLaunch(velFlight: Vec3, spinFlight: Vec3): Launch {
  const speed = norm(velFlight);
  const horiz = Math.hypot(velFlight[0], velFlight[1]);
  const launchAngleRad =
    horiz < 1e-12 ? Math.PI / 2.0 : Math.atan2(velFlight[2], horiz);
  const azimuthRad = horiz > 1e-12 ? Math.atan2(velFlight[1], velFlight[0]) : 0.0;
  const spinRadS = norm(spinFlight);
  const spinAxis: Vec3 =
    spinRadS > 1e-12 ? scale(spinFlight, 1.0 / spinRadS) : [0, -1, 0];
  return {
    ballSpeedMps: speed,
    launchAngleRad,
    azimuthRad,
    spinRpm: spinRadS / RPM_TO_RAD_S,
    spinAxis,
  };
}

export interface FlightPoint {
  time: number;
  position: Vec3; // flight frame
  velocity: Vec3;
}

/** Flight sample with the complete signed angular state required at landing. */
export interface AngularFlightPoint extends FlightPoint {
  angularVelocityRadS: Vec3; // flight frame
}

export interface FlightResult {
  trajectory: AngularFlightPoint[];
  carryM: number;
  maxHeightM: number;
  flightTimeS: number;
  landingAngleDeg: number;
  lateralM: number;
}

export interface FlightSimulationOptions {
  readonly maxTimeS?: number;
  readonly stepS?: number;
  readonly sampleEvery?: number;
  /** Positive while separated; zero at physical sphere/surface contact. */
  readonly terminalGapM?: (positionFlightM: Vec3) => number;
}

/**
 * Waterloo/Penner model (quadratic Cd, power-law Cl), fixed-step RK4 with
 * linear interpolation to the ground crossing. Python uses scipy RK45; the
 * parity tests band the difference.
 */
export function simulateFlight(
  launch: Launch,
  maxTime = 10.0,
  dt = 0.001,
  sampleEvery = 10,
): FlightResult {
  return simulateFlightWithOptions(launch, {
    maxTimeS: maxTime,
    stepS: dt,
    sampleEvery,
  });
}

/** Simulate to an explicit physical contact surface or the launch plane. */
export function simulateFlightWithOptions(
  launch: Launch,
  options: FlightSimulationOptions = {},
): FlightResult {
  return integrateFlight(launch, options);
}
