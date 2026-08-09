/**
 * Launch derivation + Waterloo/Penner flight model for the web clone
 * (epic #4103). Ports swing_sim/flight/launch.py and the Waterloo/Penner
 * model of swing_sim/flight/models.py with fixed-step RK4 (Python uses
 * scipy RK45; the parity tests band the difference).
 *
 * NOTE (P7): replaced by the tools-core ball_flight WASM kernel.
 */

import {
  AIR_DENSITY_KG_M3,
  GOLF_BALL_MASS_KG,
  GOLF_BALL_RADIUS_M,
  GRAVITY_M_S2,
  MAX_LIFT_COEFFICIENT,
  add,
  cross,
  norm,
  scale,
  sub,
  type Vec3,
} from "./impactPhysics";
import { windVelocityAt, type WindScenario } from "./wind";

const RPM_TO_RAD_S = (2.0 * Math.PI) / 60.0;
const deg = (r: number): number => (r * 180.0) / Math.PI;

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

export interface FlightResult {
  trajectory: FlightPoint[];
  carryM: number;
  maxHeightM: number;
  flightTimeS: number;
  landingAngleDeg: number;
  lateralM: number;
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
  const [cd0, cd1, cd2, cl1, cl2] = [0.21, 0.05, 0.02, 0.7, 0.645];
  const area = Math.PI * GOLF_BALL_RADIUS_M ** 2;
  const omega = scale(launch.spinAxis, launch.spinRpm * RPM_TO_RAD_S);
  const omegaMag = norm(omega);

  const accel = (time: number, position: Vec3, velocity: Vec3): Vec3 => {
    const wind = launch.windScenario
      ? windVelocityAt(launch.windScenario, time, position)
      : [0, 0, 0] as Vec3;
    const v = sub(velocity, wind);
    const speed = norm(v);
    if (speed < 0.1) return [0, 0, -GRAVITY_M_S2];
    const vu = scale(v, 1.0 / speed);
    const s = (omegaMag * GOLF_BALL_RADIUS_M) / speed;
    const cd = cd0 + cd1 * s + cd2 * s * s;
    const cl = Math.min(MAX_LIFT_COEFFICIENT, s > 0 ? cl1 * s ** cl2 : 0.0);
    const q = 0.5 * AIR_DENSITY_KG_M3 * speed * speed * (area / GOLF_BALL_MASS_KG);
    let acc = scale(vu, -q * cd);
    if (omegaMag > 0) {
      const c = cross(scale(omega, 1.0 / omegaMag), vu);
      const cNorm = norm(c);
      if (cNorm > 1e-10) acc = add(acc, scale(c, (q * cl) / cNorm));
    }
    return [acc[0], acc[1], acc[2] - GRAVITY_M_S2];
  };

  let pos: Vec3 = [0, 0, 0];
  let vel: Vec3 = [
    launch.ballSpeedMps * Math.cos(launch.launchAngleRad) * Math.cos(launch.azimuthRad),
    launch.ballSpeedMps * Math.cos(launch.launchAngleRad) * Math.sin(launch.azimuthRad),
    launch.ballSpeedMps * Math.sin(launch.launchAngleRad),
  ];
  const trajectory: FlightPoint[] = [{ time: 0, position: pos, velocity: vel }];
  let maxHeight = 0.0;
  let t = 0.0;
  let step = 0;

  while (t < maxTime) {
    // RK4 on (pos, vel).
    const k1v = accel(t, pos, vel);
    const k1p = vel;
    const halfTime = t + dt / 2;
    const k2Position = add(pos, scale(k1p, dt / 2));
    const k2p = add(vel, scale(k1v, dt / 2));
    const k2v = accel(halfTime, k2Position, k2p);
    const k3Position = add(pos, scale(k2p, dt / 2));
    const k3p = add(vel, scale(k2v, dt / 2));
    const k3v = accel(halfTime, k3Position, k3p);
    const k4Position = add(pos, scale(k3p, dt));
    const k4p = add(vel, scale(k3v, dt));
    const k4v = accel(t + dt, k4Position, k4p);
    const nextVel = add(
      vel,
      scale(add(add(k1v, scale(add(k2v, k3v), 2)), k4v), dt / 6),
    );
    const nextPos = add(
      pos,
      scale(add(add(k1p, scale(add(k2p, k3p), 2)), k4p), dt / 6),
    );
    t += dt;
    step += 1;

    if (nextPos[2] < 0.0 && t > dt) {
      // Linear interpolation to the ground crossing. A launch that starts at
      // z = 0 travelling downward is already below ground by the time the
      // t > dt guard lets this run, which makes the raw ratio negative and
      // would extrapolate backwards past t = 0. Clamping keeps the reported
      // crossing inside the step that detected it.
      const drop = pos[2] - nextPos[2];
      const frac = drop > 0 ? Math.min(1, Math.max(0, pos[2] / drop)) : 0;
      const tGround = t - dt + frac * dt;
      const posGround = add(pos, scale(sub(nextPos, pos), frac));
      const velGround = add(vel, scale(sub(nextVel, vel), frac));
      trajectory.push({ time: tGround, position: posGround, velocity: velGround });
      pos = posGround;
      vel = velGround;
      t = tGround;
      break;
    }
    pos = nextPos;
    vel = nextVel;
    maxHeight = Math.max(maxHeight, pos[2]);
    if (step % sampleEvery === 0) {
      trajectory.push({ time: t, position: pos, velocity: vel });
    }
  }

  const vHoriz = Math.hypot(vel[0], vel[1]);
  return {
    trajectory,
    carryM: Math.hypot(pos[0], pos[1]),
    maxHeightM: maxHeight,
    flightTimeS: t,
    landingAngleDeg: vHoriz > 0.1 ? deg(Math.atan2(-vel[2], vHoriz)) : 90.0,
    lateralM: pos[1],
  };
}

