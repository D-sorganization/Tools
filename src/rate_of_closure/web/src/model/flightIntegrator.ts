/** Small fixed-step RK4 kernel behind the public flight-model facade. */

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
import type {
  AngularFlightPoint,
  FlightResult,
  FlightSimulationOptions,
  Launch,
} from "./flight";
import { windVelocityAt } from "./wind";

const RPM_TO_RAD_S = (2 * Math.PI) / 60;
const BALL_AREA_M2 = Math.PI * GOLF_BALL_RADIUS_M ** 2;
/** Hard ceiling that keeps synchronous UI-thread RK4 work bounded. */
export const MAX_FLIGHT_INTEGRATION_STEPS = 50_000;
type Acceleration = (time: number, position: Vec3, velocity: Vec3) => Vec3;

interface IntegratorContext {
  readonly maxTimeS: number;
  readonly stepLimit: number;
  readonly stepS: number;
  readonly sampleEvery: number;
  readonly terminalGapM: (position: Vec3) => number;
  readonly omega: Vec3;
  readonly acceleration: Acceleration;
}

interface IntegratorState {
  readonly time: number;
  readonly position: Vec3;
  readonly velocity: Vec3;
  readonly step: number;
  readonly maxHeightM: number;
}

const validateLaunch = (launch: Launch): void => {
  const scalars = [launch.ballSpeedMps, launch.launchAngleRad, launch.azimuthRad, launch.spinRpm];
  if (scalars.some((value) => !Number.isFinite(value))) {
    throw new RangeError("launch scalars must be finite");
  }
  if (launch.ballSpeedMps < 0) throw new RangeError("ballSpeedMps must be nonnegative");
  if (launch.spinRpm < 0) throw new RangeError("spinRpm must be nonnegative; sign belongs in spinAxis");
  if (launch.spinAxis.some((value) => !Number.isFinite(value))
    || Math.abs(norm(launch.spinAxis) - 1) > 1e-9) {
    throw new RangeError("spinAxis must be a finite unit vector");
  }
};

const validateOptions = (options: FlightSimulationOptions): void => {
  const maxTime = options.maxTimeS ?? 10;
  const step = options.stepS ?? 0.001;
  const sampleEvery = options.sampleEvery ?? 10;
  if (!Number.isFinite(maxTime) || maxTime <= 0) throw new RangeError("maxTimeS must be positive");
  if (!Number.isFinite(step) || step <= 0) throw new RangeError("stepS must be positive");
  if (!Number.isInteger(sampleEvery) || sampleEvery < 1) {
    throw new RangeError("sampleEvery must be a positive integer");
  }
  if (Math.ceil(maxTime / step) > MAX_FLIGHT_INTEGRATION_STEPS) {
    throw new RangeError(
      "integration step budget exceeds the synchronous limit of 50,000",
    );
  }
};

const aerodynamicAcceleration = (
  relativeVelocity: Vec3,
  omega: Vec3,
): Vec3 => {
  const speed = norm(relativeVelocity);
  if (speed < 0.1) return [0, 0, -GRAVITY_M_S2];
  const direction = scale(relativeVelocity, 1 / speed);
  const spin = norm(omega);
  const spinRatio = (spin * GOLF_BALL_RADIUS_M) / speed;
  const drag = 0.21 + 0.05 * spinRatio + 0.02 * spinRatio ** 2;
  const lift = Math.min(
    MAX_LIFT_COEFFICIENT,
    spinRatio > 0 ? 0.7 * spinRatio ** 0.645 : 0,
  );
  const pressure = 0.5 * AIR_DENSITY_KG_M3 * speed ** 2 * (BALL_AREA_M2 / GOLF_BALL_MASS_KG);
  let acceleration = scale(direction, -pressure * drag);
  if (spin > 0) {
    const liftDirection = cross(scale(omega, 1 / spin), direction);
    const liftNorm = norm(liftDirection);
    if (liftNorm > 1e-10) {
      acceleration = add(acceleration, scale(liftDirection, pressure * lift / liftNorm));
    }
  }
  return [acceleration[0], acceleration[1], acceleration[2] - GRAVITY_M_S2];
};

const createAcceleration = (launch: Launch, omega: Vec3): Acceleration =>
  (time, position, velocity) => {
    const wind: Vec3 = launch.windScenario
      ? windVelocityAt(launch.windScenario, time, position)
      : [0, 0, 0];
    return aerodynamicAcceleration(sub(velocity, wind), omega);
  };

const createContext = (
  launch: Launch,
  options: FlightSimulationOptions,
): IntegratorContext => {
  validateLaunch(launch);
  validateOptions(options);
  const omega = scale(launch.spinAxis, launch.spinRpm * RPM_TO_RAD_S);
  return {
    maxTimeS: options.maxTimeS ?? 10,
    stepLimit: Math.ceil((options.maxTimeS ?? 10) / (options.stepS ?? 0.001)),
    stepS: options.stepS ?? 0.001,
    sampleEvery: options.sampleEvery ?? 10,
    terminalGapM: options.terminalGapM ?? ((position) => position[2]),
    omega,
    acceleration: createAcceleration(launch, omega),
  };
};

const launchVelocity = (launch: Launch): Vec3 => [
  launch.ballSpeedMps * Math.cos(launch.launchAngleRad) * Math.cos(launch.azimuthRad),
  launch.ballSpeedMps * Math.cos(launch.launchAngleRad) * Math.sin(launch.azimuthRad),
  launch.ballSpeedMps * Math.sin(launch.launchAngleRad),
];

const initialState = (launch: Launch): IntegratorState => ({
  time: 0,
  position: [0, 0, 0],
  velocity: launchVelocity(launch),
  step: 0,
  maxHeightM: 0,
});

const rk4Step = (
  state: IntegratorState,
  context: IntegratorContext,
  stepS: number,
): IntegratorState => {
  const { acceleration } = context;
  const dt = stepS;
  const { time, position, velocity } = state;
  const k1v = acceleration(time, position, velocity);
  const k2p = add(velocity, scale(k1v, dt / 2));
  const k2v = acceleration(time + dt / 2, add(position, scale(velocity, dt / 2)), k2p);
  const k3p = add(velocity, scale(k2v, dt / 2));
  const k3v = acceleration(time + dt / 2, add(position, scale(k2p, dt / 2)), k3p);
  const k4p = add(velocity, scale(k3v, dt));
  const k4v = acceleration(time + dt, add(position, scale(k3p, dt)), k4p);
  const nextVelocity = add(
    velocity,
    scale(add(add(k1v, scale(add(k2v, k3v), 2)), k4v), dt / 6),
  );
  const nextPosition = add(
    position,
    scale(add(add(velocity, scale(add(k2p, k3p), 2)), k4p), dt / 6),
  );
  return {
    time: time + dt,
    position: nextPosition,
    velocity: nextVelocity,
    step: state.step + 1,
    maxHeightM: Math.max(state.maxHeightM, nextPosition[2]),
  };
};

const interpolatedContact = (
  current: IntegratorState,
  next: IntegratorState,
  context: IntegratorContext,
): IntegratorState | null => {
  const currentGap = context.terminalGapM(current.position);
  const nextGap = context.terminalGapM(next.position);
  if (!Number.isFinite(currentGap) || !Number.isFinite(nextGap)) {
    throw new RangeError("terminalGapM must return finite values");
  }
  if (currentGap <= 0 || nextGap > 0) return null;
  const stepS = next.time - current.time;
  const fraction = currentGap / (currentGap - nextGap);
  return {
    time: current.time + fraction * stepS,
    position: add(current.position, scale(sub(next.position, current.position), fraction)),
    velocity: add(current.velocity, scale(sub(next.velocity, current.velocity), fraction)),
    step: next.step,
    maxHeightM: next.maxHeightM,
  };
};

const sample = (state: IntegratorState, omega: Vec3): AngularFlightPoint => ({
  time: state.time,
  position: state.position,
  velocity: state.velocity,
  angularVelocityRadS: omega,
});

const integrate = (
  launch: Launch,
  context: IntegratorContext,
): readonly [IntegratorState, AngularFlightPoint[]] => {
  let state = initialState(launch);
  const trajectory = [sample(state, context.omega)];
  while (state.step < context.stepLimit) {
    const remainingS = context.maxTimeS - state.time;
    if (remainingS <= 0) break;
    const finalStep = state.step + 1 === context.stepLimit;
    const stepS = finalStep ? remainingS : Math.min(context.stepS, remainingS);
    const next = rk4Step(state, context, stepS);
    const contact = interpolatedContact(state, next, context);
    if (contact) {
      state = contact;
      trajectory.push(sample(state, context.omega));
      break;
    }
    state = next;
    if (state.step % context.sampleEvery === 0) {
      trajectory.push(sample(state, context.omega));
    }
  }
  if (trajectory[trajectory.length - 1]?.time !== state.time) {
    trajectory.push(sample(state, context.omega));
  }
  return [state, trajectory];
};

const degrees = (radians: number): number => radians * 180 / Math.PI;

/** Integrate one launch and preserve full signed angular state at every sample. */
export function integrateFlight(
  launch: Launch,
  options: FlightSimulationOptions,
): FlightResult {
  const context = createContext(launch, options);
  const [state, trajectory] = integrate(launch, context);
  const horizontalSpeed = Math.hypot(state.velocity[0], state.velocity[1]);
  return {
    trajectory,
    carryM: Math.hypot(state.position[0], state.position[1]),
    maxHeightM: state.maxHeightM,
    flightTimeS: state.time,
    landingAngleDeg: horizontalSpeed > 0.1
      ? degrees(Math.atan2(-state.velocity[2], horizontalSpeed)) : 90,
    lateralM: state.position[1],
  };
}
