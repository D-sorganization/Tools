/**
 * Simulation session physics for the web clone (epic #4103).
 *
 * Minimal TypeScript port of the Python session pipeline — double-pendulum
 * swing (RK4, port of shared/python/swing_sim/reference.py), rigid-body COR
 * impact with the 2/7 rolling-cap friction spin (scalar-MOI path of
 * swing_sim/impact/models.py), launch derivation, and the Waterloo/Penner
 * flight model (swing_sim/flight/models.py) integrated with fixed-step RK4.
 *
 * Parity: pinned in simulation.test.ts against the pytest numbers (tight
 * for the shared-formula pendulum/impact/launch math, banded for flight
 * where scipy RK45 and this RK4 differ by integration error only).
 *
 * NOTE (P7): this hand port is a stopgap — the swing-core / tools-core
 * WASM kernels replace the double/triple RK4 kernels in epic phase P7 and
 * add the gear-effect model plus screw-axis overlay to the web.
 *
 * Frames: app frame is x target, y up, z right; the flight math runs in
 * the UpstreamDrift flight frame (x forward, y left, z up).
 */

import {
  deriveLaunch,
  simulateFlight,
  type FlightPoint,
} from "./flight";
import { golfTripleParameters, simulateTriplePendulum } from "./triplePendulum";
import {
  PASSIVE_DOUBLE_PENDULUM_RUN,
  golfDefaultParams,
  inPlaneGravity,
  simulateConfiguredPendulum,
  summarizeDoublePendulumRun,
  type DoublePendulumRunConfig,
  type PendulumState,
  type PendulumParams,
} from "./doublePendulum";
import {
  assessFixedContact,
  deliveryInspectionOutcome,
  type ContactMode,
  type ImpactOutcomeTs,
} from "./contact";
import {
  GOLF_BALL_RADIUS_M,
  ballCenterPosition,
  resolveBallSetup,
  type BallSetup,
} from "./ballSetup";
import {
  MPH_PER_MPS,
  add,
  fromFlightFrame,
  norm,
  scale,
  solveImpact,
  sub,
  toFlightFrame,
  type DeliveryInput,
  type ImpactClubProperties,
  type Vec3,
} from "./impactPhysics";

export { deriveLaunch, simulateFlight } from "./flight";
export type { FlightPoint, FlightResult, Launch } from "./flight";
export {
  golfDefaultParams,
  inPlaneGravity,
  pendulumRk4Step,
  simulatePendulum,
} from "./doublePendulum";
export type { PendulumParams, PendulumState } from "./doublePendulum";
const rad = (deg: number): number => (deg * Math.PI) / 180.0;
const deg = (r: number): number => (r * 180.0) / Math.PI;
export * from "./impactPhysics";
export { GOLF_BALL_RADIUS_M } from "./ballSetup";

// --- Session orchestration ----------------------------------------------

export const BALL_POSITION: Vec3 = [0.0, GOLF_BALL_RADIUS_M, 0.0];

export type WebSourceKind = "manual" | "double_pendulum" | "triple_pendulum";

export interface SimulationInput {
  sourceKind: WebSourceKind;
  clubheadSpeedMph: number; // manual source
  omegaDps: Vec3; // manual source angular velocity (app frame, deg/s)
  loftDeg: number;
  impactOffsetToeMm: number;
  impactOffsetHighMm: number;
  planeYawDeg: number;
  planeSideTiltDeg: number;
  planeForwardTiltDeg: number;
  impactTimeS: number | null; // null = auto (max clubhead speed)
  swingDurationS: number;
  /** Optional passive double-pendulum parameters for trace studies. */
  pendulumParameters?: PendulumParams;
  /** Offset from the automatic peak-speed inspection time [s]. */
  impactTimeOffsetS?: number;
  club?: ImpactClubProperties;
  /** Defaults to delivery inspection for backward-compatible studies. */
  contactMode?: ContactMode;
  /** Defaults to passive; prescribed mode is valid only for double pendulum. */
  doublePendulumRun?: DoublePendulumRunConfig;
  /** θ1, θ2 [rad], then their relative angular rates [rad/s]. */
  doublePendulumInitialState?: PendulumState;
  /** Defaults to Ground for backward compatibility with older saved scenarios. */
  ballSetup?: BallSetup;
}

export interface SwingSampleTs {
  t: number;
  position: Vec3; // app frame; aligned only in delivery-inspection mode
  velocity: Vec3;
  joints: Vec3[]; // pivot -> articulated joints -> clubhead
}

export interface SimulationLaunchTs {
  ballSpeedMph: number;
  launchAngleDeg: number;
  launchAzimuthDeg: number;
  spinRpm: number;
  carryM: number;
  maxHeightM: number;
  flightTimeS: number;
  landingAngleDeg: number;
}

export interface SimulationRunTs {
  sourceKind: WebSourceKind;
  torqueRun: ReturnType<typeof summarizeDoublePendulumRun>;
  swing: SwingSampleTs[];
  impactOutcome: ImpactOutcomeTs;
  impactTimeS: number | null;
  totalDurationS: number;
  launch: SimulationLaunchTs | null;
  flight: FlightPoint[]; // app frame, ball-aligned positions
  ballSetup: BallSetup;
  ballPositionM: Vec3;
}

const clampAngle = (value: number): number => Math.max(-89, Math.min(89, value));

function swingSamples(input: SimulationInput): SwingSampleTs[] {
  const dt = 1e-3;
  const runConfig = input.doublePendulumRun ?? PASSIVE_DOUBLE_PENDULUM_RUN;
  if (
    input.sourceKind !== "double_pendulum" &&
    (runConfig.mode === "prescribed" || runConfig.jointLocks.lockedJointIds.length > 0)
  ) {
    throw new Error("prescribed torque and joint locks require the double-pendulum source");
  }
  if (input.sourceKind === "manual") {
    const duration = 0.06;
    const speed = input.clubheadSpeedMph / MPH_PER_MPS;
    const omega = scale(input.omegaDps, Math.PI / 180.0);
    const samples: SwingSampleTs[] = [];
    for (let t = 0.0; t <= duration + 1e-9; t += dt) {
      const rel = t - duration / 2.0;
      // Straight-line reference travel; rotation only affects the pose,
      // which the web scene does not render — velocity is constant.
      void omega;
      samples.push({
        t,
        position: [speed * rel, 0, 0],
        velocity: [speed, 0, 0],
        joints: [],
      });
    }
    return samples;
  }
  // Pendulum on the oriented plane (swing frame), adapted to app.
  const doubleParameters = input.pendulumParameters ?? golfDefaultParams();
  const g = inPlaneGravity(
    rad(input.planeYawDeg),
    rad(input.planeSideTiltDeg),
    rad(input.planeForwardTiltDeg),
  );
  const nSteps = Math.round(input.swingDurationS / dt);
  const initialState = input.doublePendulumInitialState ?? [-Math.PI / 2, 0, 0, 0];
  if (initialState.some((value) => !Number.isFinite(value))) {
    throw new Error("double-pendulum initial state must contain four finite values");
  }
  const states =
    input.sourceKind === "double_pendulum"
      ? simulateConfiguredPendulum(
          doubleParameters,
          initialState,
          g,
          dt,
          nSteps,
          runConfig,
        )
      : simulateTriplePendulum(g, dt, nSteps);
  // Plane axes in the swing world frame, then app frame via
  // (x, y, z)_app = (x, z, -y)_swing.
  const yaw = rad(input.planeYawDeg);
  const side = rad(input.planeSideTiltDeg);
  const fwd = rad(input.planeForwardTiltDeg);
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cs = Math.cos(side);
  const ss = Math.sin(side);
  const cf = Math.cos(fwd);
  const sf = Math.sin(fwd);
  // Columns of Rz(yaw) Rx(side) Ry(fwd): local x (col 0) and up (col 2).
  const xAxisSwing: Vec3 = [cy * cf - sy * ss * sf, sy * cf + cy * ss * sf, -cs * sf];
  const upAxisSwing: Vec3 = [cy * sf + sy * ss * cf, sy * sf - cy * ss * cf, cs * cf];
  const xAxis = fromFlightFrame(xAxisSwing);
  const upAxis = fromFlightFrame(upAxisSwing);
  return states.map((state, index) => {
    const triple = golfTripleParameters();
    const angles = input.sourceKind === "double_pendulum"
      ? [state[0], state[0] + state[1]]
      : state.slice(0, 3);
    const rates = input.sourceKind === "double_pendulum"
      ? [state[2], state[2] + state[3]]
      : state.slice(3, 6);
    const lengths = input.sourceKind === "double_pendulum"
      ? [doubleParameters.l1, doubleParameters.l2]
      : triple.length;
    const localJoints: Array<[number, number]> = [[0, 0]];
    let x = 0;
    let yLoc = 0;
    let vx = 0;
    let vy = 0;
    angles.forEach((angle, linkIndex) => {
      const length = lengths[linkIndex];
      x += length * Math.sin(angle);
      yLoc -= length * Math.cos(angle);
      vx += length * Math.cos(angle) * rates[linkIndex];
      vy += length * Math.sin(angle) * rates[linkIndex];
      localJoints.push([x, yLoc]);
    });
    return {
      t: index * dt,
      position: add(scale(xAxis, x), scale(upAxis, yLoc)),
      velocity: add(scale(xAxis, vx), scale(upAxis, vy)),
      joints: [
        ...localJoints.map(([jointX, jointY]) =>
          add(scale(xAxis, jointX), scale(upAxis, jointY)),
        ),
      ],
    };
  });
}

/** Run the full swing -> impact -> flight pipeline (web parity port). */
export function runSimulation(input: SimulationInput): SimulationRunTs {
  const ballSetup = resolveBallSetup(input.ballSetup);
  const ballPositionM = ballCenterPosition(ballSetup);
  const swing = swingSamples(input);
  const impactTimeOffsetS = input.impactTimeOffsetS ?? 0;
  if (!Number.isFinite(impactTimeOffsetS)) {
    throw new Error("impactTimeOffsetS must be finite");
  }
  const torqueRun = summarizeDoublePendulumRun(
    input.doublePendulumRun,
    input.sourceKind === "double_pendulum" ? swing.map((sample) => sample.t) : [],
  );
  let impactIndex: number;
  if (input.impactTimeS === null) {
    let best = 0;
    let bestSpeed = -1;
    swing.forEach((sample, index) => {
      const speed = norm(sample.velocity);
      if (speed > bestSpeed) {
        bestSpeed = speed;
        best = index;
      }
    });
    impactIndex = best;
  } else {
    const clamped = Math.max(
      0,
      Math.min(input.impactTimeS, swing[swing.length - 1].t),
    );
    impactIndex = Math.round(clamped / (swing[1].t - swing[0].t));
  }
  if (impactTimeOffsetS !== 0) {
    const shifted = swing[impactIndex].t + impactTimeOffsetS;
    const clamped = Math.max(0, Math.min(shifted, swing[swing.length - 1].t));
    impactIndex = Math.round(clamped / (swing[1].t - swing[0].t));
  }
  const impactSample = swing[impactIndex];
  const contactMode = input.contactMode ?? "delivery_inspection";
  const impactOutcome =
    contactMode === "fixed_ball_contact"
      ? assessFixedContact(swing, ballPositionM, GOLF_BALL_RADIUS_M)
      : deliveryInspectionOutcome(
          impactSample.t,
          ballPositionM,
          GOLF_BALL_RADIUS_M,
        );
  const candidate = swing.reduce((best, sample) =>
    Math.abs(sample.t - impactOutcome.candidateTimeS) <
    Math.abs(best.t - impactOutcome.candidateTimeS)
      ? sample
      : best,
  );
  const aligned =
    contactMode === "fixed_ball_contact"
      ? swing
      : alignSwingToBall(swing, candidate.position, ballPositionM);

  if (impactOutcome.status === "miss") {
    return {
      sourceKind: input.sourceKind,
      torqueRun,
      swing: aligned,
      impactOutcome,
      impactTimeS: null,
      totalDurationS: aligned[aligned.length - 1].t,
      launch: null,
      flight: [],
      ballSetup,
      ballPositionM,
    };
  }

  const v = candidate.velocity;
  const speed = norm(v);
  const delivery: DeliveryInput = {
    clubheadSpeedMps: speed,
    clubPathDeg: clampAngle(deg(Math.atan2(v[2], v[0]))),
    faceAngleDeg: 0.0,
    attackAngleDeg: clampAngle(deg(Math.atan2(v[1], Math.hypot(v[0], v[2])))),
    dynamicLoftDeg: input.loftDeg,
    impactOffsetToeMm: input.impactOffsetToeMm,
    impactOffsetHighMm: input.impactOffsetHighMm,
    club: input.club,
  };
  const impact = solveImpact(delivery);
  const launch = deriveLaunch(
    toFlightFrame(impact.ballVelocity),
    toFlightFrame(impact.ballAngularVelocity),
  );
  const flightResult = simulateFlight(launch);
  const flight = flightResult.trajectory.map((point) => ({
    ...point,
    position: add(fromFlightFrame(point.position), ballPositionM),
    velocity: fromFlightFrame(point.velocity),
  }));

  return {
    sourceKind: input.sourceKind,
    torqueRun,
    swing: aligned,
    impactOutcome,
    impactTimeS: candidate.t,
    totalDurationS: aligned[aligned.length - 1].t + flightResult.flightTimeS,
    launch: {
      ballSpeedMph: launch.ballSpeedMps * MPH_PER_MPS,
      launchAngleDeg: deg(launch.launchAngleRad),
      launchAzimuthDeg: -deg(launch.azimuthRad),
      spinRpm: launch.spinRpm,
      carryM: flightResult.carryM,
      maxHeightM: flightResult.maxHeightM,
      flightTimeS: flightResult.flightTimeS,
      landingAngleDeg: flightResult.landingAngleDeg,
    },
    flight,
    ballSetup,
    ballPositionM,
  };
}

function alignSwingToBall(
  swing: readonly SwingSampleTs[],
  candidatePosition: Vec3,
  ballPositionM: Vec3,
): SwingSampleTs[] {
  const offset = sub(ballPositionM, candidatePosition);
  return swing.map((sample) => ({
    ...sample,
    position: add(sample.position, offset),
    joints: sample.joints.map((joint) => add(joint, offset)),
  }));
}
