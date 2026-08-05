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

export type Vec3 = [number, number, number];

import {
  deriveLaunch,
  simulateFlight,
  type FlightPoint,
} from "./flight";
import { golfTripleParameters, simulateTriplePendulum } from "./triplePendulum";
import {
  assessFixedContact,
  deliveryInspectionOutcome,
  type ContactMode,
  type ImpactOutcomeTs,
} from "./contact";

export { deriveLaunch, simulateFlight } from "./flight";
export type { FlightPoint, FlightResult, Launch } from "./flight";

// --- Constants (vendored, same citations as the Python packages) --------
export const GRAVITY_M_S2 = 9.80665;
export const AIR_DENSITY_KG_M3 = 1.225;
export const GOLF_BALL_MASS_KG = 0.04593;
export const GOLF_BALL_RADIUS_M = 0.04267 / 2.0;
export const GOLF_BALL_MOI_KG_M2 =
  (2.0 / 5.0) * GOLF_BALL_MASS_KG * GOLF_BALL_RADIUS_M ** 2;
export const DRIVER_COR = 0.83;
export const DRIVER_MASS_KG = 0.2;
export const DRIVER_MOI_KG_M2 = 4.5e-4;
export const MAX_LIFT_COEFFICIENT = 0.155;
export const MPH_PER_MPS = 1.0 / 0.44704;
const SPHERE_ROLLING_CAP = 2.0 / 7.0;
const FRICTION_COEFFICIENT = 0.4;

/** Club properties consumed by the scalar-MOI impact model. */
export interface ImpactClubProperties {
  headMassKg: number;
  moiAboutShaftKgM2: number;
  /** Optional until the club library carries measured COR values. */
  coefficientOfRestitution?: number;
}

type ResolvedImpactClubProperties = Required<ImpactClubProperties>;

/** Legacy driver values used when a direct caller does not provide a club. */
export const DEFAULT_IMPACT_CLUB: Readonly<ResolvedImpactClubProperties> =
  Object.freeze({
    headMassKg: DRIVER_MASS_KG,
    moiAboutShaftKgM2: DRIVER_MOI_KG_M2,
    coefficientOfRestitution: DRIVER_COR,
  });

// --- Small vector helpers ------------------------------------------------
export const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const norm = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);
export const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

/** App frame (x target, y up, z right) -> flight frame (x fwd, y left, z up). */
export const toFlightFrame = (v: Vec3): Vec3 => [v[0], -v[2], v[1]];
/** Flight frame -> app frame. */
export const fromFlightFrame = (v: Vec3): Vec3 => [v[0], v[2], -v[1]];

// --- Double pendulum (port of swing_sim/reference.py) --------------------

export interface PendulumParams {
  m1: number;
  l1: number;
  lc1: number;
  i1: number;
  m2: number;
  l2: number;
  lc2: number;
  i2: number;
  d1: number;
  d2: number;
}

/** UpstreamDrift golf defaults — same segment formulas as the Rust kernel. */
export function golfDefaultParams(): PendulumParams {
  const m1 = 7.5;
  const l1 = 0.75;
  const lc1 = l1 * 0.45;
  const i1 = (1.0 / 12.0) * m1 * l1 * l1 + m1 * lc1 * lc1;
  const l2 = 1.0;
  const ms = 0.15;
  const mh = 0.2;
  const m2 = ms + mh;
  const shaftCom = l2 * 0.43;
  const lc2 = (shaftCom * ms + l2 * mh) / m2;
  const iShaft = (1.0 / 12.0) * ms * l2 * l2;
  const parallel = ms * (shaftCom - lc2) ** 2 + mh * (l2 - lc2) ** 2;
  const i2 = iShaft + parallel + m2 * lc2 * lc2;
  return { m1, l1, lc1, i1, m2, l2, lc2, i2, d1: 0.4, d2: 0.25 };
}

export type PendulumState = [number, number, number, number]; // th1, th2, w1, w2

/** In-plane gravity components for the three sequential plane tilts (rad). */
export function inPlaneGravity(
  yaw: number,
  sideTilt: number,
  fwdTilt: number,
  g = GRAVITY_M_S2,
): [number, number] {
  // g_world = (0, 0, -g) projected on the local x (col 0) and up (col 2)
  // axes of Rz(yaw) Rx(side) Ry(fwd). Yaw (about world z) drops out of
  // the world-z row: R[2][0] = -cos(side) sin(fwd), R[2][2] =
  // cos(side) cos(fwd).
  void yaw;
  const cs = Math.cos(sideTilt);
  const cf = Math.cos(fwdTilt);
  const sf = Math.sin(fwdTilt);
  return [g * cs * sf, -g * cs * cf];
}

function pendulumDerivatives(
  p: PendulumParams,
  y: PendulumState,
  gInplane: [number, number],
): PendulumState {
  const [th1, th2, w1, w2] = y;
  const cos2 = Math.cos(th2);
  const m11 = p.i1 + p.i2 + p.m2 * p.l1 * p.l1 + 2.0 * p.m2 * p.l1 * p.lc2 * cos2;
  const m12 = p.i2 + p.m2 * p.l1 * p.lc2 * cos2;
  const m22 = p.i2;
  const h = -p.m2 * p.l1 * p.lc2 * Math.sin(th2);
  const c1 = h * (2.0 * w1 * w2 + w2 * w2);
  const c2 = -h * w1 * w1;
  const [gx, gy] = gInplane;
  const t12 = th1 + th2;
  const a1 = p.m1 * p.lc1 + p.m2 * p.l1;
  const a2 = p.m2 * p.lc2;
  const g1 =
    -a1 * (gx * Math.cos(th1) + gy * Math.sin(th1)) -
    a2 * (gx * Math.cos(t12) + gy * Math.sin(t12));
  const g2 = -a2 * (gx * Math.cos(t12) + gy * Math.sin(t12));
  const d1 = p.d1 * w1;
  const d2 = p.d2 * w2;
  const det = m11 * m22 - m12 * m12;
  const rhs1 = -(c1 + g1 + d1);
  const rhs2 = -(c2 + g2 + d2);
  const acc1 = (m22 * rhs1 - m12 * rhs2) / det;
  const acc2 = (-m12 * rhs1 + m11 * rhs2) / det;
  return [w1, w2, acc1, acc2];
}

/** One classical RK4 step (same evaluation order as the Python oracle). */
export function pendulumRk4Step(
  p: PendulumParams,
  y: PendulumState,
  gInplane: [number, number],
  dt: number,
): PendulumState {
  const f = (v: PendulumState) => pendulumDerivatives(p, v, gInplane);
  const addS = (a: PendulumState, s: number, b: PendulumState): PendulumState => [
    a[0] + s * b[0],
    a[1] + s * b[1],
    a[2] + s * b[2],
    a[3] + s * b[3],
  ];
  const k1 = f(y);
  const k2 = f(addS(y, dt / 2.0, k1));
  const k3 = f(addS(y, dt / 2.0, k2));
  const k4 = f(addS(y, dt, k3));
  return [0, 1, 2, 3].map(
    (i) => y[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]),
  ) as PendulumState;
}

/** Integrate nSteps RK4 steps; returns nSteps + 1 states incl. the initial. */
export function simulatePendulum(
  p: PendulumParams,
  initial: PendulumState,
  gInplane: [number, number],
  dt: number,
  nSteps: number,
): PendulumState[] {
  const out: PendulumState[] = [initial];
  let current = initial;
  for (let i = 0; i < nSteps; i += 1) {
    current = pendulumRk4Step(p, current, gInplane, dt);
    out.push(current);
  }
  return out;
}

// --- Impact (scalar-MOI path of swing_sim/impact/models.py) --------------

export interface DeliveryInput {
  clubheadSpeedMps: number;
  clubPathDeg: number;
  faceAngleDeg: number;
  attackAngleDeg: number;
  dynamicLoftDeg: number;
  impactOffsetToeMm: number;
  impactOffsetHighMm: number;
  club?: ImpactClubProperties;
}

export interface ImpactOutput {
  ballVelocity: Vec3; // app frame [m/s]
  ballAngularVelocity: Vec3; // app frame [rad/s]
}

const rad = (deg: number): number => (deg * Math.PI) / 180.0;
const deg = (r: number): number => (r * 180.0) / Math.PI;

function resolveImpactClub(
  club?: ImpactClubProperties,
): ResolvedImpactClubProperties {
  const resolved = {
    headMassKg: club?.headMassKg ?? DEFAULT_IMPACT_CLUB.headMassKg,
    moiAboutShaftKgM2:
      club?.moiAboutShaftKgM2 ?? DEFAULT_IMPACT_CLUB.moiAboutShaftKgM2,
    coefficientOfRestitution:
      club?.coefficientOfRestitution ??
      DEFAULT_IMPACT_CLUB.coefficientOfRestitution,
  };
  if (!Number.isFinite(resolved.headMassKg) || resolved.headMassKg <= 0) {
    throw new RangeError("Club head mass must be a positive finite value.");
  }
  if (
    !Number.isFinite(resolved.moiAboutShaftKgM2) ||
    resolved.moiAboutShaftKgM2 <= 0
  ) {
    throw new RangeError("Club MOI must be a positive finite value.");
  }
  if (
    !Number.isFinite(resolved.coefficientOfRestitution) ||
    resolved.coefficientOfRestitution < 0 ||
    resolved.coefficientOfRestitution > 1
  ) {
    throw new RangeError("Club coefficient of restitution must be between 0 and 1.");
  }
  return resolved;
}

/**
 * Rigid-body COR impulse solve in the app frame (ball initially at rest).
 * Off-center offsets reduce the effective club mass via the scalar MOI;
 * friction spin uses the 2/7 rolling cap with the t x n axis (bug-fixed
 * sign, matching the Python port). Gear effect: P7 (WASM).
 */
export function solveImpact(input: DeliveryInput): ImpactOutput {
  const club = resolveImpactClub(input.club);
  const path = rad(input.clubPathDeg);
  const face = rad(input.faceAngleDeg);
  const aoa = rad(input.attackAngleDeg);
  const loft = rad(input.dynamicLoftDeg);

  const vHat: Vec3 = [
    Math.cos(aoa) * Math.cos(path),
    Math.sin(aoa),
    Math.cos(aoa) * Math.sin(path),
  ];
  const n: Vec3 = [
    Math.cos(loft) * Math.cos(face),
    Math.sin(loft),
    Math.cos(loft) * Math.sin(face),
  ];
  const vClub = scale(vHat, input.clubheadSpeedMps);

  const rOffset = Math.hypot(
    input.impactOffsetToeMm / 1000.0,
    input.impactOffsetHighMm / 1000.0,
  );
  const mClubEff =
    rOffset > 1e-6
      ? 1.0 /
        (1.0 / club.headMassKg +
          (rOffset * rOffset) / club.moiAboutShaftKgM2)
      : club.headMassKg;

  const vApproach = dot(vClub, n);
  const mEff =
    (GOLF_BALL_MASS_KG * mClubEff) / (GOLF_BALL_MASS_KG + mClubEff);
  const j = (1.0 + club.coefficientOfRestitution) * mEff * vApproach;
  const ballVelocity = scale(n, j / GOLF_BALL_MASS_KG);

  // Friction spin (2/7 rolling cap, axis t x n).
  const vTangent = sub(vClub, scale(n, vApproach));
  const tangentMag = norm(vTangent);
  let ballAngularVelocity: Vec3 = [0, 0, 0];
  if (tangentMag > 1e-6) {
    const tDir = scale(vTangent, 1.0 / tangentMag);
    const spinAxis = cross(tDir, n);
    const jFriction = Math.min(
      FRICTION_COEFFICIENT * j,
      GOLF_BALL_MASS_KG * tangentMag * SPHERE_ROLLING_CAP,
    );
    const spinMagnitude = jFriction / (GOLF_BALL_MOI_KG_M2 / GOLF_BALL_RADIUS_M);
    ballAngularVelocity = scale(spinAxis, spinMagnitude);
  }
  return { ballVelocity, ballAngularVelocity };
}

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
  club?: ImpactClubProperties;
  /** Defaults to delivery inspection for backward-compatible studies. */
  contactMode?: ContactMode;
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
  swing: SwingSampleTs[];
  impactOutcome: ImpactOutcomeTs;
  impactTimeS: number | null;
  totalDurationS: number;
  launch: SimulationLaunchTs | null;
  flight: FlightPoint[]; // app frame, ball-aligned positions
}

const clampAngle = (value: number): number => Math.max(-89, Math.min(89, value));

function swingSamples(input: SimulationInput): SwingSampleTs[] {
  const dt = 1e-3;
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
  const doubleParameters = golfDefaultParams();
  const g = inPlaneGravity(
    rad(input.planeYawDeg),
    rad(input.planeSideTiltDeg),
    rad(input.planeForwardTiltDeg),
  );
  const nSteps = Math.round(input.swingDurationS / dt);
  const states =
    input.sourceKind === "double_pendulum"
      ? simulatePendulum(doubleParameters, [-Math.PI / 2, 0, 0, 0], g, dt, nSteps)
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
  const swing = swingSamples(input);
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
  const impactSample = swing[impactIndex];
  const contactMode = input.contactMode ?? "delivery_inspection";
  const impactOutcome =
    contactMode === "fixed_ball_contact"
      ? assessFixedContact(swing, BALL_POSITION, GOLF_BALL_RADIUS_M)
      : deliveryInspectionOutcome(
          impactSample.t,
          BALL_POSITION,
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
      : alignSwingToBall(swing, candidate.position);

  if (impactOutcome.status === "miss") {
    return {
      sourceKind: input.sourceKind,
      swing: aligned,
      impactOutcome,
      impactTimeS: null,
      totalDurationS: aligned[aligned.length - 1].t,
      launch: null,
      flight: [],
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
    position: add(fromFlightFrame(point.position), BALL_POSITION),
    velocity: fromFlightFrame(point.velocity),
  }));

  return {
    sourceKind: input.sourceKind,
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
  };
}

function alignSwingToBall(
  swing: readonly SwingSampleTs[],
  candidatePosition: Vec3,
): SwingSampleTs[] {
  const offset = sub(BALL_POSITION, candidatePosition);
  return swing.map((sample) => ({
    ...sample,
    position: add(sample.position, offset),
    joints: sample.joints.map((joint) => add(joint, offset)),
  }));
}
