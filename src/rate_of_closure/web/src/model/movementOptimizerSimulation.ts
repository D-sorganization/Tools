/**
 * Movement-optimizer swing delivery simulation for the web client (epic #4103).
 *
 * Implements deterministic synthesis of reproducible swing deliveries from
 * minimal golfer anthropometry inputs (height, body mass, and optional segment
 * overrides), maintaining parity with the Python GolferAnthropometry authority.
 */

import {
  type GolferAnthropometryTs,
  type SimulationInput,
  type SwingSampleTs,
} from "./simulationTypes";
import {
  inPlaneGravity,
  simulateConfiguredPendulum,
  type PendulumParams,
  type PendulumState,
} from "./doublePendulum";
import {
  add,
  cross,
  fromFlightFrame,
  scale,
  type Vec3,
} from "./impactPhysics";
import { rotationFromColumns } from "./rotation";

export const DEFAULT_GOLFER_HEIGHT_M = 1.75;
export const DEFAULT_GOLFER_MASS_KG = 75.0;
export const STANDARD_ARM_LENGTH_FRACTION = 0.42;
export const STANDARD_ARM_MASS_FRACTION = 0.10;

export const DEFAULT_GOLFER_ANTHROPOMETRY: GolferAnthropometryTs = {
  heightM: DEFAULT_GOLFER_HEIGHT_M,
  bodyMassKg: DEFAULT_GOLFER_MASS_KG,
  armLengthM: DEFAULT_GOLFER_HEIGHT_M * STANDARD_ARM_LENGTH_FRACTION,
  armMassKg: DEFAULT_GOLFER_MASS_KG * STANDARD_ARM_MASS_FRACTION,
};

const ARM_COM_FRACTION = 0.45;
const ARM_INERTIA_SCALING = 1.0 / 12.0;
const DEFAULT_SHAFT_LENGTH_M = 1.0;
const DEFAULT_SHAFT_MASS_KG = 0.15;
const DEFAULT_CLUBHEAD_MASS_KG = 0.20;
const SHAFT_COM_FRACTION = 0.43;
const DEFAULT_DAMPING_SHOULDER = 0.4;
const DEFAULT_DAMPING_WRIST = 0.25;

const rad = (deg: number): number => (deg * Math.PI) / 180.0;

export function golferAnthropometryToParams(
  anthro?: GolferAnthropometryTs,
): PendulumParams {
  const heightM = anthro?.heightM ?? DEFAULT_GOLFER_HEIGHT_M;
  const massKg = anthro?.bodyMassKg ?? DEFAULT_GOLFER_MASS_KG;

  if (heightM <= 0 || !Number.isFinite(heightM)) {
    throw new Error(`Golfer height must be finite and positive: ${heightM}`);
  }
  if (massKg <= 0 || !Number.isFinite(massKg)) {
    throw new Error(`Golfer mass must be finite and positive: ${massKg}`);
  }

  let l1 = heightM * STANDARD_ARM_LENGTH_FRACTION;
  if (anthro?.armLengthM !== undefined && anthro?.armLengthM !== null) {
    if (!Number.isFinite(anthro.armLengthM) || anthro.armLengthM <= 0) {
      throw new Error(`Arm length must be finite and positive: ${anthro.armLengthM}`);
    }
    l1 = anthro.armLengthM;
  }

  let m1 = massKg * STANDARD_ARM_MASS_FRACTION;
  if (anthro?.armMassKg !== undefined && anthro?.armMassKg !== null) {
    if (!Number.isFinite(anthro.armMassKg) || anthro.armMassKg <= 0) {
      throw new Error(`Arm mass must be finite and positive: ${anthro.armMassKg}`);
    }
    m1 = anthro.armMassKg;
  }
  const lc1 = l1 * ARM_COM_FRACTION;
  const i1Com = ARM_INERTIA_SCALING * m1 * l1 * l1;
  const i1 = i1Com + m1 * lc1 * lc1;

  const l2 = DEFAULT_SHAFT_LENGTH_M;
  const ms = DEFAULT_SHAFT_MASS_KG;
  const mh = DEFAULT_CLUBHEAD_MASS_KG;
  const m2 = ms + mh;
  const shaftCom = l2 * SHAFT_COM_FRACTION;
  const lc2 = (shaftCom * ms + l2 * mh) / m2;
  const shaftInertiaCom = (1.0 / 12.0) * ms * l2 * l2;
  const parallelAxis =
    ms * (shaftCom - lc2) * (shaftCom - lc2) + mh * (l2 - lc2) * (l2 - lc2);
  const i2Com = shaftInertiaCom + parallelAxis;
  const i2 = i2Com + m2 * lc2 * lc2;

  return {
    m1,
    l1,
    lc1,
    i1,
    m2,
    l2,
    lc2,
    i2,
    d1: DEFAULT_DAMPING_SHOULDER,
    d2: DEFAULT_DAMPING_WRIST,
  };
}

export function simulateMovementOptimizerDelivery(
  input: SimulationInput,
  dt = 1e-3,
): SwingSampleTs[] {
  const params = golferAnthropometryToParams(input.golferAnthropometry);
  const g = inPlaneGravity(
    rad(input.planeYawDeg),
    rad(input.planeSideTiltDeg),
    rad(input.planeForwardTiltDeg),
  );
  const nSteps = Math.round(input.swingDurationS / dt);
  const initialState: PendulumState = [-Math.PI / 2, 0, 0, 0];

  const states = simulateConfiguredPendulum(
    params,
    initialState,
    g,
    dt,
    nSteps,
  );

  const yaw = rad(input.planeYawDeg);
  const side = rad(input.planeSideTiltDeg);
  const fwd = rad(input.planeForwardTiltDeg);
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cs = Math.cos(side);
  const ss = Math.sin(side);
  const cf = Math.cos(fwd);
  const sf = Math.sin(fwd);

  const xAxisSwing: Vec3 = [
    cy * cf - sy * ss * sf,
    sy * cf + cy * ss * sf,
    -cs * sf,
  ];
  const upAxisSwing: Vec3 = [
    cy * sf + sy * ss * cf,
    sy * sf - cy * ss * cf,
    cs * cf,
  ];
  const xAxis = fromFlightFrame(xAxisSwing);
  const upAxis = fromFlightFrame(upAxisSwing);
  const planeNormal = cross(upAxis, xAxis);

  const samples: SwingSampleTs[] = [];
  const lengths = [params.l1, params.l2];

  for (let i = 0; i <= nSteps; i++) {
    const t = i * dt;
    const [theta1, theta2, omega1, omega2] = states[i];
    const angles = [theta1, theta1 + theta2];
    const rates = [omega1, omega1 + omega2];

    const localJoints: Array<[number, number]> = [[0, 0]];
    let x = 0;
    let yLoc = 0;
    let vx = 0;
    let vy = 0;

    for (let linkIndex = 0; linkIndex < 2; linkIndex++) {
      const length = lengths[linkIndex];
      const angle = angles[linkIndex];
      const rate = rates[linkIndex];
      x += length * Math.sin(angle);
      yLoc -= length * Math.cos(angle);
      vx += length * Math.cos(angle) * rate;
      vy += length * Math.sin(angle) * rate;
      localJoints.push([x, yLoc]);
    }

    const clubAngle = angles[1];
    const cosine = Math.cos(clubAngle);
    const sine = Math.sin(clubAngle);
    const headX = add(scale(xAxis, cosine), scale(upAxis, -sine));
    const headZ = add(scale(xAxis, sine), scale(upAxis, cosine));

    samples.push({
      t,
      position: add(scale(xAxis, x), scale(upAxis, yLoc)),
      velocity: add(scale(xAxis, vx), scale(upAxis, vy)),
      angularVelocity: scale(planeNormal, rates[1]),
      rotation: rotationFromColumns(headX, planeNormal, headZ),
      joints: localJoints.map(([jointX, jointY]) =>
        add(scale(xAxis, jointX), scale(upAxis, jointY)),
      ),
    });
  }
  return samples;
}
