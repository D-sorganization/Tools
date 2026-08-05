/** Passive and prescribed-torque double-pendulum integration. */

import { PrescribedTorqueProfile } from "./torqueProfiles";

export const DOUBLE_PENDULUM_MODEL_ID = "model.double_pendulum.v1";
export const SHOULDER_JOINT_ID = "joint.shoulder";
export const WRIST_JOINT_ID = "joint.wrist";
export const DOUBLE_PENDULUM_JOINT_IDS = Object.freeze([
  SHOULDER_JOINT_ID,
  WRIST_JOINT_ID,
] as const);

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

export type PendulumState = [number, number, number, number];
export type JointTorquesNm = readonly [number, number];

export type DoublePendulumRunConfig =
  | Readonly<{ mode: "passive" }>
  | Readonly<{ mode: "prescribed"; profile: PrescribedTorqueProfile }>;

export interface DoublePendulumRunSummary {
  mode: "passive" | "prescribed";
  profileId: string | null;
  appliedTorqueHistory: readonly AppliedTorqueSample[];
}

export interface AppliedTorqueSample {
  timeS: number;
  torquesNm: Readonly<Record<string, number>>;
}

export const PASSIVE_DOUBLE_PENDULUM_RUN: DoublePendulumRunConfig =
  Object.freeze({ mode: "passive" });

export function prescribedDoublePendulumRun(
  profile: PrescribedTorqueProfile,
): DoublePendulumRunConfig {
  if (!(profile instanceof PrescribedTorqueProfile)) {
    throw new Error("prescribed mode requires a torque profile");
  }
  return Object.freeze({ mode: "prescribed", profile });
}

export function summarizeDoublePendulumRun(
  config: DoublePendulumRunConfig = PASSIVE_DOUBLE_PENDULUM_RUN,
  sampleTimes: readonly number[] = [],
): DoublePendulumRunSummary {
  const history = (torqueAt: (timeS: number) => readonly [number, number]) =>
    Object.freeze(sampleTimes.map((timeS) => {
      const [shoulder, wrist] = torqueAt(timeS);
      return Object.freeze({
        timeS,
        torquesNm: Object.freeze({
          [SHOULDER_JOINT_ID]: shoulder,
          [WRIST_JOINT_ID]: wrist,
        }),
      });
    }));
  if (config.mode === "passive") {
    return { mode: "passive", profileId: null, appliedTorqueHistory: history(() => [0, 0]) };
  }
  if (config.mode !== "prescribed" || !(config.profile instanceof PrescribedTorqueProfile)) {
    throw new Error("invalid double-pendulum run configuration");
  }
  return {
    mode: "prescribed",
    profileId: config.profile.profileId,
    appliedTorqueHistory: history((timeS) => {
      const values = config.profile.evaluate(timeS);
      return [values[SHOULDER_JOINT_ID], values[WRIST_JOINT_ID]];
    }),
  };
}

/** UpstreamDrift golf defaults — same segment formulas as the Rust kernel. */
export function golfDefaultParams(): PendulumParams {
  const m1 = 7.5;
  const l1 = 0.75;
  const lc1 = l1 * 0.45;
  const i1 = (1 / 12) * m1 * l1 * l1 + m1 * lc1 * lc1;
  const l2 = 1;
  const ms = 0.15;
  const mh = 0.2;
  const m2 = ms + mh;
  const shaftCom = l2 * 0.43;
  const lc2 = (shaftCom * ms + l2 * mh) / m2;
  const iShaft = (1 / 12) * ms * l2 * l2;
  const parallel = ms * (shaftCom - lc2) ** 2 + mh * (l2 - lc2) ** 2;
  const i2 = iShaft + parallel + m2 * lc2 * lc2;
  return { m1, l1, lc1, i1, m2, l2, lc2, i2, d1: 0.4, d2: 0.25 };
}

/** In-plane gravity components for the three sequential plane tilts (rad). */
export function inPlaneGravity(
  yaw: number,
  sideTilt: number,
  forwardTilt: number,
  gravity = 9.80665,
): [number, number] {
  void yaw;
  const cosSide = Math.cos(sideTilt);
  return [
    gravity * cosSide * Math.sin(forwardTilt),
    -gravity * cosSide * Math.cos(forwardTilt),
  ];
}

function pendulumDerivatives(
  p: PendulumParams,
  state: PendulumState,
  gravity: [number, number],
  torques: JointTorquesNm,
): PendulumState {
  const [theta1, theta2, omega1, omega2] = state;
  const cos2 = Math.cos(theta2);
  const m11 = p.i1 + p.i2 + p.m2 * p.l1 ** 2 + 2 * p.m2 * p.l1 * p.lc2 * cos2;
  const m12 = p.i2 + p.m2 * p.l1 * p.lc2 * cos2;
  const m22 = p.i2;
  const h = -p.m2 * p.l1 * p.lc2 * Math.sin(theta2);
  const c1 = h * (2 * omega1 * omega2 + omega2 ** 2);
  const c2 = -h * omega1 ** 2;
  const totalAngle = theta1 + theta2;
  const [gx, gy] = gravity;
  const arm = p.m1 * p.lc1 + p.m2 * p.l1;
  const club = p.m2 * p.lc2;
  const g1 = -arm * (gx * Math.cos(theta1) + gy * Math.sin(theta1))
    - club * (gx * Math.cos(totalAngle) + gy * Math.sin(totalAngle));
  const g2 = -club * (gx * Math.cos(totalAngle) + gy * Math.sin(totalAngle));
  const rhs1 = torques[0] - (c1 + g1 + p.d1 * omega1);
  const rhs2 = torques[1] - (c2 + g2 + p.d2 * omega2);
  const det = m11 * m22 - m12 ** 2;
  return [
    omega1,
    omega2,
    (m22 * rhs1 - m12 * rhs2) / det,
    (-m12 * rhs1 + m11 * rhs2) / det,
  ];
}

const addScaled = (
  state: PendulumState,
  scale: number,
  slope: PendulumState,
): PendulumState => state.map((value, index) =>
  value + scale * slope[index],
) as PendulumState;

export function pendulumRk4StepForced(
  params: PendulumParams,
  state: PendulumState,
  gravity: [number, number],
  timeS: number,
  dt: number,
  torqueAt: (timeS: number) => JointTorquesNm,
): PendulumState {
  if (!Number.isFinite(timeS) || !Number.isFinite(dt) || dt <= 0) {
    throw new Error("RK4 time and step must be finite with step > 0");
  }
  const slope = (time: number, value: PendulumState): PendulumState => {
    const torque = torqueAt(time);
    if (torque.length !== 2 || torque.some((item) => !Number.isFinite(item))) {
      throw new Error("joint torques must contain two finite values");
    }
    return pendulumDerivatives(params, value, gravity, torque);
  };
  const k1 = slope(timeS, state);
  const k2 = slope(timeS + dt / 2, addScaled(state, dt / 2, k1));
  const k3 = slope(timeS + dt / 2, addScaled(state, dt / 2, k2));
  const k4 = slope(timeS + dt, addScaled(state, dt, k3));
  return state.map((value, index) => value + (dt / 6) * (
    k1[index] + 2 * k2[index] + 2 * k3[index] + k4[index]
  )) as PendulumState;
}

export function pendulumRk4Step(
  params: PendulumParams,
  state: PendulumState,
  gravity: [number, number],
  dt: number,
): PendulumState {
  return pendulumRk4StepForced(params, state, gravity, 0, dt, () => [0, 0]);
}

export function simulatePendulum(
  params: PendulumParams,
  initial: PendulumState,
  gravity: [number, number],
  dt: number,
  nSteps: number,
): PendulumState[] {
  const output: PendulumState[] = [initial];
  let current = initial;
  for (let index = 0; index < nSteps; index += 1) {
    current = pendulumRk4Step(params, current, gravity, dt);
    output.push(current);
  }
  return output;
}

function validateProfile(profile: PrescribedTorqueProfile, durationS: number): void {
  if (profile.modelId !== DOUBLE_PENDULUM_MODEL_ID) {
    throw new Error(`profile model_id must be ${DOUBLE_PENDULUM_MODEL_ID}`);
  }
  const actual = new Set(profile.assignments.map((item) => item.jointId));
  if (
    actual.size !== DOUBLE_PENDULUM_JOINT_IDS.length ||
    DOUBLE_PENDULUM_JOINT_IDS.some((jointId) => !actual.has(jointId))
  ) {
    throw new Error("profile joint assignments must contain exactly shoulder and wrist");
  }
  if (profile.timeDomainS[0] > 0 || profile.timeDomainS[1] < durationS) {
    throw new Error("profile time domain must cover the complete simulation run");
  }
}

export function simulateConfiguredPendulum(
  params: PendulumParams,
  initial: PendulumState,
  gravity: [number, number],
  dt: number,
  nSteps: number,
  config: DoublePendulumRunConfig = PASSIVE_DOUBLE_PENDULUM_RUN,
): PendulumState[] {
  if (config.mode === "passive") {
    return simulatePendulum(params, initial, gravity, dt, nSteps);
  }
  const durationS = nSteps * dt;
  validateProfile(config.profile, durationS);
  const torqueAt = (timeS: number): JointTorquesNm => {
    const values = config.profile.evaluate(Math.min(Math.max(timeS, 0), durationS));
    return [values[SHOULDER_JOINT_ID], values[WRIST_JOINT_ID]];
  };
  const output: PendulumState[] = [initial];
  let current = initial;
  for (let index = 0; index < nSteps; index += 1) {
    current = pendulumRk4StepForced(
      params, current, gravity, index * dt, dt, torqueAt,
    );
    output.push(current);
  }
  return output;
}
