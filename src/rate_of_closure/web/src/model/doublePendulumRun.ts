/** Run-mode, lock, sampled-torque-history, and localized command declarations. */

import {
  JointLockConfig,
  NO_JOINT_LOCKS,
  SHOULDER_JOINT_ID,
  WRIST_JOINT_ID,
} from "./jointLocks";
import {
  addLocalizedTorqueOffsets,
  normalizeLocalizedTorqueOffsets,
  type LocalizedTorqueOffsetTs,
} from "./localizedTorque";
import { PrescribedTorqueProfile } from "./torqueProfiles";

export type DoublePendulumRunConfig =
  | Readonly<{
      mode: "passive";
      jointLocks: JointLockConfig;
      commandedTorqueOffsets?: readonly LocalizedTorqueOffsetTs[];
    }>
  | Readonly<{
      mode: "prescribed";
      profile: PrescribedTorqueProfile;
      jointLocks: JointLockConfig;
      commandedTorqueOffsets?: readonly LocalizedTorqueOffsetTs[];
    }>;

export interface AppliedTorqueSample {
  timeS: number;
  torquesNm: Readonly<Record<string, number>>;
}

export interface DoublePendulumRunSummary {
  mode: "passive" | "prescribed";
  profileId: string | null;
  lockedJointIds: readonly string[];
  appliedTorqueHistory: readonly AppliedTorqueSample[];
}

export const PASSIVE_DOUBLE_PENDULUM_RUN: DoublePendulumRunConfig = Object.freeze({
  mode: "passive",
  jointLocks: NO_JOINT_LOCKS,
  commandedTorqueOffsets: Object.freeze([]),
});

export function passiveDoublePendulumRun(
  jointLocks: JointLockConfig = NO_JOINT_LOCKS,
  commandedTorqueOffsets: readonly LocalizedTorqueOffsetTs[] = [],
): DoublePendulumRunConfig {
  return Object.freeze({
    mode: "passive",
    jointLocks,
    commandedTorqueOffsets: normalizeLocalizedTorqueOffsets(commandedTorqueOffsets),
  });
}

export function prescribedDoublePendulumRun(
  profile: PrescribedTorqueProfile,
  jointLocks: JointLockConfig = NO_JOINT_LOCKS,
  commandedTorqueOffsets: readonly LocalizedTorqueOffsetTs[] = [],
): DoublePendulumRunConfig {
  if (!(profile instanceof PrescribedTorqueProfile)) {
    throw new Error("prescribed mode requires a torque profile");
  }
  return Object.freeze({
    mode: "prescribed",
    profile,
    jointLocks,
    commandedTorqueOffsets: normalizeLocalizedTorqueOffsets(commandedTorqueOffsets),
  });
}

export function withJointLocks(
  config: DoublePendulumRunConfig,
  jointLocks: JointLockConfig,
): DoublePendulumRunConfig {
  return config.mode === "prescribed"
    ? prescribedDoublePendulumRun(config.profile, jointLocks, config.commandedTorqueOffsets ?? [])
    : passiveDoublePendulumRun(jointLocks, config.commandedTorqueOffsets ?? []);
}

export function summarizeDoublePendulumRun(
  config: DoublePendulumRunConfig = PASSIVE_DOUBLE_PENDULUM_RUN,
  sampleTimes: readonly number[] = [],
): DoublePendulumRunSummary {
  const offsets = normalizeLocalizedTorqueOffsets(config.commandedTorqueOffsets ?? []);
  const history = (torqueAt: (timeS: number) => readonly [number, number]) =>
    Object.freeze(sampleTimes.map((timeS) => {
      const [shoulder, wrist] = addLocalizedTorqueOffsets(
        torqueAt(timeS), offsets, timeS,
      );
      return Object.freeze({
        timeS,
        torquesNm: Object.freeze({
          [SHOULDER_JOINT_ID]: shoulder,
          [WRIST_JOINT_ID]: wrist,
        }),
      });
    }));
  if (config.mode === "passive") {
    return {
      mode: "passive", profileId: null,
      lockedJointIds: config.jointLocks.lockedJointIds,
      appliedTorqueHistory: history(() => [0, 0]),
    };
  }
  if (config.mode !== "prescribed" || !(config.profile instanceof PrescribedTorqueProfile)) {
    throw new Error("invalid double-pendulum run configuration");
  }
  return {
    mode: "prescribed", profileId: config.profile.profileId,
    lockedJointIds: config.jointLocks.lockedJointIds,
    appliedTorqueHistory: history((timeS) => {
      const values = config.profile.evaluate(timeS);
      return [values[SHOULDER_JOINT_ID], values[WRIST_JOINT_ID]];
    }),
  };
}
