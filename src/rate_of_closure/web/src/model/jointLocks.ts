/** Stable double-pendulum joint IDs and immutable ideal-lock configuration. */

export const DOUBLE_PENDULUM_MODEL_ID = "model.double_pendulum.v1";
export const SHOULDER_JOINT_ID = "joint.shoulder";
export const WRIST_JOINT_ID = "joint.wrist";
export const DOUBLE_PENDULUM_JOINT_IDS = Object.freeze([
  SHOULDER_JOINT_ID,
  WRIST_JOINT_ID,
] as const);

export class JointLockConfig {
  readonly lockedJointIds: readonly string[];

  constructor(lockedJointIds: readonly string[] = []) {
    if (new Set(lockedJointIds).size !== lockedJointIds.length) {
      throw new Error("locked joint IDs must be unique");
    }
    const unknown = lockedJointIds.filter(
      (jointId) => !DOUBLE_PENDULUM_JOINT_IDS.includes(
        jointId as (typeof DOUBLE_PENDULUM_JOINT_IDS)[number],
      ),
    );
    if (unknown.length > 0) {
      throw new Error(
        "locked joint IDs must belong to the double-pendulum model: " +
        unknown.join(", "),
      );
    }
    this.lockedJointIds = Object.freeze(
      DOUBLE_PENDULUM_JOINT_IDS.filter((jointId) =>
        lockedJointIds.includes(jointId)),
    );
    Object.freeze(this);
  }

  get mask(): readonly [boolean, boolean] {
    return Object.freeze([
      this.isLocked(SHOULDER_JOINT_ID),
      this.isLocked(WRIST_JOINT_ID),
    ]);
  }

  isLocked(jointId: string): boolean {
    return this.lockedJointIds.includes(jointId);
  }
}

export const NO_JOINT_LOCKS = new JointLockConfig();
