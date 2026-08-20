/** App-frame swing-series geometry helpers. */

import { add, sub, type Vec3 } from "./impactPhysics";

export interface SwingSampleTs {
  t: number;
  position: Vec3;
  velocity: Vec3;
  joints: Vec3[];
}

export function alignSwingToBall(
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
