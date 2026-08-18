/** UI-neutral selection and explanation helpers for the swing screw overlay. */

import type { SimulationRunTs } from "../model/simulation";
import {
  analyzeTwist,
  jointMotionAt,
  projectMotion,
  type MotionProjectionTs,
  type ScrewMotionTs,
  type Twist6,
  type Vec3,
} from "../model/screwAnalysis";

const JOINT_LABELS: Record<string, string> = {
  "joint.shoulder": "Shoulder Joint",
  "joint.elbow": "Elbow Joint",
  "joint.wrist": "Wrist Joint",
};

export interface ScrewEntityOption {
  id: string;
  label: string;
}

export interface ScrewPresentation {
  label: string;
  motion: ScrewMotionTs;
  jointResidualMps: number | null;
}

function jointIds(run: SimulationRunTs): string[] {
  const count = Math.max(0, (run.swing[0]?.joints.length ?? 1) - 1);
  const canonical = run.sourceKind === "triple_pendulum"
    ? ["joint.shoulder", "joint.elbow", "joint.wrist"]
    : ["joint.shoulder", "joint.wrist"];
  return Array.from({ length: count }, (_, index) =>
    canonical[index] ?? `joint.${index + 1}`);
}

export function screwEntityOptions(run: SimulationRunTs | null): ScrewEntityOption[] {
  const options: ScrewEntityOption[] = [{ id: "club", label: "Club" }];
  if (!run || run.swing.length === 0) return options;
  jointIds(run).forEach((id) => options.push({
    id,
    label: JOINT_LABELS[id] ?? `${id.replace("joint.", "")} Joint`,
  }));
  return options;
}

function swingIndex(run: SimulationRunTs, time: number): number {
  const end = run.swing[run.swing.length - 1].t;
  const bounded = Number.isFinite(time) ? Math.max(0, Math.min(time, end)) : 0;
  const progress = end > 0 ? bounded / end : 0;
  return Math.max(0, Math.min(
    run.swing.length - 1,
    Math.round(progress * (run.swing.length - 1)),
  ));
}

export function screwPresentation(
  run: SimulationRunTs,
  time: number,
  entityId: string,
): ScrewPresentation {
  const index = swingIndex(run, time);
  const sample = run.swing[index];
  if (entityId === "club") {
    const twist: Twist6 = [...sample.angularVelocity, ...sample.velocity];
    return { label: "Club", motion: analyzeTwist(twist, sample.position), jointResidualMps: null };
  }
  const ids = jointIds(run);
  const jointIndex = ids.indexOf(entityId);
  if (jointIndex < 0) throw new Error(`unknown screw entity ${entityId}`);
  const joint = jointMotionAt(
    run.swing.map((row) => row.t),
    run.swing.map((row) => row.joints),
    ids,
    index,
  );
  const twist: Twist6 = [
    ...joint.angularVelocityRadS[jointIndex],
    ...joint.contributionVelocityMps[jointIndex],
  ];
  return {
    label: JOINT_LABELS[entityId] ?? entityId,
    motion: analyzeTwist(twist, sample.position),
    jointResidualMps: joint.reconstructionResidualMps,
  };
}

function directionAngles(velocity: Vec3): { aoaDeg: number; pathDeg: number } | null {
  if (Math.hypot(...velocity) < 1e-10) return null;
  return {
    aoaDeg: Math.atan2(velocity[1], Math.hypot(velocity[0], velocity[2])) * 180 / Math.PI,
    pathDeg: Math.atan2(velocity[2], velocity[0]) * 180 / Math.PI,
  };
}

export function screwExplanation(
  presentation: ScrewPresentation,
  faceNormal: Vec3 = [1, 0, 0],
): string {
  const { label, motion, jointResidualMps } = presentation;
  if (motion.kind === "translation") {
    return `${label}: pure translation. The screw axis is at infinity; the arrow shows translation direction.`;
  }
  if (motion.kind === "stationary") {
    return `${label}: stationary at this sample, so no instantaneous axis exists.`;
  }
  const projections = projectMotion(motion, {
    target: [1, 0, 0],
    vertical: [0, 1, 0],
    lateral: [0, 0, 1],
    faceNormal,
  });
  const angles = directionAngles(motion.referenceVelocityMps);
  const angleText = angles
    ? ` AoA ${angles.aoaDeg.toFixed(2)} deg; path ${angles.pathDeg.toFixed(2)} deg.`
    : " AoA and path are undefined at zero speed.";
  const rateDps = motion.angularRateRadS * 180 / Math.PI;
  const residual = jointResidualMps === null
    ? ""
    : ` Joint contribution residual ${jointResidualMps.toExponential(2)} m/s.`;
  const projectionRows: Array<[string, MotionProjectionTs]> = [
    ["target", projections.target],
    ["vertical/AoA", projections.vertical],
    ["lateral/path", projections.lateral],
    ["face normal", projections.faceNormal],
  ];
  const projectionText = projectionRows.map(([name, projection]) =>
    `${name} ${projection.totalMps.toFixed(3)} = `
    + `${projection.orbitalMps.toFixed(3)} orbital + `
    + `${projection.axialMps.toFixed(3)} axial`).join("; ");
  return `${label}: finite screw in app/world frame (x target, y up, z right). `
    + `The axis arrow follows angular velocity; the wrapped arc shows handedness. `
    + `Rate ${rateDps.toFixed(1)} deg/s; pitch ${motion.pitchMPerRad!.toFixed(4)} m/rad; `
    + `axial speed ${motion.axialSpeedMps.toFixed(3)} m/s; R_ISA ${motion.radiusM!.toFixed(3)} m. `
    + `Orbital + axial velocity reconstructs the selected-point velocity. `
    + `Signed projection breakdown [total = orbital + axial] (m/s): ${projectionText}.`
    + `${angleText} Orbital/axial direction angles are diagnostics, not additive.${residual}`;
}
