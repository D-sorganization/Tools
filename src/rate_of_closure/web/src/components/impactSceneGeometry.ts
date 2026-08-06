/** UI-neutral 3-D primitives for the interactive impact inspection still. */

import { add, cross, norm, scale, sub, type Vec3 } from "../model/impactPhysics";
import type { ImpactKinematicsTs } from "../model/impactKinematics";

export interface ImpactSceneLineTs {
  key: string;
  label: string;
  points: Vec3[];
  color: string;
  width: number;
  dash?: number[];
  arrow?: boolean;
}

export interface ImpactSceneGeometryTs {
  lines: ImpactSceneLineTs[];
  contactPoint: Vec3;
  ballCenter: Vec3;
}

const COLORS = {
  shaft: "#cbd5e1", face: "#38bdf8", edge: "#facc15",
  normal: "#22c55e", arc: "#a78bfa", screw: "#e879f9",
  total: "#fb7185", axisTranslation: "#38bdf8", shaftRotation: "#f97316",
  otherRotation: "#c084fc", withoutShaft: "#94a3b8", ground: "#334155",
};

const centered = (point: Vec3, center: Vec3): Vec3 => sub(point, center);

export function impactSceneGeometry(
  scene: ImpactKinematicsTs,
  visibleVectors: ReadonlySet<string>,
): ImpactSceneGeometryTs {
  const center = scene.contactPointM;
  const contact: Vec3 = [0, 0, 0];
  const shaftPoint = centered(scene.shaftAxisPointM, center);
  const faceUp = cross(scene.leadingEdgeUnit, scene.faceNormalUnit);
  const faceCorners = [
    add(scale(scene.leadingEdgeUnit, -0.06), scale(faceUp, -0.035)),
    add(scale(scene.leadingEdgeUnit, 0.06), scale(faceUp, -0.035)),
    add(scale(scene.leadingEdgeUnit, 0.06), scale(faceUp, 0.035)),
    add(scale(scene.leadingEdgeUnit, -0.06), scale(faceUp, 0.035)),
    add(scale(scene.leadingEdgeUnit, -0.06), scale(faceUp, -0.035)),
  ];
  const backCorners = faceCorners.map((point) =>
    add(point, scale(scene.faceNormalUnit, -0.05)));
  const lines: ImpactSceneLineTs[] = [
    { key: "shaft", label: "Physical Shaft Axis", points: [
      add(shaftPoint, scale(scene.shaftAxisUnit, -0.08)),
      add(shaftPoint, scale(scene.shaftAxisUnit, 0.45)),
    ], color: COLORS.shaft, width: 3 },
    { key: "face", label: "Wedge Face", points: faceCorners, color: COLORS.face, width: 2 },
    { key: "back", label: "Wedge Back", points: backCorners, color: "#64748b", width: 1.5 },
    { key: "edge", label: "Leading Edge", points: [
      scale(scene.leadingEdgeUnit, -0.06), scale(scene.leadingEdgeUnit, 0.06),
    ], color: COLORS.edge, width: 3 },
    { key: "normal", label: "Face Normal", points: [contact, scale(scene.faceNormalUnit, 0.15)], color: COLORS.normal, width: 2, arrow: true },
    { key: "arc", label: "Arc Tangent", points: [contact, scale(scene.arcTangentUnit, 0.15)], color: COLORS.arc, width: 2, arrow: true },
  ];
  for (let index = 0; index < 4; index += 1) {
    lines.push({
      key: `body-${index}`,
      label: "Wedge Body",
      points: [faceCorners[index], backCorners[index]],
      color: "#64748b",
      width: 1.2,
    });
  }
  for (const offset of [-0.15, -0.075, 0, 0.075, 0.15]) {
    lines.push(
      { key: `ground-x-${offset}`, label: "Ground", points: [[-0.15, -center[1], offset], [0.15, -center[1], offset]], color: COLORS.ground, width: 1 },
      { key: `ground-z-${offset}`, label: "Ground", points: [[offset, -center[1], -0.15], [offset, -center[1], 0.15]], color: COLORS.ground, width: 1 },
    );
  }
  const maxSpeed = Math.max(...scene.vectors.map((vector) => norm(vector.vectorMps)), 1e-12);
  for (const vector of scene.vectors) {
    if (!visibleVectors.has(vector.key)) continue;
    lines.push({
      key: vector.key,
      label: vector.label,
      points: [contact, scale(vector.vectorMps, 0.18 / maxSpeed)],
      color: COLORS[vector.key],
      width: vector.key === "total" ? 3 : 2,
      dash: vector.key === "withoutShaft" ? [7, 5] : undefined,
      arrow: true,
    });
  }
  if (scene.screwAxis) {
    const point = centered(scene.screwAxis.pointM, center);
    lines.push({
      key: "screw", label: "Instantaneous Screw Axis", points: [
        add(point, scale(scene.screwAxis.directionUnit, -0.3)),
        add(point, scale(scene.screwAxis.directionUnit, 0.3)),
      ], color: COLORS.screw, width: 2, dash: [3, 4],
    });
  }
  return {
    lines,
    contactPoint: contact,
    ballCenter: centered(scene.ballCenterM, center),
  };
}
