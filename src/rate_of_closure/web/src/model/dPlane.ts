/** Reference-frame-explicit three-dimensional D-plane geometry. */

import { cross, dot, norm, scale, sub, type Vec3 } from "./impactPhysics";

const EPSILON = 1e-12;
const DEG = 180 / Math.PI;

export type DPlaneStatus = "defined" | "zero_travel" | "parallel" | "antiparallel";

export interface DPlaneAnalysisTs {
  status: DPlaneStatus;
  frameId: string;
  travelDirectionUnit: Vec3 | null;
  faceNormalUnit: Vec3;
  targetUnit: Vec3;
  upUnit: Vec3;
  rightUnit: Vec3;
  dplaneNormalUnit: Vec3 | null;
  groundIntersectionUnit: Vec3 | null;
  spinLoft3dDeg: number | null;
  planarSpinLoftDeg: number | null;
  signedPlanarGapDeg: number | null;
  spinLoftResidualDeg: number | null;
  clubPathDeg: number | null;
  attackAngleDeg: number | null;
  faceAngleDeg: number | null;
  dynamicLoftDeg: number;
  faceToPathDeg: number | null;
  dplaneNormalAzimuthDeg: number | null;
  dplaneTiltDeg: number | null;
  dplaneInclinationDeg: number | null;
  groundIntersectionAzimuthDeg: number | null;
}

const finiteVector = (value: Vec3, name: string): Vec3 => {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${name} must contain three finite components`);
  }
  return value;
};

const unit = (value: Vec3, name: string): Vec3 => {
  const magnitude = norm(value);
  if (!(magnitude > EPSILON)) throw new RangeError(`${name} must be nonzero`);
  return scale(value, 1 / magnitude);
};

const horizontal = (value: Vec3, up: Vec3): Vec3 => sub(value, scale(up, dot(value, up)));

const headingDeg = (value: Vec3, target: Vec3, right: Vec3, up: Vec3): number | null => {
  const projected = horizontal(value, up);
  if (norm(projected) <= EPSILON) return null;
  return Math.atan2(dot(projected, right), dot(projected, target)) * DEG;
};

const elevationDeg = (value: Vec3, up: Vec3): number =>
  Math.atan2(dot(value, up), norm(horizontal(value, up))) * DEG;

const wrappedDeltaDeg = (first: number, second: number): number =>
  ((first - second + 540) % 360) - 180;

const emptyResult = (
  face: Vec3,
  target: Vec3,
  up: Vec3,
  right: Vec3,
  frameId: string,
): DPlaneAnalysisTs => ({
  status: "zero_travel", frameId, travelDirectionUnit: null, faceNormalUnit: face,
  targetUnit: target, upUnit: up, rightUnit: right, dplaneNormalUnit: null,
  groundIntersectionUnit: null, spinLoft3dDeg: null, planarSpinLoftDeg: null,
  signedPlanarGapDeg: null, spinLoftResidualDeg: null, clubPathDeg: null,
  attackAngleDeg: null, faceAngleDeg: headingDeg(face, target, right, up),
  dynamicLoftDeg: elevationDeg(face, up), faceToPathDeg: null,
  dplaneNormalAzimuthDeg: null, dplaneTiltDeg: null, dplaneInclinationDeg: null,
  groundIntersectionAzimuthDeg: null,
});

export function analyzeDPlane(
  travelVector: Vec3,
  faceNormal: Vec3,
  targetInput: Vec3 = [1, 0, 0],
  upInput: Vec3 = [0, 1, 0],
  frameId = "app_frame:x_target,y_up,z_right",
): DPlaneAnalysisTs {
  const travel = finiteVector(travelVector, "travelVector");
  const face = unit(finiteVector(faceNormal, "faceNormal"), "faceNormal");
  const target = unit(finiteVector(targetInput, "target"), "target");
  const up = unit(finiteVector(upInput, "up"), "up");
  if (Math.abs(dot(target, up)) > 1e-10) {
    throw new RangeError("target and up axes must be orthogonal");
  }
  const right = unit(cross(target, up), "target x up right axis");
  if (norm(travel) <= EPSILON) return emptyResult(face, target, up, right, frameId);

  const travelUnit = unit(travel, "travelVector");
  const clubPath = headingDeg(travelUnit, target, right, up);
  const attackAngle = elevationDeg(travelUnit, up);
  const faceAngle = headingDeg(face, target, right, up);
  const dynamicLoft = elevationDeg(face, up);
  const faceToPath = faceAngle === null || clubPath === null
    ? null : wrappedDeltaDeg(faceAngle, clubPath);
  const signedPlanarGap = dynamicLoft - attackAngle;
  const planarSpinLoft = Math.abs(signedPlanarGap);
  const cosine = Math.max(-1, Math.min(1, dot(travelUnit, face)));
  const spinLoft = Math.acos(cosine) * DEG;
  const normalRaw = cross(travelUnit, face);
  const normalMagnitude = norm(normalRaw);
  const common = {
    frameId, travelDirectionUnit: travelUnit, faceNormalUnit: face,
    targetUnit: target, upUnit: up, rightUnit: right, spinLoft3dDeg: spinLoft,
    planarSpinLoftDeg: planarSpinLoft, signedPlanarGapDeg: signedPlanarGap,
    spinLoftResidualDeg: spinLoft - planarSpinLoft, clubPathDeg: clubPath,
    attackAngleDeg: attackAngle, faceAngleDeg: faceAngle, dynamicLoftDeg: dynamicLoft,
    faceToPathDeg: faceToPath,
  };
  if (normalMagnitude <= EPSILON) {
    return {
      ...common, status: cosine >= 0 ? "parallel" : "antiparallel",
      dplaneNormalUnit: null, groundIntersectionUnit: null,
      dplaneNormalAzimuthDeg: null, dplaneTiltDeg: null,
      dplaneInclinationDeg: null, groundIntersectionAzimuthDeg: null,
    };
  }

  const normal = scale(normalRaw, 1 / normalMagnitude);
  const normalHorizontal = horizontal(normal, up);
  const groundRaw = cross(up, normal);
  let ground: Vec3 | null = norm(groundRaw) <= EPSILON
    ? null : scale(groundRaw, 1 / norm(groundRaw));
  if (ground && dot(ground, target) < 0) ground = scale(ground, -1);
  return {
    ...common, status: "defined", dplaneNormalUnit: normal,
    groundIntersectionUnit: ground,
    dplaneNormalAzimuthDeg: headingDeg(normal, target, right, up),
    dplaneTiltDeg: Math.atan2(-dot(normal, up), norm(normalHorizontal)) * DEG,
    dplaneInclinationDeg: Math.acos(Math.max(0, Math.min(1, Math.abs(dot(normal, up))))) * DEG,
    groundIntersectionAzimuthDeg: ground === null ? null : headingDeg(ground, target, right, up),
  };
}

export function spinLoftSectorDirections(
  analysis: DPlaneAnalysisTs,
  segments = 24,
): Vec3[] {
  if (!Number.isInteger(segments) || segments < 2) {
    throw new RangeError("segments must be an integer of at least two");
  }
  if (analysis.status !== "defined" || analysis.travelDirectionUnit === null ||
      analysis.spinLoft3dDeg === null) return [];
  const angle = analysis.spinLoft3dDeg / DEG;
  const sine = Math.sin(angle);
  if (Math.abs(sine) <= EPSILON) throw new RangeError("defined D-plane must have a nonzero sector");
  return Array.from({ length: segments + 1 }, (_, index) => {
    const fraction = index / segments;
    const travelWeight = Math.sin((1 - fraction) * angle) / sine;
    const faceWeight = Math.sin(fraction * angle) / sine;
    return unit([
      travelWeight * analysis.travelDirectionUnit![0] + faceWeight * analysis.faceNormalUnit[0],
      travelWeight * analysis.travelDirectionUnit![1] + faceWeight * analysis.faceNormalUnit[1],
      travelWeight * analysis.travelDirectionUnit![2] + faceWeight * analysis.faceNormalUnit[2],
    ], "sector direction");
  });
}
