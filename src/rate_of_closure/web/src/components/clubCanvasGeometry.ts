import { type ImpactScenario } from "../model/impact";
import { applyRotation, rodrigues } from "../model/rotation";

export type Vec3 = [number, number, number];

const FACE_W = 0.058;
const FACE_H = 0.028;
const BODY_DEPTH = 0.11;
export const SHAFT_LEN = 0.3;

export { rodrigues };
export const apply = applyRotation;

export const add = (first: Vec3, second: Vec3): Vec3 => [
  first[0] + second[0], first[1] + second[1], first[2] + second[2],
];

/** Orthographic projection under a user-controlled orbit camera. */
export function project(
  vector: Vec3,
  width: number,
  height: number,
  zoom: number,
  azimuth: number,
  elevation: number,
): [number, number] {
  const sinAzimuth = Math.sin(azimuth);
  const cosAzimuth = Math.cos(azimuth);
  const sinElevation = Math.sin(elevation);
  const cosElevation = Math.cos(elevation);
  const screenX = vector[0] * sinAzimuth - vector[2] * cosAzimuth;
  const screenY =
    -sinElevation * cosAzimuth * vector[0] +
    cosElevation * vector[1] -
    sinElevation * sinAzimuth * vector[2];
  const scale = Math.min(width, height) * zoom;
  return [width / 2 + screenX * scale, height * 0.62 - screenY * scale];
}

/** Project an application-frame point around an explicit camera target. */
export function projectAroundTarget(
  vector: Vec3,
  target: Vec3,
  width: number,
  height: number,
  zoom: number,
  azimuth: number,
  elevation: number,
): [number, number] {
  return project(
    [vector[0] - target[0], vector[1] - target[1], vector[2] - target[2]],
    width,
    height,
    zoom,
    azimuth,
    elevation,
  );
}

export function headParts(scenario: ImpactScenario) {
  const depth = scenario.comToFaceMm / 1000;
  const lie = (scenario.lieAngleDeg * Math.PI) / 180;
  const face: Vec3[] = [
    [depth, -FACE_H, -FACE_W], [depth, -FACE_H, FACE_W],
    [depth, FACE_H, FACE_W], [depth, FACE_H, -FACE_W],
    [depth, -FACE_H, -FACE_W],
  ];
  const back = face.map((point): Vec3 => [point[0] - BODY_DEPTH, point[1], point[2]]);
  const hosel: Vec3 = [depth - 0.02, FACE_H, -FACE_W];
  const shaftEnd: Vec3 = [
    hosel[0], hosel[1] + Math.sin(lie) * SHAFT_LEN, hosel[2] - Math.cos(lie) * SHAFT_LEN,
  ];
  const impact: Vec3 = [
    depth, scenario.impactOffsetHighMm / 1000, scenario.impactOffsetToeMm / 1000,
  ];
  return { face, back, hosel, shaftEnd, impact };
}
