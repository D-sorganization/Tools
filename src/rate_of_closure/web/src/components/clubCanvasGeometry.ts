import { type ImpactScenario } from "../model/impact";

export type Vec3 = [number, number, number];

const FACE_W = 0.058;
const FACE_H = 0.028;
const BODY_DEPTH = 0.11;
export const SHAFT_LEN = 0.3;

export function rodrigues(omega: Vec3, dt: number): number[][] {
  const magnitude = Math.hypot(...omega);
  const theta = magnitude * dt;
  if (Math.abs(theta) < 1e-12) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const [x, y, z] = omega.map((component) => component / magnitude);
  const cosine = Math.cos(theta);
  const sine = Math.sin(theta);
  const complement = 1 - cosine;
  return [
    [complement * x * x + cosine, complement * x * y - sine * z, complement * x * z + sine * y],
    [complement * x * y + sine * z, complement * y * y + cosine, complement * y * z - sine * x],
    [complement * x * z - sine * y, complement * y * z + sine * x, complement * z * z + cosine],
  ];
}

export function apply(matrix: number[][], vector: Vec3): Vec3 {
  return [
    matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
    matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
    matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
  ];
}

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
