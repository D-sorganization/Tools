/** Small immutable 3-D rotation helpers shared by simulation and rendering. */

import type { Vec3 } from "./impactPhysics";

export type Mat3 = [Vec3, Vec3, Vec3];
type Quaternion = [number, number, number, number];

export const IDENTITY_ROTATION: Mat3 = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

export function rodrigues(omega: Vec3, dt: number): Mat3 {
  const magnitude = Math.hypot(...omega);
  const theta = magnitude * dt;
  if (Math.abs(theta) < 1e-12) return IDENTITY_ROTATION.map((row) => [...row]) as Mat3;
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

export function applyRotation(matrix: Mat3, vector: Vec3): Vec3 {
  return [
    matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
    matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
    matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
  ];
}

export function rotationFromColumns(first: Vec3, second: Vec3, third: Vec3): Mat3 {
  return [
    [first[0], second[0], third[0]],
    [first[1], second[1], third[1]],
    [first[2], second[2], third[2]],
  ];
}

function quaternionFromRotation(matrix: Mat3): Quaternion {
  const trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
  let quaternion: Quaternion;
  if (trace > 0) {
    const scale = Math.sqrt(trace + 1) * 2;
    quaternion = [
      scale / 4,
      (matrix[2][1] - matrix[1][2]) / scale,
      (matrix[0][2] - matrix[2][0]) / scale,
      (matrix[1][0] - matrix[0][1]) / scale,
    ];
  } else {
    const axis = matrix[0][0] > matrix[1][1]
      ? (matrix[0][0] > matrix[2][2] ? 0 : 2)
      : (matrix[1][1] > matrix[2][2] ? 1 : 2);
    const next = (axis + 1) % 3;
    const last = (axis + 2) % 3;
    const scale = Math.sqrt(1 + matrix[axis][axis] - matrix[next][next] - matrix[last][last]) * 2;
    const vector: Vec3 = [0, 0, 0];
    vector[axis] = scale / 4;
    vector[next] = (matrix[next][axis] + matrix[axis][next]) / scale;
    vector[last] = (matrix[last][axis] + matrix[axis][last]) / scale;
    quaternion = [
      (matrix[last][next] - matrix[next][last]) / scale,
      vector[0], vector[1], vector[2],
    ];
  }
  const magnitude = Math.hypot(...quaternion);
  return quaternion.map((value) => value / magnitude) as Quaternion;
}

function rotationFromQuaternion([w, x, y, z]: Quaternion): Mat3 {
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}

/** Shortest-arc interpolation of two proper rotation matrices. */
export function slerpRotation(start: Mat3, end: Mat3, alpha: number): Mat3 {
  if (!Number.isFinite(alpha) || alpha < 0 || alpha > 1) {
    throw new RangeError("alpha must be finite and within [0, 1]");
  }
  const first = quaternionFromRotation(start);
  let second = quaternionFromRotation(end);
  let cosine = first.reduce((sum, value, index) => sum + value * second[index], 0);
  if (cosine < 0) {
    second = second.map((value) => -value) as Quaternion;
    cosine = -cosine;
  }
  let result: Quaternion;
  if (cosine > 0.9995) {
    result = first.map((value, index) => value + alpha * (second[index] - value)) as Quaternion;
  } else {
    const angle = Math.acos(Math.max(-1, Math.min(1, cosine)));
    const denominator = Math.sin(angle);
    const startWeight = Math.sin((1 - alpha) * angle) / denominator;
    const endWeight = Math.sin(alpha * angle) / denominator;
    result = first.map((value, index) =>
      startWeight * value + endWeight * second[index]) as Quaternion;
  }
  const magnitude = Math.hypot(...result);
  return rotationFromQuaternion(result.map((value) => value / magnitude) as Quaternion);
}
