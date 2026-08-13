/** Numerically bounded 3-D covariance helpers for variation geometry. */

import type { Vec3 } from "./simulation";
import type { DispersionAdequacyTs } from "./variationGeometry";

interface Eigenpairs {
  values: Vec3;
  axes: [Vec3, Vec3, Vec3];
}

export interface SampleDispersionTs {
  mean: Vec3;
  rmsRadiusM: number;
  eigenvaluesM2: Vec3;
  principalSigmaM: number;
  principalAxis: Vec3;
  adequacy: DispersionAdequacyTs;
}

const MIN_FULL_RANK_SAMPLES = 4;
const EIGENVALUE_ROUNDOFF_FACTOR = 64;

export function sampleDispersion(points: Vec3[]): SampleDispersionTs {
  const mean = vectorMean(points);
  const centered = points.map((point) => subtract(point, mean));
  const rmsRadiusM = Math.sqrt(
    centered.reduce((sum, point) => sum + dot(point, point), 0) / points.length,
  );
  const eigenpairs = symmetricEigenpairs(covarianceMatrix(centered));
  return {
    mean,
    rmsRadiusM,
    eigenvaluesM2: eigenpairs.values,
    principalSigmaM: Math.sqrt(Math.max(eigenpairs.values[0], 0)),
    principalAxis: eigenpairs.axes[0],
    adequacy: classifyAdequacy(points.length, eigenpairs.values),
  };
}

export function confidenceRadiusScale(probability: number): number {
  const targetSurvival = 1 - probability;
  let lower = 0;
  let upper = 1;
  while (chiSquareThreeSurvival(upper) > targetSurvival) upper *= 2;
  for (let iteration = 0; iteration < 96; iteration += 1) {
    const midpoint = (lower + upper) / 2;
    if (chiSquareThreeSurvival(midpoint) > targetSurvival) lower = midpoint;
    else upper = midpoint;
  }
  return Math.sqrt((lower + upper) / 2);
}

const vectorMean = (points: Vec3[]): Vec3 => [0, 1, 2].map(
  (axis) => points.reduce((sum, point) => sum + point[axis], 0) / points.length,
) as Vec3;
const subtract = (left: Vec3, right: Vec3): Vec3 => [
  left[0] - right[0], left[1] - right[1], left[2] - right[2],
];
const dot = (left: Vec3, right: Vec3): number => (
  left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
);

function covarianceMatrix(centered: Vec3[]): number[][] {
  if (centered.length < 2) return [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  return [0, 1, 2].map((row) => [0, 1, 2].map((column) =>
    centered.reduce((sum, point) => sum + point[row] * point[column], 0)
      / (centered.length - 1),
  ));
}

function symmetricEigenpairs(matrix: number[][]): Eigenpairs {
  const values = matrix.map((row) => [...row]);
  const vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let iteration = 0; iteration < 48; iteration += 1) {
    const [row, column, magnitude] = largestOffDiagonal(values);
    const scale = Math.max(...values.flat().map(Math.abs), Number.MIN_VALUE);
    if (magnitude <= 1e-14 * scale) break;
    rotateJacobi(values, vectors, row, column);
  }
  const pairs = [0, 1, 2].map((index) => ({
    value: values[index][index],
    axis: canonicalAxis(vectors.map((row) => row[index]) as Vec3),
  })).sort((left, right) => right.value - left.value);
  return {
    values: pairs.map((pair) => pair.value) as Vec3,
    axes: pairs.map((pair) => pair.axis) as [Vec3, Vec3, Vec3],
  };
}

function largestOffDiagonal(matrix: number[][]): [number, number, number] {
  let selected: [number, number, number] = [0, 1, Math.abs(matrix[0][1])];
  ([[0, 2], [1, 2]] as Array<[number, number]>).forEach(([row, column]) => {
    const magnitude = Math.abs(matrix[row][column]);
    if (magnitude > selected[2]) selected = [row, column, magnitude];
  });
  return selected;
}

function rotateJacobi(
  matrix: number[][],
  vectors: number[][],
  row: number,
  column: number,
): void {
  const angle = 0.5 * Math.atan2(
    2 * matrix[row][column], matrix[column][column] - matrix[row][row],
  );
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const app = matrix[row][row];
  const aqq = matrix[column][column];
  const apq = matrix[row][column];
  matrix[row][row] = cosine * cosine * app - 2 * sine * cosine * apq + sine * sine * aqq;
  matrix[column][column] = sine * sine * app + 2 * sine * cosine * apq + cosine * cosine * aqq;
  matrix[row][column] = 0;
  matrix[column][row] = 0;
  for (let index = 0; index < 3; index += 1) {
    if (index !== row && index !== column) {
      const air = matrix[index][row];
      const aiq = matrix[index][column];
      matrix[index][row] = cosine * air - sine * aiq;
      matrix[row][index] = matrix[index][row];
      matrix[index][column] = sine * air + cosine * aiq;
      matrix[column][index] = matrix[index][column];
    }
    const vir = vectors[index][row];
    const viq = vectors[index][column];
    vectors[index][row] = cosine * vir - sine * viq;
    vectors[index][column] = sine * vir + cosine * viq;
  }
}

function canonicalAxis(axis: Vec3): Vec3 {
  const largest = axis.reduce(
    (best, value, index) => Math.abs(value) > Math.abs(axis[best]) ? index : best,
    0,
  );
  return axis[largest] < 0 ? axis.map((value) => -value) as Vec3 : axis;
}

function classifyAdequacy(count: number, eigenvalues: Vec3): DispersionAdequacyTs {
  if (count < 2) return "insufficient-samples";
  if (!eigenvalues.every(Number.isFinite)) return "invalid-covariance";
  const scale = Math.max(...eigenvalues.map(Math.abs), Number.MIN_VALUE);
  const tolerance = EIGENVALUE_ROUNDOFF_FACTOR * Number.EPSILON * scale;
  if (Math.min(...eigenvalues) < -tolerance) return "invalid-covariance";
  const rank = eigenvalues.filter((value) => value > tolerance).length;
  return count < MIN_FULL_RANK_SAMPLES || eigenvalues[0] === 0 || rank < 3
    ? "rank-deficient" : "estimable";
}

function chiSquareThreeSurvival(value: number): number {
  const root = Math.sqrt(value / 2);
  return complementaryErrorFunction(root)
    + Math.sqrt(2 * value / Math.PI) * Math.exp(-value / 2);
}

function complementaryErrorFunction(value: number): number {
  const scaled = 1 / (1 + 0.5 * Math.abs(value));
  const polynomial = scaled * Math.exp(
    -value * value - 1.26551223 + scaled * (
      1.00002368 + scaled * (
        0.37409196 + scaled * (
          0.09678418 + scaled * (
            -0.18628806 + scaled * (
              0.27886807 + scaled * (
                -1.13520398 + scaled * (
                  1.48851587 + scaled * (-0.82215223 + scaled * 0.17087277)
                )
              )
            )
          )
        )
      )
    ),
  );
  return value >= 0 ? polynomial : 2 - polynomial;
}
