/** Focused three-dimensional mass-property and rigid-transform mathematics. */

export type Vec3 = [number, number, number];
export type Matrix3 = [Vec3, Vec3, Vec3];

const TOLERANCE = 1e-10;

function determinant(matrix: Matrix3): number {
  const [a, b, c] = matrix;
  return (
    a[0] * (b[1] * c[2] - b[2] * c[1]) -
    a[1] * (b[0] * c[2] - b[2] * c[0]) +
    a[2] * (b[0] * c[1] - b[1] * c[0])
  );
}

export function transpose(matrix: Matrix3): Matrix3 {
  return matrix[0].map((_, column) =>
    matrix.map((row) => row[column]),
  ) as Matrix3;
}

export function multiply(left: Matrix3, right: Matrix3): Matrix3 {
  const rightT = transpose(right);
  return left.map((row) =>
    rightT.map((column) =>
      row.reduce((sum, value, index) => sum + value * column[index], 0),
    ),
  ) as Matrix3;
}

export function matrixVector(matrix: Matrix3, vector: Vec3): Vec3 {
  return matrix.map((row) =>
    row.reduce((sum, value, index) => sum + value * vector[index], 0),
  ) as Vec3;
}

function symmetricEigenvalues(matrix: Matrix3): Vec3 {
  const traceThird = (matrix[0][0] + matrix[1][1] + matrix[2][2]) / 3;
  const offDiagonal = matrix[0][1] ** 2 + matrix[0][2] ** 2 + matrix[1][2] ** 2;
  const spread =
    (matrix[0][0] - traceThird) ** 2 +
    (matrix[1][1] - traceThird) ** 2 +
    (matrix[2][2] - traceThird) ** 2 +
    2 * offDiagonal;
  const scale = Math.sqrt(spread / 6);
  if (scale === 0) return [traceThird, traceThird, traceThird];
  const centered = matrix.map((row, i) =>
    row.map((value, j) => (value - (i === j ? traceThird : 0)) / scale),
  ) as Matrix3;
  const angle =
    Math.acos(Math.max(-1, Math.min(1, determinant(centered) / 2))) / 3;
  const largest = traceThird + 2 * scale * Math.cos(angle);
  const smallest = traceThird + 2 * scale * Math.cos(angle + (2 * Math.PI) / 3);
  const middle = 3 * traceThird - largest - smallest;
  return [smallest, middle, largest].sort((a, b) => a - b) as Vec3;
}

export function validateInertia(
  matrix: Matrix3,
  name: string,
  positive: boolean,
): void {
  for (let row = 0; row < 3; row += 1) {
    for (let column = row + 1; column < 3; column += 1) {
      if (Math.abs(matrix[row][column] - matrix[column][row]) > TOLERANCE) {
        throw new Error(`${name} must be symmetric`);
      }
    }
  }
  const moments = symmetricEigenvalues(matrix);
  if (moments[0] < -TOLERANCE || (positive && moments[0] <= 0)) {
    throw new Error(
      `${name} must be ${positive ? "positive definite" : "positive semidefinite"}`,
    );
  }
  if (moments[2] - moments[1] - moments[0] > TOLERANCE) {
    throw new Error(
      `${name} principal moments must satisfy the triangle inequality`,
    );
  }
}

export function validateRotation(rotation: Matrix3): void {
  const product = multiply(transpose(rotation), rotation);
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      const expected = row === column ? 1 : 0;
      if (Math.abs(product[row][column] - expected) > TOLERANCE) {
        throw new Error("rotation must be proper orthonormal");
      }
    }
  }
  if (Math.abs(determinant(rotation) - 1) > TOLERANCE) {
    throw new Error("rotation must be proper orthonormal");
  }
}
