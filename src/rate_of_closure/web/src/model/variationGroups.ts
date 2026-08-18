import type { NoiseSpecTs, PerturbationGroupTs } from "./variationSchema";

const MATRIX_TOLERANCE = 1e-12;

const identityMatrix = (size: number): number[][] =>
  Array.from({ length: size }, (_row, i) =>
    Array.from({ length: size }, (_column, j) => (i === j ? 1 : 0)),
  );

/** Deterministic Jacobi eigensolver for the small symmetric plan matrices. */
const symmetricEigen = (
  source: number[][],
): { values: number[]; vectors: number[][] } => {
  const size = source.length;
  const matrix = source.map((row) => [...row]);
  const vectors = identityMatrix(size);
  const maxIterations = Math.max(1, 50 * size * size);
  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    let p = 0;
    let q = 0;
    let largest = 0;
    for (let i = 0; i < size; i += 1) {
      for (let j = i + 1; j < size; j += 1) {
        const magnitude = Math.abs(matrix[i][j]);
        if (magnitude > largest) {
          largest = magnitude;
          p = i;
          q = j;
        }
      }
    }
    if (largest <= MATRIX_TOLERANCE) break;

    const angle = 0.5 * Math.atan2(2 * matrix[p][q], matrix[q][q] - matrix[p][p]);
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    const app = matrix[p][p];
    const aqq = matrix[q][q];
    const apq = matrix[p][q];
    matrix[p][p] = cosine * cosine * app - 2 * sine * cosine * apq + sine * sine * aqq;
    matrix[q][q] = sine * sine * app + 2 * sine * cosine * apq + cosine * cosine * aqq;
    matrix[p][q] = 0;
    matrix[q][p] = 0;
    for (let i = 0; i < size; i += 1) {
      if (i !== p && i !== q) {
        const aip = matrix[i][p];
        const aiq = matrix[i][q];
        matrix[i][p] = cosine * aip - sine * aiq;
        matrix[p][i] = matrix[i][p];
        matrix[i][q] = sine * aip + cosine * aiq;
        matrix[q][i] = matrix[i][q];
      }
      const vip = vectors[i][p];
      const viq = vectors[i][q];
      vectors[i][p] = cosine * vip - sine * viq;
      vectors[i][q] = sine * vip + cosine * viq;
    }
  }
  return { values: matrix.map((row, i) => row[i]), vectors };
};

export function validateGroupMatrix(group: PerturbationGroupTs): void {
  const size = group.specIds.length;
  if (size < 2) throw new Error("correlation group needs at least two specIds");
  if (group.matrix.length !== size || group.matrix.some((row) => row.length !== size)) {
    throw new Error("matrix shape must match specIds");
  }
  if (group.matrix.some((row) => row.some((value) => !Number.isFinite(value)))) {
    throw new Error("matrix entries must be finite");
  }
  for (let i = 0; i < size; i += 1) {
    for (let j = i + 1; j < size; j += 1) {
      if (Math.abs(group.matrix[i][j] - group.matrix[j][i]) > MATRIX_TOLERANCE) {
        throw new Error("matrix must be symmetric");
      }
    }
  }
  const { values } = symmetricEigen(group.matrix);
  if (Math.min(...values) < -MATRIX_TOLERANCE) {
    throw new Error("matrix must be positive semidefinite");
  }
  if (group.matrixKind === "correlation") {
    if (group.matrix.some((row, i) => Math.abs(row[i] - 1) > MATRIX_TOLERANCE)) {
      throw new Error("correlation matrix must have a unit diagonal");
    }
  } else if (group.matrixKind === "covariance") {
    if (group.matrix.some((row, i) => !(row[i] > 0))) {
      throw new Error("covariance diagonal must be positive");
    }
  } else {
    throw new Error(`matrixKind must be correlation or covariance: ${String(group.matrixKind)}`);
  }
}

export function covarianceMatrix(
  group: PerturbationGroupTs,
  specs: NoiseSpecTs[],
): number[][] {
  if (group.matrixKind === "covariance") return group.matrix.map((row) => [...row]);
  return group.matrix.map((row, i) =>
    row.map((value, j) => specs[i].scale * value * specs[j].scale),
  );
}

/** Symmetric PSD square root, matching the Python engine's factor semantics. */
export function covarianceFactor(covariance: number[][]): number[][] {
  const { values, vectors } = symmetricEigen(covariance);
  const roots = values.map((value) => Math.sqrt(Math.max(value, 0)));
  return vectors.map((_row, i) =>
    vectors.map((_unused, j) =>
      roots.reduce(
        (sum, root, k) => sum + vectors[i][k] * root * vectors[j][k],
        0,
      ),
    ),
  );
}
