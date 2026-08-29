/** Browser persistence and polynomial point-fit helpers for torque profiles. */

import {
  FitMetadata,
  JointTorqueAssignment,
  PrescribedTorqueProfile,
  TorquePolynomial,
  TorqueProfileSource,
} from "./torqueProfiles";
import {
  DOUBLE_PENDULUM_MODEL_ID,
  SHOULDER_JOINT_ID,
  WRIST_JOINT_ID,
} from "./doublePendulum";

export const TORQUE_PROFILE_STORAGE_KEY =
  "rate_of_closure.torque_profiles.schema_v1";

export interface TorqueSampleRow {
  timeS: number;
  shoulderNm: number;
  wristNm: number;
}

export interface TorqueFit {
  rows: readonly TorqueSampleRow[];
  shoulder: TorquePolynomial;
  wrist: TorquePolynomial;
}

export function starterTorqueProfile(): PrescribedTorqueProfile {
  return new PrescribedTorqueProfile({
    profileId: "profile.web.starter_drive.v1",
    modelId: DOUBLE_PENDULUM_MODEL_ID,
    name: "Web Starter Drive",
    description: "Editable starter shoulder and wrist drive for the web workbench.",
    source: TorqueProfileSource.DIRECT,
    sourceMetadata: { author: "upstreamdrift", workflow: "web_starter" },
    createdAtUtc: "2026-08-05T12:00:00Z",
    modifiedAtUtc: "2026-08-05T12:00:00Z",
    timeDomainS: [0, 1.5],
    assignments: [
      new JointTorqueAssignment(
        SHOULDER_JOINT_ID,
        new TorquePolynomial([18, -12]),
      ),
      new JointTorqueAssignment(
        WRIST_JOINT_ID,
        new TorquePolynomial([-4, 6]),
      ),
    ],
  });
}

export function loadTorqueProfileLibrary(
  storage: Storage = window.localStorage,
): readonly PrescribedTorqueProfile[] {
  const text = storage.getItem(TORQUE_PROFILE_STORAGE_KEY);
  if (text === null) return Object.freeze([starterTorqueProfile()]);
  let values: unknown;
  try {
    values = JSON.parse(text);
  } catch (error) {
    throw new Error(`Invalid saved torque-profile library: ${String(error)}`);
  }
  if (!Array.isArray(values) || values.length === 0) {
    throw new Error("Saved torque-profile library must be a nonempty array");
  }
  const profiles = values.map(PrescribedTorqueProfile.fromJsonObject);
  if (new Set(profiles.map((profile) => profile.profileId)).size !== profiles.length) {
    throw new Error("Saved torque-profile IDs must be unique");
  }
  return Object.freeze(profiles);
}

export function saveTorqueProfileLibrary(
  profiles: readonly PrescribedTorqueProfile[],
  storage: Storage = window.localStorage,
): void {
  if (profiles.length === 0) throw new Error("Torque-profile library cannot be empty");
  const ordered = [...profiles].sort((left, right) =>
    left.profileId.localeCompare(right.profileId),
  );
  storage.setItem(
    TORQUE_PROFILE_STORAGE_KEY,
    JSON.stringify(ordered.map((profile) => profile.toJsonObject())),
  );
}

export function parseCoefficientText(text: string): readonly number[] {
  const tokens = text.split(/[\s,]+/).filter(Boolean);
  if (tokens.length === 0) throw new Error("At least one coefficient is required");
  const coefficients = tokens.map(Number);
  if (coefficients.some((value) => !Number.isFinite(value))) {
    throw new Error("Coefficients must be finite numbers");
  }
  return Object.freeze(coefficients);
}

export function parseTorqueSampleRows(text: string): readonly TorqueSampleRow[] {
  const rows = text.split(/\r?\n/).filter((line) => line.trim() !== "").map(
    (line, index) => {
      const values = line.split(/[\s,]+/).filter(Boolean).map(Number);
      if (values.length !== 3 || values.some((value) => !Number.isFinite(value))) {
        throw new Error(`Sample row ${index + 1} must contain time, shoulder, wrist`);
      }
      return { timeS: values[0], shoulderNm: values[1], wristNm: values[2] };
    },
  );
  if (rows.length < 2) throw new Error("At least two sample rows are required");
  for (let index = 1; index < rows.length; index += 1) {
    if (!(rows[index].timeS > rows[index - 1].timeS)) {
      throw new Error("Sample times must be strictly increasing");
    }
  }
  return Object.freeze(rows);
}

function designMatrix(times: readonly number[], degree: number): number[][] {
  // ⚡ Bolt Optimization: Avoid Array.from({ length }) and .map overhead by pre-allocating
  // arrays and using standard for-loops, eliminating closure overhead in hot math paths.
  const matrix = new Array(times.length);
  const cols = degree + 1;
  for (let i = 0; i < times.length; i++) {
    const time = times[i];
    const row = new Array(cols);
    for (let power = 0; power < cols; power++) {
      row[power] = time ** power;
    }
    matrix[i] = row;
  }
  return matrix;
}

function solveLinearSystem(matrix: number[][], values: number[]): number[] {
  const augmented = matrix.map((row, index) => [...row, values[index]]);
  for (let column = 0; column < matrix.length; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < matrix.length; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    if (Math.abs(augmented[pivot][column]) < 1e-12) throw new Error("Polynomial fit design is singular");
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    for (let j = column; j <= matrix.length; j += 1) augmented[column][j] /= divisor;
    for (let row = 0; row < matrix.length; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let j = column; j <= matrix.length; j += 1) augmented[row][j] -= factor * augmented[column][j];
    }
  }
  return augmented.map((row) => row[matrix.length]);
}

function conditionNumber(design: number[][]): number {
  const columns = design[0].length;
  if (columns === 1) return 1;
  const gram = Array.from({ length: columns }, (_, i) =>
    Array.from({ length: columns }, (_, j) => design.reduce((sum, row) => sum + row[i] * row[j], 0)));
  const eigen = gram.map((row) => [...row]);
  for (let iteration = 0; iteration < 50; iteration += 1) {
    let p = 0; let q = 1;
    for (let i = 0; i < columns; i += 1) for (let j = i + 1; j < columns; j += 1) {
      if (Math.abs(eigen[i][j]) > Math.abs(eigen[p]?.[q] ?? 0)) [p, q] = [i, j];
    }
    if (Math.abs(eigen[p]?.[q] ?? 0) < 1e-12) break;
    const angle = 0.5 * Math.atan2(2 * eigen[p][q], eigen[q][q] - eigen[p][p]);
    const c = Math.cos(angle); const s = Math.sin(angle);
    for (let k = 0; k < columns; k += 1) {
      const epk = eigen[p][k]; const eqk = eigen[q][k];
      eigen[p][k] = c * epk - s * eqk; eigen[q][k] = s * epk + c * eqk;
    }
    for (let k = 0; k < columns; k += 1) {
      const ekp = eigen[k][p]; const ekq = eigen[k][q];
      eigen[k][p] = c * ekp - s * ekq; eigen[k][q] = s * ekp + c * ekq;
    }
  }
  const values = eigen.map((row, index) => row[index]).filter((value) => value > 1e-12);
  if (values.length !== columns) throw new Error("Polynomial fit design is singular");
  return Math.sqrt(Math.max(...values) / Math.min(...values));
}

function binomial(n: number, k: number): number {
  let value = 1;
  for (let index = 1; index <= k; index += 1) value *= (n - index + 1) / index;
  return value;
}

function physicalCoefficients(
  normalized: readonly number[],
  startS: number,
  endS: number,
): number[] {
  const alpha = 2 / (endS - startS);
  const beta = -1 - alpha * startS;
  return normalized.map((_, physicalPower) => normalized.reduce((sum, value, normalizedPower) => {
    if (normalizedPower < physicalPower) return sum;
    return sum + value * binomial(normalizedPower, physicalPower)
      * alpha ** physicalPower * beta ** (normalizedPower - physicalPower);
  }, 0));
}

function fitPolynomial(
  rows: readonly TorqueSampleRow[],
  key: "shoulderNm" | "wristNm",
  degree: number,
): TorquePolynomial {
  const count = rows.length;
  const meanTorque = rows.reduce((sum, row) => sum + row[key], 0) / count;
  const startS = rows[0].timeS;
  const endS = rows[rows.length - 1].timeS;
  const normalizedTimes = rows.map((row) => 2 * (row.timeS - startS) / (endS - startS) - 1);
  const design = designMatrix(normalizedTimes, degree);
  let fitCondition: number;
  try {
    fitCondition = conditionNumber(design);
  } catch {
    throw new Error("Polynomial fit condition number exceeds 1e8");
  }
  if (fitCondition > 1e8) throw new Error("Polynomial fit condition number exceeds 1e8");
  const normal = design[0].map((_, i) => design[0].map((__, j) =>
    design.reduce((sum, row) => sum + row[i] * row[j], 0)));
  const rhs = design[0].map((_, i) => design.reduce((sum, row, index) => sum + row[i] * rows[index][key], 0));
  const coefficients = physicalCoefficients(solveLinearSystem(normal, rhs), startS, endS);
  const residuals = rows.map((row) => row[key] - coefficients.reduce(
    (sum, value, power) => sum + value * row.timeS ** power, 0));
  const squaredError = residuals.reduce((sum, value) => sum + value ** 2, 0);
  const centeredTorque = rows.reduce((sum, row) => sum + (row[key] - meanTorque) ** 2, 0);
  const metadata = new FitMetadata({
    degree,
    rmseNm: Math.sqrt(squaredError / count),
    maxAbsErrorNm: Math.max(...residuals.map(Math.abs)),
    rSquared: centeredTorque === 0 ? (squaredError === 0 ? 1 : 0) : 1 - squaredError / centeredTorque,
    conditionNumber: fitCondition,
  });
  return new TorquePolynomial(coefficients, metadata);
}

export function fitTorqueRows(text: string, degree: number): TorqueFit {
  const rows = parseTorqueSampleRows(text);
  if (!Number.isInteger(degree) || degree < 0 || degree > 3) {
    throw new Error("Polynomial degree must be an integer from 0 through 3");
  }
  if (rows.length <= degree) throw new Error(`Degree ${degree} requires at least ${degree + 1} sample rows`);
  return Object.freeze({
    rows,
    shoulder: fitPolynomial(rows, "shoulderNm", degree),
    wrist: fitPolynomial(rows, "wristNm", degree),
  });
}
