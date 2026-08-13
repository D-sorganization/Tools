/** Pure geometric variability analysis over retained swing traces. */

import type { Vec3 } from "./simulation";
import type {
  SwingTrialStatusTs,
  SwingVariationResultTs,
} from "./variationSwingEnsemble";

export type SwingPointKindTs = "pivot" | "wrist" | "clubhead";

export interface SwingTraceRowTs {
  trialIndex: number;
  points: Vec3[];
  timesS: number[];
  status: SwingTrialStatusTs;
}

export interface GeometricVariabilityTs {
  sampleTimesS: number[];
  validTrialCount: number[];
  meanPositionsM: Vec3[];
  rmsRadiusM: number[];
  principalSigmaM: number[];
  principalAxes: Vec3[];
  quietMask: boolean[];
  quietIntervals: Array<{ startIndex: number; endIndex: number }>;
  quietThresholdM: number;
  coordinateFrame: "app_frame:x_target,y_up,z_right";
  alignmentBasis: "common_simulation_time_s";
}

export function swingTraceRows(
  ensemble: SwingVariationResultTs,
  pointKind: SwingPointKindTs,
): SwingTraceRowTs[] {
  return ensemble.runs.flatMap((trial) => {
    if (trial.run === null) return [];
    return [{
      trialIndex: trial.trialIndex,
      status: trial.status,
      timesS: trial.run.swing.map((sample) => sample.t),
      points: trial.run.swing.map((sample) => {
        if (pointKind === "clubhead") return sample.position;
        const index = pointKind === "pivot"
          ? 0
          : Math.max(sample.joints.length - 2, 0);
        return sample.joints[index] ?? sample.position;
      }),
    }];
  });
}

export function geometricVariability(
  traces: SwingTraceRowTs[],
  quietThresholdM: number,
): GeometricVariabilityTs {
  if (!Number.isFinite(quietThresholdM) || quietThresholdM <= 0) {
    throw new Error("quietThresholdM must be finite and greater than zero");
  }
  if (traces.length === 0) return emptyVariability(quietThresholdM);
  const count = Math.min(...traces.map((trace) => trace.points.length));
  const sampleTimesS = traces[0].timesS.slice(0, count);
  const validTrialCount = Array(count).fill(traces.length) as number[];
  const meanPositionsM: Vec3[] = [];
  const rmsRadiusM: number[] = [];
  const principalSigmaM: number[] = [];
  const principalAxes: Vec3[] = [];
  for (let sample = 0; sample < count; sample += 1) {
    const points = traces.map((trace) => trace.points[sample]);
    const mean = vectorMean(points);
    const centered = points.map((point) => subtract(point, mean));
    meanPositionsM.push(mean);
    rmsRadiusM.push(Math.sqrt(
      centered.reduce((sum, point) => sum + dot(point, point), 0) / points.length,
    ));
    const covariance = covarianceMatrix(centered);
    const principal = largestEigenpair(covariance);
    principalSigmaM.push(Math.sqrt(Math.max(principal.value, 0)));
    principalAxes.push(principal.axis);
  }
  const quietMask = rmsRadiusM.map(
    (radius, index) => validTrialCount[index] >= 2 && radius <= quietThresholdM,
  );
  return {
    sampleTimesS,
    validTrialCount,
    meanPositionsM,
    rmsRadiusM,
    principalSigmaM,
    principalAxes,
    quietMask,
    quietIntervals: contiguousTrueIntervals(quietMask),
    quietThresholdM,
    coordinateFrame: "app_frame:x_target,y_up,z_right",
    alignmentBasis: "common_simulation_time_s",
  };
}

const emptyVariability = (quietThresholdM: number): GeometricVariabilityTs => ({
  sampleTimesS: [], validTrialCount: [], meanPositionsM: [], rmsRadiusM: [],
  principalSigmaM: [], principalAxes: [], quietMask: [], quietIntervals: [],
  quietThresholdM,
  coordinateFrame: "app_frame:x_target,y_up,z_right",
  alignmentBasis: "common_simulation_time_s",
});

const vectorMean = (points: Vec3[]): Vec3 => [0, 1, 2].map(
  (axis) => points.reduce((sum, point) => sum + point[axis], 0) / points.length,
) as Vec3;
const subtract = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (value: Vec3): number => Math.hypot(...value);

function covarianceMatrix(centered: Vec3[]): number[][] {
  if (centered.length < 2) return [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  return [0, 1, 2].map((row) => [0, 1, 2].map((column) =>
    centered.reduce((sum, point) => sum + point[row] * point[column], 0)
      / (centered.length - 1),
  ));
}

function largestEigenpair(matrix: number[][]): { value: number; axis: Vec3 } {
  let axis: Vec3 = [1, 1, 1];
  for (let iteration = 0; iteration < 24; iteration += 1) {
    const next = matrix.map((row) => dot(row as Vec3, axis)) as Vec3;
    const magnitude = norm(next);
    if (magnitude < 1e-15) return { value: 0, axis: [1, 0, 0] };
    axis = next.map((value) => value / magnitude) as Vec3;
  }
  const applied = matrix.map((row) => dot(row as Vec3, axis)) as Vec3;
  const value = dot(axis, applied);
  const largest = axis.reduce(
    (best, value, index) => Math.abs(value) > Math.abs(axis[best]) ? index : best,
    0,
  );
  if (axis[largest] < 0) axis = axis.map((value) => -value) as Vec3;
  return { value, axis };
}

function contiguousTrueIntervals(mask: boolean[]): Array<{ startIndex: number; endIndex: number }> {
  const intervals: Array<{ startIndex: number; endIndex: number }> = [];
  let start: number | null = null;
  mask.forEach((value, index) => {
    if (value && start === null) start = index;
    if (!value && start !== null) {
      intervals.push({ startIndex: start, endIndex: index - 1 });
      start = null;
    }
  });
  if (start !== null) intervals.push({ startIndex: start, endIndex: mask.length - 1 });
  return intervals;
}
