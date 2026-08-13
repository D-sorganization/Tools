/** Strict geometric dispersion parity with the shared Python authority. */

import type { Vec3 } from "./simulation";
import type {
  SwingTrialStatusTs,
  SwingVariationResultTs,
} from "./variationSwingEnsemble";
import {
  confidenceRadiusScale,
  sampleDispersion,
} from "./variationDispersionMath";

export const DISPERSION_METRICS = [
  "rms-radius",
  "largest-principal-sigma",
  "confidence-ellipsoid-volume",
] as const;
export type DispersionMetricTs = typeof DISPERSION_METRICS[number];
export type DispersionAdequacyTs =
  | "estimable"
  | "rank-deficient"
  | "insufficient-samples"
  | "invalid-covariance";

export interface DispersionCriteriaTs {
  metric: DispersionMetricTs;
  maxValue: number;
  confidenceLevel: number;
  minDurationS: number;
  minSamples: number;
}

export type SwingPointKindTs = "pivot" | "wrist" | "clubhead";

export interface SwingTraceRowTs {
  trialIndex: number;
  points: Vec3[];
  timesS: number[];
  status: SwingTrialStatusTs;
}

export interface RankedQuietIntervalTs {
  startIndex: number;
  endIndex: number;
  startTimeS: number;
  endTimeS: number;
  nSamples: number;
  meanValue: number;
  maxValue: number;
  score: number;
  rank: number;
}

export interface GeometricVariabilityTs {
  sampleTimesS: number[];
  validTrialCount: number[];
  meanPositionsM: Vec3[];
  rmsRadiusM: number[];
  principalSigmaM: number[];
  principalAxes: Vec3[];
  metric: DispersionMetricTs;
  authorityUnit: "m" | "m^3";
  displayUnit: "mm" | "mm³";
  confidenceLevel: number | null;
  interpretation: "sample-position-dispersion" | "gaussian-position-content-region";
  metricValues: number[];
  displayValues: number[];
  adequacy: DispersionAdequacyTs[];
  adequacyCounts: Record<DispersionAdequacyTs, number>;
  unavailableCount: number;
  quietMask: boolean[];
  quietIntervals: RankedQuietIntervalTs[];
  criteria: DispersionCriteriaTs;
  coordinateFrame: "app_frame:x_target,y_up,z_right";
  alignmentBasis: "common_simulation_time_s";
}

const MIN_CONFIDENCE = 1e-12;

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
        const index = pointKind === "pivot" ? 0 : Math.max(sample.joints.length - 2, 0);
        return sample.joints[index] ?? sample.position;
      }),
    }];
  });
}

export function geometricVariability(
  traces: SwingTraceRowTs[],
  criteria: DispersionCriteriaTs,
): GeometricVariabilityTs {
  validateCriteria(criteria);
  if (traces.length === 0) return emptyVariability(criteria);
  const sampleTimesS = validateCommonGrid(traces);
  const validTrialCount = Array(sampleTimesS.length).fill(traces.length) as number[];
  const meanPositionsM: Vec3[] = [];
  const rmsRadiusM: number[] = [];
  const principalSigmaM: number[] = [];
  const principalAxes: Vec3[] = [];
  const eigenvalues: Vec3[] = [];
  const adequacy: DispersionAdequacyTs[] = [];
  for (let sample = 0; sample < sampleTimesS.length; sample += 1) {
    const points = traces.map((trace) => trace.points[sample]);
    const sampleResult = sampleDispersion(points);
    meanPositionsM.push(sampleResult.mean);
    rmsRadiusM.push(sampleResult.rmsRadiusM);
    eigenvalues.push(sampleResult.eigenvaluesM2);
    principalSigmaM.push(sampleResult.principalSigmaM);
    principalAxes.push(sampleResult.principalAxis);
    adequacy.push(sampleResult.adequacy);
  }
  const metricValues = selectedMetricValues(criteria, rmsRadiusM, eigenvalues, adequacy);
  const quietIntervals = rankedQuietIntervals(
    sampleTimesS,
    metricValues,
    adequacy,
    criteria,
  );
  const quietMask = Array(sampleTimesS.length).fill(false) as boolean[];
  quietIntervals.forEach((interval) => {
    for (let index = interval.startIndex; index <= interval.endIndex; index += 1) {
      quietMask[index] = true;
    }
  });
  const authorityUnit = criteria.metric === "confidence-ellipsoid-volume" ? "m^3" : "m";
  const scale = authorityUnit === "m^3" ? 1e9 : 1e3;
  const adequacyCounts = countAdequacy(adequacy);
  return {
    sampleTimesS,
    validTrialCount,
    meanPositionsM,
    rmsRadiusM,
    principalSigmaM,
    principalAxes,
    metric: criteria.metric,
    authorityUnit,
    displayUnit: authorityUnit === "m^3" ? "mm³" : "mm",
    confidenceLevel: criteria.metric === "confidence-ellipsoid-volume"
      ? criteria.confidenceLevel : null,
    interpretation: criteria.metric === "confidence-ellipsoid-volume"
      ? "gaussian-position-content-region" : "sample-position-dispersion",
    metricValues,
    displayValues: metricValues.map((value) => value * scale),
    adequacy,
    adequacyCounts,
    unavailableCount: unavailableCount(criteria.metric, adequacyCounts),
    quietMask,
    quietIntervals,
    criteria: { ...criteria },
    coordinateFrame: "app_frame:x_target,y_up,z_right",
    alignmentBasis: "common_simulation_time_s",
  };
}

function validateCriteria(criteria: DispersionCriteriaTs): void {
  if (!DISPERSION_METRICS.includes(criteria.metric)) throw new Error("unknown dispersion metric");
  if (!Number.isFinite(criteria.maxValue) || criteria.maxValue <= 0) {
    throw new Error("maxValue must be finite and greater than zero");
  }
  if (!Number.isFinite(criteria.confidenceLevel)
    || criteria.confidenceLevel < MIN_CONFIDENCE
    || criteria.confidenceLevel >= 1) {
    throw new Error("confidenceLevel must be finite and in [1e-12, 1)");
  }
  if (!Number.isFinite(criteria.minDurationS) || criteria.minDurationS < 0) {
    throw new Error("minDurationS must be finite and non-negative");
  }
  if (!Number.isInteger(criteria.minSamples) || criteria.minSamples < 1) {
    throw new Error("minSamples must be an integer >= 1");
  }
}

function validateCommonGrid(traces: SwingTraceRowTs[]): number[] {
  const reference = traces[0].timesS;
  if (reference.length === 0 || reference.length !== traces[0].points.length) {
    throw new Error("each trace requires a non-empty aligned common time grid");
  }
  if (!reference.every((time, index) => Number.isFinite(time)
    && (index === 0 || time > reference[index - 1]))) {
    throw new Error("sample times must be finite and strictly increasing");
  }
  traces.forEach((trace) => {
    if (trace.timesS.length !== reference.length || trace.points.length !== reference.length) {
      throw new Error("all traces must use the exact common time grid");
    }
    trace.timesS.forEach((time, index) => {
      if (time !== reference[index]) throw new Error("all traces must use the exact common time grid");
    });
    if (!trace.points.every((point) => point.length === 3 && point.every(Number.isFinite))) {
      throw new Error("trace positions must contain finite Cartesian coordinates");
    }
  });
  return [...reference];
}

function selectedMetricValues(
  criteria: DispersionCriteriaTs,
  rmsRadiusM: number[],
  eigenvalues: Vec3[],
  adequacy: DispersionAdequacyTs[],
): number[] {
  if (criteria.metric === "rms-radius") return [...rmsRadiusM];
  if (criteria.metric === "largest-principal-sigma") {
    return eigenvalues.map((values) => Math.sqrt(Math.max(values[0], 0)));
  }
  const radiusScale = confidenceRadiusScale(criteria.confidenceLevel);
  return eigenvalues.map((values, index) => {
    if (adequacy[index] !== "estimable") return Number.NaN;
    const semiAxes = values.map((value) => radiusScale * Math.sqrt(Math.max(value, 0)));
    return 4 * Math.PI / 3 * semiAxes[0] * semiAxes[1] * semiAxes[2];
  });
}

function rankedQuietIntervals(
  times: number[],
  values: number[],
  adequacy: DispersionAdequacyTs[],
  criteria: DispersionCriteriaTs,
): RankedQuietIntervalTs[] {
  const eligible = adequacy.map((state) => criteria.metric === "confidence-ellipsoid-volume"
    ? state === "estimable"
    : state === "estimable" || state === "rank-deficient");
  const qualifying = values.map((value, index) => (
    eligible[index] && Number.isFinite(value) && value <= criteria.maxValue
  ));
  const candidates = trueRuns(qualifying).flatMap(([startIndex, endIndex]) => {
    const nSamples = endIndex - startIndex + 1;
    const duration = times[endIndex] - times[startIndex];
    if (nSamples < criteria.minSamples || duration < criteria.minDurationS) return [];
    const selected = values.slice(startIndex, endIndex + 1);
    const meanValue = selected.reduce((sum, value) => sum + value, 0) / selected.length;
    return [{
      startIndex,
      endIndex,
      startTimeS: times[startIndex],
      endTimeS: times[endIndex],
      nSamples,
      meanValue,
      maxValue: Math.max(...selected),
      score: meanValue / criteria.maxValue,
      rank: 1,
    }];
  }).sort((left, right) => left.score - right.score
    || left.startIndex - right.startIndex || left.endIndex - right.endIndex);
  let rank = 0;
  let previousScore: number | null = null;
  return candidates.map((interval) => {
    if (previousScore === null || interval.score !== previousScore) {
      rank += 1;
      previousScore = interval.score;
    }
    return { ...interval, rank };
  });
}

function trueRuns(mask: boolean[]): Array<[number, number]> {
  const intervals: Array<[number, number]> = [];
  let start: number | null = null;
  mask.forEach((value, index) => {
    if (value && start === null) start = index;
    if (!value && start !== null) {
      intervals.push([start, index - 1]);
      start = null;
    }
  });
  if (start !== null) intervals.push([start, mask.length - 1]);
  return intervals;
}

function countAdequacy(
  adequacy: DispersionAdequacyTs[],
): Record<DispersionAdequacyTs, number> {
  const result: Record<DispersionAdequacyTs, number> = {
    estimable: 0,
    "rank-deficient": 0,
    "insufficient-samples": 0,
    "invalid-covariance": 0,
  };
  adequacy.forEach((state) => { result[state] += 1; });
  return result;
}

function unavailableCount(
  metric: DispersionMetricTs,
  counts: Record<DispersionAdequacyTs, number>,
): number {
  if (metric === "confidence-ellipsoid-volume") {
    return counts["rank-deficient"] + counts["insufficient-samples"]
      + counts["invalid-covariance"];
  }
  return counts["insufficient-samples"] + counts["invalid-covariance"];
}

function emptyVariability(criteria: DispersionCriteriaTs): GeometricVariabilityTs {
  const authorityUnit = criteria.metric === "confidence-ellipsoid-volume" ? "m^3" : "m";
  return {
    sampleTimesS: [], validTrialCount: [], meanPositionsM: [], rmsRadiusM: [],
    principalSigmaM: [], principalAxes: [], metric: criteria.metric, authorityUnit,
    displayUnit: authorityUnit === "m^3" ? "mm³" : "mm",
    confidenceLevel: criteria.metric === "confidence-ellipsoid-volume"
      ? criteria.confidenceLevel : null,
    interpretation: criteria.metric === "confidence-ellipsoid-volume"
      ? "gaussian-position-content-region" : "sample-position-dispersion",
    metricValues: [], displayValues: [], adequacy: [], adequacyCounts: countAdequacy([]),
    unavailableCount: 0, quietMask: [], quietIntervals: [], criteria: { ...criteria },
    coordinateFrame: "app_frame:x_target,y_up,z_right",
    alignmentBasis: "common_simulation_time_s",
  };
}
