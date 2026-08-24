/**
 * Analysis + export over web variation datasets (epic #4120, V3).
 *
 * TypeScript mirror of shared/python/swing_sim/variation/analysis.py and
 * dataset_io.py: per-output summary statistics, one-at-a-time
 * sensitivity (paired draws via the engine's per-variable RNG streams),
 * Spearman rank correlation, the 2-sigma landing-dispersion ellipse,
 * and CSV/JSON dataset serialization in the documented Python schema.
 */

import {
  runVariation,
  type VariationDatasetTs,
  type VariationPlanTs,
} from "./variation";
import { variationExecutionDocument } from "./variationExecutionMetadata";

export interface OutputStatsTs {
  name: string;
  mean: number;
  std: number;
  p5: number;
  p50: number;
  p95: number;
  n: number;
}

const okColumn = (dataset: VariationDatasetTs, j: number): number[] => {
  const values: number[] = [];
  dataset.outputs.forEach((row, i) => {
    const v = row[j];
    if (dataset.success[i] && v !== null && Number.isFinite(v)) values.push(v);
  });
  return values;
};

const percentile = (sorted: number[], q: number): number => {
  if (sorted.length === 0) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
};

const sampleStd = (values: number[], mean: number): number => {
  if (values.length < 2) return NaN;
  const ss = values.reduce((acc, v) => acc + (v - mean) ** 2, 0);
  return Math.sqrt(ss / (values.length - 1));
};

/** Per-output mean/std/percentiles over the successful runs. */
export function summaryStats(dataset: VariationDatasetTs): OutputStatsTs[] {
  return dataset.outputNames.map((name, j) => {
    const values = okColumn(dataset, j);
    const n = values.length;
    if (n === 0) {
      return { name, mean: NaN, std: NaN, p5: NaN, p50: NaN, p95: NaN, n: 0 };
    }
    const mean = values.reduce((a, v) => a + v, 0) / n;
    const sorted = [...values].sort((a, b) => a - b);
    return {
      name,
      mean,
      std: sampleStd(values, mean),
      p5: percentile(sorted, 0.05),
      p50: percentile(sorted, 0.5),
      p95: percentile(sorted, 0.95),
      n,
    };
  });
}

export interface SensitivityResultTs {
  inputKeys: string[];
  outputNames: string[];
  matrix: number[][]; // std induced per (input, output)
  normalized: number[][]; // column-normalized, 1 = dominant input
}

/** One-at-a-time sensitivity: rerun with a single spec active at a time. */
export function oneAtATimeSensitivity(
  plan: VariationPlanTs,
  onTrialComplete?: () => void,
): SensitivityResultTs {
  const rows: number[][] = [];
  let outputNames: string[] = [];
  for (const spec of plan.noise) {
    // A one-at-a-time study evaluates the selected marginal independently;
    // retaining a multi-member correlation group would leave dangling IDs.
    const dataset = runVariation(
      { ...plan, noise: [spec], groups: [] },
      onTrialComplete,
    );
    outputNames = dataset.outputNames;
    rows.push(
      dataset.outputNames.map((_name, j) => {
        const values = okColumn(dataset, j);
        const mean = values.reduce((a, v) => a + v, 0) / (values.length || 1);
        return sampleStd(values, mean);
      }),
    );
  }
  const normalized = normalizeSensitivityMatrix(rows, outputNames.length);
  return {
    inputKeys: plan.noise.map((s) => s.variableKey),
    outputNames,
    matrix: rows,
    normalized,
  };
}

export function normalizeSensitivityMatrix(
  rows: number[][],
  outputCount: number,
): number[][] {
  const normalized = rows.map((row) => row.slice());
  for (let j = 0; j < outputCount; j += 1) {
    let max = 0;
    let finiteCount = 0;
    for (const row of rows) {
      const v = Math.abs(row[j]);
      if (Number.isFinite(v)) {
        finiteCount += 1;
        if (v > max) max = v;
      }
    }
    for (let i = 0; i < rows.length; i += 1) {
      const value = rows[i][j];
      normalized[i][j] = !Number.isFinite(value) || finiteCount === 0
        ? Number.NaN
        : max > 0
          ? Math.abs(value) / max
          : 0;
    }
  }
  return normalized;
}

const ranks = (values: number[]): number[] => {
  const order = values
    .map((v, i) => [v, i] as const)
    .sort((a, b) => a[0] - b[0]);
  const out = new Array<number>(values.length);
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j += 1;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k += 1) out[order[k][1]] = avg;
    i = j + 1;
  }
  return out;
};

const finitePair = (
  dataset: VariationDatasetTs,
  inputIndex: number,
  outputIndex: number,
): ReadonlyArray<readonly [number, number]> => {
  const pair: Array<readonly [number, number]> = [];
  dataset.success.forEach((successful, trialIndex) => {
    const input = dataset.inputs[trialIndex]?.[inputIndex];
    const output = dataset.outputs[trialIndex]?.[outputIndex];
    if (successful && Number.isFinite(input) && Number.isFinite(output)) {
      pair.push([input, output as number]);
    }
  });
  return pair;
};

/** Spearman rank correlation, inputs (rows) x outputs (columns). */
export function spearmanMatrix(dataset: VariationDatasetTs): number[][] {
  const shapeRows = dataset.inputNames.length;
  const shapeCols = dataset.outputNames.length;
  const stats = (r: number[]): { mean: number; std: number } => {
    const mean = r.reduce((a, v) => a + v, 0) / r.length;
    const std = Math.sqrt(r.reduce((a, v) => a + (v - mean) ** 2, 0) / r.length);
    return { mean, std };
  };
  return Array.from({ length: shapeRows }, (_unused, inputIndex) =>
    Array.from({ length: shapeCols }, (_unusedOutput, outputIndex) => {
      const pair = finitePair(dataset, inputIndex, outputIndex);
      if (pair.length < 3) return NaN;
      const ri = ranks(pair.map(([input]) => input));
      const rj = ranks(pair.map(([, output]) => output));
      const si = stats(ri);
      const sj = stats(rj);
      if (si.std <= 0 || sj.std <= 0) return NaN;
      let cov = 0;
      for (let k = 0; k < ri.length; k += 1) {
        cov += (ri[k] - si.mean) * (rj[k] - sj.mean);
      }
      cov /= ri.length;
      return cov / (si.std * sj.std);
    }),
  );
}

export interface DispersionEllipseTs {
  centerCarryM: number;
  centerLateralM: number;
  semiMajorM: number;
  semiMinorM: number;
  angleDeg: number; // CCW from the carry axis toward + lateral
  n: number;
}

export interface LandingPointTs {
  readonly trialIndex: number;
  readonly carryM: number;
  readonly lateralM: number;
}

/** Return only successful rows with a paired finite landing coordinate. */
export function pairedLandingPoints(
  dataset: VariationDatasetTs,
): LandingPointTs[] {
  if (dataset.outputs.length !== dataset.success.length) {
    throw new Error("variation outputs and success flags must align by trial");
  }
  const carryIndex = dataset.outputNames.indexOf("carry_m");
  const lateralIndex = dataset.outputNames.indexOf("lateral_m");
  if (carryIndex < 0 || lateralIndex < 0) return [];
  const points: LandingPointTs[] = [];
  dataset.outputs.forEach((row, trialIndex) => {
    const carryM = row[carryIndex];
    const lateralM = row[lateralIndex];
    if (
      dataset.success[trialIndex] &&
      carryM !== null && Number.isFinite(carryM) &&
      lateralM !== null && Number.isFinite(lateralM)
    ) {
      points.push({ trialIndex, carryM, lateralM });
    }
  });
  return points;
}

/** 2-sigma landing ellipse from the carry/lateral sample covariance. */
export function dispersionEllipse(
  dataset: VariationDatasetTs,
  nSigma = 2.0,
): DispersionEllipseTs | null {
  const points = pairedLandingPoints(dataset);
  const carry = points.map((point) => point.carryM);
  const lateral = points.map((point) => point.lateralM);
  const n = points.length;
  if (n < 2) return null;
  const mc = carry.reduce((a, v) => a + v, 0) / n;
  const ml = lateral.reduce((a, v) => a + v, 0) / n;
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  for (let i = 0; i < n; i += 1) {
    sxx += (carry[i] - mc) ** 2;
    syy += (lateral[i] - ml) ** 2;
    sxy += (carry[i] - mc) * (lateral[i] - ml);
  }
  sxx /= n - 1;
  syy /= n - 1;
  sxy /= n - 1;
  // Closed-form 2x2 symmetric eigen-decomposition.
  const trace = sxx + syy;
  const det = sxx * syy - sxy * sxy;
  const disc = Math.sqrt(Math.max((trace * trace) / 4 - det, 0));
  const l1 = trace / 2 + disc; // major
  const l2 = Math.max(trace / 2 - disc, 0); // minor
  const angleRad =
    Math.abs(sxy) < 1e-15 && sxx >= syy ? 0 : Math.atan2(l1 - sxx, sxy);
  return {
    centerCarryM: mc,
    centerLateralM: ml,
    semiMajorM: nSigma * Math.sqrt(Math.max(l1, 0)),
    semiMinorM: nSigma * Math.sqrt(l2),
    angleDeg: (angleRad * 180.0) / Math.PI,
    n,
  };
}

/** CSV in the Python dataset_io.write_csv schema. */
export function datasetToCsv(dataset: VariationDatasetTs): string {
  const header = ["run", "success", ...dataset.inputNames, ...dataset.outputNames];
  const lines = [header.join(",")];
  for (let i = 0; i < dataset.plan.nRuns; i += 1) {
    lines.push(
      [
        i,
        dataset.success[i] ? 1 : 0,
        ...dataset.inputs[i],
        ...dataset.outputs[i].map((v) => (v === null ? "" : v)),
      ].join(","),
    );
  }
  return `${lines.join("\n")}\n`;
}

/** JSON in the Python dataset_io.to_json_dict schema. */
export function datasetToJson(dataset: VariationDatasetTs): string {
  return JSON.stringify(
    {
      schema_version: 2,
      plan_document: variationExecutionDocument(dataset.plan),
      input_names: dataset.inputNames,
      output_names: dataset.outputNames,
      inputs: dataset.inputs,
      outputs: dataset.outputs,
      success: dataset.success,
      elapsed_s: 0.0,
    },
    null,
    2,
  );
}
