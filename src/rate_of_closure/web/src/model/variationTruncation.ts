/**
 * Truncation mean-shift detection in web variation studies (#4253 item d).
 */

import type { VariationDatasetTs } from "./variation";

export interface TruncationShiftNoteTs {
  variableKey: string;
  nominalBase: number;
  realizedMean: number;
  meanShift: number;
  nominalScale: number;
  realizedStd: number;
  lowerBound: number | null;
  upperBound: number | null;
  truncatedLowerCount: number;
  truncatedUpperCount: number;
  hasShift: boolean;
  note: string;
}

const sampleStd = (values: number[], mean: number): number => {
  if (values.length < 2) return NaN;
  const ss = values.reduce((acc, v) => acc + (v - mean) ** 2, 0);
  return Math.sqrt(ss / (values.length - 1));
};

/** Detect parameter mean-shift caused by truncated input bounds. */
export function detectTruncationMeanShifts(
  dataset: VariationDatasetTs,
  thresholdFraction = 0.05,
): TruncationShiftNoteTs[] {
  const notes: TruncationShiftNoteTs[] = [];
  dataset.plan.noise.forEach((spec, idx) => {
    if (spec.lower === null && spec.upper === null) return;
    const base = dataset.plan.baseVariables[spec.variableKey] ?? 0;
    const vals: number[] = [];
    dataset.inputs.forEach((row, r) => {
      if (dataset.success[r] && Number.isFinite(row[idx])) vals.push(row[idx]);
    });
    if (vals.length === 0) return;
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const shift = mean - base;
    const std = sampleStd(vals, mean);
    let loCount = 0;
    let hiCount = 0;
    vals.forEach((v) => {
      if (spec.lower !== null && v <= spec.lower + 1e-9) loCount += 1;
      if (spec.upper !== null && v >= spec.upper - 1e-9) hiCount += 1;
    });
    const hasShift = Math.abs(shift) > thresholdFraction * spec.scale || loCount > 0 || hiCount > 0;
    if (hasShift) {
      notes.push({
        variableKey: spec.variableKey,
        nominalBase: base,
        realizedMean: mean,
        meanShift: shift,
        nominalScale: spec.scale,
        realizedStd: std,
        lowerBound: spec.lower,
        upperBound: spec.upper,
        truncatedLowerCount: loCount,
        truncatedUpperCount: hiCount,
        hasShift,
        note: `${spec.variableKey}: nominal ${base.toFixed(2)} shifted by ${shift > 0 ? "+" : ""}${shift.toFixed(2)} to ${mean.toFixed(2)} via bounds [${spec.lower ?? "-inf"}, ${spec.upper ?? "+inf"}] (${loCount} lower / ${hiCount} upper clamps).`,
      });
    }
  });
  return notes;
}
