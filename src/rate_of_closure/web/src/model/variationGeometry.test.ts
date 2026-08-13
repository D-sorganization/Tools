import golden from "./__fixtures__/dispersion_metrics_golden_v1.json";
import { describe, expect, it } from "vitest";

import type { DispersionMetricTs, SwingTraceRowTs } from "./variationGeometry";
import { geometricVariability } from "./variationGeometry";

const criteria = (metric: DispersionMetricTs, maxValue: number) => ({
  metric,
  maxValue,
  confidenceLevel: 0.95,
  minDurationS: 0,
  minSamples: 1,
} as const);

const traces = (): SwingTraceRowTs[] => [
  {
    trialIndex: 0,
    status: "evaluated_hit",
    timesS: [0, 0.01, 0.02],
    points: [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
  },
  {
    trialIndex: 1,
    status: "evaluated_no_impact",
    timesS: [0, 0.01, 0.02],
    points: [[0, 1, 0], [1, 1, 0], [2, 1, 0]],
  },
];

describe("geometric variation plot data", () => {
  it("pins RMS radius, covariance principal spread, and quiet intervals", () => {
    const result = geometricVariability(traces(), criteria("rms-radius", 0.6));

    expect(result.rmsRadiusM).toEqual([0.5, 0.5, 0.5]);
    result.principalSigmaM.forEach((value) => expect(value).toBeCloseTo(Math.sqrt(0.5)));
    result.principalAxes.forEach((axis) => {
      expect(axis[0]).toBeCloseTo(0);
      expect(axis[1]).toBeCloseTo(1);
      expect(axis[2]).toBeCloseTo(0);
    });
    expect(result.quietMask).toEqual([true, true, true]);
    expect(result.quietIntervals.map(({ startIndex, endIndex, rank }) => ({
      startIndex, endIndex, rank,
    }))).toEqual([{ startIndex: 0, endIndex: 2, rank: 1 }]);
    expect(result.alignmentBasis).toBe("common_simulation_time_s");
  });

  it("retains measured dispersion when no sample meets the threshold", () => {
    const result = geometricVariability(traces(), criteria("rms-radius", 0.4));

    expect(result.quietMask).toEqual([false, false, false]);
    expect(result.quietIntervals).toEqual([]);
    expect(result.rmsRadiusM).toEqual([0.5, 0.5, 0.5]);
  });

  it("reports rank-deficient confidence volume as unavailable", () => {
    const result = geometricVariability(
      traces(), criteria("confidence-ellipsoid-volume", 1),
    );

    expect(result.adequacy).toEqual([
      "rank-deficient", "rank-deficient", "rank-deficient",
    ]);
    expect(result.unavailableCount).toBe(3);
    result.metricValues.forEach((value) => expect(value).toBeNaN());
    expect(result.quietMask).toEqual([false, false, false]);
    expect(result.quietIntervals).toEqual([]);
  });

  it("rejects non-physical quiet-zone thresholds", () => {
    expect(() => geometricVariability(
      traces(), criteria("rms-radius", 0),
    )).toThrow(/greater than zero/);
  });

  it("matches the Python authority for every selectable metric", () => {
    const goldenTraces: SwingTraceRowTs[] = golden.traces.map((trace) => ({
      trialIndex: trace.trial_index,
      status: trace.status as SwingTraceRowTs["status"],
      timesS: golden.times_s,
      points: trace.points_m as SwingTraceRowTs["points"],
    }));
    Object.entries(golden.expected).forEach(([metric, expected]) => {
      const result = geometricVariability(
        goldenTraces,
        criteria(metric as DispersionMetricTs, expected.threshold),
      );
      result.metricValues.forEach((value, index) => {
        const relativeError = Math.abs(value - expected.values[index])
          / Math.max(Math.abs(expected.values[index]), Number.MIN_VALUE);
        expect(relativeError).toBeLessThan(1e-6);
      });
      expect(result.adequacy).toEqual(expected.adequacy);
      expect(result.quietIntervals.map((interval) => ({
        start_index: interval.startIndex,
        end_index: interval.endIndex,
        rank: interval.rank,
      }))).toEqual(expected.intervals.map((interval) => ({
        start_index: interval.start_index,
        end_index: interval.end_index,
        rank: interval.rank,
      })));
    });
  });

  it("rejects unequal grids and nonfinite samples instead of truncating", () => {
    const unequal = traces();
    unequal[1] = { ...unequal[1], timesS: [0, 0.01], points: unequal[1].points.slice(0, 2) };
    expect(() => geometricVariability(
      unequal, criteria("rms-radius", 1),
    )).toThrow(/common time grid/);
    const nonfinite = traces();
    nonfinite[1].points[1] = [Number.NaN, 0, 0];
    expect(() => geometricVariability(
      nonfinite, criteria("rms-radius", 1),
    )).toThrow(/finite/);
  });
});
