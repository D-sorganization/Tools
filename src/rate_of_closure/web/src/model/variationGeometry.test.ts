import { describe, expect, it } from "vitest";

import type { SwingTraceRowTs } from "./variationGeometry";
import { geometricVariability } from "./variationGeometry";

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
    const result = geometricVariability(traces(), 0.6);

    expect(result.rmsRadiusM).toEqual([0.5, 0.5, 0.5]);
    result.principalSigmaM.forEach((value) => expect(value).toBeCloseTo(Math.sqrt(0.5)));
    result.principalAxes.forEach((axis) => {
      expect(axis[0]).toBeCloseTo(0);
      expect(axis[1]).toBeCloseTo(1);
      expect(axis[2]).toBeCloseTo(0);
    });
    expect(result.quietMask).toEqual([true, true, true]);
    expect(result.quietIntervals).toEqual([{ startIndex: 0, endIndex: 2 }]);
    expect(result.alignmentBasis).toBe("common_simulation_time_s");
  });

  it("retains measured dispersion when no sample meets the threshold", () => {
    const result = geometricVariability(traces(), 0.4);

    expect(result.quietMask).toEqual([false, false, false]);
    expect(result.quietIntervals).toEqual([]);
    expect(result.rmsRadiusM).toEqual([0.5, 0.5, 0.5]);
  });

  it("rejects non-physical quiet-zone thresholds", () => {
    expect(() => geometricVariability(traces(), 0)).toThrow(/greater than zero/);
  });
});
