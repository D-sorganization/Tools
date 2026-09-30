import { describe, expect, it } from "vitest";

import {
  analyzeDispersion,
  analyzeSessionTrend,
  calculateStrokesGained,
  calculateTargetError,
} from "./launchMonitorPerformance";

describe("launch monitor performance adapter", () => {
  it("reports unit-labeled left/right dispersion", () => {
    const result = analyzeDispersion([
      { offline_m: -9.144, carry_m: 100 }, { offline_m: 0, carry_m: 101 },
      { offline_m: 4.572, carry_m: 102 },
    ], { lateralColumn: "offline_m", carryColumn: "carry_m", lateralUnit: "m", carryUnit: "m" });
    expect(result).toMatchObject({ unit: "yd", leftCount: 1, centerCount: 1, rightCount: 1 });
    expect(result.points[0].lateralYards).toBeCloseTo(-10);
  });

  it("preserves dispersion validation and finite-row filtering", () => {
    const request = {
      lateralColumn: "offline",
      carryColumn: "carry",
      lateralUnit: "yd" as const,
      carryUnit: "yd" as const,
    };

    expect(() => analyzeDispersion([], request)).toThrow(
      "Columns are unavailable: offline, carry",
    );
    expect(() => analyzeDispersion([{ offline: 1 }], request)).toThrow(
      "Columns are unavailable: carry",
    );
    expect(() => analyzeDispersion([
      { offline: null, carry: 1 },
      { offline: Number.NaN, carry: 2 },
      { offline: Number.POSITIVE_INFINITY, carry: 3 },
      { offline: Number.NEGATIVE_INFINITY, carry: 4 },
      { offline: 5, carry: Number.NaN },
    ], request)).toThrow("Dispersion requires finite lateral and carry values");
  });

  it("preserves a single valid point and zero sample deviation", () => {
    const result = analyzeDispersion([
      { offline: Number.NaN, carry: 1 },
      { offline: -4, carry: 120 },
      { offline: null, carry: 2 },
    ], {
      lateralColumn: "offline", carryColumn: "carry",
      lateralUnit: "yd", carryUnit: "yd",
    });

    expect(result.points).toEqual([{ sourceIndex: 1, lateralYards: -4, carryYards: 120 }]);
    expect(result).toMatchObject({
      meanLateralYards: -4,
      standardDeviationYards: 0,
      rmsYards: 4,
      leftCount: 1,
      centerCount: 0,
      rightCount: 0,
    });
  });

  it("preserves source order, indices, and mixed-sign sample statistics", () => {
    const result = analyzeDispersion([
      { offline: Number.NaN, carry: 10 },
      { offline: -3, carry: 11 },
      { offline: null, carry: 12 },
      { offline: 0, carry: 13 },
      { offline: 3, carry: 14 },
      { offline: Number.POSITIVE_INFINITY, carry: 15 },
      { offline: 6, carry: 16 },
      { offline: Number.NEGATIVE_INFINITY, carry: 17 },
      { offline: 1, carry: Number.NaN },
    ], {
      lateralColumn: "offline", carryColumn: "carry",
      lateralUnit: "yd", carryUnit: "yd",
    });

    expect(result.points.map(({ sourceIndex, lateralYards }) => [sourceIndex, lateralYards]))
      .toEqual([[1, -3], [3, 0], [4, 3], [6, 6]]);
    expect(result.meanLateralYards).toBe(1.5);
    expect(result.standardDeviationYards).toBe(Math.sqrt(15));
    expect(result.rmsYards).toBe(Math.sqrt(13.5));
    expect(result).toMatchObject({ leftCount: 1, centerCount: 1, rightCount: 2 });
  });

  it("labels user-supplied expected-strokes SG as not source-backed", () => {
    const rows = [{ before: 3.2, after: 2.0 }, { before: 3.1, after: 1.8 }];
    expect(() => calculateStrokesGained(rows, {
      expectedBeforeColumn: "before", expectedAfterColumn: "after", baselineSourceUrl: "",
    })).toThrow(/source/i);
    const result = calculateStrokesGained(rows, {
      expectedBeforeColumn: "before", expectedAfterColumn: "after",
      baselineSourceUrl: "https://datagolf.com/frequently-asked-questions",
    });
    expect(result.metricName).toBe("user_supplied_expected_strokes_sg");
    expect(result.mean).toBeCloseTo(0.25);
  });

  it("names launch-only performance radial target error, never strokes gained", () => {
    const result = calculateTargetError([{ carry: 150, offline: 12 }], {
      carryColumn: "carry", lateralColumn: "offline", carryUnit: "yd",
      lateralUnit: "yd", targetDistanceYards: 160,
    });
    expect(result.metricName).toBe("radial_target_error");
    expect(result.values[0]).toBeCloseTo(Math.hypot(10, 12));
  });

  it("requires explicit trusted player/session identity and session ordering", () => {
    const rows = [
      { player: "p1", session: "a", order: 1, speed: 100 },
      { player: "p1", session: "a", order: 1, speed: 102 },
      { player: "p1", session: "b", order: 2, speed: 104 },
      { player: "p1", session: "b", order: 2, speed: 106 },
    ];
    expect(() => analyzeSessionTrend(rows, {
      metricColumn: "speed", sessionColumn: "session", sessionOrderColumn: "order",
      playerColumn: "player", sessionIdentityAttested: true, playerIdentityAttested: false,
    })).toThrow(/attested/i);
    const result = analyzeSessionTrend(rows, {
      metricColumn: "speed", sessionColumn: "session", sessionOrderColumn: "order",
      playerColumn: "player", sessionIdentityAttested: true, playerIdentityAttested: true,
    });
    expect(result.points.map(({ sessionId, mean, cumulativeMean }) =>
      [sessionId, mean, cumulativeMean])).toEqual([["a", 101, 101], ["b", 105, 103]]);
  });
});
