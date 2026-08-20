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
