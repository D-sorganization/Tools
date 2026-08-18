import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/putting_sample_inspector_golden_v1.json";
import {
  MAX_PUTTING_DISPLAY_SAMPLES,
  navigatePuttingSamples,
  nearestPuttingSample,
  planPuttingSamples,
  validatePuttingResultSummary,
} from "./puttingSampleInspector";
import { simulatePutt } from "./putting";

describe("putting sample inspector", () => {
  it("matches the Python-owned geometry, phase, navigation, and pixel tie golden", () => {
    const plan = planPuttingSamples(fixture.series, 6);
    const tiePlan = planPuttingSamples({
      path_x_m: Array.from({ length: 8 }, (_, index) => index),
      path_y_m: Array(8).fill(0), speeds_mps: Array.from({ length: 8 }, (_, index) => 8 - index),
      times_s: Array.from({ length: 8 }, (_, index) => index), skid_end_index: 0,
    }, 5);
    expect(tiePlan.displayedRawIndices)
      .toEqual(fixture.expected.half_tie_displayed_raw_indices_at_cap_5);
    expect(plan.samples.map((sample) => sample.rawIndex))
      .toEqual(fixture.expected.displayed_raw_indices_at_cap_6);
    plan.cumulativeDistanceM.forEach((value, index) =>
      expect(value).toBeCloseTo(fixture.expected.cumulative_distance_m[index], 12));
    expect(Array.from({ length: plan.rawCount }, (_, index) => plan.rawSample(index).phase))
      .toEqual(fixture.expected.phases);
    Object.entries(fixture.expected.navigation_from_3).forEach(([command, expected]) => {
      expect(navigatePuttingSamples(plan, 3, command as "next")).toBe(expected);
    });
    expect(nearestPuttingSample(
      fixture.expected.nearest.projected.map((item) =>
        item as [number, number, number]),
      fixture.expected.nearest.pointer as [number, number],
      fixture.expected.nearest.hit_radius_px,
    )).toBe(fixture.expected.nearest.raw_index);
  });

  it("keeps the fixed plan deeply immutable and never inserts selection", () => {
    const plan = planPuttingSamples({
      path_x_m: Array.from({ length: 100 }, (_, index) => index),
      path_y_m: Array(100).fill(0),
      speeds_mps: Array.from({ length: 100 }, (_, index) => 100 - index),
      times_s: Array.from({ length: 100 }, (_, index) => index * 0.002),
      skid_end_index: 17,
    }, 8);
    const before = plan.displayedRawIndices;
    const undisplayed = Array.from({ length: 100 }, (_, index) => index)
      .find((index) => !before.includes(index));
    expect(undisplayed).not.toBeUndefined();
    expect(navigatePuttingSamples(plan, undisplayed ?? null, "next")).toBe(before[0]);
    expect(plan.displayedRawIndices).toBe(before);
    expect(Object.isFrozen(plan)).toBe(true);
    expect(Object.isFrozen(plan.samples)).toBe(true);
    expect(Object.isFrozen(plan.samples[0])).toBe(true);
  });

  it("caps a legal solver-produced 30,001-row putt at 1,024", () => {
    const result = simulatePutt({
      ballSpeedMps: 0.2,
      launchAngleDeg: 0,
      horizontalSpeedMps: 0.2,
      spinRadS: 0,
      effectiveLoftDeg: 0,
    }, { stimpFt: 13, gradePercent: 10, aspectDeg: 0 }, 40);
    expect(result.timesS).toHaveLength(30_001);
    const plan = planPuttingSamples({
      path_x_m: result.pathXM,
      path_y_m: result.pathYM,
      speeds_mps: result.speedsMps,
      times_s: result.timesS,
      skid_end_index: result.skidEndIndex,
    });
    expect(plan.displayedCount).toBe(MAX_PUTTING_DISPLAY_SAMPLES);
    expect(plan.displayedRawIndices[0]).toBe(0);
    expect(plan.displayedRawIndices[plan.displayedRawIndices.length - 1]).toBe(30_000);
    expect(plan.displayedRawIndices).toContain(result.skidEndIndex);
  });

  it("treats split as first pure-roll sample and split zero as no skid", () => {
    const plan = planPuttingSamples({
      path_x_m: [0, 1, 2], path_y_m: [0, 0, 0], speeds_mps: [2, 1, 0],
      times_s: [0, 0.002, 0.004], skid_end_index: 0,
    });
    expect(plan.skidPolylineIndices).toEqual([]);
    expect(plan.pureRollPolylineIndices).toEqual([0, 1, 2]);
    expect(plan.rawSample(0).phase).toBe("pure-roll");
  });

  it("rejects malformed aligned evidence and out-of-radius pointers", () => {
    expect(() => planPuttingSamples({
      path_x_m: [0, Number.NaN], path_y_m: [0, 0], speeds_mps: [1, 0],
      times_s: [0, 0.002], skid_end_index: 1,
    })).toThrow(/finite/);
    expect(() => planPuttingSamples({
      path_x_m: [0, 1], path_y_m: [0, 0], speeds_mps: [1, 0],
      times_s: [0, 0], skid_end_index: 1,
    })).toThrow(/strictly increasing/);
    expect(nearestPuttingSample([[4, 10, 10]], [30, 30])).toBeNull();
    expect(() => planPuttingSamples({
      path_x_m: [Number.MAX_VALUE, -Number.MAX_VALUE], path_y_m: [0, 0],
      speeds_mps: [1, 0], times_s: [0, 0.002], skid_end_index: 1,
    })).toThrow(/distance must remain finite/);
    expect(() => planPuttingSamples({
      path_x_m: [0, 0], path_y_m: [0, Number.MAX_VALUE],
      speeds_mps: [Number.MAX_VALUE, 0], times_s: [0, 0.002], skid_end_index: 1,
    })).toThrow(/display envelope/);
  });

  it("rejects summary evidence that contradicts exact raw samples", () => {
    const result = simulatePutt(
      { ballSpeedMps: 2, launchAngleDeg: 0, horizontalSpeedMps: 2,
        spinRadS: 0, effectiveLoftDeg: 0 },
      { stimpFt: 10, gradePercent: 0, aspectDeg: 0 }, 3,
    );
    const plan = planPuttingSamples({ path_x_m: result.pathXM, path_y_m: result.pathYM,
      speeds_mps: result.speedsMps, times_s: result.timesS,
      skid_end_index: result.skidEndIndex });
    validatePuttingResultSummary(result, plan);
    expect(() => validatePuttingResultSummary({ ...result, totalDistanceM: 1 }, plan))
      .toThrow(/exact raw/);
  });
});
