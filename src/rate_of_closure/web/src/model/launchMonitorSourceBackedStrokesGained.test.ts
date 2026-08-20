import { describe, expect, it } from "vitest";

import {
  baselineTableHash,
  calculateSourceBackedStrokesGained,
  parseStrokesGainedBaseline,
} from "./launchMonitorSourceBackedStrokesGained";

const states = [
  { lie: "fairway", distance_yards: 100, expected_strokes: 2.8 },
  { lie: "fairway", distance_yards: 200, expected_strokes: 3.8 },
  { lie: "green", distance_yards: 0, expected_strokes: 0 },
  { lie: "green", distance_yards: 20, expected_strokes: 1.5 },
];

describe("source-backed strokes gained", () => {
  it("verifies the artifact hash and interpolates within each course lie", async () => {
    const baseline = await parseStrokesGainedBaseline(JSON.stringify({
      contract_version: "launch-monitor-strokes-gained-baseline/1.0.0",
      baseline_id: "licensed-test", version: "2026.1",
      source_url: "https://example.org/method", license: "test-only",
      table_sha256: await baselineTableHash(states), states,
    }));
    const result = calculateSourceBackedStrokesGained([
      { before_lie: "fairway", before_distance: 150, after_lie: "green", after_distance: 20 },
    ], baseline, {
      beforeLieColumn: "before_lie", beforeDistanceColumn: "before_distance",
      afterLieColumn: "after_lie", afterDistanceColumn: "after_distance",
      beforeDistanceUnit: "yd", afterDistanceUnit: "yd",
    });
    expect(result.values[0]).toBeCloseTo(0.8);
    expect(result.backingRows[0].expectedBefore).toBeCloseTo(3.3);
    expect(result.baselineId).toBe("licensed-test");
  });

  it("fails closed on table tamper and missing lie coverage", async () => {
    const payload = {
      contract_version: "launch-monitor-strokes-gained-baseline/1.0.0",
      baseline_id: "licensed-test", version: "2026.1",
      source_url: "https://example.org/method", license: "test-only",
      table_sha256: "0".repeat(64), states,
    };
    await expect(parseStrokesGainedBaseline(JSON.stringify(payload))).rejects.toThrow(/SHA-256/);
  });
});
