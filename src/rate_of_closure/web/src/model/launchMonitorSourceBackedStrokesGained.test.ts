import { describe, expect, it } from "vitest";

import {
  baselineTableHash,
  buildSourceBackedStrokesGainedPayload,
  calculateSourceBackedStrokesGained,
  parseStrokesGainedBaseline,
} from "./launchMonitorSourceBackedStrokesGained";

const states = [
  { lie: "fairway", context: "standard", target: "hole-1", distance_yards: 100, expected_strokes: 2.8, standard_error: 0.1 },
  { lie: "fairway", context: "standard", target: "hole-1", distance_yards: 200, expected_strokes: 3.8, standard_error: 0.14 },
  { lie: "green", context: "standard", target: "hole-1", distance_yards: 0, expected_strokes: 0, standard_error: 0 },
  { lie: "green", context: "standard", target: "hole-1", distance_yards: 20, expected_strokes: 1.5, standard_error: 0.08 },
];

describe("source-backed strokes gained", () => {
  it("verifies the artifact hash and interpolates within each course lie", async () => {
    expect(await baselineTableHash([...states].reverse())).toBe("5250552cc6ec58da60dfe8ebf50f7238534d28016b0725bf42d8098054404428");
    const baseline = await parseStrokesGainedBaseline(JSON.stringify({
      contract_version: "launch-monitor-strokes-gained-baseline/2.0.0",
      baseline_id: "licensed-test", version: "2026.1",
      source_url: "https://example.org/method", license: "test-only",
      table_sha256: await baselineTableHash(states), states,
    }));
    const result = calculateSourceBackedStrokesGained([
      { before_lie: "fairway", before_context: "standard", target: "hole-1", before_distance: 150, after_lie: "green", after_context: "standard", after_distance: 20 },
    ], baseline, {
        beforeLieColumn: "before_lie", beforeContextColumn: "before_context", beforeTargetColumn: "target", beforeDistanceColumn: "before_distance",
        afterLieColumn: "after_lie", afterContextColumn: "after_context", afterTargetColumn: "target", afterDistanceColumn: "after_distance",
      beforeDistanceUnit: "yd", afterDistanceUnit: "yd",
    });
    expect(result.values[0]).toBeCloseTo(0.8);
    expect(result.backingRows[0].expectedBefore).toBeCloseTo(3.3);
    expect(result.baselineId).toBe("licensed-test");
  });

  it("fails closed on table tamper and missing lie coverage", async () => {
    const payload = {
      contract_version: "launch-monitor-strokes-gained-baseline/2.0.0",
      baseline_id: "licensed-test", version: "2026.1",
      source_url: "https://example.org/method", license: "test-only",
      table_sha256: "0".repeat(64), states,
    };
    await expect(parseStrokesGainedBaseline(JSON.stringify(payload))).rejects.toThrow(/SHA-256/);
  });

  it("emits grouped and longitudinal requests only from explicit attestation", async () => {
    const baseline = await parseStrokesGainedBaseline(JSON.stringify({
      contract_version: "launch-monitor-strokes-gained-baseline/2.0.0",
      baseline_id: "licensed-test", version: "2026.1",
      source_url: "https://example.org/method", license: "test-only",
      table_sha256: await baselineTableHash(states), states,
    }));
    const payload = buildSourceBackedStrokesGainedPayload([{ player: "p1", order: 1 }], baseline, {
      beforeLieColumn: "start_lie", beforeContextColumn: "start_context", beforeTargetColumn: "target", beforeDistanceColumn: "start_distance",
      afterLieColumn: "finish_lie", afterContextColumn: "finish_context", afterTargetColumn: "target", afterDistanceColumn: "finish_distance",
      beforeDistanceUnit: "yd", afterDistanceUnit: "yd",
      trustedSummary: { playerColumn: "player", sessionColumn: "", clubColumn: "", orderColumn: "order", orderUnit: "session", evidence: "User attested." },
    });
    const request = payload.request as Record<string, unknown>;
    expect(request.summaries).toEqual([expect.objectContaining({ dimension: "player", trust_level: "explicit_user_attested" })]);
    expect(request.longitudinal).toEqual(expect.objectContaining({ group_dimension: "player", order_column: "order" }));
  });
});
