import { describe, expect, it } from "vitest";

import {
  STROKES_GAINED_EXCLUSION_REASONS,
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

const columnRequest = {
  beforeLieColumn: "before_lie", beforeContextColumn: "before_context",
  beforeTargetColumn: "target", beforeDistanceColumn: "before_distance",
  afterLieColumn: "after_lie", afterContextColumn: "after_context",
  afterTargetColumn: "target", afterDistanceColumn: "after_distance",
  beforeDistanceUnit: "yd" as const, afterDistanceUnit: "yd" as const,
};

const cleanRow = () => ({
  before_lie: "fairway", before_context: "standard", target: "hole-1", before_distance: 150,
  after_lie: "green", after_context: "standard", after_distance: 20,
});

const loadedBaseline = async () => parseStrokesGainedBaseline(JSON.stringify({
  contract_version: "launch-monitor-strokes-gained-baseline/2.0.0",
  baseline_id: "licensed-test", version: "2026.1",
  source_url: "https://example.org/method", license: "test-only",
  table_sha256: await baselineTableHash(states), states,
}));

describe("source-backed strokes gained", () => {
  it("verifies the artifact hash and interpolates within each course lie", async () => {
    expect(await baselineTableHash([...states].reverse())).toBe("5250552cc6ec58da60dfe8ebf50f7238534d28016b0725bf42d8098054404428"); // pragma: allowlist secret
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
    expect(result.status).toBe("available");
    expect(result.excludedRows).toEqual([]);
    expect(result.exclusions).toEqual({ inputRowCount: 1, includedRowCount: 1, totalExcluded: 0, byReason: {} });
  });

  // ADR-0048 decision G1-D3: exclude-and-audit. These cases mirror the Python
  // twin's parametrised suite one-for-one so the two runtimes classify the
  // same malformed row identically.
  it.each([
    [{ before_lie: "rough" }, "outside_baseline"],
    [{ before_context: "windy" }, "outside_baseline"],
    [{ before_distance: 400 }, "outside_baseline"],
    [{ before_lie: "   " }, "missing_course_state"],
    [{ before_lie: null }, "missing_course_state"],
    [{ after_distance: null }, "missing_course_state"],
    [{ after_distance: "not-a-number" }, "missing_course_state"],
    [{ after_distance: -1.5 }, "invalid_distance"],
    [{ after_distance: true }, "invalid_distance"],
  ])("excludes and audits a malformed row (%j) instead of throwing", async (overrides, reasonCode) => {
    const baseline = await loadedBaseline();
    const result = calculateSourceBackedStrokesGained(
      [cleanRow(), { ...cleanRow(), ...overrides }, cleanRow()], baseline, columnRequest,
    );

    expect(result.status).toBe("partial");
    expect(result.mean).toBeCloseTo(0.8);
    expect(result.values).toHaveLength(2);
    expect(result.excludedRows.map((row) => row.sourceIndex)).toEqual([1]);
    expect(result.excludedRows[0].reasonCode).toBe(reasonCode);
    expect(result.excludedRows[0].message).toBeTruthy();
    expect(result.exclusions).toEqual({
      inputRowCount: 3, includedRowCount: 2, totalExcluded: 1, byReason: { [reasonCode as string]: 1 },
    });
  });

  it("reports an audited unavailable result when no row survives", async () => {
    const baseline = await loadedBaseline();
    const result = calculateSourceBackedStrokesGained(
      [{ ...cleanRow(), before_lie: "rough" }, { ...cleanRow(), before_lie: "   " }], baseline, columnRequest,
    );

    expect(result.status).toBe("unavailable");
    expect(result.mean).toBeNull();
    expect(result.values).toEqual([]);
    expect(result.exclusions).toEqual({
      inputRowCount: 2, includedRowCount: 0, totalExcluded: 2,
      byReason: { outside_baseline: 1, missing_course_state: 1 },
    });
  });

  it("accounts for every supplied row, dropping none in silence", async () => {
    const baseline = await loadedBaseline();
    const rows = [cleanRow(), { ...cleanRow(), before_lie: "rough" },
      { ...cleanRow(), after_context: "" }, { ...cleanRow(), before_distance: -2 }];
    const result = calculateSourceBackedStrokesGained(rows, baseline, columnRequest);
    const summary = result.exclusions;

    expect(summary.inputRowCount).toBe(rows.length);
    expect(summary.includedRowCount + summary.totalExcluded).toBe(rows.length);
    expect(summary.includedRowCount).toBe(result.backingRows.length);
    expect(summary.totalExcluded).toBe(result.excludedRows.length);
    expect(Object.values(summary.byReason).reduce((sum, count) => sum + count, 0)).toBe(summary.totalExcluded);
    expect(Object.keys(summary.byReason).every((code) =>
      (STROKES_GAINED_EXCLUSION_REASONS as readonly string[]).includes(code))).toBe(true);
    expect(summary.byReason).toEqual({ outside_baseline: 1, missing_course_state: 1, invalid_distance: 1 });
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
