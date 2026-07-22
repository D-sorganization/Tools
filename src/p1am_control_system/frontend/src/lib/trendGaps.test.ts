import { describe, it, expect } from "vitest";

import { bridgeIsolatedDropouts, bridgeTimedSeries } from "./trendGaps";

describe("bridgeIsolatedDropouts", () => {
  it("interpolates a single spurious zero between solid neighbours", () => {
    // The real dropout pattern from the historian: 34.4, 0.0, 34.3.
    const out = bridgeIsolatedDropouts([34.4, 0, 34.3]);
    expect(out[0]).toBe(34.4);
    expect(out[2]).toBe(34.3);
    expect(out[1]).toBeCloseTo(34.35, 5); // interpolated, not 0
  });

  it("bridges a short run up to maxRun", () => {
    const out = bridgeIsolatedDropouts([100, 0, 0, 100], { maxRun: 2 });
    expect(out[1]).toBeCloseTo(100, 5);
    expect(out[2]).toBeCloseTo(100, 5);
  });

  it("leaves a run longer than maxRun untouched", () => {
    const input = [100, 0, 0, 0, 100];
    expect(bridgeIsolatedDropouts(input, { maxRun: 2 })).toEqual(input);
  });

  it("leaves a sustained / genuinely-zero channel untouched", () => {
    expect(bridgeIsolatedDropouts([0, 0, 0, 0])).toEqual([0, 0, 0, 0]);
  });

  it("does not bridge when the neighbours are not clearly non-zero", () => {
    // A signal genuinely near zero (neighbours below minNeighbor) is real data.
    const input = [0.5, 0, 0.5];
    expect(bridgeIsolatedDropouts(input, { minNeighbor: 5 })).toEqual(input);
  });

  it("leaves an edge dropout (no bracket) untouched", () => {
    expect(bridgeIsolatedDropouts([0, 34, 34])).toEqual([0, 34, 34]);
    expect(bridgeIsolatedDropouts([34, 34, 0])).toEqual([34, 34, 0]);
  });

  it("preserves a true zero-crossing of a bipolar signal", () => {
    // +10, 0, −10 straddles zero; interpolation restores ≈0 (the real value).
    const out = bridgeIsolatedDropouts([10, 0, -10]);
    expect(out[1]).toBeCloseTo(0, 5);
  });

  it("lifts a same-level dropout back to the signal level", () => {
    const out = bridgeIsolatedDropouts([475, 0, 475]);
    expect(out[1]).toBeCloseTo(475, 5);
  });

  it("handles multiple separate dropouts in one series", () => {
    const out = bridgeIsolatedDropouts([50, 0, 50, 60, 0, 60]);
    expect(out[1]).toBeCloseTo(50, 5);
    expect(out[4]).toBeCloseTo(60, 5);
  });

  it("returns a copy, not the input array", () => {
    const input = [10, 10];
    expect(bridgeIsolatedDropouts(input)).not.toBe(input);
  });

  it("throws TypeError on a non-array input (DbC)", () => {
    // @ts-expect-error deliberate wrong type
    expect(() => bridgeIsolatedDropouts(null)).toThrow(TypeError);
  });

  it("throws TypeError on a negative option (DbC)", () => {
    expect(() => bridgeIsolatedDropouts([1], { floor: -1 })).toThrow(TypeError);
  });
});

describe("bridgeTimedSeries", () => {
  it("bridges the v field while preserving t and other fields", () => {
    const pts = [
      { t: 1, v: 50, extra: "a" },
      { t: 2, v: 0, extra: "b" },
      { t: 3, v: 50, extra: "c" },
    ];
    const out = bridgeTimedSeries(pts);
    expect(out.map((p) => p.t)).toEqual([1, 2, 3]);
    expect(out.map((p) => p.extra)).toEqual(["a", "b", "c"]);
    expect(out[1].v).toBeCloseTo(50, 5);
  });

  it("returns new objects, not the inputs", () => {
    const pts = [{ t: 1, v: 10 }];
    expect(bridgeTimedSeries(pts)[0]).not.toBe(pts[0]);
  });
});
