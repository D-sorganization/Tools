import { describe, it, expect } from "vitest";

import {
  effectiveHz,
  stalenessMs,
  formatStaleness,
  formatClockMillis,
  formatFeedLine,
} from "./diagnostics";

describe("effectiveHz", () => {
  it("returns 0 for fewer than two frames", () => {
    expect(effectiveHz([])).toBe(0);
    expect(effectiveHz([1000])).toBe(0);
  });

  it("computes the rate from evenly spaced timestamps", () => {
    // 11 frames, 100 ms apart -> 10 intervals over 1.0 s -> 10 Hz.
    const t = Array.from({ length: 11 }, (_, i) => 1000 + i * 100);
    expect(effectiveHz(t)).toBeCloseTo(10, 5);
  });

  it("uses only the trailing window", () => {
    // Old fast frames then recent 100 ms frames; a small window sees ~10 Hz.
    const old = Array.from({ length: 50 }, (_, i) => i * 10); // 100 Hz
    const recent = Array.from({ length: 5 }, (_, i) => 1000 + i * 100); // 10 Hz
    expect(effectiveHz([...old, ...recent], 5)).toBeCloseTo(10, 1);
  });

  it("throws TypeError on a non-array (DbC)", () => {
    // @ts-expect-error deliberate wrong type
    expect(() => effectiveHz(null)).toThrow(TypeError);
  });
});

describe("stalenessMs", () => {
  it("is Infinity when there are no frames", () => {
    expect(stalenessMs([], 5000)).toBe(Number.POSITIVE_INFINITY);
  });

  it("measures ms since the last frame", () => {
    expect(stalenessMs([1000, 2000, 3000], 3400)).toBe(400);
  });

  it("clamps a skewed clock to 0 (never negative)", () => {
    expect(stalenessMs([1000, 5000], 4000)).toBe(0);
  });

  it("throws TypeError on a non-finite now (DbC)", () => {
    expect(() => stalenessMs([1], Number.NaN)).toThrow(TypeError);
  });
});

describe("formatStaleness", () => {
  it("shows one decimal under 10 s and whole seconds above", () => {
    expect(formatStaleness(120)).toBe("0.1s");
    expect(formatStaleness(3400)).toBe("3.4s");
    expect(formatStaleness(12300)).toBe("12s");
  });

  it("shows an em-dash for a non-finite staleness", () => {
    expect(formatStaleness(Number.POSITIVE_INFINITY)).toBe("—");
  });
});

describe("formatClockMillis", () => {
  it("zero-pads to HH:MM:SS.mmm", () => {
    expect(formatClockMillis(0)).toMatch(/^\d{2}:\d{2}:\d{2}\.\d{3}$/);
  });
});

describe("formatFeedLine", () => {
  const tags = [35.9, 475.4, 0, 0];

  it("renders the clock plus the requested tag fields verbatim", () => {
    const line = formatFeedLine(0, tags, [0, 1, 2]);
    expect(line).toMatch(/^\d{2}:\d{2}:\d{2}\.\d{3}\s+T0=35\.9\s+T1=475\.4\s+T2=0\.0$/);
  });

  it("marks a non-finite tag with a dot rather than crashing", () => {
    const line = formatFeedLine(0, [Number.NaN], [0]);
    expect(line).toContain("T0=·");
  });

  it("throws TypeError on a non-array tags (DbC)", () => {
    // @ts-expect-error deliberate wrong type
    expect(() => formatFeedLine(0, null, [0])).toThrow(TypeError);
  });
});
