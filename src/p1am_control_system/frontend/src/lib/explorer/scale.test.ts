import { describe, it, expect } from "vitest";
import { clamp, extent, linearScale, niceTicks } from "./scale";

describe("clamp", () => {
  it("restricts to [min, max]", () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(-3, 0, 10)).toBe(0);
    expect(clamp(42, 0, 10)).toBe(10);
    expect(clamp(0, 0, 10)).toBe(0);
    expect(clamp(10, 0, 10)).toBe(10);
  });

  it("throws on non-finite args (DbC)", () => {
    expect(() => clamp(NaN, 0, 1)).toThrow(TypeError);
    expect(() => clamp(1, Infinity, 2)).toThrow(TypeError);
  });

  it("throws RangeError when min > max", () => {
    expect(() => clamp(1, 5, 0)).toThrow(RangeError);
  });
});

describe("linearScale", () => {
  it("maps the domain endpoints onto the range endpoints", () => {
    const s = linearScale([0, 10], [0, 100]);
    expect(s(0)).toBeCloseTo(0, 9);
    expect(s(10)).toBeCloseTo(100, 9);
    expect(s(5)).toBeCloseTo(50, 9);
    expect(s.domain).toEqual([0, 10]);
    expect(s.range).toEqual([0, 100]);
  });

  it("handles an inverted (pixel) range, as used for the y-axis", () => {
    const s = linearScale([0, 1], [200, 0]);
    expect(s(0)).toBeCloseTo(200, 9);
    expect(s(1)).toBeCloseTo(0, 9);
    expect(s(0.5)).toBeCloseTo(100, 9);
  });

  it("inverts back to the domain", () => {
    const s = linearScale([0, 10], [0, 100]);
    expect(s.invert(50)).toBeCloseTo(5, 9);
    expect(s.invert(0)).toBeCloseTo(0, 9);
    expect(s.invert(100)).toBeCloseTo(10, 9);
  });

  it("maps a zero-width domain to the range midpoint", () => {
    const s = linearScale([5, 5], [0, 100]);
    expect(s(5)).toBe(50);
    expect(s(999)).toBe(50);
  });

  it("throws on non-finite bounds (DbC)", () => {
    expect(() => linearScale([NaN, 1], [0, 1])).toThrow(TypeError);
    expect(() => linearScale([0, 1], [0, Infinity])).toThrow(TypeError);
  });
});

describe("niceTicks", () => {
  it("returns round 1/2/5 step ticks spanning the data", () => {
    const ticks = niceTicks(0, 10, 5);
    expect(ticks).toEqual([0, 2, 4, 6, 8, 10]);
  });

  it("rounds the endpoints outward", () => {
    const ticks = niceTicks(1, 9, 5);
    // step ~ nice(8/5=1.6) -> 2 ; floor(1/2)*2=0, ceil(9/2)*2=10
    expect(ticks[0]).toBe(0);
    expect(ticks[ticks.length - 1]).toBe(10);
  });

  it("produces increasing ticks and ~count intervals", () => {
    const ticks = niceTicks(0, 100, 5);
    expect(ticks).toEqual([0, 20, 40, 60, 80, 100]);
    for (let i = 1; i < ticks.length; i += 1) {
      expect(ticks[i]).toBeGreaterThan(ticks[i - 1]);
    }
  });

  it("handles fractional domains without float noise", () => {
    const ticks = niceTicks(0, 1, 5);
    expect(ticks).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
  });

  it("returns a single tick when min === max", () => {
    expect(niceTicks(7, 7, 5)).toEqual([7]);
  });

  it("throws on non-finite bounds / bad count / inverted range (DbC)", () => {
    expect(() => niceTicks(NaN, 1, 5)).toThrow(TypeError);
    expect(() => niceTicks(0, 1, 0)).toThrow(RangeError);
    expect(() => niceTicks(0, 1, 2.5)).toThrow(RangeError);
    expect(() => niceTicks(10, 0, 5)).toThrow(RangeError);
  });
});

describe("extent", () => {
  it("returns [min, max] ignoring nulls and non-finite values", () => {
    expect(extent([3, 1, 4, 1, 5, 9, 2, 6])).toEqual([1, 9]);
    expect(extent([null, 2, null, -1, NaN, 8])).toEqual([-1, 8]);
    expect(extent([Infinity, 5, -Infinity, 3])).toEqual([3, 5]);
  });

  it("returns [NaN, NaN] for an all-gap / empty column", () => {
    expect(extent([]).every(Number.isNaN)).toBe(true);
    expect(extent([null, null]).every(Number.isNaN)).toBe(true);
    expect(extent([NaN, Infinity]).every(Number.isNaN)).toBe(true);
  });

  it("throws TypeError when not given an array", () => {
    // @ts-expect-error testing runtime guard
    expect(() => extent("nope")).toThrow(TypeError);
  });
});
