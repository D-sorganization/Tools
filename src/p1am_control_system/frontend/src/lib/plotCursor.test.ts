import { describe, expect, it } from "vitest";

import { nearestIndexByX, placeTooltip, type PlotRect } from "./plotCursor";

describe("nearestIndexByX", () => {
  const xs = [0, 10, 20, 30, 40];

  it("returns null for an empty array", () => {
    expect(nearestIndexByX([], 5)).toBeNull();
  });

  it("clamps below the first sample to index 0", () => {
    expect(nearestIndexByX(xs, -100)).toBe(0);
    expect(nearestIndexByX(xs, 0)).toBe(0);
  });

  it("clamps above the last sample to the last index", () => {
    expect(nearestIndexByX(xs, 100)).toBe(4);
    expect(nearestIndexByX(xs, 40)).toBe(4);
  });

  it("finds the nearest interior sample", () => {
    expect(nearestIndexByX(xs, 11)).toBe(1); // closer to 10
    expect(nearestIndexByX(xs, 19)).toBe(2); // closer to 20
    expect(nearestIndexByX(xs, 26)).toBe(3); // closer to 30
  });

  it("resolves an exact midpoint to the lower index (tie)", () => {
    expect(nearestIndexByX(xs, 15)).toBe(1); // equidistant 10/20 -> lower
    expect(nearestIndexByX(xs, 25)).toBe(2);
  });

  it("returns the exact index when targetX matches a sample", () => {
    expect(nearestIndexByX(xs, 20)).toBe(2);
    expect(nearestIndexByX(xs, 30)).toBe(3);
  });

  it("handles a single-element array", () => {
    expect(nearestIndexByX([42], 0)).toBe(0);
    expect(nearestIndexByX([42], 99)).toBe(0);
  });

  it("works on large ascending arrays (binary search)", () => {
    const big = Array.from({ length: 10000 }, (_, i) => i * 2); // 0,2,...,19998
    expect(nearestIndexByX(big, 5001)).toBe(2500); // nearest to 5000
    expect(nearestIndexByX(big, 4)).toBe(2);
  });

  it("throws TypeError on a non-array xs (DbC)", () => {
    // @ts-expect-error deliberate wrong type
    expect(() => nearestIndexByX(null, 5)).toThrow(TypeError);
  });

  it("throws TypeError on a non-finite targetX (DbC)", () => {
    expect(() => nearestIndexByX(xs, Number.NaN)).toThrow(TypeError);
    expect(() => nearestIndexByX(xs, Number.POSITIVE_INFINITY)).toThrow(TypeError);
  });
});

describe("placeTooltip", () => {
  const bounds: PlotRect = { x0: 0, y0: 0, x1: 200, y1: 100 };
  const size = { w: 40, h: 20 };

  it("places up-and-right of the anchor when there is room", () => {
    const pos = placeTooltip({ x: 50, y: 50 }, size, bounds, 8);
    expect(pos.x).toBe(58); // 50 + 8
    expect(pos.y).toBe(22); // 50 - 20 - 8
  });

  it("flips to the left when it would overflow the right edge", () => {
    const pos = placeTooltip({ x: 190, y: 50 }, size, bounds, 8);
    // 190 + 8 + 40 = 238 > 200 -> flip left: 190 - 8 - 40 = 142
    expect(pos.x).toBe(142);
  });

  it("flips below when it would overflow the top edge", () => {
    const pos = placeTooltip({ x: 50, y: 5 }, size, bounds, 8);
    // 5 - 20 - 8 = -23 < 0 -> flip below: 5 + 8 = 13
    expect(pos.y).toBe(13);
  });

  it("clamps so the box never leaves the plot rectangle", () => {
    const pos = placeTooltip({ x: 199, y: 99 }, size, bounds, 8);
    expect(pos.x).toBeLessThanOrEqual(bounds.x1 - size.w);
    expect(pos.x).toBeGreaterThanOrEqual(bounds.x0);
    expect(pos.y).toBeLessThanOrEqual(bounds.y1 - size.h);
    expect(pos.y).toBeGreaterThanOrEqual(bounds.y0);
  });

  it("throws TypeError on a non-finite anchor (DbC)", () => {
    expect(() => placeTooltip({ x: Number.NaN, y: 0 }, size, bounds)).toThrow(
      TypeError,
    );
  });
});
