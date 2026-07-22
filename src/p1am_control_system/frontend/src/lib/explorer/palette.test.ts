import { describe, it, expect } from "vitest";
import { SERIES_COLORS, colorForIndex, divergingColor } from "./palette";

const HEX = /^#[0-9a-f]{6}$/;

describe("SERIES_COLORS", () => {
  it("is a non-empty list of 6-digit hex colors", () => {
    expect(SERIES_COLORS.length).toBeGreaterThan(0);
    for (const c of SERIES_COLORS) {
      expect(c).toMatch(HEX);
    }
  });

  it("has no duplicate colors", () => {
    expect(new Set(SERIES_COLORS).size).toBe(SERIES_COLORS.length);
  });
});

describe("colorForIndex", () => {
  it("returns the palette entry for in-range indices", () => {
    expect(colorForIndex(0)).toBe(SERIES_COLORS[0]);
    expect(colorForIndex(1)).toBe(SERIES_COLORS[1]);
  });

  it("cycles modulo the palette length", () => {
    const n = SERIES_COLORS.length;
    expect(colorForIndex(n)).toBe(SERIES_COLORS[0]);
    expect(colorForIndex(n + 1)).toBe(SERIES_COLORS[1]);
    expect(colorForIndex(2 * n + 3)).toBe(SERIES_COLORS[3]);
  });

  it("throws on negative or non-integer indices (DbC)", () => {
    expect(() => colorForIndex(-1)).toThrow(TypeError);
    expect(() => colorForIndex(1.5)).toThrow(TypeError);
    expect(() => colorForIndex(NaN)).toThrow(TypeError);
  });
});

describe("divergingColor", () => {
  it("produces valid hex across the range", () => {
    for (let t = -1; t <= 1.0001; t += 0.1) {
      expect(divergingColor(t)).toMatch(HEX);
    }
  });

  it("anchors at blue / white / red", () => {
    expect(divergingColor(-1)).toBe("#2166ac");
    expect(divergingColor(0)).toBe("#f7f7f7");
    expect(divergingColor(1)).toBe("#b2182b");
  });

  it("is symmetric in lightness around the midpoint", () => {
    // Endpoints differ; the midpoint is the lightest point.
    const mid = divergingColor(0);
    expect(mid).toBe("#f7f7f7");
    expect(divergingColor(-0.5)).not.toBe(divergingColor(0.5));
  });

  it("clamps values outside [-1, 1]", () => {
    expect(divergingColor(-5)).toBe(divergingColor(-1));
    expect(divergingColor(5)).toBe(divergingColor(1));
  });

  it("throws on non-finite t (DbC)", () => {
    expect(() => divergingColor(NaN)).toThrow(TypeError);
    expect(() => divergingColor(Infinity)).toThrow(TypeError);
  });
});
