/** Course scene palette + layout (epic #4125 H7a) — Python mirror. */

import { describe, expect, it } from "vitest";

import { courseColors, DEFAULT_COURSE_LAYOUT } from "./course";
import { blend, getChartColor, withAlpha } from "./theme";

describe("blend", () => {
  it("hits both endpoints and the midpoint", () => {
    expect(blend("#000000", "#ffffff", 0)).toBe("#000000");
    expect(blend("#000000", "#ffffff", 1)).toBe("#ffffff");
    expect(blend("#000000", "#ffffff", 0.5)).toBe("#808080");
  });

  it("clamps the fraction and passes non-hex through", () => {
    expect(blend("#102030", "#ffffff", 2)).toBe("#ffffff");
    expect(blend("C1", "#000000", 0.5)).toBe("C1");
  });
});

describe("courseColors", () => {
  it("derives every tone from the chart palette (Python parity)", () => {
    const grass = getChartColor(1);
    const tones = courseColors();
    expect(tones.rough).toBe(blend(grass, "#000000", 0.45));
    expect(tones.fairway).toBe(blend(grass, "#000000", 0.15));
    expect(tones.green).toBe(blend(grass, "#ffffff", 0.2));
    expect(tones.hole).toBe(blend(grass, "#000000", 0.85));
    expect(tones.flag).toBe(getChartColor(3));
    expect(tones.tee).toBe(getChartColor(6));
  });

  it("orders the grass family dark to light", () => {
    const luminance = (c: string) =>
      0.299 * parseInt(c.slice(1, 3), 16) +
      0.587 * parseInt(c.slice(3, 5), 16) +
      0.114 * parseInt(c.slice(5, 7), 16);
    const tones = courseColors();
    expect(luminance(tones.hole)).toBeLessThan(luminance(tones.rough));
    expect(luminance(tones.rough)).toBeLessThan(luminance(tones.fairway));
    expect(luminance(tones.fairway)).toBeLessThan(luminance(tones.green));
  });
});

describe("layout + alpha helper", () => {
  it("default layout is sane", () => {
    expect(DEFAULT_COURSE_LAYOUT.greenDistanceM).toBeGreaterThan(
      DEFAULT_COURSE_LAYOUT.greenRadiusM,
    );
    expect(DEFAULT_COURSE_LAYOUT.fairwayHalfWidthM).toBeGreaterThan(0);
  });

  it("withAlpha appends a two-digit channel", () => {
    expect(withAlpha("#30d158", 1)).toBe("#30d158ff");
    expect(withAlpha("#30d158", 0)).toBe("#30d15800");
  });
});
