import { describe, it, expect } from "vitest";
import {
  createViewport,
  isLive,
  resolveVisible,
  panBy,
  zoomBy,
  zoomToRange,
  reset,
  setPaused,
  type SpanLimits,
} from "./trendViewport";

const bounds = { min: 0, max: 1000 }; // e.g. 1000 units of history available
const limits: SpanLimits = { minSpan: 10, maxSpan: 5000 };

describe("createViewport", () => {
  it("makes a live viewport with the given span", () => {
    const vp = createViewport(100);
    expect(vp).toEqual({ paused: false, span: 100, offset: 0 });
    expect(isLive(vp)).toBe(true);
  });
  it("rejects a non-positive or non-finite span (DbC)", () => {
    expect(() => createViewport(0)).toThrow(RangeError);
    expect(() => createViewport(Number.NaN)).toThrow(TypeError);
  });
});

describe("resolveVisible", () => {
  it("follows the live edge when offset is 0", () => {
    expect(resolveVisible(createViewport(100), bounds, limits)).toEqual({
      start: 900,
      end: 1000,
    });
  });
  it("shows the panned-back window when offset > 0", () => {
    const vp = { paused: false, span: 100, offset: 200 };
    expect(resolveVisible(vp, bounds, limits)).toEqual({ start: 700, end: 800 });
  });
  it("clamps the window to the oldest sample", () => {
    const vp = { paused: false, span: 100, offset: 99999 };
    const r = resolveVisible(vp, bounds, limits);
    expect(r.start).toBe(0);
    expect(r.end).toBe(100);
  });
  it("clamps span to the limits", () => {
    const wide = resolveVisible({ paused: false, span: 1, offset: 0 }, bounds, limits);
    expect(wide.end - wide.start).toBe(limits.minSpan);
  });
});

describe("panBy", () => {
  it("pans back into history (positive delta) and stops following", () => {
    const vp = panBy(createViewport(100), 300, bounds, limits);
    expect(vp.offset).toBe(300);
    expect(isLive(vp)).toBe(false);
  });
  it("clamps at the live edge (offset never negative)", () => {
    const vp = panBy({ paused: false, span: 100, offset: 50 }, -200, bounds, limits);
    expect(vp.offset).toBe(0);
    expect(isLive(vp)).toBe(true);
  });
  it("clamps at the oldest data (cannot pan past the buffer)", () => {
    const vp = panBy(createViewport(100), 100000, bounds, limits);
    expect(vp.offset).toBe(bounds.max - bounds.min - 100); // 900
  });
  it("rejects a non-finite delta (DbC)", () => {
    expect(() => panBy(createViewport(100), Number.NaN, bounds, limits)).toThrow(
      TypeError,
    );
  });
});

describe("zoomBy", () => {
  it("zooms in (factor<1) keeping the focus point stationary", () => {
    const vp = createViewport(100); // visible [900,1000]
    const focus = 950; // middle
    const z = zoomBy(vp, 0.5, focus, bounds, limits);
    expect(z.span).toBe(50);
    const r = resolveVisible(z, bounds, limits);
    // focus stays at the same fraction (middle) -> [925, 975]
    expect(r.start).toBeCloseTo(925, 6);
    expect(r.end).toBeCloseTo(975, 6);
  });
  it("zooms out (factor>1) and clamps span to maxSpan", () => {
    const z = zoomBy(createViewport(3000), 5, 500, bounds, limits);
    expect(z.span).toBe(limits.maxSpan);
  });
  it("rejects factor <= 0 (DbC)", () => {
    expect(() => zoomBy(createViewport(100), 0, 950, bounds, limits)).toThrow(
      RangeError,
    );
  });
});

describe("zoomToRange (drag-select)", () => {
  it("sets span to the selection width and right edge to the selection end", () => {
    const z = zoomToRange(createViewport(1000), 300, 500, bounds, limits);
    expect(z.span).toBe(200);
    const r = resolveVisible(z, bounds, limits);
    expect(r).toEqual({ start: 300, end: 500 });
  });
  it("normalizes a backwards selection (b < a)", () => {
    const z = zoomToRange(createViewport(1000), 500, 300, bounds, limits);
    const r = resolveVisible(z, bounds, limits);
    expect(r).toEqual({ start: 300, end: 500 });
  });
  it("clamps a too-small selection to minSpan", () => {
    const z = zoomToRange(createViewport(1000), 500, 502, bounds, limits);
    expect(z.span).toBe(limits.minSpan);
  });
});

describe("reset / setPaused", () => {
  it("reset returns a live viewport at the span", () => {
    const vp = reset(120);
    expect(vp).toEqual({ paused: false, span: 120, offset: 0 });
  });
  it("setPaused toggles the flag and leaves live-following false while paused", () => {
    const paused = setPaused(createViewport(100), true);
    expect(paused.paused).toBe(true);
    expect(isLive(paused)).toBe(false);
    expect(() => setPaused(createViewport(100), "yes" as unknown as boolean)).toThrow(
      TypeError,
    );
  });
});
