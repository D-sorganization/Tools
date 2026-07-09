import { describe, it, expect } from "vitest";

import {
  buildCrosshairModel,
  finitePairs,
  makeProjector,
  type HoverSeries,
} from "./projection";

const ZERO_MARGIN = { top: 0, right: 0, bottom: 0, left: 0 };

describe("AxisProjector.invert", () => {
  it("inverts a linear x-axis (data value from an inner pixel)", () => {
    const { x } = makeProjector({
      width: 200,
      height: 100,
      xDomain: [0, 10],
      yDomain: [0, 4],
      margin: ZERO_MARGIN,
    });
    expect(x.invert(0)).toBeCloseTo(0, 6);
    expect(x.invert(200)).toBeCloseTo(10, 6);
    expect(x.invert(100)).toBeCloseTo(5, 6);
  });

  it("inverts an inverted linear y-axis (top pixel is the data max)", () => {
    const { y } = makeProjector({
      width: 200,
      height: 100,
      xDomain: [0, 10],
      yDomain: [0, 4],
      margin: ZERO_MARGIN,
    });
    // y pixels run [innerHeight..0] as the value climbs [0..4].
    expect(y.invert(0)).toBeCloseTo(4, 6); // top
    expect(y.invert(100)).toBeCloseTo(0, 6); // bottom
    expect(y.invert(50)).toBeCloseTo(2, 6); // middle
  });

  it("inverts a log10 axis back out of log space", () => {
    const { x } = makeProjector({
      width: 200,
      height: 100,
      xDomain: [1, 100],
      yDomain: [0, 1],
      logX: true,
      margin: ZERO_MARGIN,
    });
    expect(x.invert(0)).toBeCloseTo(1, 6);
    expect(x.invert(200)).toBeCloseTo(100, 6);
    // The pixel midpoint is log10(10) -> the geometric mean of 1..100.
    expect(x.invert(100)).toBeCloseTo(10, 4);
  });

  it("round-trips x(invert(p)) ≈ p on linear and log axes", () => {
    const linear = makeProjector({
      width: 200,
      height: 120,
      xDomain: [-3, 7],
      yDomain: [10, 50],
      margin: ZERO_MARGIN,
    });
    const log = makeProjector({
      width: 200,
      height: 120,
      xDomain: [0.5, 5000],
      yDomain: [1, 1000],
      logX: true,
      logY: true,
      margin: ZERO_MARGIN,
    });
    for (const px of [0, 25, 60, 137, 200]) {
      expect(linear.x(linear.x.invert(px))).toBeCloseTo(px, 4);
      expect(log.x(log.x.invert(px))).toBeCloseTo(px, 4);
    }
    for (const py of [0, 30, 60, 90, 120]) {
      expect(linear.y(linear.y.invert(py))).toBeCloseTo(py, 4);
      expect(log.y(log.y.invert(py))).toBeCloseTo(py, 4);
    }
  });

  it("falls back to the domain midpoint on a zero-width pixel range", () => {
    // width == left+right margins -> innerWidth 0 -> no invertible pixel span.
    const { x } = makeProjector({
      width: 100,
      height: 100,
      xDomain: [2, 8],
      yDomain: [0, 1],
      margin: { top: 0, right: 50, bottom: 0, left: 50 },
    });
    expect(x.invert(0)).toBeCloseTo(5, 6); // (2 + 8) / 2
  });
});

describe("finitePairs", () => {
  it("splits points into parallel arrays, dropping non-finite pairs", () => {
    const { xs, ys } = finitePairs([
      [0, 1],
      [1, Number.NaN],
      [2, 3],
      [Number.POSITIVE_INFINITY, 4],
    ]);
    expect(xs).toEqual([0, 2]);
    expect(ys).toEqual([1, 3]);
  });

  it("throws TypeError when points is not an array (DbC)", () => {
    // @ts-expect-error deliberate wrong type
    expect(() => finitePairs(null)).toThrow(TypeError);
  });
});

describe("buildCrosshairModel", () => {
  const proj = makeProjector({
    width: 200,
    height: 100,
    xDomain: [0, 2],
    yDomain: [0, 20],
    margin: ZERO_MARGIN,
  });

  it("resolves the nearest sample of each series and snaps to the first", () => {
    const series: HoverSeries[] = [
      { label: "a", color: "#f00", xs: [0, 1, 2], ys: [0, 10, 20] },
    ];
    const model = buildCrosshairModel(100, series, proj.x, proj.y);
    expect(model).not.toBeNull();
    expect(model?.dataX).toBe(1); // snapped x
    expect(model?.lineX).toBeCloseTo(100, 6); // x(1) with inner width 200
    expect(model?.markers).toHaveLength(1);
    expect(model?.markers[0].value).toBe(10);
    expect(model?.markers[0].px).toBeCloseTo(100, 6);
    expect(model?.markers[0].py).toBeCloseTo(50, 6); // y(10) inverted
  });

  it("produces a marker per series, snapping the line to the first", () => {
    const series: HoverSeries[] = [
      { label: "a", color: "#f00", xs: [0, 10], ys: [1, 2] },
      { label: "b", color: "#0f0", xs: [3, 7], ys: [5, 6] },
    ];
    const p = makeProjector({
      width: 200,
      height: 100,
      xDomain: [0, 10],
      yDomain: [0, 10],
      margin: ZERO_MARGIN,
    });
    // innerX 80 -> dataX 4: A nearest -> 0 (idx 0), B nearest -> 3 (idx 0).
    const model = buildCrosshairModel(80, series, p.x, p.y);
    expect(model?.markers).toHaveLength(2);
    expect(model?.dataX).toBe(0); // first series' nearest x
  });

  it("returns null when no series has a finite sample", () => {
    expect(
      buildCrosshairModel(50, [], proj.x, proj.y),
    ).toBeNull();
    expect(
      buildCrosshairModel(
        50,
        [{ label: "empty", color: "#000", xs: [], ys: [] }],
        proj.x,
        proj.y,
      ),
    ).toBeNull();
  });

  it("throws TypeError on a non-finite innerX (DbC)", () => {
    expect(() =>
      buildCrosshairModel(Number.NaN, [], proj.x, proj.y),
    ).toThrow(TypeError);
  });

  it("throws TypeError when series is not an array (DbC)", () => {
    // @ts-expect-error deliberate wrong type
    expect(() => buildCrosshairModel(10, null, proj.x, proj.y)).toThrow(
      TypeError,
    );
  });
});
