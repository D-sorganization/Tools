import { describe, it, expect } from "vitest";
import {
  linearFit,
  fitSeries,
  getFitMethod,
  fitOptions,
  heatUpRateFromFit,
  pointsInLastWindow,
  formatHeatUpRate,
  NO_FIT_ID,
  type FitPoint,
} from "./curveFit";

const line = (m: number, b: number, n: number): FitPoint[] =>
  Array.from({ length: n }, (_, i) => ({ x: i, y: m * i + b }));

describe("linearFit", () => {
  it("recovers slope and intercept of a perfect line", () => {
    const r = linearFit.fit(line(2, 1, 10));
    expect(r.coeffs[0]).toBeCloseTo(2, 9);
    expect(r.coeffs[1]).toBeCloseTo(1, 9);
    expect(r.r2).toBeCloseTo(1, 9);
    expect(r.predict(5)).toBeCloseTo(11, 9);
  });

  it("fits the least-squares line through noisy points", () => {
    const r = linearFit.fit([
      { x: 0, y: 1 },
      { x: 1, y: 3 },
      { x: 2, y: 4 },
      { x: 3, y: 7 },
    ]);
    // slope from least squares = 1.9, intercept = 0.9
    expect(r.coeffs[0]).toBeCloseTo(1.9, 6);
    expect(r.coeffs[1]).toBeCloseTo(0.9, 6);
    expect(r.r2).toBeGreaterThan(0.95);
  });

  it("renders an equation string", () => {
    expect(linearFit.fit(line(2, 1, 5)).equation).toBe("y = 2·x + 1");
    expect(linearFit.fit(line(1.5, -3, 5)).equation).toBe("y = 1.5·x − 3");
  });

  it("throws on too few points or zero x-variance (DbC)", () => {
    expect(() => linearFit.fit([{ x: 0, y: 0 }])).toThrow(RangeError);
    expect(() =>
      linearFit.fit([
        { x: 2, y: 0 },
        { x: 2, y: 5 },
      ]),
    ).toThrow(RangeError);
  });
});

describe("fitSeries", () => {
  it("returns a result for a known method", () => {
    expect(fitSeries(line(3, 0, 5), "linear")?.coeffs[0]).toBeCloseTo(3, 9);
  });

  it("returns null for none / unknown method / too few points", () => {
    expect(fitSeries(line(1, 0, 5), NO_FIT_ID)).toBeNull();
    expect(fitSeries(line(1, 0, 5), "nope")).toBeNull();
    expect(fitSeries([{ x: 0, y: 0 }], "linear")).toBeNull();
  });

  it("drops non-finite points before fitting", () => {
    const pts = [...line(2, 1, 4), { x: NaN, y: 5 }, { x: 5, y: Infinity }];
    expect(fitSeries(pts, "linear")?.coeffs[0]).toBeCloseTo(2, 9);
  });

  it("never throws on degenerate input", () => {
    expect(() =>
      fitSeries(
        [
          { x: 1, y: 1 },
          { x: 1, y: 2 },
        ],
        "linear",
      ),
    ).not.toThrow();
  });
});

describe("registry", () => {
  it("exposes options with None first", () => {
    const opts = fitOptions();
    expect(opts[0]).toEqual({ id: NO_FIT_ID, label: "None" });
    expect(opts.map((o) => o.id)).toContain("linear");
  });

  it("looks methods up by id", () => {
    expect(getFitMethod("linear")?.label).toBe("Linear");
    expect(getFitMethod("missing")).toBeNull();
  });
});

describe("pointsInLastWindow", () => {
  it("keeps only points within the window of the latest x", () => {
    // x in ms; keep the last 2000 ms => x >= 5000-2000 = 3000.
    const pts: FitPoint[] = [
      { x: 0, y: 0 },
      { x: 3000, y: 3 },
      { x: 4000, y: 4 },
      { x: 5000, y: 5 },
    ];
    expect(pointsInLastWindow(pts, 2000).map((p) => p.x)).toEqual([3000, 4000, 5000]);
  });

  it("returns all points when the window is non-positive or non-finite", () => {
    const pts: FitPoint[] = [
      { x: 0, y: 0 },
      { x: 10, y: 1 },
    ];
    expect(pointsInLastWindow(pts, 0)).toBe(pts);
    expect(pointsInLastWindow(pts, -5)).toBe(pts);
    expect(pointsInLastWindow(pts, Number.NaN)).toBe(pts);
  });

  it("handles an empty series and rejects non-arrays (DbC)", () => {
    expect(pointsInLastWindow([], 1000)).toEqual([]);
    // @ts-expect-error deliberate wrong type
    expect(() => pointsInLastWindow(null, 1000)).toThrow(TypeError);
  });
});

describe("heatUpRateFromFit", () => {
  it("converts a ms-based linear slope into deg/min and deg/hr", () => {
    // 0.01 deg C per ms => 600 deg/min, 36000 deg/hr.
    const fit = linearFit.fit([
      { x: 0, y: 0 },
      { x: 1000, y: 10 },
    ]);
    const rate = heatUpRateFromFit(fit, 1);
    expect(rate?.perMinute).toBeCloseTo(600, 6);
    expect(rate?.perHour).toBeCloseTo(36000, 6);
  });

  it("reports a negative rate while cooling", () => {
    const fit = linearFit.fit([
      { x: 0, y: 100 },
      { x: 60000, y: 40 },
    ]); // -60 deg over 60 s => -1 deg/min
    const rate = heatUpRateFromFit(fit, 1);
    expect(rate?.perMinute).toBeCloseTo(-60, 6); // slope=-0.001 deg/ms => -60/min
  });

  it("returns null for no fit / non-linear fit", () => {
    expect(heatUpRateFromFit(null)).toBeNull();
    expect(
      heatUpRateFromFit({
        methodId: "poly",
        label: "x",
        coeffs: [1, 0],
        r2: 1,
        predict: (x) => x,
        equation: "",
      }),
    ).toBeNull();
  });

  it("rejects a non-positive x-unit scale (DbC)", () => {
    const fit = linearFit.fit([
      { x: 0, y: 0 },
      { x: 1, y: 1 },
    ]);
    expect(() => heatUpRateFromFit(fit, 0)).toThrow(RangeError);
    expect(() => heatUpRateFromFit(fit, -1)).toThrow(RangeError);
  });
});

describe("formatHeatUpRate", () => {
  it("formats both units with an explicit sign", () => {
    expect(formatHeatUpRate({ perMinute: 12.34, perHour: 740.4 })).toBe(
      "+12.3 °C/min · +740 °C/hr",
    );
    expect(formatHeatUpRate({ perMinute: -5, perHour: -300 })).toBe(
      "−5.0 °C/min · −300 °C/hr",
    );
  });

  it("shows a placeholder when there is no rate", () => {
    expect(formatHeatUpRate(null)).toBe("—");
  });
});
