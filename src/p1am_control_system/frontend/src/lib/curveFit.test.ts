import { describe, it, expect } from "vitest";
import {
  linearFit,
  fitSeries,
  getFitMethod,
  fitOptions,
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
