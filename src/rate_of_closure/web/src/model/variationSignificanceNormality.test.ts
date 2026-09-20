/**
 * Unit tests for Mardia bivariate normality, 2D convex hull fallback,
 * Spearman permutation significance & bootstrap CIs, and truncation mean-shift analysis (#4253).
 */

import { describe, expect, it } from "vitest";

import {
  convexHull2d,
  mardiaBivariateNormality,
} from "./variationNormality";
import {
  detectTruncationMeanShifts,
} from "./variationTruncation";
import {
  spearmanAnalysis,
  dispersionEllipse,
} from "./variationAnalysis";
import {
  runVariation,
  type VariationDatasetTs,
  type VariationPlanTs,
} from "./variation";
import {
  CATEGORY_DELIVERY,
  CATEGORY_LAUNCH,
} from "./variationRegistry";

const SPEED = `${CATEGORY_DELIVERY}.clubhead_speed_mps`;
const BALL = `${CATEGORY_LAUNCH}.ball_speed_mph`;

describe("mardiaBivariateNormality", () => {
  it("handles empty and small point arrays gracefully", () => {
    const diag = mardiaBivariateNormality([]);
    expect(diag.isNormal).toBe(true);
    expect(diag.skewnessStat).toBe(0);

    const diagSmall = mardiaBivariateNormality([[0, 0], [1, 1]]);
    expect(diagSmall.isNormal).toBe(true);
  });

  it("classifies standard symmetric Gaussian points as bivariate normal", () => {
    // Generate 100 points around (0, 0)
    const points: Array<[number, number]> = [];
    let state = 12345;
    const rand = (): number => {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 4294967296;
    };
    for (let i = 0; i < 150; i += 1) {
      // Box-Muller
      const u1 = Math.max(1e-9, rand());
      const u2 = rand();
      const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
      points.push([z0, z1]);
    }
    const diag = mardiaBivariateNormality(points);
    expect(diag.isNormal).toBe(true);
    expect(diag.skewnessPValue).toBeGreaterThan(0.01);
    expect(diag.kurtosisPValue).toBeGreaterThan(0.01);
  });

  it("detects highly skewed or outlier-heavy distributions as non-normal", () => {
    const points: Array<[number, number]> = [];
    // Exponentially skewed points with strong outliers
    for (let i = 0; i < 100; i += 1) {
      points.push([Math.exp(i / 20), Math.exp(i / 25)]);
    }
    const diag = mardiaBivariateNormality(points);
    expect(diag.isNormal).toBe(false);
  });
});

describe("convexHull2d", () => {
  it("returns input for fewer than 3 points", () => {
    expect(convexHull2d([])).toEqual([]);
    expect(convexHull2d([[1, 2]])).toEqual([[1, 2]]);
    expect(convexHull2d([[1, 2], [3, 4]])).toEqual([[1, 2], [3, 4]]);
  });

  it("finds the convex hull of a square with interior points", () => {
    const points: Array<[number, number]> = [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10],
      [5, 5],
      [2, 3],
      [7, 8],
    ];
    const hull = convexHull2d(points);
    expect(hull.length).toBe(4);
    // All 4 hull vertices must be corners
    const corners = new Set(["0,0", "10,0", "10,10", "0,10"]);
    for (const [x, y] of hull) {
      expect(corners.has(`${x},${y}`)).toBe(true);
    }
  });
});

describe("dispersionEllipse with normality diagnostic", () => {
  it("attaches diagnostic to dispersion ellipse", () => {
    const plan: VariationPlanTs = {
      mode: "launch",
      baseVariables: {},
      noise: [
        { variableKey: BALL, distribution: "normal", scale: 2.0, lower: null, upper: null },
        { variableKey: `${CATEGORY_LAUNCH}.launch_azimuth_deg`, distribution: "normal", scale: 1.5, lower: null, upper: null },
      ],
      nRuns: 32,
      seed: 42,
      flightModel: "waterloo_penner",
    };
    const dataset = runVariation(plan);
    const ellipse = dispersionEllipse(dataset);
    expect(ellipse).not.toBeNull();
    if (ellipse) {
      expect(ellipse.diagnostic).toBeDefined();
      expect(typeof ellipse.diagnostic?.isNormal).toBe("boolean");
      if (!ellipse.diagnostic?.isNormal) {
        expect(ellipse.convexHull).toBeDefined();
        expect(ellipse.convexHull?.length).toBeGreaterThan(2);
      }
    }
  });

  it("populates convex hull fallback when landing points are non-normal", () => {
    // Construct dataset with bimodal/skewed landing coordinates
    const fakeDataset: VariationDatasetTs = {
      plan: {
        mode: "launch",
        baseVariables: {},
        noise: [],
        nRuns: 30,
        seed: 1,
        flightModel: "waterloo_penner",
      },
      inputNames: ["in"],
      outputNames: ["carry_m", "lateral_m"],
      inputs: Array.from({ length: 30 }, (_, i) => [i]),
      outputs: Array.from({ length: 30 }, (_, i) => [
        i < 15 ? 100 + i * 0.1 : 250 + i * 2, // extreme bimodal carry
        i < 15 ? -20 - i * 0.1 : 30 + i * 2,
      ]),
      success: Array(30).fill(true),
    };
    const ellipse = dispersionEllipse(fakeDataset);
    expect(ellipse).not.toBeNull();
    expect(ellipse?.diagnostic?.isNormal).toBe(false);
    expect(ellipse?.convexHull).toBeDefined();
    expect(ellipse?.convexHull?.length).toBeGreaterThan(2);
  });
});

describe("spearmanAnalysis", () => {
  it("computes correlation, permutation p-values, and bootstrap CIs", () => {
    const fakeDataset: VariationDatasetTs = {
      plan: {
        mode: "launch",
        baseVariables: {},
        noise: [],
        nRuns: 30,
        seed: 1,
        flightModel: "waterloo_penner",
      },
      inputNames: ["inA", "inB"],
      outputNames: ["outA", "outB"],
      inputs: [],
      outputs: [],
      success: Array(30).fill(true),
    };
    for (let i = 0; i < 30; i += 1) {
      // inA strictly monotonic with outA; inB random noise
      fakeDataset.inputs.push([i, Math.sin(i * 99)]);
      fakeDataset.outputs.push([i * 2 + 1, Math.cos(i * 33)]);
    }

    const res = spearmanAnalysis(fakeDataset, 100, 0.05, 42);
    expect(res.matrix.length).toBe(2);
    expect(res.matrix[0].length).toBe(2);

    // inA vs outA has perfect monotonic rank correlation
    expect(res.matrix[0][0]).toBeCloseTo(1.0, 4);
    expect(res.pValues[0][0]).toBeLessThan(0.05);
    expect(res.significant[0][0]).toBe(true);
    expect(res.ciLower[0][0]).toBeGreaterThan(0.9);
    expect(res.ciUpper[0][0]).toBeCloseTo(1.0, 2);
  });
});

describe("detectTruncationMeanShifts", () => {
  it("detects shift when samples are clamped by truncation boundaries", () => {
    const plan: VariationPlanTs = {
      mode: "delivery",
      baseVariables: { [SPEED]: 45.0 },
      noise: [
        {
          variableKey: SPEED,
          distribution: "normal",
          scale: 5.0,
          lower: 47.0, // Force clamping/shifting above nominal base
          upper: 55.0,
        },
      ],
      nRuns: 40,
      seed: 7,
      flightModel: "waterloo_penner",
    };
    const dataset = runVariation(plan);
    const notes = detectTruncationMeanShifts(dataset);
    expect(notes.length).toBeGreaterThan(0);
    expect(notes[0].variableKey).toBe(SPEED);
    expect(notes[0].note).toContain("shifted by");
  });

  it("reports no shift for symmetric un-truncated noise", () => {
    const plan: VariationPlanTs = {
      mode: "delivery",
      baseVariables: { [SPEED]: 45.0 },
      noise: [
        {
          variableKey: SPEED,
          distribution: "normal",
          scale: 0.5,
          lower: null,
          upper: null,
        },
      ],
      nRuns: 50,
      seed: 7,
      flightModel: "waterloo_penner",
    };
    const dataset = runVariation(plan);
    const notes = detectTruncationMeanShifts(dataset);
    expect(notes.length).toBe(0);
  });
});
