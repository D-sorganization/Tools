/**
 * Web variation engine tests (epic #4120, V3): seed reproducibility,
 * truncation, subset-stable streams, the planted-dominant-variable
 * sensitivity check, plan JSON schema interop with Python, the bounded
 * run cap, export formats, and the loose statistical parity pin against
 * the Python-generated fixture (exact RNG parity deliberately not
 * required — see variation.ts).
 */

import { describe, expect, it } from "vitest";

import spearmanPairwiseFixture from "../vendored/variation_spearman_pairwise_finite.json";
import parityFixture from "./__fixtures__/variation_parity.json";
import {
  CATEGORY_DELIVERY,
  CATEGORY_LAUNCH,
  MAX_RUNS,
  keysForMode,
  mulberry32,
  planFromJson,
  planToJson,
  runVariation,
  sampleInputs,
  type VariationDatasetTs,
  type VariationPlanTs,
} from "./variation";
import {
  datasetToCsv,
  datasetToJson,
  dispersionEllipse,
  oneAtATimeSensitivity,
  normalizeSensitivityMatrix,
  spearmanMatrix,
  summaryStats,
} from "./variationAnalysis";

const FACE = `${CATEGORY_DELIVERY}.face_angle_deg`;
const SPEED = `${CATEGORY_DELIVERY}.clubhead_speed_mps`;
const BALL = `${CATEGORY_LAUNCH}.ball_speed_mph`;

const launchPlan = (overrides: Partial<VariationPlanTs> = {}): VariationPlanTs => ({
  mode: "launch",
  baseVariables: {},
  noise: [
    { variableKey: BALL, distribution: "normal", scale: 1.0, lower: null, upper: null },
  ],
  nRuns: 16,
  seed: 3,
  flightModel: "waterloo_penner",
  ...overrides,
});

describe("seeded sampling", () => {
  it("same plan+seed reproduces the exact dataset; seeds differ", () => {
    const a = runVariation(launchPlan());
    const b = runVariation(launchPlan());
    const c = runVariation(launchPlan({ seed: 4 }));
    expect(a.inputs).toEqual(b.inputs);
    expect(a.outputs).toEqual(b.outputs);
    expect(a.inputs).not.toEqual(c.inputs);
  });

  it("respects truncation for every distribution", () => {
    for (const distribution of ["normal", "uniform", "triangular"] as const) {
      const samples = sampleInputs(
        launchPlan({
          nRuns: 400,
          noise: [
            { variableKey: BALL, distribution, scale: 10.0, lower: 148.0, upper: 153.0 },
          ],
        }),
      );
      for (const [value] of samples) {
        expect(value).toBeGreaterThanOrEqual(148.0);
        expect(value).toBeLessThanOrEqual(153.0);
      }
    }
  });

  it("streams are subset-stable per variable (paired OAT draws)", () => {
    const both = launchPlan({
      nRuns: 32,
      noise: [
        { variableKey: BALL, distribution: "normal", scale: 1, lower: null, upper: null },
        {
          variableKey: `${CATEGORY_LAUNCH}.spin_rpm`,
          distribution: "normal",
          scale: 100,
          lower: null,
          upper: null,
        },
      ],
    });
    const only = launchPlan({ nRuns: 32, noise: [both.noise[0]] });
    const a = sampleInputs(both).map((row) => row[0]);
    const b = sampleInputs(only).map((row) => row[0]);
    expect(a).toEqual(b);
  });

  it("mulberry32 is deterministic and uniform-ish", () => {
    const rng = mulberry32(123);
    const rng2 = mulberry32(123);
    const a = Array.from({ length: 5 }, rng);
    expect(Array.from({ length: 5 }, rng2)).toEqual(a);
    for (const v of a) expect(v).toBeGreaterThanOrEqual(0);
  });
});

describe("plan schema interop", () => {
  it("round-trips through the Python snake_case schema", () => {
    const plan = launchPlan({
      baseVariables: { [BALL]: 155.0 },
      noise: [
        { variableKey: BALL, distribution: "triangular", scale: 2, lower: 150, upper: 160 },
      ],
    });
    expect(planFromJson(planToJson(plan))).toMatchObject(plan);
  });

  it("parses the Python-generated fixture plan verbatim", () => {
    const plan = planFromJson(JSON.stringify(parityFixture.plan));
    expect(plan.mode).toBe("launch");
    expect(plan.nRuns).toBe(300);
    expect(plan.noise).toHaveLength(4);
  });

  it("rejects unknown modes, oversize runs, and illegal keys", () => {
    expect(() => runVariation(launchPlan({ nRuns: MAX_RUNS + 1 }))).toThrow(/nRuns/);
    expect(() =>
      runVariation({ ...launchPlan(), mode: "unsupported" as never }),
    ).toThrow(/mode/);
    expect(() =>
      runVariation(
        launchPlan({
          noise: [
            { variableKey: FACE, distribution: "normal", scale: 1, lower: null, upper: null },
          ],
        }),
      ),
    ).toThrow(/not legal/);
  });
});

describe("sensitivity", () => {
  const plantedPlan = (): VariationPlanTs => ({
    mode: "delivery",
    baseVariables: {},
    noise: [
      { variableKey: FACE, distribution: "normal", scale: 3.0, lower: null, upper: null },
      { variableKey: SPEED, distribution: "normal", scale: 0.045, lower: null, upper: null },
    ],
    nRuns: 40,
    seed: 9,
    flightModel: "waterloo_penner",
  });

  it("identifies the planted dominant variable (face -> lateral)", () => {
    const result = oneAtATimeSensitivity(plantedPlan());
    const lat = result.outputNames.indexOf("lateral_m");
    const face = result.inputKeys.indexOf(FACE);
    const speed = result.inputKeys.indexOf(SPEED);
    expect(result.matrix[face][lat]).toBeGreaterThan(10 * result.matrix[speed][lat]);
    expect(result.normalized[face][lat]).toBeCloseTo(1.0, 6);
  });

  it("keeps all-unavailable normalization distinct from finite zero spread", () => {
    const normalized = normalizeSensitivityMatrix(
      [[Number.NaN, 0, Number.NaN], [Number.NaN, 0, 2]],
      3,
    );
    expect(normalized.every((row) => Number.isNaN(row[0]))).toBe(true);
    expect(normalized.every((row) => row[1] === 0)).toBe(true);
    expect(Number.isNaN(normalized[0][2])).toBe(true);
    expect(normalized[1][2]).toBe(1);
  });

  it("spearman corroborates the dominance on the full dataset", () => {
    const dataset = runVariation(plantedPlan());
    const rho = spearmanMatrix(dataset);
    const lat = dataset.outputNames.indexOf("lateral_m");
    const face = dataset.inputNames.indexOf(FACE);
    const speed = dataset.inputNames.indexOf(SPEED);
    expect(Math.abs(rho[face][lat])).toBeGreaterThan(0.9);
    expect(Math.abs(rho[face][lat])).toBeGreaterThan(Math.abs(rho[speed][lat]));
  });

  it("uses pairwise finite rows with Python fixture parity", () => {
    const fixture = spearmanPairwiseFixture;
    const dataset: VariationDatasetTs = {
      plan: launchPlan({ nRuns: fixture.success.length }),
      inputNames: fixture.input_names,
      inputs: fixture.inputs.map((row) =>
        row.map((value) => value ?? Number.NaN),
      ),
      outputNames: fixture.output_names,
      outputs: fixture.outputs,
      success: fixture.success,
    };

    const actual = spearmanMatrix(dataset);

    fixture.expected.forEach((row, inputIndex) => {
      row.forEach((expected, outputIndex) => {
        if (expected === null) {
          expect(Number.isNaN(actual[inputIndex][outputIndex])).toBe(true);
        } else {
          expect(actual[inputIndex][outputIndex]).toBeCloseTo(expected, 12);
        }
      });
    });
  });
});

describe("statistical parity with the Python engine", () => {
  it("matches the fixture dispersion within the loose band", () => {
    const plan = planFromJson(JSON.stringify(parityFixture.plan));
    const dataset = runVariation(plan);
    expect(dataset.success.every(Boolean)).toBe(true);
    const stats = new Map(summaryStats(dataset).map((s) => [s.name, s]));
    const band = parityFixture.web_band;
    for (const [name, expected] of Object.entries(parityFixture.python_stats)) {
      const actual = stats.get(name)!;
      const meanTol = (band.mean_abs_tolerance as Record<string, number>)[name];
      expect(Math.abs(actual.mean - expected.mean)).toBeLessThan(meanTol);
      expect(Math.abs(actual.std - expected.std) / expected.std).toBeLessThan(
        band.std_rel_tolerance,
      );
    }
  });
});

describe("analysis + export", () => {
  it("summary stats and ellipse are finite on a real run", () => {
    const dataset = runVariation(
      launchPlan({
        nRuns: 48,
        noise: [
          { variableKey: BALL, distribution: "normal", scale: 1, lower: null, upper: null },
          {
            variableKey: `${CATEGORY_LAUNCH}.launch_azimuth_deg`,
            distribution: "normal",
            scale: 0.8,
            lower: null,
            upper: null,
          },
        ],
      }),
    );
    const stats = summaryStats(dataset);
    expect(stats.find((s) => s.name === "carry_m")!.mean).toBeGreaterThan(100);
    const ellipse = dispersionEllipse(dataset)!;
    expect(ellipse.semiMajorM).toBeGreaterThan(0);
    expect(ellipse.semiMinorM).toBeGreaterThan(0);
    expect(ellipse.n).toBe(48);
  });

  it("fits landing dispersion only from rows with a paired finite position", () => {
    const dataset = {
      plan: launchPlan({ nRuns: 4 }),
      inputNames: [BALL],
      inputs: [[150], [151], [152], [153]],
      outputNames: ["carry_m", "lateral_m"],
      outputs: [[100, 10], [1000, null], [null, 100], [300, 30]],
      success: [true, true, true, true],
    };

    const ellipse = dispersionEllipse(dataset);

    expect(ellipse).not.toBeNull();
    expect(ellipse?.n).toBe(2);
    expect(ellipse?.centerCarryM).toBe(200);
    expect(ellipse?.centerLateralM).toBe(20);
  });

  it("CSV/JSON exports carry the documented schemas", () => {
    const dataset = runVariation(launchPlan({ nRuns: 4 }));
    const csv = datasetToCsv(dataset);
    const [header, ...rows] = csv.trim().split("\n");
    expect(header).toBe(`run,success,${BALL},${dataset.outputNames.join(",")}`);
    expect(rows).toHaveLength(4);
    const json = JSON.parse(datasetToJson(dataset)) as Record<string, unknown>;
    expect(json.schema_version).toBe(2);
    expect(json).not.toHaveProperty("plan");
    const planDocument = json.plan_document as Record<string, unknown>;
    expect(planDocument.schema_version).toBe(3);
    expect((planDocument.plan as Record<string, unknown>).n_runs).toBe(4);
  });

  it("failed runs export as empty CSV cells and null JSON entries", () => {
    const dataset = runVariation(
      launchPlan({
        nRuns: 32,
        baseVariables: { [BALL]: 0.5 },
        noise: [
          { variableKey: BALL, distribution: "normal", scale: 3, lower: null, upper: null },
        ],
      }),
    );
    expect(dataset.success.some((ok) => !ok)).toBe(true);
    expect(dataset.success.some((ok) => ok)).toBe(true);
    const failedIndex = dataset.success.indexOf(false);
    expect(dataset.outputs[failedIndex].every((v) => v === null)).toBe(true);
  });

  it("registry keys per mode mirror the Python categories", () => {
    // 7, not 5: the two regional-ground keys are part of the launch
    // category. Python reaches the same count -- its registry holds 5
    // launch keys until register_ground_variation_variables() adds exactly
    // ground_normal_restitution and ground_rolling_resistance, which
    // regional_ground_variation_request does from inside its parse path.
    expect(keysForMode("launch")).toHaveLength(7);
    expect(keysForMode("delivery")).toHaveLength(7); // club category is desktop-only
  });
});
