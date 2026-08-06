import { describe, expect, it } from "vitest";

import type { VariationDatasetTs } from "./variation";
import {
  buildScalarMarginal,
  buildScalarPlotVariables,
  buildScalarScatter,
  distributionMatrixToCsv,
  distributionMatrixToSvg,
} from "./variationPlotData";

const SPEED = "swing_sim.impact.delivery.clubhead_speed_mps";

const dataset = (): VariationDatasetTs => ({
  plan: {
    mode: "delivery",
    baseVariables: {},
    noise: [{
      variableKey: SPEED,
      distribution: "normal",
      scale: 1,
      lower: null,
      upper: null,
    }],
    nRuns: 3,
    seed: 1,
    flightModel: "waterloo_penner",
  },
  inputNames: [SPEED],
  inputs: [[44], [45], [46]],
  outputNames: ["club_path_deg", "carry_m"],
  outputs: [[-1, 100], [0, null], [1, 110]],
  success: [true, false, true],
});

describe("variation plot data", () => {
  it("exposes unit-bearing input, impact, and shot variables", () => {
    const variables = buildScalarPlotVariables(dataset());

    expect(variables).toEqual(expect.arrayContaining([
      expect.objectContaining({ key: `input:${SPEED}`, unit: "m/s", kind: "input" }),
      expect.objectContaining({ key: "output:club_path_deg", unit: "deg", kind: "impact" }),
      expect.objectContaining({ key: "output:carry_m", unit: "m", kind: "shot" }),
    ]));
  });

  it("pairs finite rows and accounts for unavailable failures", () => {
    const result = buildScalarScatter(
      dataset(),
      `input:${SPEED}`,
      "output:carry_m",
    );

    expect(result.points).toEqual([
      { trialIndex: 0, x: 44, y: 100, cohort: "evaluated" },
      { trialIndex: 2, x: 46, y: 110, cohort: "evaluated" },
    ]);
    expect(result.cohorts).toEqual({
      evaluated: { total: 2, plotted: 2, unavailable: 0 },
      failure: { total: 1, plotted: 0, unavailable: 1 },
    });
  });

  it("builds deterministic marginals without fabricating missing outputs", () => {
    const marginal = buildScalarMarginal(dataset(), "output:carry_m", 2);

    expect(marginal.binEdges).toEqual([100, 105, 110]);
    expect(marginal.counts).toEqual([1, 1]);
    expect(marginal.nAvailable).toBe(2);
    expect(marginal.nMissing).toBe(1);
  });

  it("exports every selected matrix row including unavailable values", () => {
    const csv = distributionMatrixToCsv(dataset(), [
      `input:${SPEED}`,
      "output:carry_m",
    ]);

    expect(csv.split("\n")).toEqual([
      `trial_index,outcome,input:${SPEED},output:carry_m`,
      "0,evaluated,44,100",
      "1,failure,45,",
      "2,evaluated,46,110",
    ]);
    expect(distributionMatrixToSvg(dataset(), [
      `input:${SPEED}`,
      "output:carry_m",
    ])).toContain("<svg");
  });
});
