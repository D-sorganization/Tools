import { describe, expect, it } from "vitest";

import { runSwingVariation } from "./variationSwingEnsemble";
import {
  makeVariationPlotDefinition,
  swingResultFingerprint,
  variationPlotDefinitionToJson,
} from "./variationPlotDefinition";
import { CATEGORY_SWING, type VariationPlanTs } from "./variation";

const YAW = `${CATEGORY_SWING}.yaw_deg`;

const plan = (nRuns: number, seed = 0): VariationPlanTs => ({
  mode: "swing",
  baseVariables: { [YAW]: 0 },
  noise: [{
    variableKey: YAW, distribution: "uniform", scale: 0.2,
    lower: null, upper: null,
  }],
  nRuns,
  seed,
  flightModel: "waterloo_penner",
});

describe("variation plot definitions", () => {
  it("pins a stable result fingerprint and complete geometric state", () => {
    const ensemble = runSwingVariation(plan(2, 19));
    const definition = makeVariationPlotDefinition(ensemble, {
      plotType: "swing_arc_overlay",
      coordinateFrame: ensemble.coordinateFrame,
      xVariableKey: null,
      yVariableKey: null,
      pointId: "swing.clubhead.reference",
      positionUnit: "m",
      alignmentBasis: "common_simulation_time_s",
      dispersionMetric: "confidence-ellipsoid-volume",
      dispersionUnit: "m^3",
      quietThreshold: 1.25e-7,
      confidenceLevel: 0.95,
      minQuietDurationS: 0.02,
      minQuietSamples: 3,
      selectedTrialIndex: 1,
      cameraYawDeg: -37,
      cameraPitchDeg: 22,
      cameraZoom: 1.2,
      outcomeFilter: "evaluated_hit",
      phaseEndFraction: 0.75,
      perturbationSourceKey: "swing_motion.yaw_deg",
      perturbationBand: "upper",
      variableKeys: null,
    });

    expect(definition.schemaVersion).toBe(2);
    expect(definition.resultId).toBe(swingResultFingerprint(ensemble));
    expect(JSON.parse(variationPlotDefinitionToJson(definition))).toEqual(definition);
  });

  it("rejects an invalid quiet-zone threshold", () => {
    const ensemble = runSwingVariation(plan(2));
    expect(() => makeVariationPlotDefinition(ensemble, {
      plotType: "geometric_variability", coordinateFrame: ensemble.coordinateFrame,
      xVariableKey: null, yVariableKey: null, pointId: "swing.wrist", positionUnit: "m",
      alignmentBasis: "common_simulation_time_s",
      dispersionMetric: "rms-radius", dispersionUnit: "m", quietThreshold: 0,
      confidenceLevel: null, minQuietDurationS: 0, minQuietSamples: 1,
      selectedTrialIndex: null, cameraYawDeg: null, cameraPitchDeg: null, cameraZoom: null,
      outcomeFilter: null, phaseEndFraction: null,
      perturbationSourceKey: null, perturbationBand: null,
      variableKeys: null,
    })).toThrow(/greater than zero/);
  });

  it("pins distribution-matrix variable selection", () => {
    const ensemble = runSwingVariation(plan(2, 31));
    const definition = makeVariationPlotDefinition(ensemble, {
      plotType: "distribution_matrix", coordinateFrame: null,
      xVariableKey: null, yVariableKey: null, pointId: null, positionUnit: null,
      alignmentBasis: null, dispersionMetric: null, dispersionUnit: null,
      quietThreshold: null, confidenceLevel: null,
      minQuietDurationS: null, minQuietSamples: null, selectedTrialIndex: null,
      cameraYawDeg: null, cameraPitchDeg: null, cameraZoom: null,
      outcomeFilter: null, phaseEndFraction: null,
      perturbationSourceKey: null, perturbationBand: null,
      variableKeys: ["input:swing_sim.swing.yaw_deg", "output:carry_m"],
    });
    expect(definition.variableKeys).toHaveLength(2);
  });
});
