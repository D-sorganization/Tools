import { describe, expect, it } from "vitest";

import { runSwingVariation } from "./variationSwingEnsemble";
import {
  makeVariationPlotDefinition,
  parseVariationPlotDefinition,
  swingResultFingerprint,
  type VariationPlotDefinitionInputTs,
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

const completeGeometricInput = (): VariationPlotDefinitionInputTs => ({
  plotType: "swing_arc_overlay",
  coordinateFrame: "app_frame:x_target,y_up,z_right",
  xVariableKey: null,
  yVariableKey: null,
  pointId: "swing.clubhead.reference",
  positionUnit: "m",
  alignmentBasis: "common_simulation_time_s",
  dispersionMetric: "rms-radius",
  dispersionUnit: "m",
  quietThreshold: 0.005,
  confidenceLevel: null,
  minQuietDurationS: 0,
  minQuietSamples: 1,
  selectedTrialIndex: 0,
  cameraYawDeg: -37,
  cameraPitchDeg: 22,
  cameraZoom: 1.2,
  outcomeFilter: "evaluated_hit",
  phaseEndFraction: 0.75,
  perturbationSourceKey: "swing_sim.swing.yaw_deg",
  perturbationBand: "upper",
  variableKeys: null,
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
    expect(parseVariationPlotDefinition(variationPlotDefinitionToJson(definition))).toEqual(definition);
  });

  it("strictly migrates v1 geometric defaults", () => {
    const migrated = parseVariationPlotDefinition(JSON.stringify({
      schemaVersion: 1,
      resultId: "ensemble-v1",
      plotType: "geometric_variability",
      coordinateFrame: "app_frame:x_target,y_up,z_right",
      xVariableKey: null,
      yVariableKey: null,
      pointId: "swing.clubhead.reference",
      positionUnit: "m",
      alignmentBasis: "common_simulation_time_s",
      quietThresholdM: null,
      selectedTrialIndex: null,
      cameraYawDeg: null,
      cameraPitchDeg: null,
      cameraZoom: null,
      outcomeFilter: null,
      phaseEndFraction: null,
      perturbationSourceKey: null,
      perturbationBand: null,
      variableKeys: null,
    }));

    expect(migrated).toMatchObject({
      schemaVersion: 2,
      dispersionMetric: "rms-radius",
      dispersionUnit: "m",
      quietThreshold: 0.005,
      confidenceLevel: null,
      minQuietDurationS: 0,
      minQuietSamples: 1,
    });
  });

  it.each([true, 2.5, "2", 3])("rejects coercive or unknown schema %s", (schemaVersion) => {
    expect(() => parseVariationPlotDefinition(JSON.stringify({ schemaVersion }))).toThrow();
  });

  it("rejects unknown fields", () => {
    const definition = makeVariationPlotDefinition(runSwingVariation(plan(2)), {
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
    expect(() => parseVariationPlotDefinition(JSON.stringify({
      ...definition, unexpected: true,
    }))).toThrow(/fields/);
    const missing = { ...definition } as Record<string, unknown>;
    delete missing.cameraZoom;
    expect(() => parseVariationPlotDefinition(JSON.stringify(missing))).toThrow(/fields/);
    expect(() => parseVariationPlotDefinition(JSON.stringify({
      ...definition, minQuietSamples: "1",
    }))).toThrow(/integer/);
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

  it.each([
    ["plotType", "unknown"],
    ["coordinateFrame", " "],
    ["pointId", "swing.clubhead.reference "],
    ["positionUnit", "mm"],
    ["alignmentBasis", "sample-index"],
    ["selectedTrialIndex", true],
    ["selectedTrialIndex", 1.5],
    ["cameraYawDeg", true],
    ["cameraYawDeg", Number.NaN],
    ["cameraYawDeg", Number.POSITIVE_INFINITY],
    ["cameraPitchDeg", true],
    ["cameraPitchDeg", -90.0001],
    ["cameraPitchDeg", 90.0001],
    ["cameraZoom", true],
    ["cameraZoom", Number.NaN],
    ["cameraZoom", Number.POSITIVE_INFINITY],
    ["phaseEndFraction", true],
    ["phaseEndFraction", Number.NaN],
    ["phaseEndFraction", Number.POSITIVE_INFINITY],
    ["phaseEndFraction", 1.0001],
    ["outcomeFilter", "hit"],
    ["perturbationSourceKey", " swing_sim.swing.yaw_deg"],
    ["perturbationBand", "outer"],
  ])("rejects malformed full-object field %s", (field, value) => {
    const input = { ...completeGeometricInput(), [field]: value } as VariationPlotDefinitionInputTs;
    expect(() => makeVariationPlotDefinition(runSwingVariation(plan(2)), input)).toThrow();
  });

  it("requires a source for a perturbation band", () => {
    const input = { ...completeGeometricInput(), perturbationSourceKey: null };
    expect(() => makeVariationPlotDefinition(runSwingVariation(plan(2)), input)).toThrow(/source/i);
  });

  it.each([
    ["resultId", " "],
    ["selectedTrialIndex", true],
    ["cameraYawDeg", Number.NaN],
    ["cameraPitchDeg", Number.NaN],
    ["cameraZoom", Number.NaN],
    ["outcomeFilter", "hit"],
  ] as const)(
    "validates tampered %s before JSON.stringify",
    (field, value) => {
      const definition = makeVariationPlotDefinition(
        runSwingVariation(plan(2)), completeGeometricInput(),
      );
      const tampered = { ...definition, [field]: value };
      expect(() => variationPlotDefinitionToJson(tampered)).toThrow();
    },
  );

  it("rejects undeclared constructor fields that could override result identity", () => {
    const input = {
      ...completeGeometricInput(), resultId: "attacker-selected-result",
    } as VariationPlotDefinitionInputTs;
    expect(() => makeVariationPlotDefinition(runSwingVariation(plan(2)), input))
      .toThrow(/fields/);
  });
});
