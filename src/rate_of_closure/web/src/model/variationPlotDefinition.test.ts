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
  showConfidenceEllipsoids: false,
});

const scatterInput = (): VariationPlotDefinitionInputTs => ({
  plotType: "scalar_scatter", coordinateFrame: null,
  xVariableKey: "input:swing.speed", yVariableKey: "output:carry_m",
  pointId: null, positionUnit: null, alignmentBasis: null,
  dispersionMetric: null, dispersionUnit: null, quietThreshold: null,
  confidenceLevel: null, minQuietDurationS: null, minQuietSamples: null,
  selectedTrialIndex: 1, cameraYawDeg: null, cameraPitchDeg: null,
  cameraZoom: null, outcomeFilter: null, phaseEndFraction: null,
  perturbationSourceKey: null, perturbationBand: null, variableKeys: null,
  showConfidenceEllipsoids: null,
});

const matrixInput = (): VariationPlotDefinitionInputTs => ({
  plotType: "distribution_matrix", coordinateFrame: null,
  xVariableKey: null, yVariableKey: null, pointId: null, positionUnit: null,
  alignmentBasis: null, dispersionMetric: null, dispersionUnit: null,
  quietThreshold: null, confidenceLevel: null, minQuietDurationS: null,
  minQuietSamples: null, selectedTrialIndex: null, cameraYawDeg: null,
  cameraPitchDeg: null, cameraZoom: null, outcomeFilter: null,
  phaseEndFraction: null, perturbationSourceKey: null, perturbationBand: null,
  variableKeys: ["input:swing.speed", "output:carry_m"],
  showConfidenceEllipsoids: null,
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
      showConfidenceEllipsoids: true,
    });

    expect(definition.schemaVersion).toBe(3);
    expect(definition.resultId).toBe(swingResultFingerprint(ensemble));
    expect(JSON.parse(variationPlotDefinitionToJson(definition))).toEqual(definition);
    expect(parseVariationPlotDefinition(variationPlotDefinitionToJson(definition))).toEqual(definition);
  });

  it("strictly migrates v1 geometric defaults", () => {
    const legacy = {
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
    };
    const migrated = parseVariationPlotDefinition(JSON.stringify(legacy));

    expect(migrated).toMatchObject({
      schemaVersion: 3,
      dispersionMetric: "rms-radius",
      dispersionUnit: "m",
      quietThreshold: 0.005,
      confidenceLevel: null,
      minQuietDurationS: 0,
      minQuietSamples: 1,
    });
    expect(() => parseVariationPlotDefinition(JSON.stringify({
      ...legacy, variableKeys: ["input:a", "output:b"],
    }))).toThrow(/applicable/);
  });

  it("strictly migrates v2 with ellipsoid surfaces disabled", () => {
    const current = makeVariationPlotDefinition(runSwingVariation(plan(2)), matrixInput());
    const legacy = { ...current } as Record<string, unknown>;
    delete legacy.showConfidenceEllipsoids;
    const migrated = parseVariationPlotDefinition(JSON.stringify({
      ...legacy, schemaVersion: 2,
    }));
    expect(migrated.schemaVersion).toBe(3);
    expect(migrated.showConfidenceEllipsoids).toBeNull();
  });

  it.each([
    ["scalar_scatter", "input:speed", "output:carry_m", 2, null],
    ["distribution_matrix", null, null, null, ["input:speed", "output:carry_m"]],
  ] as const)("normalizes the authentic v1 %s frame", (
    plotType, xVariableKey, yVariableKey, selectedTrialIndex, variableKeys,
  ) => {
    const legacy = {
      schemaVersion: 1, resultId: "historical-v1", plotType,
      coordinateFrame: "app_frame:x_target,y_up,z_right",
      xVariableKey, yVariableKey, pointId: null, positionUnit: null,
      alignmentBasis: null, quietThresholdM: null, selectedTrialIndex,
      cameraYawDeg: null, cameraPitchDeg: null, cameraZoom: null,
      outcomeFilter: null, phaseEndFraction: null, perturbationSourceKey: null,
      perturbationBand: null, variableKeys,
    };

    expect(parseVariationPlotDefinition(JSON.stringify(legacy)).coordinateFrame).toBeNull();
    expect(() => parseVariationPlotDefinition(JSON.stringify({
      ...legacy, coordinateFrame: "other_frame",
    }))).toThrow(/legacy coordinateFrame/);
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
      showConfidenceEllipsoids: null,
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
      showConfidenceEllipsoids: null,
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
      showConfidenceEllipsoids: null,
    });
    expect(definition.variableKeys).toHaveLength(2);
  });

  it.each([
    ["plotType", "unknown"],
    ["coordinateFrame", " "],
    ["coordinateFrame", "app_frame:x_target,z_up,y_right"],
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

  it.each([
    [scatterInput, "coordinateFrame", "app_frame:x_target,y_up,z_right"],
    [scatterInput, "pointId", "swing.clubhead.reference"],
    [scatterInput, "cameraZoom", 1],
    [scatterInput, "variableKeys", ["input:a", "output:b"]],
    [matrixInput, "coordinateFrame", "app_frame:x_target,y_up,z_right"],
    [matrixInput, "xVariableKey", "input:swing.speed"],
    [matrixInput, "selectedTrialIndex", 0],
    [completeGeometricInput, "xVariableKey", "input:swing.speed"],
    [completeGeometricInput, "variableKeys", ["input:a", "output:b"]],
  ] as const)("rejects inapplicable %s state in %s", (factory, field, value) => {
    const input = { ...factory(), [field]: value } as VariationPlotDefinitionInputTs;
    expect(() => makeVariationPlotDefinition(runSwingVariation(plan(2)), input))
      .toThrow(/applicable/);
  });

  it.each([
    [completeGeometricInput, "pointId", "swing\u0000clubhead"],
    [completeGeometricInput, "perturbationSourceKey", "swing_sim\u0080yaw"],
    [scatterInput, "xVariableKey", "input\u007fspeed"],
    [matrixInput, "variableKeys", ["input:a", "output:\u0081b"]],
  ] as const)("rejects control characters in stable %s IDs", (factory, field, value) => {
    const input = { ...factory(), [field]: value } as VariationPlotDefinitionInputTs;
    expect(() => makeVariationPlotDefinition(runSwingVariation(plan(2)), input))
      .toThrow(/control/);
  });

  it("rejects control and inapplicable state through parser and writer paths", () => {
    const definition = makeVariationPlotDefinition(
      runSwingVariation(plan(2)), completeGeometricInput(),
    );
    const controlled = { ...definition, resultId: "result\u001fidentity" };
    expect(() => variationPlotDefinitionToJson(controlled)).toThrow(/control/);
    expect(() => parseVariationPlotDefinition(JSON.stringify(controlled))).toThrow(/control/);

    const inapplicable = { ...definition, variableKeys: ["input:a", "output:b"] };
    expect(() => variationPlotDefinitionToJson(inapplicable)).toThrow(/applicable/);
    expect(() => parseVariationPlotDefinition(JSON.stringify(inapplicable))).toThrow(/applicable/);
  });

  it("rejects boxed numeric objects rather than relying on JSON coercion", () => {
    const input = {
      ...completeGeometricInput(), cameraZoom: new Number(1.2),
    } as VariationPlotDefinitionInputTs;
    expect(() => makeVariationPlotDefinition(runSwingVariation(plan(2)), input)).toThrow();
  });
});
