/** Versioned, toolkit-independent variation plot definitions. */

import type { SwingVariationResultTs } from "./variationSwingEnsemble";
import type { VariationDatasetTs } from "./variation";
import type { DispersionMetricTs } from "./variationGeometry";

export const VARIATION_PLOT_DEFINITION_SCHEMA_VERSION = 2;

export type VariationPlotTypeTs =
  | "scalar_scatter"
  | "swing_arc_overlay"
  | "geometric_variability"
  | "distribution_matrix";

export interface VariationPlotDefinitionTs {
  schemaVersion: 2;
  resultId: string;
  plotType: VariationPlotTypeTs;
  coordinateFrame: string | null;
  xVariableKey: string | null;
  yVariableKey: string | null;
  pointId: string | null;
  positionUnit: string | null;
  alignmentBasis: string | null;
  dispersionMetric: DispersionMetricTs | null;
  dispersionUnit: "m" | "m^3" | null;
  quietThreshold: number | null;
  confidenceLevel: number | null;
  minQuietDurationS: number | null;
  minQuietSamples: number | null;
  selectedTrialIndex: number | null;
  cameraYawDeg: number | null;
  cameraPitchDeg: number | null;
  cameraZoom: number | null;
  outcomeFilter: string | null;
  phaseEndFraction: number | null;
  perturbationSourceKey: string | null;
  perturbationBand: string | null;
  variableKeys: string[] | null;
}

export type VariationPlotDefinitionInputTs = Omit<
  VariationPlotDefinitionTs,
  "schemaVersion" | "resultId"
>;

export function makeVariationPlotDefinition(
  result: SwingVariationResultTs | VariationDatasetTs,
  input: VariationPlotDefinitionInputTs,
): VariationPlotDefinitionTs {
  validateDispersionState(input);
  if (input.selectedTrialIndex !== null && input.selectedTrialIndex < 0) {
    throw new Error("selectedTrialIndex must be non-negative");
  }
  if (input.phaseEndFraction !== null
    && (input.phaseEndFraction <= 0 || input.phaseEndFraction > 1)) {
    throw new Error("phaseEndFraction must be in (0, 1]");
  }
  if (input.plotType === "distribution_matrix") {
    if (input.variableKeys === null
      || input.variableKeys.length < 2
      || input.variableKeys.length > 8) {
      throw new Error("distribution matrix requires 2 to 8 variableKeys");
    }
    if (new Set(input.variableKeys).size !== input.variableKeys.length) {
      throw new Error("distribution matrix variableKeys must be unique");
    }
  }
  return {
    schemaVersion: VARIATION_PLOT_DEFINITION_SCHEMA_VERSION,
    resultId: variationResultFingerprint(result),
    ...input,
  };
}

function validateDispersionState(input: VariationPlotDefinitionInputTs): void {
  const geometric = input.plotType === "swing_arc_overlay"
    || input.plotType === "geometric_variability";
  if (!geometric) return;
  if (input.dispersionMetric === null) throw new Error("geometric plot requires dispersionMetric");
  const expectedUnit = input.dispersionMetric === "confidence-ellipsoid-volume" ? "m^3" : "m";
  if (input.dispersionUnit !== expectedUnit) throw new Error("invalid dispersionUnit");
  if (input.quietThreshold === null
    || !Number.isFinite(input.quietThreshold)
    || input.quietThreshold <= 0) {
    throw new Error("quietThreshold must be finite and greater than zero");
  }
  if (input.dispersionMetric === "confidence-ellipsoid-volume") {
    if (input.confidenceLevel === null
      || !Number.isFinite(input.confidenceLevel)
      || input.confidenceLevel < 1e-12
      || input.confidenceLevel >= 1) {
      throw new Error("volume requires confidenceLevel in [1e-12, 1)");
    }
  } else if (input.confidenceLevel !== null) {
    throw new Error("confidenceLevel applies only to confidence-ellipsoid volume");
  }
  if (input.minQuietDurationS === null
    || !Number.isFinite(input.minQuietDurationS)
    || input.minQuietDurationS < 0) {
    throw new Error("minQuietDurationS must be finite and non-negative");
  }
  if (input.minQuietSamples === null
    || !Number.isInteger(input.minQuietSamples)
    || input.minQuietSamples < 1) {
    throw new Error("minQuietSamples must be an integer >= 1");
  }
}

export const variationPlotDefinitionToJson = (
  definition: VariationPlotDefinitionTs,
): string => JSON.stringify(definition, null, 2);

export function swingResultFingerprint(ensemble: SwingVariationResultTs): string {
  return variationResultFingerprint(ensemble);
}

export function variationResultFingerprint(
  result: SwingVariationResultTs | VariationDatasetTs,
): string {
  const dataset = "dataset" in result ? result.dataset : result;
  const identity = JSON.stringify({
    plan: dataset.plan,
    outputs: dataset.outputNames,
    success: dataset.success,
    statuses: "runs" in result ? result.runs.map((run) => run.status) : null,
    samples: "runs" in result
      ? result.runs.map((run) => run.run?.swing.length ?? 0)
      : null,
    frame: "coordinateFrame" in result ? result.coordinateFrame : null,
  });
  let hash = 0x811c9dc5;
  for (let index = 0; index < identity.length; index += 1) {
    hash ^= identity.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return `variation-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}
