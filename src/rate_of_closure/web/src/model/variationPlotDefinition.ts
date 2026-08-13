/** Versioned, toolkit-independent variation plot definitions. */

import type { SwingVariationResultTs } from "./variationSwingEnsemble";
import type { VariationDatasetTs } from "./variation";

export const VARIATION_PLOT_DEFINITION_SCHEMA_VERSION = 1;

export type VariationPlotTypeTs =
  | "scalar_scatter"
  | "swing_arc_overlay"
  | "geometric_variability"
  | "distribution_matrix";

export interface VariationPlotDefinitionTs {
  schemaVersion: 1;
  resultId: string;
  plotType: VariationPlotTypeTs;
  coordinateFrame: string | null;
  xVariableKey: string | null;
  yVariableKey: string | null;
  pointId: string | null;
  positionUnit: string | null;
  alignmentBasis: string | null;
  quietThresholdM: number | null;
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
  if (input.quietThresholdM !== null && input.quietThresholdM <= 0) {
    throw new Error("quietThresholdM must be greater than zero");
  }
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
