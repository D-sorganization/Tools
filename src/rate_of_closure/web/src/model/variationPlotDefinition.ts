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

const PLOT_TYPES: VariationPlotTypeTs[] = [
  "scalar_scatter", "swing_arc_overlay", "geometric_variability", "distribution_matrix",
];
const OUTCOME_FILTERS = [
  "evaluated_hit", "evaluated_no_impact", "numerical_failure",
] as const;
const PERTURBATION_BANDS = [
  "lower", "middle", "upper",
  "Lower Half", "Upper Half", "Lower Third", "Middle Third", "Upper Third",
] as const;
const V2_FIELDS = [
  "schemaVersion", "resultId", "plotType", "coordinateFrame", "xVariableKey",
  "yVariableKey", "pointId", "positionUnit", "alignmentBasis", "dispersionMetric",
  "dispersionUnit", "quietThreshold", "confidenceLevel", "minQuietDurationS",
  "minQuietSamples", "selectedTrialIndex", "cameraYawDeg", "cameraPitchDeg",
  "cameraZoom", "outcomeFilter", "phaseEndFraction", "perturbationSourceKey",
  "perturbationBand", "variableKeys",
] as const;
const V1_FIELDS = [
  "schemaVersion", "resultId", "plotType", "coordinateFrame", "xVariableKey",
  "yVariableKey", "pointId", "positionUnit", "alignmentBasis", "quietThresholdM",
  "selectedTrialIndex", "cameraYawDeg", "cameraPitchDeg", "cameraZoom",
  "outcomeFilter", "phaseEndFraction", "perturbationSourceKey", "perturbationBand",
  "variableKeys",
] as const;
const INPUT_FIELDS = V2_FIELDS.filter(
  (field) => field !== "schemaVersion" && field !== "resultId",
);
const LEGACY_RMS_THRESHOLD_M = 0.005;

export function makeVariationPlotDefinition(
  result: SwingVariationResultTs | VariationDatasetTs,
  input: VariationPlotDefinitionInputTs,
): VariationPlotDefinitionTs {
  exactFields(jsonRecord(input), INPUT_FIELDS);
  return parseV2({
    schemaVersion: VARIATION_PLOT_DEFINITION_SCHEMA_VERSION,
    resultId: variationResultFingerprint(result),
    ...input,
  });
}

function validateDefinitionInput(input: VariationPlotDefinitionInputTs): void {
  validateDispersionState(input);
  if (input.plotType === "scalar_scatter"
    && (input.xVariableKey === null || input.yVariableKey === null)) {
    throw new Error("scalar scatter requires both variable keys");
  }
  if ((input.plotType === "swing_arc_overlay" || input.plotType === "geometric_variability")
    && (input.pointId === null || input.coordinateFrame === null)) {
    throw new Error("geometric plot requires pointId and coordinateFrame");
  }
  if (input.selectedTrialIndex !== null && input.selectedTrialIndex < 0) {
    throw new Error("selectedTrialIndex must be non-negative");
  }
  if (input.cameraPitchDeg !== null
    && (input.cameraPitchDeg < -90 || input.cameraPitchDeg > 90)) {
    throw new Error("cameraPitchDeg must be in [-90, 90]");
  }
  if (input.cameraZoom !== null && input.cameraZoom <= 0) {
    throw new Error("cameraZoom must be greater than zero");
  }
  if (input.phaseEndFraction !== null
    && (input.phaseEndFraction <= 0 || input.phaseEndFraction > 1)) {
    throw new Error("phaseEndFraction must be in (0, 1]");
  }
  const geometric = input.plotType === "swing_arc_overlay"
    || input.plotType === "geometric_variability";
  if (geometric) {
    if (input.positionUnit !== "m") throw new Error("geometric positionUnit must be m");
    if (input.alignmentBasis !== "common_simulation_time_s") {
      throw new Error("geometric alignmentBasis must be common_simulation_time_s");
    }
    if (input.outcomeFilter !== null && !OUTCOME_FILTERS.includes(
      input.outcomeFilter as typeof OUTCOME_FILTERS[number],
    )) throw new Error("unknown outcomeFilter");
    if (input.perturbationBand !== null && !PERTURBATION_BANDS.includes(
      input.perturbationBand as typeof PERTURBATION_BANDS[number],
    )) throw new Error("unknown perturbationBand");
    if (input.perturbationBand !== null && input.perturbationSourceKey === null) {
      throw new Error("perturbationBand requires a perturbation source");
    }
  } else {
    if ([input.dispersionMetric, input.dispersionUnit, input.quietThreshold,
      input.confidenceLevel, input.minQuietDurationS, input.minQuietSamples]
      .some((value) => value !== null)) {
      throw new Error("non-geometric plots cannot contain dispersion state");
    }
    if ([input.outcomeFilter, input.phaseEndFraction, input.perturbationSourceKey,
      input.perturbationBand].some((value) => value !== null)) {
      throw new Error("non-geometric plots cannot contain arc filter state");
    }
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
): string => JSON.stringify(parseV2(jsonRecord(definition)), null, 2);

export function parseVariationPlotDefinition(document: string): VariationPlotDefinitionTs {
  const root = jsonRecord(JSON.parse(document));
  const version = integer(root.schemaVersion, "schemaVersion");
  if (version === 1) return parseV2(migrateV1(root));
  if (version !== VARIATION_PLOT_DEFINITION_SCHEMA_VERSION) {
    throw new Error("unsupported variation plot definition schema");
  }
  return parseV2(root);
}

function parseV2(root: Record<string, unknown>): VariationPlotDefinitionTs {
  exactFields(root, V2_FIELDS);
  if (integer(root.schemaVersion, "schemaVersion") !== 2) throw new Error("unsupported schema");
  const plotType = oneOf(root.plotType, PLOT_TYPES, "plotType");
  const definition: VariationPlotDefinitionTs = {
    schemaVersion: 2,
    resultId: requiredString(root.resultId, "resultId"),
    plotType,
    coordinateFrame: nullableString(root.coordinateFrame, "coordinateFrame"),
    xVariableKey: nullableString(root.xVariableKey, "xVariableKey"),
    yVariableKey: nullableString(root.yVariableKey, "yVariableKey"),
    pointId: nullableString(root.pointId, "pointId"),
    positionUnit: nullableString(root.positionUnit, "positionUnit"),
    alignmentBasis: nullableString(root.alignmentBasis, "alignmentBasis"),
    dispersionMetric: nullableMetric(root.dispersionMetric),
    dispersionUnit: nullableUnit(root.dispersionUnit),
    quietThreshold: nullableNumber(root.quietThreshold, "quietThreshold"),
    confidenceLevel: nullableNumber(root.confidenceLevel, "confidenceLevel"),
    minQuietDurationS: nullableNumber(root.minQuietDurationS, "minQuietDurationS"),
    minQuietSamples: nullableInteger(root.minQuietSamples, "minQuietSamples"),
    selectedTrialIndex: nullableInteger(root.selectedTrialIndex, "selectedTrialIndex"),
    cameraYawDeg: nullableNumber(root.cameraYawDeg, "cameraYawDeg"),
    cameraPitchDeg: nullableNumber(root.cameraPitchDeg, "cameraPitchDeg"),
    cameraZoom: nullableNumber(root.cameraZoom, "cameraZoom"),
    outcomeFilter: nullableString(root.outcomeFilter, "outcomeFilter"),
    phaseEndFraction: nullableNumber(root.phaseEndFraction, "phaseEndFraction"),
    perturbationSourceKey: nullableString(root.perturbationSourceKey, "perturbationSourceKey"),
    perturbationBand: nullableString(root.perturbationBand, "perturbationBand"),
    variableKeys: nullableStrings(root.variableKeys, "variableKeys"),
  };
  validateDefinitionInput(definition);
  return definition;
}

function migrateV1(root: Record<string, unknown>): Record<string, unknown> {
  exactFields(root, V1_FIELDS);
  const plotType = oneOf(root.plotType, PLOT_TYPES, "plotType");
  const threshold = nullableNumber(root.quietThresholdM, "quietThresholdM");
  if (threshold !== null && threshold <= 0) throw new Error("quietThresholdM must be positive");
  const geometric = plotType === "swing_arc_overlay" || plotType === "geometric_variability";
  const migrated = { ...root };
  delete migrated.quietThresholdM;
  return {
    ...migrated,
    schemaVersion: 2,
    dispersionMetric: geometric ? "rms-radius" : null,
    dispersionUnit: geometric ? "m" : null,
    quietThreshold: geometric ? threshold ?? LEGACY_RMS_THRESHOLD_M : null,
    confidenceLevel: null,
    minQuietDurationS: geometric ? 0 : null,
    minQuietSamples: geometric ? 1 : null,
  };
}

function jsonRecord(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("plot definition must be an object");
  }
  return value as Record<string, unknown>;
}

function exactFields(root: Record<string, unknown>, expected: readonly string[]): void {
  const actual = Object.keys(root).sort();
  const declared = [...expected].sort();
  if (actual.length !== declared.length || actual.some((key, index) => key !== declared[index])) {
    throw new Error("invalid plot definition fields");
  }
}

function integer(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isInteger(value)) throw new Error(`${name} must be integer`);
  return value;
}

function nullableInteger(value: unknown, name: string): number | null {
  return value === null ? null : integer(value, name);
}

function nullableNumber(value: unknown, name: string): number | null {
  if (value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${name} must be finite`);
  return value;
}

function requiredString(value: unknown, name: string): string {
  const result = nullableString(value, name);
  if (result === null) throw new Error(`${name} is required`);
  return result;
}

function nullableString(value: unknown, name: string): string | null {
  if (value === null) return null;
  if (typeof value !== "string" || value.length === 0 || value.trim() !== value) {
    throw new Error(`${name} must be a non-empty trimmed string`);
  }
  return value;
}

function nullableStrings(value: unknown, name: string): string[] | null {
  if (value === null) return null;
  if (!Array.isArray(value)) throw new Error(`${name} must be an array`);
  return value.map((item) => requiredString(item, name));
}

function oneOf<T extends string>(value: unknown, allowed: readonly T[], name: string): T {
  const result = requiredString(value, name);
  if (!allowed.includes(result as T)) throw new Error(`unknown ${name}`);
  return result as T;
}

function nullableMetric(value: unknown): DispersionMetricTs | null {
  if (value === null) return null;
  return oneOf(value, [
    "rms-radius", "largest-principal-sigma", "confidence-ellipsoid-volume",
  ] as const, "dispersionMetric");
}

function nullableUnit(value: unknown): "m" | "m^3" | null {
  if (value === null) return null;
  return oneOf(value, ["m", "m^3"] as const, "dispersionUnit");
}

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
