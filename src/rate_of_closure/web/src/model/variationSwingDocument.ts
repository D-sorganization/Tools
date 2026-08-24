/** Strict finite JSON persistence for browser swing ensembles. */

import { planToJson, type VariationDatasetTs } from "./variation";
import type { VariationPlanTs } from "./variationSchema";
import {
  parseVariationExecutionDocument,
  variationExecutionDocument,
} from "./variationExecutionMetadata";
import { sampleInputs } from "./variationSampling";
import { parseUniqueJson } from "./strictJson";
import type { SwingVariationResultTs } from "./variationSwingEnsemble";
import {
  assertJsonFinite, documentSwingInputAuthority,
  validateLocalizedTrialCommands, validateSwingTrialPayload,
} from "./variationSwingResultValidation";

export const SWING_ENSEMBLE_EXPORT_SCHEMA_VERSION = 3;
const FRAME = "app_frame:x_target,y_up,z_right" as const;
const ROOT_FIELDS = [
  "schemaVersion", "coordinateFrame", "positionUnit", "timeUnit", "dataset", "trials",
] as const;

const record = (value: unknown, label: string): Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
};

const exactFields = (value: Record<string, unknown>, fields: readonly string[], label: string) => {
  if (Object.keys(value).sort().join("|") !== [...fields].sort().join("|")) {
    throw new Error(`${label} fields are invalid`);
  }
};

function validateDataset(datasetValue: unknown, expectedPlan?: VariationPlanTs): VariationDatasetTs {
  const dataset = record(datasetValue, "swing ensemble dataset");
  exactFields(
    dataset,
    ["planDocument", "inputNames", "inputs", "outputNames", "outputs", "success"],
    "swing ensemble dataset",
  );
  const plan = parseVariationExecutionDocument(
    JSON.stringify(record(dataset.planDocument, "swing ensemble plan document")),
  ).plan;
  if (expectedPlan && planToJson(plan) !== planToJson(expectedPlan)) {
    throw new Error("swing ensemble plan does not match the expected plan");
  }
  const inputNames = dataset.inputNames;
  const inputs = dataset.inputs;
  const outputNames = dataset.outputNames;
  const outputs = dataset.outputs;
  const success = dataset.success;
  const expectedInputs = plan.noise.map((spec) => spec.variableKey);
  if (!Array.isArray(inputNames) || JSON.stringify(inputNames) !== JSON.stringify(expectedInputs) ||
      !Array.isArray(inputs) || JSON.stringify(inputs) !== JSON.stringify(sampleInputs(plan)) ||
      !Array.isArray(outputNames) || !outputNames.every((name) => typeof name === "string") ||
      !Array.isArray(outputs) || outputs.length !== plan.nRuns ||
      !outputs.every((row) => Array.isArray(row) && row.length === outputNames.length &&
        row.every((cell) => cell === null ||
          (typeof cell === "number" && Number.isFinite(cell)))) ||
      !Array.isArray(success) || success.length !== plan.nRuns ||
      !success.every((item) => typeof item === "boolean")) {
    throw new Error("swing ensemble dataset does not match its plan");
  }
  return {
    plan,
    inputNames,
    inputs,
    outputNames,
    outputs,
    success,
  } as VariationDatasetTs;
}

export function validateSwingEnsembleDocument(
  value: unknown,
  expectedPlan?: VariationPlanTs,
): SwingVariationResultTs {
  assertJsonFinite(value, "swing ensemble");
  const root = record(value, "swing ensemble document");
  exactFields(root, ROOT_FIELDS, "swing ensemble document");
  if (root.schemaVersion !== SWING_ENSEMBLE_EXPORT_SCHEMA_VERSION ||
      root.coordinateFrame !== FRAME || root.positionUnit !== "m" || root.timeUnit !== "s") {
    throw new Error("swing ensemble document metadata is invalid");
  }
  const dataset = validateDataset(root.dataset, expectedPlan);
  const plan = dataset.plan;
  if (!Array.isArray(root.trials) || root.trials.length !== plan.nRuns) {
    throw new Error("swing ensemble trial count is invalid");
  }
  const firstTrial = record(root.trials[0], "swing ensemble trial 0");
  const baseInput = documentSwingInputAuthority(
    firstTrial.input, plan, dataset.inputs[0],
  );
  if (baseInput === null) throw new Error("swing ensemble trial 0 input is invalid");
  root.trials.forEach((raw, index) => {
    const trial = record(raw, `swing ensemble trial ${index}`);
    exactFields(
      trial,
      ["trialIndex", "status", "input", "run", "error", "localizedTorqueCommands"],
      `swing ensemble trial ${index}`,
    );
    const inputRow = dataset.inputs[index];
    const outputRow = dataset.outputs[index];
    const available = dataset.success[index] === (trial.status !== "numerical_failure") &&
      (trial.status === "numerical_failure"
        ? outputRow.every((cell) => cell === null)
        : trial.status === "evaluated_no_impact"
          ? outputRow.slice(0, 3).every((cell) => typeof cell === "number") &&
            outputRow.slice(3).every((cell) => cell === null)
          : outputRow.every((cell) => typeof cell === "number" && Number.isFinite(cell)));
    if (trial.trialIndex !== index ||
        !available ||
        !validateSwingTrialPayload(trial, index, plan, inputRow, baseInput) ||
        !validateLocalizedTrialCommands(trial, plan, dataset.inputNames, inputRow)) {
      throw new Error(`swing ensemble trial ${index} is invalid`);
    }
  });
  return { dataset, runs: root.trials as SwingVariationResultTs["runs"], coordinateFrame: FRAME };
}

export function swingEnsembleToJsonDocument(result: SwingVariationResultTs): string {
  const value = swingEnsembleDocumentValue(result);
  validateSwingEnsembleDocument(value, result.dataset.plan);
  return JSON.stringify(value, null, 2);
}

export function assertSwingEnsembleResult(result: SwingVariationResultTs): void {
  validateSwingEnsembleDocument(swingEnsembleDocumentValue(result), result.dataset.plan);
}

const swingEnsembleDocumentValue = (result: SwingVariationResultTs) =>
  ({
    schemaVersion: SWING_ENSEMBLE_EXPORT_SCHEMA_VERSION,
    coordinateFrame: result.coordinateFrame,
    positionUnit: "m",
    timeUnit: "s",
    dataset: {
      planDocument: variationExecutionDocument(result.dataset.plan),
      inputNames: result.dataset.inputNames,
      inputs: result.dataset.inputs,
      outputNames: result.dataset.outputNames,
      outputs: result.dataset.outputs,
      success: result.dataset.success,
    },
    trials: result.runs,
  });

export function swingEnsembleFromJson(
  text: string,
  expectedPlan?: VariationPlanTs,
): SwingVariationResultTs {
  return validateSwingEnsembleDocument(
    parseUniqueJson(text, "swing ensemble JSON"), expectedPlan,
  );
}
