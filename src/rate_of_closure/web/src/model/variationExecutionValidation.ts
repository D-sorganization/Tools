import { outputsForMode, planToJson, validatePlan, type VariationDatasetTs, type VariationPlanTs } from "./variation";
import type { SensitivityResultTs } from "./variationAnalysis";
import { SWING_VARIATION_OUTPUT_NAMES, type SwingVariationResultTs } from "./variationSwingEnsemble";
import type { VariationExecutionRequest, VariationExecutionResult } from "./variationExecutionService";

export const MAX_WORKER_ERROR_LENGTH = 512;

const EXECUTION_POLICIES: ReadonlySet<string> = new Set([
  "all_together", "individual", "both",
]);

export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

export const failProtocol = (detail: string): never => {
  throw new Error(`Invalid variation worker ${detail}.`);
};

export const validateExecutionRequest = (request: VariationExecutionRequest): void => {
  if (!EXECUTION_POLICIES.has(request.analysisExecution)) {
    throw new Error("Invalid variation analysis execution policy.");
  }
  validatePlan(request.plan);
};

const stringArray = (value: unknown, field: string): string[] => {
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
    return failProtocol(`${field} string array`);
  }
  return value;
};

const matrix = (
  value: unknown,
  rows: number,
  columns: number,
  field: string,
  nullable: boolean,
): Array<Array<number | null>> => {
  if (!Array.isArray(value) || value.length !== rows) {
    return failProtocol(`${field} row count`);
  }
  const validCell = (cell: unknown) =>
    (nullable && cell === null) || (typeof cell === "number" && Number.isFinite(cell));
  if (!value.every((row) =>
    Array.isArray(row) && row.length === columns && row.every(validCell))) {
    return failProtocol(`${field} matrix`);
  }
  return value as Array<Array<number | null>>;
};

const expectedOutputs = (plan: VariationPlanTs): string[] => plan.mode === "swing"
  ? [...SWING_VARIATION_OUTPUT_NAMES]
  : outputsForMode(plan.mode);

const validateDataset = (
  value: unknown,
  request: VariationExecutionRequest,
): VariationDatasetTs => {
  if (!isRecord(value) || !isRecord(value.plan)) return failProtocol("result dataset");
  try {
    if (planToJson(value.plan as unknown as VariationPlanTs) !== planToJson(request.plan)) {
      return failProtocol("result plan identity");
    }
  } catch {
    return failProtocol("result plan");
  }
  const inputNames = stringArray(value.inputNames, "result inputNames");
  const outputNames = stringArray(value.outputNames, "result outputNames");
  const expectedInputs = request.plan.noise.map((spec) => spec.variableKey);
  const outputIdentity = expectedOutputs(request.plan);
  if (inputNames.length !== expectedInputs.length
      || inputNames.some((name, index) => name !== expectedInputs[index])) {
    return failProtocol("result input identity");
  }
  if (outputNames.length !== outputIdentity.length
      || outputNames.some((name, index) => name !== outputIdentity[index])) {
    return failProtocol("result output identity");
  }
  matrix(value.inputs, request.plan.nRuns, inputNames.length, "result inputs", false);
  const outputs = matrix(value.outputs, request.plan.nRuns, outputNames.length, "result outputs", true);
  if (!Array.isArray(value.success) || value.success.length !== request.plan.nRuns
      || !value.success.every((item) => typeof item === "boolean")) {
    return failProtocol("result success flags");
  }
  if (value.success.some((succeeded, index) =>
    !succeeded && outputs[index].some((item) => item !== null))) {
    return failProtocol("failed result availability");
  }
  return value as unknown as VariationDatasetTs;
};

const validateSensitivity = (
  value: unknown,
  request: VariationExecutionRequest,
): SensitivityResultTs => {
  if (!isRecord(value)) return failProtocol("sensitivity result");
  const inputKeys = stringArray(value.inputKeys, "sensitivity inputKeys");
  const outputNames = stringArray(value.outputNames, "sensitivity outputNames");
  const outputIdentity = expectedOutputs(request.plan);
  if (inputKeys.length !== request.plan.noise.length
      || inputKeys.some((key, index) => key !== request.plan.noise[index].variableKey)) {
    return failProtocol("sensitivity input identity");
  }
  if (outputNames.length !== outputIdentity.length
      || outputNames.some((name, index) => name !== outputIdentity[index])) {
    return failProtocol("sensitivity output identity");
  }
  const validMatrix = (candidate: unknown) =>
    Array.isArray(candidate) && candidate.length === inputKeys.length
    && candidate.every((row) => Array.isArray(row) && row.length === outputNames.length
      && row.every((item) => typeof item === "number" && ![Infinity, -Infinity].includes(item)));
  if (!validMatrix(value.matrix) || !validMatrix(value.normalized)) {
    return failProtocol("sensitivity matrix");
  }
  return value as unknown as SensitivityResultTs;
};

const validateEnsemble = (
  value: unknown,
  request: VariationExecutionRequest,
): SwingVariationResultTs => {
  if (!isRecord(value)
      || value.coordinateFrame !== "app_frame:x_target,y_up,z_right"
      || !Array.isArray(value.runs)
      || value.runs.length !== request.plan.nRuns) {
    return failProtocol("swing ensemble");
  }
  validateDataset(value.dataset, request);
  const validStatuses = new Set(["evaluated_hit", "evaluated_no_impact", "numerical_failure"]);
  if (!value.runs.every((trial, index) => isRecord(trial)
      && trial.trialIndex === index && validStatuses.has(String(trial.status)))) {
    return failProtocol("swing ensemble trials");
  }
  return value as unknown as SwingVariationResultTs;
};

export const validateResult = (
  value: unknown,
  request: VariationExecutionRequest,
): VariationExecutionResult => {
  if (!isRecord(value)) return failProtocol("result message");
  const runJoint = request.analysisExecution !== "individual";
  const runIndividual = request.analysisExecution !== "all_together";
  const expectedEnsemble = runJoint && request.plan.mode === "swing";
  if ((value.dataset === null) === runJoint
      || (value.sensitivity === null) === runIndividual
      || (value.ensemble === null) === expectedEnsemble) {
    return failProtocol("result availability");
  }
  const dataset = value.dataset === null ? null : validateDataset(value.dataset, request);
  const sensitivity = value.sensitivity === null
    ? null
    : validateSensitivity(value.sensitivity, request);
  const ensemble = value.ensemble === null ? null : validateEnsemble(value.ensemble, request);
  return { dataset, sensitivity, ensemble };
};

export const workerError = (value: unknown): Error | DOMException => {
  if (value instanceof Error || value instanceof DOMException) return value;
  return new Error(String(value).slice(0, MAX_WORKER_ERROR_LENGTH));
};
