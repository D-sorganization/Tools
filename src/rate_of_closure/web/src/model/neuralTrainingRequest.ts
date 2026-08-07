/** Portable metadata-only request contract for the private neural-training CLI. */

export const TRAINING_REQUEST_SCHEMA = "launch-monitor-neural-training/v1";

export interface TrainingDatasetReference {
  fileName: string;
  rowCount: number;
  columns: string[];
  sha256?: string;
}

export interface TrainingRequestInput {
  vendor: string;
  dataset: TrainingDatasetReference;
  featureColumns: string[];
  outputColumns: string[];
  hiddenLayers: number[];
  activation: "relu" | "tanh" | "linear";
  alpha: number;
  epochs: number;
  learningRate: number;
  validationFraction: number;
  randomSeed: number;
}

export interface NeuralTrainingRequest extends TrainingRequestInput {
  schema: typeof TRAINING_REQUEST_SCHEMA;
  objective: "vendor-comparable-regression";
}

const assertIntegerBetween = (value: number, minimum: number, maximum: number, label: string) => {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`${label} must be an integer from ${minimum} to ${maximum}`);
  }
};

/** Validate user configuration and create a request that does not embed source rows. */
export function createTrainingRequest(input: TrainingRequestInput): NeuralTrainingRequest {
  if (!input.vendor.trim()) throw new RangeError("vendor is required");
  if (!input.dataset.fileName.trim() || input.dataset.rowCount < 1) throw new RangeError("dataset metadata is incomplete");
  const available = new Set(input.dataset.columns);
  if (!input.featureColumns.length || !input.outputColumns.length) throw new RangeError("features and outputs are required");
  if ([...input.featureColumns, ...input.outputColumns].some((column) => !available.has(column))) {
    throw new RangeError("selected columns must exist in the dataset");
  }
  if (input.featureColumns.some((column) => input.outputColumns.includes(column))) {
    throw new RangeError("a column cannot be both feature and output because that creates target leakage");
  }
  if (new Set(input.featureColumns).size !== input.featureColumns.length ||
    new Set(input.outputColumns).size !== input.outputColumns.length) throw new RangeError("selected columns must be unique");
  if (!input.hiddenLayers.length || input.hiddenLayers.length > 8) throw new RangeError("one to eight hidden layers are required");
  input.hiddenLayers.forEach((width) => assertIntegerBetween(width, 1, 1024, "hidden-layer width"));
  assertIntegerBetween(input.epochs, 1, 100000, "epochs");
  assertIntegerBetween(input.randomSeed, 0, 2147483647, "random seed");
  if (!(input.learningRate > 0 && input.learningRate <= 1)) throw new RangeError("learning rate must be greater than 0 and at most 1");
  if (!(input.alpha >= 0 && input.alpha <= 1)) throw new RangeError("regularization alpha must be from 0 to 1");
  if (!(input.validationFraction >= 0.05 && input.validationFraction <= 0.5)) {
    throw new RangeError("validation fraction must be from 0.05 to 0.5");
  }
  return { schema: TRAINING_REQUEST_SCHEMA, objective: "vendor-comparable-regression", ...input };
}
