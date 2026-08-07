/** Validated, portable dense-network inference for launch-monitor model artifacts. */

import { parseUniqueJson } from "./strictJson";

export const PORTABLE_MODEL_SCHEMA = "launch-monitor-neural-bundle/v1";
const MAX_FEATURES = 64;
const MAX_OUTPUTS = 32;
const MAX_LAYERS = 16;
const MAX_LAYER_WIDTH = 1024;

export interface ModelVariable {
  name: string;
  unit: string;
  mean: number;
  scale: number;
  min?: number;
  max?: number;
}

export interface DenseLayer {
  activation: "linear" | "relu" | "tanh";
  weights: number[][];
  bias: number[];
}

export interface ModelMetric {
  name: string;
  value: number;
  unit?: string;
  split: string;
}

export interface LearningCurvePoint {
  epoch?: number;
  trainingRows?: number;
  trainingFraction?: number;
  trainingLoss?: number;
  validationLoss: number;
}

export interface ModelProvenance {
  datasetSha256: string;
  sampleCount: number;
  details: Record<string, unknown>;
}

export interface PortableModelBundle {
  schema: typeof PORTABLE_MODEL_SCHEMA;
  modelId: string;
  vendor: string;
  createdAt: string;
  features: ModelVariable[];
  outputs: ModelVariable[];
  layers: DenseLayer[];
  metrics: ModelMetric[];
  learningCurve: LearningCurvePoint[];
  provenance: ModelProvenance;
}

const recordOf = (value: unknown, label: string): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new RangeError(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
};

const textOf = (value: unknown, label: string): string => {
  if (typeof value !== "string" || !value.trim()) throw new RangeError(`${label} must be non-empty text`);
  return value;
};

const finiteOf = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new RangeError(`${label} must be finite`);
  return value;
};

function variablesOf(value: unknown, label: string, maximum: number, bounded: boolean): ModelVariable[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > maximum) {
    throw new RangeError(`${label} count must be between 1 and ${maximum}`);
  }
  const variables = value.map((item, index) => {
    const row = recordOf(item, `${label}[${index}]`);
    const scale = finiteOf(row.scale, `${label}[${index}].scale`);
    if (scale <= 0) throw new RangeError(`${label}[${index}].scale must be positive`);
    const minimum = bounded ? finiteOf(row.min, `${label} min`) : undefined;
    const maximumValue = bounded ? finiteOf(row.max, `${label} max`) : undefined;
    if (bounded && (minimum ?? 0) > (maximumValue ?? 0)) throw new RangeError(`${label} applicability bounds are reversed`);
    return { name: textOf(row.name, `${label} name`), unit: typeof row.unit === "string" ? row.unit : "unitless",
      mean: finiteOf(row.mean, `${label} mean`), scale,
      ...(bounded ? { min: minimum, max: maximumValue } : {}) };
  });
  if (new Set(variables.map(({ name }) => name)).size !== variables.length) {
    throw new RangeError(`${label} names must be unique`);
  }
  return variables;
}

function layersOf(value: unknown, inputWidth: number, outputWidth: number): DenseLayer[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > MAX_LAYERS) {
    throw new RangeError(`layers count must be between 1 and ${MAX_LAYERS}`);
  }
  let expectedInputs = inputWidth;
  const layers = value.map((item, layerIndex) => {
    const row = recordOf(item, `layers[${layerIndex}]`);
    if (!Array.isArray(row.weights) || !Array.isArray(row.bias) || row.bias.length < 1 ||
      row.bias.length > MAX_LAYER_WIDTH || row.weights.length !== row.bias.length) {
      throw new RangeError(`layer ${layerIndex} dimension mismatch`);
    }
    const weights = row.weights.map((rawWeights, nodeIndex) => {
      if (!Array.isArray(rawWeights) || rawWeights.length !== expectedInputs) {
        throw new RangeError(`layer ${layerIndex} dimension mismatch at node ${nodeIndex}`);
      }
      return rawWeights.map((weight, index) => finiteOf(weight, `weight ${layerIndex}:${nodeIndex}:${index}`));
    });
    const bias = row.bias.map((itemBias, index) => finiteOf(itemBias, `bias ${layerIndex}:${index}`));
    const activation = row.activation;
    if (activation !== "linear" && activation !== "relu" && activation !== "tanh") {
      throw new RangeError(`layer ${layerIndex} activation is unsupported`);
    }
    expectedInputs = bias.length;
    return { activation: activation as DenseLayer["activation"], weights, bias };
  });
  if (expectedInputs !== outputWidth) throw new RangeError("final layer dimension does not match outputs");
  return layers;
}

function metricsOf(value: unknown, outputUnits: Map<string, string>): ModelMetric[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item, index) => {
    const row = recordOf(item, `metrics[${index}]`);
    const target = typeof row.target === "string" ? row.target : "output";
    const split = typeof row.split === "string" ? row.split : "unspecified";
    return ["mae", "rmse", "r2"].flatMap((name) => row[name] === undefined ? [] : [{
      name: `${target} ${name.toUpperCase()}`, value: finiteOf(row[name], `${name} metric`),
      split, unit: name === "r2" ? "unitless" : outputUnits.get(target) ?? "output unit",
    }]);
  });
}

const curveOf = (value: unknown): LearningCurvePoint[] => Array.isArray(value) ? value.map((item, index) => {
  const row = recordOf(item, `learningCurve[${index}]`);
  const validation = row.validation_standardized_rmse ?? row.validationLoss;
  return { validationLoss: finiteOf(validation, "validation loss"),
    ...(row.epoch === undefined ? {} : { epoch: finiteOf(row.epoch, "epoch") }),
    ...(row.training_rows === undefined ? {} : { trainingRows: finiteOf(row.training_rows, "training rows") }),
    ...(row.training_fraction === undefined ? {} : { trainingFraction: finiteOf(row.training_fraction, "training fraction") }),
    ...(row.trainLoss === undefined ? {} : { trainingLoss: finiteOf(row.trainLoss, "training loss") }) };
}) : [];

function provenanceOf(value: unknown): ModelProvenance {
  const row = recordOf(value, "provenance");
  const digest = textOf(row.datasetSha256 ?? row.dataset_sha256, "dataset SHA-256");
  if (!/^[a-f\d]{1,64}$/i.test(digest)) throw new RangeError("dataset SHA-256 must be hexadecimal text");
  const sampleCount = finiteOf(row.rowCount ?? row.row_count, "sample count");
  if (!Number.isInteger(sampleCount) || sampleCount < 1) throw new RangeError("sample count must be a positive integer");
  return { datasetSha256: digest, sampleCount, details: row };
}

/** Parse an untrusted artifact and enforce all inference-shape invariants. */
export function parsePortableModelBundle(text: string): PortableModelBundle {
  const row = recordOf(parseUniqueJson(text), "model bundle");
  if (row.schema !== PORTABLE_MODEL_SCHEMA) throw new RangeError(`schema must be ${PORTABLE_MODEL_SCHEMA}`);
  const features = variablesOf(row.features, "features", MAX_FEATURES, true);
  const outputs = variablesOf(row.outputs, "outputs", MAX_OUTPUTS, false);
  return { schema: PORTABLE_MODEL_SCHEMA, modelId: textOf(row.modelId, "modelId"),
    vendor: textOf(row.vendor, "vendor"), createdAt: textOf(row.createdAt, "createdAt"), features, outputs,
    layers: layersOf(row.layers, features.length, outputs.length),
    metrics: metricsOf(row.metrics, new Map(outputs.map(({ name, unit }) => [name, unit]))),
    learningCurve: curveOf(row.learningCurve), provenance: provenanceOf(row.provenance) };
}

/** Report inputs outside the artifact's recorded feature applicability range. */
export function applicabilityWarnings(bundle: PortableModelBundle, inputs: Record<string, number>): string[] {
  return bundle.features.flatMap((feature) => {
    const value = inputs[feature.name];
    return value < (feature.min ?? -Infinity) || value > (feature.max ?? Infinity)
      ? [`${feature.name} is outside [${feature.min}, ${feature.max}] ${feature.unit}.`] : [];
  });
}

const activate = (name: DenseLayer["activation"], value: number): number =>
  name === "relu" ? Math.max(0, value) : name === "tanh" ? Math.tanh(value) : value;

/** Run deterministic local inference after applying stored standardization. */
export function inferPortableModel(bundle: PortableModelBundle, inputs: Record<string, number>): Record<string, number> {
  let values = bundle.features.map((feature) => {
    const value = inputs[feature.name];
    if (!Number.isFinite(value)) throw new RangeError(`input ${feature.name} must be finite`);
    return (value - feature.mean) / feature.scale;
  });
  for (const layer of bundle.layers) {
    values = layer.weights.map((weights, outputIndex) => activate(layer.activation,
      weights.reduce((sum, weight, inputIndex) => sum + weight * values[inputIndex], layer.bias[outputIndex])));
  }
  return Object.fromEntries(bundle.outputs.map((output, index) =>
    [output.name, values[index] * output.scale + output.mean]));
}
