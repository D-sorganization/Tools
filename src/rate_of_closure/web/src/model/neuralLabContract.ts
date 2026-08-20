/** Safe reference-only private training and portable inference contracts. */

import type { LaunchMonitorRow } from "./launchMonitorAnalysisTypes";
import { sha256Text } from "./launchMonitorFingerprint";

export const CAPABILITY_SCHEMA = "launch-monitor-capability-manifest/v1" as const;
export const TRAINING_SCHEMA = "launch-monitor-neural-training/v2" as const;
export const PORTABLE_MODEL_SCHEMA = "launch-monitor-neural-bundle/v2" as const;
const forbiddenGroups = new Set(["shot_id", "source_row_number", "row_index"]);

const record = (value: unknown, label: string): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new RangeError(`${label} must be an object`);
  return value as Record<string, unknown>;
};
const text = (value: unknown, label: string): string => {
  if (typeof value !== "string" || !value.trim()) throw new RangeError(`${label} must be non-empty text`);
  return value.trim();
};
const finite = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new RangeError(`${label} must be finite`);
  return value;
};
const integer = (value: unknown, label: string): number => {
  const result = finite(value, label); if (!Number.isSafeInteger(result) || result < 0) throw new RangeError(`${label} is invalid`); return result;
};
const digest = (value: unknown, label: string): string => {
  const result = text(value, label).toLowerCase();
  if (!/^[a-f\d]{64}$/.test(result)) throw new RangeError(`${label} must be a SHA-256`); return result;
};

export interface VendorCapability { vendor: string; state: "available" | "unavailable"; rowCount: number;
  strictRowCount: number; artifactState: string; blockers: string[] }
export interface CapabilityManifest { schema: typeof CAPABILITY_SCHEMA; policySha256: string; vendors: VendorCapability[] }

export function parseCapabilityManifest(value: unknown): CapabilityManifest {
  const root = record(value, "capability manifest");
  if (root.schema !== CAPABILITY_SCHEMA || !Array.isArray(root.vendors)) throw new RangeError("Unsupported capability manifest");
  const labels: Record<string, string> = { trackman: "TrackMan", foresight: "Foresight", flightscope: "FlightScope", unconfirmed: "Unconfirmed" };
  return { schema: CAPABILITY_SCHEMA, policySha256: digest(root.policy_sha256, "capability policy SHA-256"), vendors: root.vendors.map((item) => {
    const row = record(item, "vendor capability");
    const key = text(row.vendor_key, "vendor key"); const allowed = record(row.allowed_operations, "allowed operations");
    const blockers = record(row.training_blockers, "training blockers");
    return { vendor: labels[key] ?? key, state: allowed.vendor_training === true ? "available" : "unavailable",
      rowCount: integer(row.rows, "row count"), strictRowCount: integer(row.strict_model_input_rows, "strict row count"),
      artifactState: text(row.current_surrogate_artifact_status, "artifact state"),
      blockers: Object.entries(blockers).map(([reason, count]) => `${reason.replace(/_/g, " ")}: ${integer(count, "blocker count")} source-metric policy decisions`) };
  }) };
}

export interface DatasetAuthority { datasetId: string; repository: string; commit: string; datasetPath: string; sha256: string; rowCount: number }
export interface TrainingSelection { vendor: string; features: string[]; targets: string[]; splitGroup: string; splitGroupPolicyApproved: boolean }
export interface GroupSummary { column: string; distinctGroups: number; repeatedGroups: number }

export function validateTrainingGroups(rows: LaunchMonitorRow[], column: string, policyApproved: boolean): GroupSummary {
  const name = text(column, "split group");
  if (forbiddenGroups.has(name.toLowerCase())) throw new RangeError(`${name} is a forbidden row-like split group`);
  if (!policyApproved) throw new RangeError("Split group must be explicitly policy-approved");
  const counts = new Map<string, number>();
  rows.forEach((row) => { if (row[name] !== null && row[name] !== undefined) {
    const group = String(row[name]); counts.set(group, (counts.get(group) ?? 0) + 1); } });
  const result = { column: name, distinctGroups: counts.size, repeatedGroups: [...counts.values()].filter((count) => count >= 2).length };
  if (result.distinctGroups < 3) throw new RangeError("Split group requires at least three distinct groups");
  if (result.repeatedGroups < 1) throw new RangeError("Split group requires at least one repeated group");
  return result;
}

export function buildTrainingManifest(dataset: DatasetAuthority, rows: LaunchMonitorRow[], selection: TrainingSelection) {
  [dataset.datasetId, dataset.repository, dataset.datasetPath].forEach((item) => text(item, "dataset reference"));
  if (!/^[a-f\d]{40}$/i.test(dataset.commit)) throw new RangeError("Dataset commit must be a git commit");
  digest(dataset.sha256, "dataset SHA-256");
  if (dataset.rowCount !== rows.length || rows.length < 1) throw new RangeError("Dataset reference row count does not match loaded rows");
  const features = selection.features.map((name) => text(name, "feature"));
  const targets = selection.targets.map((name) => text(name, "target"));
  if (!features.length || !targets.length || features.some((name) => targets.includes(name))) throw new RangeError("Features and targets must be non-empty and disjoint");
  const columns = new Set(rows.flatMap((row) => Object.keys(row)));
  if ([...features, ...targets].some((name) => !columns.has(name))) throw new RangeError("Dataset is missing selected columns");
  const groups = validateTrainingGroups(rows, selection.splitGroup, selection.splitGroupPolicyApproved);
  return { schema: TRAINING_SCHEMA, dataset: { dataset_id: dataset.datasetId, repository: dataset.repository,
    commit: dataset.commit, dataset_path: dataset.datasetPath, sha256: dataset.sha256, row_count: dataset.rowCount },
    vendor: text(selection.vendor, "vendor"), features, targets,
    split: { column: groups.column, distinctGroups: groups.distinctGroups, repeatedGroups: groups.repeatedGroups, policyApproved: true } };
}

export interface ModelVariable { name: string; unit: string; mean: number; scale: number; min?: number; max?: number }
export interface DenseLayer { activation: "linear" | "relu" | "tanh"; weights: number[][]; bias: number[] }
export interface PortableModel { schema: typeof PORTABLE_MODEL_SCHEMA; modelId: string; vendor: string;
  trainingManifestSha256: string; datasetSha256: string; features: ModelVariable[]; targets: ModelVariable[];
  layers: DenseLayer[]; modelCard: Record<string, unknown>; metrics: Record<string, unknown>[];
  trainingManifest: Record<string, unknown>;
  residuals: { state: "available" | "unavailable"; reason?: string; rows?: Record<string, unknown>[] } }

export function canonicalTrainingManifestJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalTrainingManifestJson).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value as Record<string, unknown>)
    .sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => `${JSON.stringify(key)}:${canonicalTrainingManifestJson(item)}`).join(",")}}`;
  return JSON.stringify(value);
}

function variables(value: unknown, bounded: boolean, label: string): ModelVariable[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > 64) throw new RangeError(`${label} count is unsafe`);
  return value.map((item) => { const row = record(item, label); const scale = finite(row.scale, `${label} scale`);
    if (scale <= 0) throw new RangeError(`${label} scale must be positive`);
    const variable = { name: text(row.name, `${label} name`), unit: typeof row.unit === "string" ? row.unit : "unitless",
      mean: finite(row.mean, `${label} mean`), scale } as ModelVariable;
    if (bounded) { variable.min = finite(row.min, `${label} min`); variable.max = finite(row.max, `${label} max`);
      if (variable.min > variable.max) throw new RangeError(`${label} bounds are reversed`); }
    return variable;
  });
}

export function parsePortableModel(value: unknown): PortableModel {
  const root = record(value, "portable model");
  if (root.schema !== PORTABLE_MODEL_SCHEMA) throw new RangeError(`Schema must be ${PORTABLE_MODEL_SCHEMA}`);
  const features = variables(root.features, true, "feature"); const targets = variables(root.targets, false, "target");
  if (!Array.isArray(root.layers) || root.layers.length < 1 || root.layers.length > 16) throw new RangeError("Layer count is unsafe");
  let width = features.length;
  const layers = root.layers.map((item, layerIndex) => { const row = record(item, "layer");
    if (row.activation !== "linear" && row.activation !== "relu" && row.activation !== "tanh") throw new RangeError("Unsupported activation");
    if (!Array.isArray(row.weights) || !Array.isArray(row.bias) || row.weights.length !== row.bias.length || row.bias.length > 4096) throw new RangeError("Layer dimension mismatch");
    const weights = row.weights.map((line, node) => { if (!Array.isArray(line) || line.length !== width) throw new RangeError(`Layer ${layerIndex}:${node} dimension mismatch`);
      return line.map((itemWeight) => finite(itemWeight, "weight")); });
    const bias = row.bias.map((itemBias) => finite(itemBias, "bias")); width = bias.length;
    return { activation: row.activation as DenseLayer["activation"], weights, bias }; });
  if (width !== targets.length) throw new RangeError("Final layer dimension does not match targets");
  const residuals = record(root.residuals, "residual availability");
  if (residuals.state !== "available" && residuals.state !== "unavailable") throw new RangeError("Residual availability must be explicit");
  const trainingManifest = record(root.training_manifest, "embedded training manifest");
  const trainingManifestSha256 = digest(root.training_manifest_sha256, "training manifest SHA-256");
  if (sha256Text(canonicalTrainingManifestJson(trainingManifest)) !== trainingManifestSha256) throw new RangeError("Embedded training manifest SHA-256 does not match");
  const datasetSha256 = digest(root.dataset_sha256, "dataset SHA-256");
  if (record(trainingManifest.dataset, "training manifest dataset").sha256 !== datasetSha256) throw new RangeError("Training manifest dataset SHA-256 does not match model");
  return { schema: PORTABLE_MODEL_SCHEMA, modelId: text(root.model_id, "model id"), vendor: text(root.vendor, "vendor"),
    trainingManifestSha256, datasetSha256, trainingManifest,
    features, targets, layers, modelCard: record(root.model_card, "model card"),
    metrics: Array.isArray(root.metrics) ? root.metrics.map((item) => record(item, "metric")) : [],
    residuals: { state: residuals.state, ...(typeof residuals.reason === "string" ? { reason: residuals.reason } : {}),
      ...(Array.isArray(residuals.rows) ? { rows: residuals.rows.map((item) => record(item, "residual row")) } : {}) } };
}

export function inferPortableModel(model: PortableModel, inputs: Record<string, number>) {
  let values = model.features.map((feature) => (finite(inputs[feature.name], feature.name) - feature.mean) / feature.scale);
  const warnings = model.features.flatMap((feature) => inputs[feature.name] < (feature.min ?? -Infinity) || inputs[feature.name] > (feature.max ?? Infinity)
    ? [`${feature.name} is outside training range [${feature.min}, ${feature.max}] ${feature.unit}.`] : []);
  for (const layer of model.layers) values = layer.weights.map((weights, index) => {
    const raw = weights.reduce((sum, weight, input) => sum + weight * values[input], layer.bias[index]);
    return layer.activation === "relu" ? Math.max(0, raw) : layer.activation === "tanh" ? Math.tanh(raw) : raw;
  });
  return { values: Object.fromEntries(model.targets.map((target, index) => [target.name, values[index] * target.scale + target.mean])), warnings };
}
