/** Strict plan-v2 execution sidecar with resolved registry provenance. */

import { sha256Text } from "./launchMonitorFingerprint";
import { planFromJson, planToJson, type VariationPlanTs } from "./variationSchema";
import {
  keysForMode, variableDef, variableDimension, type VariationMode,
} from "./variationRegistry";
import { resolvedBase } from "./variationSampling";

export const EXECUTION_DOCUMENT_SCHEMA_ID = "rate-of-closure/variation-execution-document";
export const EXECUTION_DOCUMENT_SCHEMA_VERSION = 1;
export const EXECUTION_METADATA_SCHEMA_ID = "rate-of-closure/variation-execution-metadata";
export const EXECUTION_METADATA_SCHEMA_VERSION = 1;
export const VARIABLE_REGISTRY_SCHEMA_ID = "swing-sim/variation-variable-registry";
export const VARIABLE_REGISTRY_SCHEMA_VERSION = 1;
export const LEGACY_CURRENT_REGISTRY_WARNING =
  "Legacy plan has no historical execution sidecar; resolved against the current variable registry. " +
  "This is not evidence of historical reproducibility.";

export interface ResolvedVariableSnapshotTs {
  readonly variableKey: string;
  readonly value: number;
  readonly unit: string;
  readonly dimension: string;
}

export interface VariationExecutionMetadataTs {
  readonly schemaId: typeof EXECUTION_METADATA_SCHEMA_ID;
  readonly schemaVersion: typeof EXECUTION_METADATA_SCHEMA_VERSION;
  readonly planSha256: string;
  readonly mode: VariationMode;
  readonly flightModel: string;
  readonly registrySchemaId: typeof VARIABLE_REGISTRY_SCHEMA_ID;
  readonly registrySchemaVersion: typeof VARIABLE_REGISTRY_SCHEMA_VERSION;
  readonly registrySha256: string;
  readonly resolvedVariables: readonly ResolvedVariableSnapshotTs[];
}

export interface VariationExecutionResolutionTs {
  readonly metadata: VariationExecutionMetadataTs;
  readonly warning: string | null;
}

export interface ParsedVariationExecutionDocumentTs extends VariationExecutionResolutionTs {
  readonly plan: VariationPlanTs;
}

interface MetadataWire {
  schema_id: string;
  schema_version: number;
  plan_sha256: string;
  mode: string;
  flight_model: string;
  registry_schema_id: string;
  registry_schema_version: number;
  registry_sha256: string;
  resolved_variables: Array<{
    variable_key: string;
    value: number;
    unit: string;
    dimension: string;
  }>;
}

interface ExecutionDocumentWire {
  schema_id: string;
  schema_version: number;
  plan: Record<string, unknown>;
  metadata: MetadataWire;
}

const DOCUMENT_FIELDS = ["metadata", "plan", "schema_id", "schema_version"];
const METADATA_FIELDS = [
  "flight_model", "mode", "plan_sha256", "registry_schema_id",
  "registry_schema_version", "registry_sha256", "resolved_variables",
  "schema_id", "schema_version",
];
const VARIABLE_FIELDS = ["dimension", "unit", "value", "variable_key"];
const PLAN_FIELDS = [
  "base_variables", "flight_model", "groups", "mode", "n_runs", "noise",
  "schema_version", "seed",
];
const PLAN_FIELDS_WITH_BALL_SETUP = [
  "ball_setup", ...PLAN_FIELDS,
].sort();

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const exactRecord = (
  value: unknown, fields: readonly string[], name: string,
): Record<string, unknown> => {
  if (!isRecord(value) || JSON.stringify(Object.keys(value).sort()) !== JSON.stringify(fields)) {
    throw new Error(`${name} fields mismatch`);
  }
  return value;
};

const text = (value: unknown, name: string): string => {
  if (typeof value !== "string") throw new Error(`${name} must be text`);
  return value;
};

const integer = (value: unknown, name: string): number => {
  if (!Number.isInteger(value)) throw new Error(`${name} must be an integer`);
  return value as number;
};

const finite = (value: unknown, name: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${name} must be finite`);
  }
  return value;
};

const float64Hex = (value: number): string => {
  const buffer = new ArrayBuffer(8);
  const view = new DataView(buffer);
  view.setFloat64(0, value, false);
  return view.getUint32(0, false).toString(16).padStart(8, "0") +
    view.getUint32(4, false).toString(16).padStart(8, "0");
};

const digestValue = (value: unknown): unknown => {
  if (value === null || typeof value === "boolean" || typeof value === "string") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("digest numbers must be finite");
    return { $f64: float64Hex(value) };
  }
  if (Array.isArray(value)) return value.map(digestValue);
  if (isRecord(value)) {
    return Object.fromEntries(
      Object.keys(value).sort().map((key) => [key, digestValue(value[key])]),
    );
  }
  throw new Error(`unsupported canonical digest value: ${typeof value}`);
};

const digest = (value: unknown): string => sha256Text(JSON.stringify(digestValue(value)));

const canonicalEqual = (left: unknown, right: unknown): boolean =>
  JSON.stringify(digestValue(left)) === JSON.stringify(digestValue(right));

const canonicalPlanWire = (plan: VariationPlanTs): Record<string, unknown> =>
  JSON.parse(planToJson(plan)) as Record<string, unknown>;

const snapshots = (plan: VariationPlanTs): readonly ResolvedVariableSnapshotTs[] => {
  const values = resolvedBase(plan);
  return Object.freeze([...keysForMode(plan.mode, plan.ballSetup)].sort().map((variableKey) => {
    const definition = variableDef(variableKey);
    if (definition === undefined) throw new Error(`unknown registry variable ${variableKey}`);
    return Object.freeze({
      variableKey,
      value: values[variableKey],
      unit: definition.unit,
      dimension: variableDimension(definition.unit),
    });
  }));
};

const registryDigest = (plan: VariationPlanTs): string => digest({
  schema_id: VARIABLE_REGISTRY_SCHEMA_ID,
  schema_version: VARIABLE_REGISTRY_SCHEMA_VERSION,
  variables: [...keysForMode(plan.mode, plan.ballSetup)].sort().map((variableKey) => {
    const definition = variableDef(variableKey);
    if (definition === undefined) throw new Error(`unknown registry variable ${variableKey}`);
    return {
      variable_key: variableKey,
      default: definition.default,
      unit: definition.unit,
      dimension: variableDimension(definition.unit),
    };
  }),
});

export const makeVariationExecutionMetadata = (
  plan: VariationPlanTs,
): VariationExecutionMetadataTs => Object.freeze({
  schemaId: EXECUTION_METADATA_SCHEMA_ID,
  schemaVersion: EXECUTION_METADATA_SCHEMA_VERSION,
  planSha256: digest(canonicalPlanWire(plan)),
  mode: plan.mode,
  flightModel: plan.flightModel,
  registrySchemaId: VARIABLE_REGISTRY_SCHEMA_ID,
  registrySchemaVersion: VARIABLE_REGISTRY_SCHEMA_VERSION,
  registrySha256: registryDigest(plan),
  resolvedVariables: snapshots(plan),
});

const snapshotWire = (item: ResolvedVariableSnapshotTs) => ({
  variable_key: item.variableKey,
  value: item.value,
  unit: item.unit,
  dimension: item.dimension,
});

const metadataWire = (metadata: VariationExecutionMetadataTs): MetadataWire => ({
  schema_id: metadata.schemaId,
  schema_version: metadata.schemaVersion,
  plan_sha256: metadata.planSha256,
  mode: metadata.mode,
  flight_model: metadata.flightModel,
  registry_schema_id: metadata.registrySchemaId,
  registry_schema_version: metadata.registrySchemaVersion,
  registry_sha256: metadata.registrySha256,
  resolved_variables: metadata.resolvedVariables.map(snapshotWire),
});

const parseMetadata = (value: unknown): VariationExecutionMetadataTs => {
  const item = exactRecord(value, METADATA_FIELDS, "metadata");
  if (!Array.isArray(item.resolved_variables)) {
    throw new Error("resolved_variables must be an array");
  }
  const resolvedVariables = Object.freeze(item.resolved_variables.map((raw, index) => {
    const variable = exactRecord(raw, VARIABLE_FIELDS, `resolved_variables[${index}]`);
    return Object.freeze({
      variableKey: text(variable.variable_key, "variable_key"),
      value: finite(variable.value, "resolved variable value"),
      unit: text(variable.unit, "unit"),
      dimension: text(variable.dimension, "dimension"),
    });
  }));
  const planSha256 = text(item.plan_sha256, "plan_sha256");
  const registrySha256 = text(item.registry_sha256, "registry_sha256");
  if (!/^[0-9a-f]{64}$/.test(planSha256) || !/^[0-9a-f]{64}$/.test(registrySha256)) {
    throw new Error("metadata digests must be lowercase SHA-256");
  }
  return Object.freeze({
    schemaId: text(item.schema_id, "metadata schema_id") as typeof EXECUTION_METADATA_SCHEMA_ID,
    schemaVersion: integer(item.schema_version, "metadata schema_version") as 1,
    planSha256,
    mode: text(item.mode, "metadata mode") as VariationMode,
    flightModel: text(item.flight_model, "metadata flight_model"),
    registrySchemaId: text(
      item.registry_schema_id, "registry schema_id",
    ) as typeof VARIABLE_REGISTRY_SCHEMA_ID,
    registrySchemaVersion: integer(item.registry_schema_version, "registry schema_version") as 1,
    registrySha256,
    resolvedVariables,
  });
};

export const validateVariationExecutionMetadata = (
  plan: VariationPlanTs, metadata: VariationExecutionMetadataTs,
): VariationExecutionMetadataTs => {
  const expected = makeVariationExecutionMetadata(plan);
  if (metadata.schemaId !== expected.schemaId) throw new Error("metadata schema ID mismatch");
  if (metadata.schemaVersion !== expected.schemaVersion) throw new Error("metadata schema version mismatch");
  if (metadata.mode !== expected.mode) throw new Error("metadata mode mismatch");
  if (metadata.flightModel !== expected.flightModel) throw new Error("metadata flight model mismatch");
  if (metadata.planSha256 !== expected.planSha256) throw new Error("plan digest mismatch");
  if (metadata.registrySchemaId !== expected.registrySchemaId) throw new Error("registry schema ID mismatch");
  if (metadata.registrySchemaVersion !== expected.registrySchemaVersion) {
    throw new Error("registry schema version mismatch");
  }
  if (JSON.stringify(metadata.resolvedVariables) !== JSON.stringify(expected.resolvedVariables)) {
    throw new Error("resolved variable snapshot mismatch");
  }
  if (metadata.registrySha256 !== expected.registrySha256) throw new Error("registry digest mismatch");
  return metadata;
};

export const resolveVariationExecutionMetadata = (
  plan: VariationPlanTs, metadata: VariationExecutionMetadataTs | null | undefined,
): VariationExecutionResolutionTs => metadata === null || metadata === undefined
  ? { metadata: makeVariationExecutionMetadata(plan), warning: LEGACY_CURRENT_REGISTRY_WARNING }
  : { metadata: validateVariationExecutionMetadata(plan, metadata), warning: null };

export const variationExecutionDocument = (
  plan: VariationPlanTs, supplied?: VariationExecutionMetadataTs,
): ExecutionDocumentWire => {
  const { metadata } = resolveVariationExecutionMetadata(plan, supplied);
  return {
    schema_id: EXECUTION_DOCUMENT_SCHEMA_ID,
    schema_version: EXECUTION_DOCUMENT_SCHEMA_VERSION,
    plan: canonicalPlanWire(plan),
    metadata: metadataWire(metadata),
  };
};

export const parseVariationExecutionDocument = (
  textValue: string,
): ParsedVariationExecutionDocumentTs => {
  const root = exactRecord(JSON.parse(textValue) as unknown, DOCUMENT_FIELDS, "execution document");
  if (root.schema_id !== EXECUTION_DOCUMENT_SCHEMA_ID ||
      root.schema_version !== EXECUTION_DOCUMENT_SCHEMA_VERSION) {
    throw new Error("execution document schema mismatch");
  }
  const rawPlan = isRecord(root.plan) && "ball_setup" in root.plan
    ? exactRecord(root.plan, PLAN_FIELDS_WITH_BALL_SETUP, "plan")
    : exactRecord(root.plan, PLAN_FIELDS, "plan");
  const planWire = rawPlan;
  if (planWire.schema_version !== 2) throw new Error("execution document requires canonical plan v2");
  const plan = planFromJson(JSON.stringify(planWire));
  if (!canonicalEqual(canonicalPlanWire(plan), planWire)) {
    throw new Error("execution document plan is not canonical v2");
  }
  const metadata = parseMetadata(root.metadata);
  return { plan, metadata: validateVariationExecutionMetadata(plan, metadata), warning: null };
};
