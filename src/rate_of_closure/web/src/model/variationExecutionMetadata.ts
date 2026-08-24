/** Strict plan-v2 execution sidecar with resolved registry provenance. */

import { sha256Text } from "./launchMonitorFingerprint";
import { planFromJson, planToJson, type VariationPlanTs } from "./variationSchema";
import {
  GROUND_NORMAL_RESTITUTION_KEY, GROUND_ROLLING_RESISTANCE_KEY,
  keysForMode, variableDef, variableDimension, type VariationMode,
} from "./variationRegistry";
import { resolvedBase } from "./variationSampling";
import { parseUniqueJson } from "./strictJson";
import {
  provenanceFromWire,
  provenanceSha256,
  provenanceToWire,
  REACT_DEFAULT_PROVENANCE,
  type PlanProducerProvenanceTs,
  type PlanProducerProvenanceWire,
} from "./variationExecutionProvenance";

export {
  makePlanProducerProvenance,
  REACT_DEFAULT_PROVENANCE,
  type PlanProducerProvenanceTs,
} from "./variationExecutionProvenance";

export const EXECUTION_DOCUMENT_SCHEMA_ID = "rate-of-closure/variation-execution-document";
export const EXECUTION_DOCUMENT_SCHEMA_VERSION = 3;
export const EXECUTION_METADATA_SCHEMA_ID = "rate-of-closure/variation-execution-metadata";
export const EXECUTION_METADATA_SCHEMA_VERSION = 3;
export const VARIABLE_REGISTRY_SCHEMA_ID = "swing-sim/variation-variable-registry";
export const VARIABLE_REGISTRY_SCHEMA_VERSION = 1;
export const LEGACY_CURRENT_REGISTRY_WARNING =
  "Legacy plan has no historical execution sidecar; resolved against the current variable registry. " +
  "This is not evidence of historical reproducibility.";
export const LEGACY_EXECUTION_DOCUMENT_MIGRATION_ERROR =
  "Execution document schema @1 or @2 lacks the complete current identity contract; load its " +
  "raw plan and resolve a fresh @3 document. Historical replay remains unproven because " +
  "source provenance was not recorded.";

export interface RngExecutionIdentityTs {
  readonly algorithmId: string;
  readonly algorithmVersion: number;
  readonly streamDerivationId: string;
  readonly streamDerivationVersion: number;
}

export interface ExecutionImplementationIdentityTs {
  readonly runtimeId: string;
  readonly runtimeVersion: number;
  readonly executorId: string;
  readonly executorVersion: number;
  readonly solverId: string;
  readonly solverVersion: number;
}

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
  readonly rngIdentity: RngExecutionIdentityTs;
  readonly implementationIdentity: ExecutionImplementationIdentityTs;
  readonly provenanceSha256: string;
}

export interface VariationExecutionResolutionTs {
  readonly metadata: VariationExecutionMetadataTs;
  readonly warning: string | null;
}

export interface ParsedVariationExecutionDocumentTs extends VariationExecutionResolutionTs {
  readonly plan: VariationPlanTs;
  readonly provenance: PlanProducerProvenanceTs;
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
  rng_identity: {
    algorithm_id: string; algorithm_version: number;
    stream_derivation_id: string; stream_derivation_version: number;
  };
  implementation_identity: {
    runtime_id: string; runtime_version: number; executor_id: string;
    executor_version: number; solver_id: string; solver_version: number;
  };
  provenance_sha256: string;
}

interface ExecutionDocumentWire {
  schema_id: string;
  schema_version: number;
  plan: Record<string, unknown>;
  metadata: MetadataWire;
  provenance: PlanProducerProvenanceWire;
}

const DOCUMENT_FIELDS = ["metadata", "plan", "provenance", "schema_id", "schema_version"];
const METADATA_FIELDS = [
  "flight_model", "mode", "plan_sha256", "registry_schema_id",
  "registry_schema_version", "registry_sha256", "resolved_variables",
  "schema_id", "schema_version",
  "rng_identity", "implementation_identity",
  "provenance_sha256",
].sort();
const VARIABLE_FIELDS = ["dimension", "unit", "value", "variable_key"];
const RUNTIME_METADATA_FIELDS = [
  "flightModel", "mode", "planSha256", "registrySchemaId",
  "registrySchemaVersion", "registrySha256", "resolvedVariables",
  "schemaId", "schemaVersion",
  "rngIdentity", "implementationIdentity",
  "provenanceSha256",
].sort();
const RUNTIME_VARIABLE_FIELDS = ["dimension", "unit", "value", "variableKey"];
const RNG_FIELDS = ["algorithm_id", "algorithm_version", "stream_derivation_id", "stream_derivation_version"];
const IMPLEMENTATION_FIELDS = ["executor_id", "executor_version", "runtime_id", "runtime_version", "solver_id", "solver_version"];
const RUNTIME_RNG_FIELDS = ["algorithmId", "algorithmVersion", "streamDerivationId", "streamDerivationVersion"];
const RUNTIME_IMPLEMENTATION_FIELDS = ["executorId", "executorVersion", "runtimeId", "runtimeVersion", "solverId", "solverVersion"];
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

const normalizedNumber = (value: number): number => Object.is(value, -0) ? 0 : value;

const digestValue = (value: unknown): unknown => {
  if (value === null || typeof value === "boolean" || typeof value === "string") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("digest numbers must be finite");
    return { $f64: float64Hex(normalizedNumber(value)) };
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

/** Return the cross-runtime digest of one canonical variation plan. */
export const variationPlanSha256 = (plan: VariationPlanTs): string =>
  digest(canonicalPlanWire(plan));

const metadataKeys = (plan: VariationPlanTs): readonly string[] => {
  const referenced = new Set([
    ...Object.keys(plan.baseVariables),
    ...plan.noise.map(({ variableKey }) => variableKey),
  ]);
  return keysForMode(plan.mode, plan.ballSetup).filter((key) =>
    ![GROUND_NORMAL_RESTITUTION_KEY, GROUND_ROLLING_RESISTANCE_KEY].includes(key) ||
    referenced.has(key));
};

const snapshots = (plan: VariationPlanTs): readonly ResolvedVariableSnapshotTs[] => {
  const values = resolvedBase(plan);
  return Object.freeze([...metadataKeys(plan)].sort().map((variableKey) => {
    const definition = variableDef(variableKey);
    if (definition === undefined) throw new Error(`unknown registry variable ${variableKey}`);
    return Object.freeze({
      variableKey,
      value: normalizedNumber(values[variableKey]),
      unit: definition.unit,
      dimension: variableDimension(definition.unit),
    });
  }));
};

const registryDigest = (plan: VariationPlanTs): string => digest({
  schema_id: VARIABLE_REGISTRY_SCHEMA_ID,
  schema_version: VARIABLE_REGISTRY_SCHEMA_VERSION,
  variables: [...metadataKeys(plan)].sort().map((variableKey) => {
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

const solverId = (mode: VariationMode): string => mode === "swing"
  ? "rate-of-closure/react-swing-rk4+rigid-impact+waterloo-fixed-rk4-flight"
  : mode === "delivery"
    ? "rate-of-closure/react-rigid-impact+waterloo-fixed-rk4-flight"
    : "rate-of-closure/react-launch+waterloo-fixed-rk4-flight";

export const makeVariationExecutionMetadata = (
  plan: VariationPlanTs,
  provenance: PlanProducerProvenanceTs = REACT_DEFAULT_PROVENANCE,
): VariationExecutionMetadataTs => {
  if (!Number.isSafeInteger(plan.seed) || !Number.isSafeInteger(plan.nRuns)) {
    throw new Error("plan seed and nRuns must be safe integers");
  }
  return Object.freeze({
  schemaId: EXECUTION_METADATA_SCHEMA_ID,
  schemaVersion: EXECUTION_METADATA_SCHEMA_VERSION,
  planSha256: variationPlanSha256(plan),
  mode: plan.mode,
  flightModel: plan.flightModel,
  registrySchemaId: VARIABLE_REGISTRY_SCHEMA_ID,
  registrySchemaVersion: VARIABLE_REGISTRY_SCHEMA_VERSION,
  registrySha256: registryDigest(plan),
  resolvedVariables: snapshots(plan),
  rngIdentity: Object.freeze({
    algorithmId: "mulberry32-u32", algorithmVersion: 1,
    streamDerivationId: "xor-low32-safe-seed-fnv1a-utf16-spec-id",
    streamDerivationVersion: 1,
  }),
  implementationIdentity: Object.freeze({
    runtimeId: "rate-of-closure/react", runtimeVersion: 1,
    executorId: "react-inline-worker-variation", executorVersion: 1,
    solverId: solverId(plan.mode), solverVersion: 1,
  }),
  provenanceSha256: provenanceSha256(provenance),
  });
};

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
  rng_identity: {
    algorithm_id: metadata.rngIdentity.algorithmId,
    algorithm_version: metadata.rngIdentity.algorithmVersion,
    stream_derivation_id: metadata.rngIdentity.streamDerivationId,
    stream_derivation_version: metadata.rngIdentity.streamDerivationVersion,
  },
  implementation_identity: {
    runtime_id: metadata.implementationIdentity.runtimeId,
    runtime_version: metadata.implementationIdentity.runtimeVersion,
    executor_id: metadata.implementationIdentity.executorId,
    executor_version: metadata.implementationIdentity.executorVersion,
    solver_id: metadata.implementationIdentity.solverId,
    solver_version: metadata.implementationIdentity.solverVersion,
  },
  provenance_sha256: metadata.provenanceSha256,
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
  const provenanceDigest = text(item.provenance_sha256, "provenance_sha256");
  if (![planSha256, registrySha256, provenanceDigest].every(
    (value) => /^[0-9a-f]{64}$/.test(value),
  )) {
    throw new Error("metadata digests must be lowercase SHA-256");
  }
  const rng = exactRecord(item.rng_identity, RNG_FIELDS, "rng_identity");
  const implementation = exactRecord(
    item.implementation_identity, IMPLEMENTATION_FIELDS, "implementation_identity",
  );
  return Object.freeze({
    schemaId: text(item.schema_id, "metadata schema_id") as typeof EXECUTION_METADATA_SCHEMA_ID,
    schemaVersion: integer(item.schema_version, "metadata schema_version") as 3,
    planSha256,
    mode: text(item.mode, "metadata mode") as VariationMode,
    flightModel: text(item.flight_model, "metadata flight_model"),
    registrySchemaId: text(
      item.registry_schema_id, "registry schema_id",
    ) as typeof VARIABLE_REGISTRY_SCHEMA_ID,
    registrySchemaVersion: integer(item.registry_schema_version, "registry schema_version") as 1,
    registrySha256,
    resolvedVariables,
    rngIdentity: Object.freeze({
      algorithmId: text(rng.algorithm_id, "RNG algorithm_id"),
      algorithmVersion: integer(rng.algorithm_version, "RNG algorithm_version"),
      streamDerivationId: text(rng.stream_derivation_id, "RNG stream_derivation_id"),
      streamDerivationVersion: integer(rng.stream_derivation_version, "RNG stream_derivation_version"),
    }),
    implementationIdentity: Object.freeze({
      runtimeId: text(implementation.runtime_id, "runtime_id"),
      runtimeVersion: integer(implementation.runtime_version, "runtime_version"),
      executorId: text(implementation.executor_id, "executor_id"),
      executorVersion: integer(implementation.executor_version, "executor_version"),
      solverId: text(implementation.solver_id, "solver_id"),
      solverVersion: integer(implementation.solver_version, "solver_version"),
    }),
    provenanceSha256: provenanceDigest,
  });
};

export const validateVariationExecutionMetadata = (
  plan: VariationPlanTs, metadata: VariationExecutionMetadataTs,
  provenance: PlanProducerProvenanceTs = REACT_DEFAULT_PROVENANCE,
): VariationExecutionMetadataTs => {
  const runtime = exactRecord(metadata, RUNTIME_METADATA_FIELDS, "metadata");
  if (!Array.isArray(runtime.resolvedVariables)) {
    throw new Error("metadata resolvedVariables must be an array");
  }
  runtime.resolvedVariables.forEach((item, index) => {
    exactRecord(item, RUNTIME_VARIABLE_FIELDS, `metadata resolvedVariables[${index}]`);
  });
  exactRecord(runtime.rngIdentity, RUNTIME_RNG_FIELDS, "metadata rngIdentity");
  exactRecord(runtime.implementationIdentity, RUNTIME_IMPLEMENTATION_FIELDS, "metadata implementationIdentity");
  const expected = makeVariationExecutionMetadata(plan, provenance);
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
  if (JSON.stringify(metadata.rngIdentity) !== JSON.stringify(expected.rngIdentity)) {
    throw new Error("RNG identity mismatch");
  }
  if (JSON.stringify(metadata.implementationIdentity) !== JSON.stringify(expected.implementationIdentity)) {
    throw new Error("implementation identity mismatch");
  }
  if (metadata.provenanceSha256 !== expected.provenanceSha256) {
    throw new Error("provenance digest mismatch");
  }
  return metadata;
};

export const resolveVariationExecutionMetadata = (
  plan: VariationPlanTs, metadata: VariationExecutionMetadataTs | null | undefined,
  provenance: PlanProducerProvenanceTs = REACT_DEFAULT_PROVENANCE,
): VariationExecutionResolutionTs => metadata === null || metadata === undefined
  ? { metadata: makeVariationExecutionMetadata(plan, provenance), warning: LEGACY_CURRENT_REGISTRY_WARNING }
  : { metadata: validateVariationExecutionMetadata(plan, metadata, provenance), warning: null };

export const variationExecutionDocument = (
  plan: VariationPlanTs, supplied?: VariationExecutionMetadataTs,
  provenance: PlanProducerProvenanceTs = REACT_DEFAULT_PROVENANCE,
): ExecutionDocumentWire => {
  const { metadata } = resolveVariationExecutionMetadata(plan, supplied, provenance);
  return {
    schema_id: EXECUTION_DOCUMENT_SCHEMA_ID,
    schema_version: EXECUTION_DOCUMENT_SCHEMA_VERSION,
    plan: canonicalPlanWire(plan),
    metadata: metadataWire(metadata),
    provenance: provenanceToWire(provenance),
  };
};

export const parseVariationExecutionDocument = (
  textValue: string,
): ParsedVariationExecutionDocumentTs => {
  const candidate = parseUniqueJson(textValue, "variation execution document");
  if (isRecord(candidate) && candidate.schema_id === EXECUTION_DOCUMENT_SCHEMA_ID &&
      (candidate.schema_version === 1 || candidate.schema_version === 2)) {
    throw new Error(LEGACY_EXECUTION_DOCUMENT_MIGRATION_ERROR);
  }
  const root = exactRecord(candidate, DOCUMENT_FIELDS, "execution document");
  if (root.schema_id !== EXECUTION_DOCUMENT_SCHEMA_ID) {
    throw new Error("execution document schema mismatch");
  }
  if (root.schema_version !== EXECUTION_DOCUMENT_SCHEMA_VERSION) {
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
  const provenance = provenanceFromWire(root.provenance);
  return {
    plan,
    metadata: validateVariationExecutionMetadata(plan, metadata, provenance),
    provenance,
    warning: null,
  };
};

const stableJsonValue = (value: unknown): unknown => {
  if (typeof value === "number") return Object.is(value, -0) ? 0 : value;
  if (Array.isArray(value)) return value.map(stableJsonValue);
  if (isRecord(value)) return Object.fromEntries(
    Object.keys(value).sort().map((key) => [key, stableJsonValue(value[key])]),
  );
  return value;
};

/** Serialize one canonical document with recursive key order and no whitespace. */
export const variationExecutionDocumentJson = (
  value: VariationPlanTs | ParsedVariationExecutionDocumentTs,
): string => {
  const document = "plan" in value
    ? variationExecutionDocument(value.plan, value.metadata, value.provenance)
    : variationExecutionDocument(value);
  return JSON.stringify(stableJsonValue(document));
};
