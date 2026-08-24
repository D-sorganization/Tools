/** Canonical variation-plan persistence with evidence-honest legacy migration. */

import { parseUniqueJson } from "./strictJson";
import {
  EXECUTION_DOCUMENT_SCHEMA_ID,
  EXECUTION_DOCUMENT_SCHEMA_VERSION,
  LEGACY_CURRENT_REGISTRY_WARNING,
  LEGACY_EXECUTION_DOCUMENT_MIGRATION_ERROR,
  parseVariationExecutionDocument,
  variationExecutionDocument,
  variationExecutionDocumentJson,
  type PlanProducerProvenanceTs,
  type VariationExecutionMetadataTs,
} from "./variationExecutionMetadata";
import { planFromJson, planToJson, type VariationPlanTs } from "./variationSchema";

export interface PersistedVariationPlanResolutionTs {
  readonly plan: VariationPlanTs;
  readonly metadata: VariationExecutionMetadataTs | null;
  readonly provenance: PlanProducerProvenanceTs | null;
  readonly warning: string | null;
}

export const PLAN_BINDING_SCHEMA_ID = "rate-of-closure/variation-plan-binding";
export const PLAN_BINDING_SCHEMA_VERSION = 1 as const;
const BINDING_FIELDS = [
  "schema_id", "schema_version", "state", "document", "legacy_plan", "legacy_warning",
] as const;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** Parse v3, or expose exactly which historical evidence a legacy plan lacks. */
export const parsePersistedVariationPlan = (
  textValue: string,
): PersistedVariationPlanResolutionTs => {
  const value = parseUniqueJson(textValue, "variation plan JSON");
  if (isRecord(value) && value.schema_id === EXECUTION_DOCUMENT_SCHEMA_ID) {
    if (value.schema_version === EXECUTION_DOCUMENT_SCHEMA_VERSION) {
      const document = parseVariationExecutionDocument(JSON.stringify(value));
      return document;
    }
    if (!isRecord(value.plan)) throw new Error("legacy execution document plan is missing");
    return Object.freeze({
      plan: planFromJson(JSON.stringify(value.plan)),
      metadata: null,
      provenance: null,
      warning: LEGACY_EXECUTION_DOCUMENT_MIGRATION_ERROR,
    });
  }
  return Object.freeze({
    plan: planFromJson(JSON.stringify(value)),
    metadata: null,
    provenance: null,
    warning: LEGACY_CURRENT_REGISTRY_WARNING,
  });
};

export const persistedVariationPlanJson = (plan: VariationPlanTs): string =>
  variationExecutionDocumentJson(plan);

const isResolution = (
  value: VariationPlanTs | PersistedVariationPlanResolutionTs,
): value is PersistedVariationPlanResolutionTs => "plan" in value && "warning" in value;

/** Retain either cohesive canonical evidence or an explicit legacy state. */
export const persistedVariationPlanBinding = (
  value: VariationPlanTs | PersistedVariationPlanResolutionTs,
): Record<string, unknown> => {
  const resolution = isResolution(value)
    ? value
    : parsePersistedVariationPlan(persistedVariationPlanJson(value));
  const canonical = resolution.metadata !== null && resolution.provenance !== null &&
    resolution.warning === null;
  if (!canonical && (resolution.metadata !== null || resolution.provenance !== null ||
      resolution.warning === null)) {
    throw new Error("partial plan evidence cannot be persisted");
  }
  return {
    schema_id: PLAN_BINDING_SCHEMA_ID,
    schema_version: PLAN_BINDING_SCHEMA_VERSION,
    state: canonical ? "canonical" : "legacy",
    document: canonical
      ? variationExecutionDocument(
        resolution.plan,
        resolution.metadata ?? undefined,
        resolution.provenance ?? undefined,
      )
      : null,
    legacy_plan: canonical ? null : JSON.parse(planToJson(resolution.plan)) as unknown,
    legacy_warning: canonical ? null : resolution.warning,
  };
};

/** Verify a persisted binding without resolving missing historical evidence. */
export const parsePersistedVariationPlanBinding = (
  value: unknown,
): PersistedVariationPlanResolutionTs => {
  if (!isRecord(value)) throw new Error("plan binding must be an object");
  const keys = Object.keys(value).sort();
  if (JSON.stringify(keys) !== JSON.stringify([...BINDING_FIELDS].sort())) {
    throw new Error("plan binding fields mismatch");
  }
  if (value.schema_id !== PLAN_BINDING_SCHEMA_ID ||
      value.schema_version !== PLAN_BINDING_SCHEMA_VERSION) {
    throw new Error("plan binding schema mismatch");
  }
  if (value.state === "canonical") {
    if (value.legacy_plan !== null || value.legacy_warning !== null) {
      throw new Error("canonical binding must not contain legacy evidence");
    }
    return parsePersistedVariationPlan(JSON.stringify(value.document));
  }
  if (value.state !== "legacy" || value.document !== null) {
    throw new Error("plan binding state mismatch");
  }
  const resolution = parsePersistedVariationPlan(JSON.stringify(value.legacy_plan));
  if (resolution.metadata !== null || resolution.provenance !== null ||
      value.legacy_warning !== resolution.warning) {
    throw new Error("legacy warning does not match the retained evidence");
  }
  return resolution;
};
