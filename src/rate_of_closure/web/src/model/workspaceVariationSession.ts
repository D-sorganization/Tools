/** Strict workspace selection around the canonical variation plan. */

import {
  outputsForMode,
  planFromJson,
  planToJson,
  validatePlan,
  type VariationPlanTs,
} from "./variation";
import { ballSetupToJson, type BallSetup } from "./ballSetup";
import {
  parsePersistedVariationPlan,
  parsePersistedVariationPlanBinding,
  persistedVariationPlanBinding,
  persistedVariationPlanJson,
  type PersistedVariationPlanResolutionTs,
} from "./variationPersistedPlan";

export const VARIATION_WORKSPACE_SCHEMA =
  "rate_of_closure.variation_workspace_selection";
export const VARIATION_WORKSPACE_SCHEMA_VERSION = 1;

export type VariationAnalysisExecution = "all_together" | "individual" | "both";

export interface VariationWorkspaceSnapshot {
  readonly plan: VariationPlanTs;
  readonly analysisExecution: VariationAnalysisExecution;
  readonly selectedOutputMetrics: readonly string[];
  readonly planEvidence?: PersistedVariationPlanResolutionTs;
}

function canonicalPlan(
  plan: VariationPlanTs,
  ballSetup?: BallSetup,
): VariationPlanTs {
  const { ballSetup: embeddedSetup, ...rootPlan } = plan;
  const context = ballSetup ?? embeddedSetup;
  const contextual =
    context === undefined ? rootPlan : { ...rootPlan, ballSetup: context };
  validatePlan(contextual);
  const normalized = planFromJson(planToJson(contextual));
  const canonical = { ...normalized };
  delete canonical.ballSetup;
  return canonical;
}

function record(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${context} must be an object`);
  }
  return value as Record<string, unknown>;
}

function exactRecord(
  value: unknown,
  fields: readonly string[],
  context: string,
): Record<string, unknown> {
  const data = record(value, context);
  const actual = Object.keys(data);
  if (
    actual.length !== fields.length ||
    fields.some((field) => !(field in data))
  ) {
    throw new TypeError(`${context} has invalid fields`);
  }
  return data;
}

/** Validate and normalize a complete authored variation workspace. */
export function validatedVariationWorkspace(
  snapshot: VariationWorkspaceSnapshot,
  ballSetup?: BallSetup,
): VariationWorkspaceSnapshot {
  const plan = canonicalPlan(snapshot.plan, ballSetup);
  if (
    !["all_together", "individual", "both"].includes(snapshot.analysisExecution)
  ) {
    throw new RangeError("unsupported variation analysis execution");
  }
  if (
    !Array.isArray(snapshot.selectedOutputMetrics) ||
    snapshot.selectedOutputMetrics.length === 0 ||
    snapshot.selectedOutputMetrics.some((metric) => typeof metric !== "string")
  ) {
    throw new TypeError("selected output metrics must be non-empty strings");
  }
  if (
    new Set(snapshot.selectedOutputMetrics).size !==
    snapshot.selectedOutputMetrics.length
  ) {
    throw new RangeError("selected output metrics must be unique");
  }
  const available = outputsForMode(plan.mode);
  const unknown = snapshot.selectedOutputMetrics.filter(
    (metric) => !available.includes(metric),
  );
  if (unknown.length > 0) {
    throw new RangeError(
      `selected output metric is not available: ${unknown.join(", ")}`,
    );
  }
  const evidence = snapshot.planEvidence ?? parsePersistedVariationPlan(
    persistedVariationPlanJson(plan),
  );
  if (planToJson(evidence.plan) !== planToJson(plan)) {
    throw new RangeError("plan evidence does not match the authored plan");
  }
  return Object.freeze({
    plan,
    analysisExecution: snapshot.analysisExecution,
    selectedOutputMetrics: Object.freeze(
      available.filter((metric) =>
        snapshot.selectedOutputMetrics.includes(metric),
      ),
    ),
    planEvidence: evidence,
  });
}

/** Serialize selection only; the canonical plan remains at the root. */
export function variationWorkspaceDocument(
  snapshot: VariationWorkspaceSnapshot,
  ballSetup?: BallSetup,
): Record<string, unknown> {
  const state = validatedVariationWorkspace(snapshot, ballSetup);
  return {
    schema: VARIATION_WORKSPACE_SCHEMA,
    schema_version: VARIATION_WORKSPACE_SCHEMA_VERSION,
    data: {
      analysis_execution: state.analysisExecution,
      selected_output_metrics: [...state.selectedOutputMetrics],
    },
  };
}

/** Parse a strict selection against its canonical root plan. */
export function variationWorkspaceFromDocument(
  value: unknown,
  plan: VariationPlanTs,
  planEvidence: PersistedVariationPlanResolutionTs,
  ballSetup?: BallSetup,
): VariationWorkspaceSnapshot {
  const envelope = exactRecord(
    value,
    ["schema", "schema_version", "data"],
    "variation workspace",
  );
  if (
    envelope.schema !== VARIATION_WORKSPACE_SCHEMA ||
    envelope.schema_version !== VARIATION_WORKSPACE_SCHEMA_VERSION
  ) {
    throw new RangeError("unsupported variation workspace selection payload");
  }
  const data = exactRecord(
    envelope.data,
    ["analysis_execution", "selected_output_metrics"],
    "variation workspace.data",
  );
  if (!Array.isArray(data.selected_output_metrics)) {
    throw new TypeError("selected_output_metrics must be a JSON array");
  }
  return validatedVariationWorkspace(
    {
      plan,
      analysisExecution: data.analysis_execution as VariationAnalysisExecution,
      selectedOutputMetrics: data.selected_output_metrics as unknown[],
      planEvidence,
    } as VariationWorkspaceSnapshot,
    ballSetup,
  );
}

/** Preserve explicit live policy unless a legacy root plan conflicts. */
export function migratedLegacyVariationFallback(
  fallback: VariationWorkspaceSnapshot,
  documentPlan: PersistedVariationPlanResolutionTs | null,
  ballSetup?: BallSetup,
): VariationWorkspaceSnapshot {
  if (
    documentPlan !== null &&
    planToJson(canonicalPlan(documentPlan.plan, ballSetup)) !==
      planToJson(canonicalPlan(fallback.plan, ballSetup))
  ) {
    throw new RangeError(
      "legacy workspace variation plan conflicts with the explicit fallback",
    );
  }
  const state = validatedVariationWorkspace(fallback, ballSetup);
  return documentPlan === null ? state : validatedVariationWorkspace({
    ...state,
    planEvidence: documentPlan,
  }, ballSetup);
}

/** Return a Python-compatible root plan while validating Tee context separately. */
export function variationPlanWorkspaceDocument(
  snapshot: VariationWorkspaceSnapshot,
  ballSetup: BallSetup,
): Record<string, unknown> {
  const state = validatedVariationWorkspace(snapshot, ballSetup);
  return persistedVariationPlanBinding(state.planEvidence ?? state.plan);
}

/** Parse a Python-compatible root plan using the separately persisted setup. */
export function variationPlanFromWorkspaceDocument(
  value: unknown,
  ballSetup: BallSetup,
  legacyRaw = false,
): PersistedVariationPlanResolutionTs {
  if (!legacyRaw) {
    const binding = record(value, "variation_plan binding");
    const document = binding.document === null
      ? null
      : record(binding.document, "variation_plan document");
    const boundPlan = document === null
      ? null
      : record(document.plan, "variation_plan document plan");
    if (boundPlan !== null && "ball_setup" in boundPlan) {
      throw new TypeError("variation_plan must not duplicate simulation ball_setup");
    }
    const resolution = parsePersistedVariationPlanBinding(value);
    const plan = canonicalPlan(resolution.plan, ballSetup);
    if (planToJson(plan) !== planToJson(resolution.plan)) {
      throw new TypeError("variation plan binding conflicts with simulation ball_setup");
    }
    return resolution;
  }
  const data = record(value, "variation_plan");
  if ("ball_setup" in data) {
    throw new TypeError(
      "variation_plan must not duplicate simulation ball_setup",
    );
  }
  const contextual = {
    ...data,
    ball_setup: ballSetupToJson(ballSetup),
  };
  const plan = canonicalPlan(planFromJson(JSON.stringify(contextual)), ballSetup);
  return parsePersistedVariationPlan(planToJson(plan));
}
