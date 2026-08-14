/** Strict workspace selection around the canonical variation plan. */

import {
  outputsForMode,
  planFromJson,
  planToJson,
  validatePlan,
  type VariationPlanTs,
} from "./variation";
import { ballSetupToJson, type BallSetup } from "./ballSetup";

export const VARIATION_WORKSPACE_SCHEMA =
  "rate_of_closure.variation_workspace_selection";
export const VARIATION_WORKSPACE_SCHEMA_VERSION = 1;

export type VariationAnalysisExecution = "all_together" | "individual" | "both";

export interface VariationWorkspaceSnapshot {
  readonly plan: VariationPlanTs;
  readonly analysisExecution: VariationAnalysisExecution;
  readonly selectedOutputMetrics: readonly string[];
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
  return Object.freeze({
    plan,
    analysisExecution: snapshot.analysisExecution,
    selectedOutputMetrics: Object.freeze(
      available.filter((metric) =>
        snapshot.selectedOutputMetrics.includes(metric),
      ),
    ),
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
    } as VariationWorkspaceSnapshot,
    ballSetup,
  );
}

/** Preserve explicit live policy unless a legacy root plan conflicts. */
export function migratedLegacyVariationFallback(
  fallback: VariationWorkspaceSnapshot,
  documentPlan: VariationPlanTs | null,
  ballSetup?: BallSetup,
): VariationWorkspaceSnapshot {
  const state = validatedVariationWorkspace(fallback, ballSetup);
  if (
    documentPlan !== null &&
    planToJson(canonicalPlan(documentPlan, ballSetup)) !==
      planToJson(state.plan)
  ) {
    throw new RangeError(
      "legacy workspace variation plan conflicts with the explicit fallback",
    );
  }
  return state;
}

/** Return a Python-compatible root plan while validating Tee context separately. */
export function variationPlanWorkspaceDocument(
  snapshot: VariationWorkspaceSnapshot,
  ballSetup: BallSetup,
): Record<string, unknown> {
  const state = validatedVariationWorkspace(snapshot, ballSetup);
  const contextual = JSON.parse(
    planToJson({ ...state.plan, ballSetup }),
  ) as Record<string, unknown>;
  delete contextual.ball_setup;
  return contextual;
}

/** Parse a Python-compatible root plan using the separately persisted setup. */
export function variationPlanFromWorkspaceDocument(
  value: unknown,
  ballSetup: BallSetup,
): VariationPlanTs {
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
  return canonicalPlan(planFromJson(JSON.stringify(contextual)), ballSetup);
}
