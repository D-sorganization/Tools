import matrixDocument from "./__fixtures__/variation_visual_state_matrix_v1.json";
import { planToJson, type VariationPlanTs } from "./variation";
import type { VariationAnalysisExecution } from "./variationAnalysisPolicy";
import { resolvedBase } from "./variationSampling";
import { defaultSwingVariationInput } from "./variationSwingInput";

export type VariationVisualEvent = "invalidate" | "start-empty" | "start-retained" |
  "succeed" | "fail-empty" | "fail-retained" | "cancel-empty" | "cancel-retained";
export type VariationVisualPhase = "empty" | "loading" | "result" | "error";
export type VariationVisualOrigin = "empty-preview" | "prior-accepted" | "current-accepted";
export type AnnouncementRole = "status" | "alert";

export interface VariationVisualState {
  readonly phase: VariationVisualPhase;
  readonly visualOrigin: VariationVisualOrigin;
  readonly announcementRole: AnnouncementRole;
}

const ROOT_KEYS = ["schema_id", "schema_version", "states"] as const;
const ROW_KEYS = ["event", "phase", "visual_origin", "announcement_role"] as const;
const EVENTS: readonly VariationVisualEvent[] = [
  "invalidate", "start-empty", "start-retained", "succeed",
  "fail-empty", "fail-retained", "cancel-empty", "cancel-retained",
];
const PHASES: readonly VariationVisualPhase[] = ["empty", "loading", "result", "error"];
const ORIGINS: readonly VariationVisualOrigin[] = [
  "empty-preview", "prior-accepted", "current-accepted",
];
const ROLES: readonly AnnouncementRole[] = ["status", "alert"];

const exactRecord = (
  value: unknown, keys: readonly string[], context: string,
): Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${context} must be an object`);
  }
  const result = value as Record<string, unknown>;
  if (Object.keys(result).sort().join("|") !== [...keys].sort().join("|")) {
    throw new TypeError(`${context} has unknown or missing fields`);
  }
  return result;
};

const member = <T extends string>(
  value: unknown, domain: readonly T[], context: string,
): T => {
  if (typeof value !== "string" || !domain.includes(value as T)) {
    throw new RangeError(`${context} is unsupported`);
  }
  return value as T;
};

export function parseVariationVisualStateMatrix(
  document: unknown,
): ReadonlyArray<readonly [VariationVisualEvent, VariationVisualState]> {
  const root = exactRecord(document, ROOT_KEYS, "visual state matrix");
  if (root.schema_id !== "rate-of-closure/variation-visual-state-matrix") {
    throw new RangeError("unsupported visual state matrix schema_id");
  }
  if (typeof root.schema_version !== "number" || !Number.isInteger(root.schema_version) ||
      root.schema_version !== 1) {
    throw new RangeError("unsupported visual state matrix schema_version");
  }
  if (!Array.isArray(root.states)) throw new TypeError("states must be an array");
  const result = new Map<VariationVisualEvent, VariationVisualState>();
  for (const raw of root.states) {
    const row = exactRecord(raw, ROW_KEYS, "visual state row");
    const event = member(row.event, EVENTS, "visual state event");
    if (result.has(event)) throw new RangeError(`duplicate visual state event: ${event}`);
    result.set(event, Object.freeze({
      phase: member(row.phase, PHASES, "visual state phase"),
      visualOrigin: member(row.visual_origin, ORIGINS, "visual state origin"),
      announcementRole: member(row.announcement_role, ROLES, "announcement role"),
    }));
  }
  if (result.size !== EVENTS.length) throw new RangeError("every visual event is required");
  return Object.freeze([...result.entries()].map(([event, state]) => (
    Object.freeze([event, state] as const)
  )));
}

const MATRIX = new Map(parseVariationVisualStateMatrix(matrixDocument));

export const variationVisualState = (event: VariationVisualEvent): VariationVisualState => {
  const state = MATRIX.get(event);
  if (state === undefined) throw new RangeError(`unsupported visual state event: ${event}`);
  return state;
};

export const variationExecutionIdentity = (
  plan: VariationPlanTs,
  analysisExecution: VariationAnalysisExecution,
): string => {
  try {
    return `valid:${JSON.stringify({
      authorityRevision: 1,
      plan: JSON.parse(planToJson(plan)) as unknown,
      resolvedBase: Object.fromEntries(Object.entries(resolvedBase(plan)).sort()),
      analysisExecution,
      swingBaseInput: plan.mode === "swing" ? defaultSwingVariationInput(plan.ballSetup) : null,
    })}`;
  } catch {
    try {
      return `invalid:${JSON.stringify({ plan, analysisExecution })}`;
    } catch {
      return "invalid:unserializable-editor-state";
    }
  }
};
