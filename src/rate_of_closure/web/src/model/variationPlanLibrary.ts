import { planFromJson, planToJson, type VariationPlanTs } from "./variationSchema";
import { parseUniqueJson } from "./strictJson";
import {
  parsePersistedVariationPlan,
  persistedVariationPlanJson,
  type PersistedVariationPlanResolutionTs,
} from "./variationPersistedPlan";
import { variationExecutionDocument } from "./variationExecutionMetadata";

export const VARIATION_PLAN_LIBRARY_KEY = "rate_of_closure.variation_plan_library";
export const VARIATION_PLAN_LIBRARY_VERSION = 2;

export interface NamedVariationPlan {
  id: string;
  name: string;
  plan: VariationPlanTs;
  evidence?: PersistedVariationPlanResolutionTs;
}

export interface VariationPlanLibraryLoad {
  plans: NamedVariationPlan[];
  warnings: string[];
}

const storageOrNull = (storage?: Storage): Storage | null => {
  try {
    const candidate = storage ?? (typeof window === "undefined" ? null : window.localStorage);
    return candidate !== null &&
      typeof candidate.getItem === "function" &&
      typeof candidate.setItem === "function"
      ? candidate
      : null;
  } catch {
    return null;
  }
};

const requiredText = (value: unknown, label: string): string => {
  if (typeof value !== "string" || value.trim().length === 0 || value !== value.trim()) {
    throw new Error(`${label} must be non-empty, trimmed text`);
  }
  return value;
};

const canonicalPlan = (plan: VariationPlanTs): VariationPlanTs =>
  planFromJson(planToJson(plan));

const exactFields = (data: Record<string, unknown>, expected: readonly string[]): void => {
  if (JSON.stringify(Object.keys(data).sort()) !== JSON.stringify([...expected].sort())) {
    throw new Error("entry fields do not match the library schema");
  }
};

const parseEntry = (value: unknown, version: number): NamedVariationPlan => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("entry must be an object");
  }
  const data = value as Record<string, unknown>;
  if (version === 1) {
    exactFields(data, ["id", "name", "plan"]);
    const evidence = parsePersistedVariationPlan(JSON.stringify(data.plan));
    return {
      id: requiredText(data.id, "plan ID"),
      name: requiredText(data.name, "plan name"),
      plan: evidence.plan,
      evidence,
    };
  }
  const canonical = "plan_document" in data;
  exactFields(data, canonical
    ? ["id", "name", "plan_document"]
    : ["id", "name", "legacy_plan", "legacy_warning"]);
  const evidence = parsePersistedVariationPlan(JSON.stringify(
    canonical ? data.plan_document : data.legacy_plan,
  ));
  if (!canonical && requiredText(data.legacy_warning, "legacy warning") !== evidence.warning) {
    throw new Error("legacy warning does not match the plan evidence");
  }
  return {
    id: requiredText(data.id, "plan ID"),
    name: requiredText(data.name, "plan name"),
    plan: evidence.plan,
    evidence,
  };
};

/** Load valid entries while isolating corrupt wrappers or partial entries. */
export function loadVariationPlanLibrary(storage?: Storage): VariationPlanLibraryLoad {
  const target = storageOrNull(storage);
  if (target === null) return { plans: [], warnings: [] };
  let text: string | null;
  try {
    text = target.getItem(VARIATION_PLAN_LIBRARY_KEY);
  } catch (error) {
    return {
      plans: [],
      warnings: [`Stored plan library could not be read: ${(error as Error).message}`],
    };
  }
  if (text === null) return { plans: [], warnings: [] };

  let data: Record<string, unknown>;
  try {
    const parsed = parseUniqueJson(text, "variation plan library");
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("library root must be an object");
    }
    data = parsed as Record<string, unknown>;
  } catch (error) {
    return {
      plans: [],
      warnings: [`Stored plan library is corrupt and was ignored: ${(error as Error).message}`],
    };
  }

  if (data.schema_version !== 1 && data.schema_version !== VARIATION_PLAN_LIBRARY_VERSION) {
    return {
      plans: [],
      warnings: [`Unsupported plan library version ${String(data.schema_version)}; stored plans were ignored.`],
    };
  }
  if (!Array.isArray(data.plans)) {
    return { plans: [], warnings: ["Stored plan library has no valid plans array."] };
  }

  const plans: NamedVariationPlan[] = [];
  const warnings: string[] = [];
  const ids = new Set<string>();
  data.plans.forEach((value, index) => {
    try {
      const entry = parseEntry(value, data.schema_version as number);
      if (ids.has(entry.id)) throw new Error(`duplicate plan ID ${entry.id}`);
      ids.add(entry.id);
      plans.push(entry);
      if (entry.evidence?.warning !== null && entry.evidence?.warning !== undefined) {
        warnings.push(`Stored plan ${index + 1} is legacy: ${entry.evidence.warning}`);
      }
    } catch (error) {
      warnings.push(`Stored plan ${index + 1} was ignored: ${(error as Error).message}`);
    }
  });
  return { plans, warnings };
}

/** Persist each named plan with its canonical binding or explicit legacy state. */
export function saveVariationPlanLibrary(
  plans: readonly NamedVariationPlan[],
  storage?: Storage,
): void {
  const target = storageOrNull(storage);
  if (target === null) return;
  const ids = new Set<string>();
  const payload = plans.map((entry) => {
    const id = requiredText(entry.id, "plan ID");
    if (ids.has(id)) throw new Error(`duplicate plan ID ${id}`);
    ids.add(id);
    const name = requiredText(entry.name, "plan name");
    const evidence = entry.evidence ?? parsePersistedVariationPlan(
      persistedVariationPlanJson(entry.plan),
    );
    if (planToJson(evidence.plan) !== planToJson(entry.plan)) {
      throw new Error(`plan evidence mismatch for ${id}`);
    }
    if (evidence.metadata !== null && evidence.provenance !== null) {
      return {
        id,
        name,
        plan_document: variationExecutionDocument(
          entry.plan, evidence.metadata, evidence.provenance,
        ),
      };
    }
    return {
      id,
      name,
      legacy_plan: JSON.parse(planToJson(entry.plan)) as unknown,
      legacy_warning: requiredText(evidence.warning, "legacy warning"),
    };
  });
  target.setItem(
    VARIATION_PLAN_LIBRARY_KEY,
    JSON.stringify({ schema_version: VARIATION_PLAN_LIBRARY_VERSION, plans: payload }),
  );
}

export function upsertVariationPlan(
  plans: readonly NamedVariationPlan[],
  entry: NamedVariationPlan,
): NamedVariationPlan[] {
  const plan = canonicalPlan(entry.plan);
  const normalized = {
    ...entry,
    plan,
    evidence: parsePersistedVariationPlan(persistedVariationPlanJson(plan)),
  };
  const existing = plans.findIndex((plan) => plan.id === entry.id);
  if (existing < 0) return [...plans, normalized];
  return plans.map((plan, index) => (index === existing ? normalized : plan));
}

export function duplicateVariationPlan(
  plans: readonly NamedVariationPlan[],
  sourceId: string,
  duplicateId: string,
): NamedVariationPlan[] {
  if (plans.some((plan) => plan.id === duplicateId)) {
    throw new Error(`duplicate plan ID ${duplicateId}`);
  }
  const source = plans.find((plan) => plan.id === sourceId);
  if (source === undefined) throw new Error(`unknown plan ID ${sourceId}`);
  return [
    ...plans,
    {
      id: requiredText(duplicateId, "plan ID"),
      name: `${source.name} Copy`,
      plan: canonicalPlan(source.plan),
      evidence: source.evidence,
    },
  ];
}

export const deleteVariationPlan = (
  plans: readonly NamedVariationPlan[],
  id: string,
): NamedVariationPlan[] => plans.filter((plan) => plan.id !== id);
