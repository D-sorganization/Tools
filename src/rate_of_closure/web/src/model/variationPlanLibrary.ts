import { planFromJson, planToJson, type VariationPlanTs } from "./variationSchema";

export const VARIATION_PLAN_LIBRARY_KEY = "rate_of_closure.variation_plan_library";
export const VARIATION_PLAN_LIBRARY_VERSION = 1;

export interface NamedVariationPlan {
  id: string;
  name: string;
  plan: VariationPlanTs;
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

const parseEntry = (value: unknown): NamedVariationPlan => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("entry must be an object");
  }
  const data = value as Record<string, unknown>;
  return {
    id: requiredText(data.id, "plan ID"),
    name: requiredText(data.name, "plan name"),
    plan: planFromJson(JSON.stringify(data.plan)),
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
    const parsed = JSON.parse(text) as unknown;
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

  if (data.schema_version !== VARIATION_PLAN_LIBRARY_VERSION) {
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
      const entry = parseEntry(value);
      if (ids.has(entry.id)) throw new Error(`duplicate plan ID ${entry.id}`);
      ids.add(entry.id);
      plans.push(entry);
    } catch (error) {
      warnings.push(`Stored plan ${index + 1} was ignored: ${(error as Error).message}`);
    }
  });
  return { plans, warnings };
}

/** Persist a validated canonical v2 snapshot of every named plan. */
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
    return {
      id,
      name: requiredText(entry.name, "plan name"),
      plan: JSON.parse(planToJson(entry.plan)) as unknown,
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
  const normalized = { ...entry, plan: canonicalPlan(entry.plan) };
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
    },
  ];
}

export const deleteVariationPlan = (
  plans: readonly NamedVariationPlan[],
  id: string,
): NamedVariationPlan[] => plans.filter((plan) => plan.id !== id);
