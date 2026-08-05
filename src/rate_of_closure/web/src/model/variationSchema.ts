import { validateGroupMatrix } from "./variationGroups";
import { keysForMode, type VariationMode } from "./variationRegistry";

export const SCHEMA_VERSION = 2;
export const MAX_RUNS = 500;

export type Distribution = "normal" | "uniform" | "triangular";

export interface NoiseSpecTs {
  variableKey: string;
  distribution: Distribution;
  scale: number;
  lower: number | null;
  upper: number | null;
  /** Stable RNG/group identifier. Defaults to variableKey for v1 plans. */
  specId?: string;
  /** Optional temporal locus; metadata only until a trajectory evaluator is available. */
  timeWindowS?: [number, number] | null;
  /** Optional spatial locus; metadata only until a trajectory evaluator is available. */
  pointIds?: string[];
}

export type MatrixKindTs = "correlation" | "covariance";

export interface PerturbationGroupTs {
  groupId: string;
  specIds: string[];
  matrix: number[][];
  matrixKind: MatrixKindTs;
}

export interface VariationPlanTs {
  mode: VariationMode;
  baseVariables: Record<string, number>;
  noise: NoiseSpecTs[];
  nRuns: number;
  seed: number;
  flightModel: string;
  /** Optional for source compatibility with v1 callers; normalized to [] on import. */
  groups?: PerturbationGroupTs[];
}

export const stableSpecId = (spec: NoiseSpecTs): string =>
  spec.specId ?? spec.variableKey;

const isStableId = (value: string): boolean =>
  value.length > 0 && value.trim() === value;

export const isGlobalSpec = (spec: NoiseSpecTs): boolean =>
  (spec.timeWindowS === undefined || spec.timeWindowS === null) &&
  (spec.pointIds?.length ?? 0) === 0;

const validateNoiseSpec = (
  spec: NoiseSpecTs,
  mode: VariationMode,
  legal: Set<string>,
): void => {
  if (!legal.has(spec.variableKey)) {
    throw new Error(`noise variable not legal in ${mode} mode: ${spec.variableKey}`);
  }
  if (!isStableId(stableSpecId(spec))) {
    throw new Error("specId must be a non-empty, trimmed stable ID");
  }
  if (!(<string[]>["normal", "uniform", "triangular"]).includes(spec.distribution)) {
    throw new Error(`unsupported distribution: ${String(spec.distribution)}`);
  }
  if (!(spec.scale > 0) || !Number.isFinite(spec.scale)) {
    throw new Error(`scale for ${spec.variableKey} must be finite and > 0`);
  }
  if (spec.lower !== null && !Number.isFinite(spec.lower)) {
    throw new Error(`lower for ${spec.variableKey} must be finite when given`);
  }
  if (spec.upper !== null && !Number.isFinite(spec.upper)) {
    throw new Error(`upper for ${spec.variableKey} must be finite when given`);
  }
  if (spec.lower !== null && spec.upper !== null && !(spec.lower < spec.upper)) {
    throw new Error(`truncation bounds for ${spec.variableKey} must be lower < upper`);
  }
  if (spec.timeWindowS !== undefined && spec.timeWindowS !== null) {
    const [start, end] = spec.timeWindowS;
    if (
      spec.timeWindowS.length !== 2 ||
      !Number.isFinite(start) ||
      !Number.isFinite(end) ||
      !(start < end)
    ) {
      throw new Error("timeWindowS must contain finite start < end");
    }
  }
  const pointIds = spec.pointIds ?? [];
  if (
    pointIds.some((pointId) => !isStableId(pointId)) ||
    new Set(pointIds).size !== pointIds.length
  ) {
    throw new Error("pointIds must be unique, non-empty stable IDs");
  }
};

const validateGroups = (
  groups: PerturbationGroupTs[],
  specsById: Map<string, NoiseSpecTs>,
): void => {
  const groupIds = new Set<string>();
  const assignedSpecIds = new Set<string>();
  for (const group of groups) {
    if (!isStableId(group.groupId)) {
      throw new Error("groupId must be a non-empty, trimmed stable ID");
    }
    if (groupIds.has(group.groupId)) throw new Error(`duplicate groupId: ${group.groupId}`);
    groupIds.add(group.groupId);
    if (
      group.specIds.some((specId) => !isStableId(specId)) ||
      new Set(group.specIds).size !== group.specIds.length
    ) {
      throw new Error("specIds must be unique, non-empty stable IDs");
    }
    validateGroupMatrix(group);
    for (const specId of group.specIds) {
      const spec = specsById.get(specId);
      if (spec === undefined) throw new Error(`group references unknown specId: ${specId}`);
      if (assignedSpecIds.has(specId)) {
        throw new Error(`a specId may belong to only one group: ${specId}`);
      }
      if (spec.distribution !== "normal") {
        throw new Error("grouped specs must use normal distributions");
      }
      assignedSpecIds.add(specId);
    }
    if (group.matrixKind === "covariance") {
      group.specIds.forEach((specId, index) => {
        const expected = specsById.get(specId)!.scale ** 2;
        const actual = group.matrix[index][index];
        if (Math.abs(actual - expected) > 1e-12 + 1e-9 * Math.abs(expected)) {
          throw new Error("covariance diagonal must equal each NoiseSpec scale squared");
        }
      });
    }
  }
};

/** DbC-style validation mirroring the Python variation plan. */
export function validatePlan(plan: VariationPlanTs): void {
  if (plan.mode !== "delivery" && plan.mode !== "launch") {
    throw new Error(`mode ${plan.mode} is not supported in the browser (desktop-only)`);
  }
  if (!Number.isInteger(plan.nRuns) || plan.nRuns < 2 || plan.nRuns > MAX_RUNS) {
    throw new Error(`nRuns must be an integer in [2, ${MAX_RUNS}]`);
  }
  if (!Number.isInteger(plan.seed) || plan.seed < 0) {
    throw new Error("seed must be a non-negative integer");
  }
  if (plan.noise.length === 0) throw new Error("plan must vary at least one variable");

  const legal = new Set(keysForMode(plan.mode));
  const seenVariables = new Set<string>();
  const specsById = new Map<string, NoiseSpecTs>();
  for (const spec of plan.noise) {
    validateNoiseSpec(spec, plan.mode, legal);
    if (seenVariables.has(spec.variableKey)) {
      throw new Error(`duplicate noise spec for ${spec.variableKey}`);
    }
    seenVariables.add(spec.variableKey);
    const specId = stableSpecId(spec);
    if (specsById.has(specId)) throw new Error(`duplicate specId: ${specId}`);
    specsById.set(specId, spec);
  }
  for (const [key, value] of Object.entries(plan.baseVariables)) {
    if (!legal.has(key)) {
      throw new Error(`base variable not legal in ${plan.mode} mode: ${key}`);
    }
    if (!Number.isFinite(value)) throw new Error(`base value must be finite: ${key}`);
  }
  validateGroups(plan.groups ?? [], specsById);
}

/** Serialize the canonical snake_case schema shared with Python. */
export function planToJson(plan: VariationPlanTs): string {
  validatePlan(plan);
  return JSON.stringify(
    {
      schema_version: SCHEMA_VERSION,
      mode: plan.mode,
      base_variables: plan.baseVariables,
      noise: plan.noise.map((spec) => ({
        variable_key: spec.variableKey,
        distribution: spec.distribution,
        scale: spec.scale,
        lower: spec.lower,
        upper: spec.upper,
        spec_id: stableSpecId(spec),
        time_window_s: spec.timeWindowS ?? null,
        point_ids: spec.pointIds ?? [],
      })),
      n_runs: plan.nRuns,
      seed: plan.seed,
      flight_model: plan.flightModel,
      groups: (plan.groups ?? []).map((group) => ({
        group_id: group.groupId,
        spec_ids: group.specIds,
        matrix_kind: group.matrixKind,
        matrix: group.matrix,
      })),
    },
    null,
    2,
  );
}

/** Parse schema v2 or migrate a schema-v1 plan into normalized model fields. */
export function planFromJson(text: string): VariationPlanTs {
  const data = JSON.parse(text) as Record<string, unknown>;
  const version = Number(data.schema_version ?? 1);
  if (version !== 1 && version !== SCHEMA_VERSION) {
    throw new Error(`unsupported schema_version ${version}`);
  }
  const noiseRaw = (data.noise ?? []) as Array<Record<string, unknown>>;
  const groupsRaw = version === 1
    ? []
    : ((data.groups ?? []) as Array<Record<string, unknown>>);
  const baseRaw = (data.base_variables ?? {}) as Record<string, unknown>;
  const plan: VariationPlanTs = {
    mode: String(data.mode) as VariationMode,
    baseVariables: Object.fromEntries(
      Object.entries(baseRaw).map(([key, value]) => [key, Number(value)]),
    ),
    noise: noiseRaw.map((spec) => ({
      variableKey: String(spec.variable_key),
      distribution: String(spec.distribution ?? "normal") as Distribution,
      scale: Number(spec.scale ?? 1),
      lower: spec.lower === null || spec.lower === undefined ? null : Number(spec.lower),
      upper: spec.upper === null || spec.upper === undefined ? null : Number(spec.upper),
      specId: spec.spec_id === null || spec.spec_id === undefined
        ? String(spec.variable_key)
        : String(spec.spec_id),
      timeWindowS: spec.time_window_s === null || spec.time_window_s === undefined
        ? null
        : (Array.from(spec.time_window_s as Iterable<unknown>, Number) as [number, number]),
      pointIds: Array.from((spec.point_ids ?? []) as Iterable<unknown>, String),
    })),
    nRuns: Number(data.n_runs ?? 200),
    seed: Number(data.seed ?? 0),
    flightModel: String(data.flight_model ?? "waterloo_penner"),
    groups: groupsRaw.map((group) => ({
      groupId: String(group.group_id),
      specIds: Array.from((group.spec_ids ?? []) as Iterable<unknown>, String),
      matrixKind: String(group.matrix_kind ?? "correlation") as MatrixKindTs,
      matrix: Array.from(group.matrix as Iterable<Iterable<unknown>>, (row) =>
        Array.from(row, Number),
      ),
    })),
  };
  validatePlan(plan);
  return plan;
}
