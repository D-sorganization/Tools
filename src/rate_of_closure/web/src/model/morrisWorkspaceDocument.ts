/** Strict, aggregate-only persistence for one Morris workspace. */

import { parseMorrisJobEnvelope, type MorrisJobEnvelope } from "./morrisAuthorityContract";
import {
  RATE_MORRIS_VARIABLE_KEYS,
  morrisAuthorityBaseIdentity,
  serializeMorrisAuthorityRequest,
  suggestedMorrisFactorDrafts,
  type MorrisAuthorityBase,
  type MorrisAuthorityRequestDocument,
  type MorrisFactorDraft,
} from "./morrisAuthorityRequest";
import { morrisJobToDocument } from "./morrisWireSerialization";
import { parseUniqueJson } from "./strictJson";

export const MORRIS_WORKSPACE_SCHEMA_ID = "rate-of-closure/morris-workspace" as const;
export const MORRIS_WORKSPACE_SCHEMA_VERSION = 1 as const;
export const MORRIS_WORKSPACE_EXPORT_SCOPE = "authority-base-and-morris-controls-only" as const;
export const MAX_MORRIS_WORKSPACE_BYTES = 2_000_000;
const MAX_JSON_DEPTH = 32;
const MAX_JSON_NODES = 25_000;
const MAX_BOUND_TEXT = 128;
const MAX_VALIDATION_TEXT = 256;
export const MAX_MORRIS_EDITOR_BOUND = 1_000_000_000;
export const INVALID_MORRIS_BOUNDS_MESSAGE = "Bounds must be finite numbers with lower < upper.";
const DECIMAL_NUMBER = /^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$/;
const MAX_MORRIS_SAMPLES = 100_000;
const MAX_MORRIS_OBSERVATION_CELLS = 1_000_000;
const MORRIS_TARGET_COUNT = 17;
const containsControlCharacter = (value: string): boolean => Array.from(value).some((character) => {
  const codePoint = character.codePointAt(0) ?? 0;
  return codePoint <= 0x1f || (codePoint >= 0x7f && codePoint <= 0x9f);
});

export interface MorrisDesignControls {
  readonly trajectories: number;
  readonly levels: number;
  readonly seed: number;
  readonly minimumEffects: number;
  readonly workerCount: number;
}
export interface MorrisWorkspaceFactorDraft {
  readonly variableKey: string;
  readonly enabled: boolean;
  readonly lower: string;
  readonly upper: string;
  readonly validationError: string | null;
}
export interface MorrisWorkspaceSetup extends MorrisDesignControls {
  readonly exportScope: typeof MORRIS_WORKSPACE_EXPORT_SCOPE;
  readonly base: MorrisAuthorityBase;
  readonly factorDrafts: readonly MorrisWorkspaceFactorDraft[];
}
export interface MorrisCompletedEvidence {
  readonly request: MorrisAuthorityRequestDocument;
  readonly job: MorrisJobEnvelope;
}
export interface MorrisWorkspaceDocument {
  readonly schemaId: typeof MORRIS_WORKSPACE_SCHEMA_ID;
  readonly schemaVersion: typeof MORRIS_WORKSPACE_SCHEMA_VERSION;
  readonly setup: MorrisWorkspaceSetup;
  readonly completedEvidence: MorrisCompletedEvidence | null;
}

const ROOT_FIELDS = ["schema_id", "schema_version", "setup", "completed_evidence"] as const;
const SETUP_FIELDS = ["export_scope", "base", "factor_drafts", "trajectories", "levels", "seed", "minimum_effects", "worker_count"] as const;
const DRAFT_FIELDS = ["variable_key", "enabled", "lower", "upper", "validation_error"] as const;
const EVIDENCE_FIELDS = ["request", "job"] as const;
const BASE_FIELDS = ["club_name", "support_mode", "tee_height_m", "plane_yaw_deg", "plane_side_tilt_deg", "plane_forward_tilt_deg", "pendulum_m1_kg", "pendulum_l1_m", "pendulum_lc1_m", "pendulum_i1_kg_m2", "pendulum_m2_kg", "pendulum_l2_m", "pendulum_lc2_m", "pendulum_i2_kg_m2", "damping_shoulder", "damping_wrist", "swing_duration_s", "flight_model", "impact_offset_toe_mm", "impact_offset_high_mm"] as const;
const REQUEST_FIELDS = ["schema_id", "schema_version", "request_id", "base", "factors", "trajectories", "levels", "seed", "minimum_effects", "worker_count"] as const;
const FACTOR_FIELDS = ["spec_id", "variable_key", "lower", "upper", "unit"] as const;

const record = (value: unknown, name: string): Record<string, unknown> => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) throw new RangeError(`${name} must be a plain object`);
  const prototype = Object.getPrototypeOf(value) as object | null;
  if (prototype !== Object.prototype && prototype !== null) throw new RangeError(`${name} must be a plain object`);
  return value as Record<string, unknown>;
};
const exact = (item: Record<string, unknown>, fields: readonly string[], name: string) => {
  const actual = Object.keys(item).sort(); const expected = [...fields].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) throw new RangeError(`${name} fields do not match the v1 allowlist`);
};
const text = (value: unknown, name: string, maximum: number, allowEmpty = false): string => {
  if (typeof value !== "string" || Array.from(value).length > maximum || containsControlCharacter(value)
      || (!allowEmpty && (value === "" || value !== value.trim()))) throw new RangeError(`${name} must be bounded text`);
  return value;
};
const rawBound = (value: unknown, name: string): string => text(value, name, MAX_BOUND_TEXT, true);
const nullableText = (value: unknown, name: string): string | null => value === null
  ? null : text(value, name, MAX_VALIDATION_TEXT);
const integerWithin = (value: unknown, minimum: number, maximum: number, name: string): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value)
      || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be an integer from ${minimum} through ${maximum}`);
  }
  return value;
};

const decimalBound = (value: string): number | null => {
  if (!DECIMAL_NUMBER.test(value)) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) && Math.abs(parsed) <= MAX_MORRIS_EDITOR_BOUND ? parsed : null;
};

const inspectJsonBudget = (value: unknown, depth = 0): number => {
  if (depth > MAX_JSON_DEPTH) throw new RangeError("Morris workspace exceeds the JSON depth limit");
  if (value === null || typeof value !== "object") return 1;
  const children = Array.isArray(value) ? value : Object.values(record(value, "JSON object"));
  const count = 1 + children.reduce((sum, child) => sum + inspectJsonBudget(child, depth + 1), 0);
  if (count > MAX_JSON_NODES) throw new RangeError("Morris workspace exceeds the JSON node limit");
  return count;
};

const baseFromWire = (value: unknown): MorrisAuthorityBase => {
  const item = record(value, "Morris workspace base"); exact(item, BASE_FIELDS, "Morris workspace base");
  return Object.freeze({
    clubName: item.club_name as string, supportMode: item.support_mode as "ground" | "tee",
    teeHeightM: item.tee_height_m as number, planeYawDeg: item.plane_yaw_deg as number,
    planeSideTiltDeg: item.plane_side_tilt_deg as number, planeForwardTiltDeg: item.plane_forward_tilt_deg as number,
    pendulumM1Kg: item.pendulum_m1_kg as number, pendulumL1M: item.pendulum_l1_m as number,
    pendulumLc1M: item.pendulum_lc1_m as number, pendulumI1KgM2: item.pendulum_i1_kg_m2 as number,
    pendulumM2Kg: item.pendulum_m2_kg as number, pendulumL2M: item.pendulum_l2_m as number,
    pendulumLc2M: item.pendulum_lc2_m as number, pendulumI2KgM2: item.pendulum_i2_kg_m2 as number,
    dampingShoulder: item.damping_shoulder as number, dampingWrist: item.damping_wrist as number,
    swingDurationS: item.swing_duration_s as number, flightModel: item.flight_model as string,
    impactOffsetToeMm: item.impact_offset_toe_mm as number, impactOffsetHighMm: item.impact_offset_high_mm as number,
  });
};

const parseDraft = (value: unknown, index: number): MorrisWorkspaceFactorDraft => {
  const item = record(value, `factor_drafts[${index}]`); exact(item, DRAFT_FIELDS, `factor_drafts[${index}]`);
  const lower = rawBound(item.lower, "factor lower"); const upper = rawBound(item.upper, "factor upper");
  if (typeof item.enabled !== "boolean") throw new RangeError("factor enabled must be boolean");
  const validationError = nullableText(item.validation_error, "factor validation_error");
  const lowerValue = decimalBound(lower); const upperValue = decimalBound(upper);
  const validBounds = lowerValue !== null && upperValue !== null && lowerValue < upperValue;
  if (item.enabled && (!validBounds || validationError !== null)) throw new RangeError("enabled factor requires valid bounds and no validation_error");
  const expectedError = validBounds ? null : INVALID_MORRIS_BOUNDS_MESSAGE;
  if (!item.enabled && validationError !== expectedError) throw new RangeError("disabled factor validation_error must exactly reflect raw bound validity");
  return Object.freeze({ variableKey: text(item.variable_key, "factor variable_key", MAX_VALIDATION_TEXT), enabled: item.enabled, lower, upper, validationError });
};

const validateDraftSet = (drafts: readonly MorrisWorkspaceFactorDraft[], base: MorrisAuthorityBase) => {
  if (drafts.length !== RATE_MORRIS_VARIABLE_KEYS.length) throw new RangeError("factor_drafts must contain the complete canonical registry");
  drafts.forEach((draft, index) => {
    if (draft.variableKey !== RATE_MORRIS_VARIABLE_KEYS[index]) throw new RangeError("factor_drafts must retain canonical order");
  });
  if (base.supportMode === "ground" && drafts[drafts.length - 1]?.enabled) throw new RangeError("ground support requires the tee-height draft to be disabled");
};

const enabledDrafts = (setup: MorrisWorkspaceSetup): readonly MorrisFactorDraft[] => setup.factorDrafts.filter((draft) => draft.enabled).map((draft) => Object.freeze({
  variableKey: draft.variableKey, enabled: true, lower: Number(draft.lower), upper: Number(draft.upper),
}));

const parseRequest = (value: unknown): MorrisAuthorityRequestDocument => {
  const item = record(value, "completed evidence request"); exact(item, REQUEST_FIELDS, "completed evidence request");
  if (!Array.isArray(item.factors)) throw new RangeError("request factors must be an array");
  item.factors.forEach((factor, index) => exact(record(factor, `request factor ${index}`), FACTOR_FIELDS, `request factor ${index}`));
  const base = baseFromWire(item.base);
  const factors = item.factors.map((factor) => {
    const row = factor as Record<string, unknown>;
    return { variableKey: row.variable_key as string, enabled: true, lower: row.lower as number, upper: row.upper as number };
  });
  const canonical = serializeMorrisAuthorityRequest({
    requestId: item.request_id as string, base, factors,
    trajectories: item.trajectories as number, levels: item.levels as number,
    seed: item.seed as number, minimumEffects: item.minimum_effects as number,
    workerCount: item.worker_count as number,
  });
  if (!jsonEqual(item, canonical)) throw new RangeError("completed evidence request is not canonical");
  return canonical;
};

const parseSetup = (value: unknown): MorrisWorkspaceSetup => {
  const item = record(value, "Morris workspace setup"); exact(item, SETUP_FIELDS, "Morris workspace setup");
  if (item.export_scope !== MORRIS_WORKSPACE_EXPORT_SCOPE) throw new RangeError("unsupported Morris workspace export_scope");
  if (!Array.isArray(item.factor_drafts)) throw new RangeError("factor_drafts must be an array");
  const base = baseFromWire(item.base); const drafts = item.factor_drafts.map(parseDraft);
  validateDraftSet(drafts, base);
  baseToWire(base);
  const trajectories = integerWithin(item.trajectories, 2, 5_000, "trajectories");
  const levels = integerWithin(item.levels, 4, 10_000, "levels");
  if (levels % 2 !== 0) throw new RangeError("levels must be even");
  const minimumEffects = integerWithin(item.minimum_effects, 2, trajectories, "minimum_effects");
  const workerCount = integerWithin(item.worker_count, 1, 32, "worker_count");
  const seed = integerWithin(item.seed, 0, 2 ** 31 - 1, "seed");
  const enabledCount = drafts.filter((draft) => draft.enabled).length;
  const totalSamples = trajectories * (enabledCount + 1);
  if (totalSamples > MAX_MORRIS_SAMPLES
      || totalSamples * MORRIS_TARGET_COUNT > MAX_MORRIS_OBSERVATION_CELLS) {
    throw new RangeError("Morris sample allocation exceeds authority resource limits");
  }
  return Object.freeze({ exportScope: MORRIS_WORKSPACE_EXPORT_SCOPE, base,
    factorDrafts: Object.freeze(drafts), trajectories, levels, seed,
    minimumEffects, workerCount });
};

const validateEvidence = (setup: MorrisWorkspaceSetup, evidence: MorrisCompletedEvidence) => {
  const { request, job } = evidence; const report = job.report;
  if (job.status !== "completed" || report === null) throw new RangeError("Morris workspace evidence must be completed and archived");
  if (job.requestId !== request.request_id) throw new RangeError("Morris workspace request identity mismatch");
  if (!jsonEqual(request.base, baseToWire(setup.base))) throw new RangeError("workspace setup and request bases differ");
  const expected = serializeMorrisAuthorityRequest({
    requestId: request.request_id, base: setup.base, factors: enabledDrafts(setup),
    trajectories: setup.trajectories, levels: setup.levels, seed: setup.seed,
    minimumEffects: setup.minimumEffects, workerCount: setup.workerCount,
  });
  if (!jsonEqual(request, expected)) throw new RangeError("workspace setup and request controls or factors differ");
  if (job.totalSamples !== request.trajectories * (request.factors.length + 1) || job.completedSamples !== job.totalSamples) throw new RangeError("workspace job sample identity mismatch");
  const design = report.design;
  if (design.trajectories !== request.trajectories || design.levels !== request.levels || design.seed !== request.seed || design.totalSamples !== job.totalSamples) throw new RangeError("workspace report design identity mismatch");
  const factors = new Map(request.factors.map((factor) => [factor.spec_id, factor]));
  const sources = new Set<string>();
  report.estimates.forEach(({ source }) => {
    const factor = factors.get(source.specId);
    if (factor === undefined || source.variableKey !== factor.variable_key || source.unit !== factor.unit
        || source.bounds[0] !== factor.lower || source.bounds[1] !== factor.upper) {
      throw new RangeError("workspace report factor provenance differs from request");
    }
    sources.add(source.specId);
  });
  if (sources.size !== request.factors.length) throw new RangeError("workspace report source set differs from request factors");
};

const parseEvidence = (value: unknown, setup: MorrisWorkspaceSetup): MorrisCompletedEvidence | null => {
  if (value === null) return null;
  const item = record(value, "completed_evidence"); exact(item, EVIDENCE_FIELDS, "completed_evidence");
  const evidence = Object.freeze({ request: parseRequest(item.request), job: parseMorrisJobEnvelope(item.job) });
  validateEvidence(setup, evidence); return evidence;
};

const baseToWire = (base: MorrisAuthorityBase): Readonly<Record<string, string | number>> => serializeMorrisAuthorityRequest({
  requestId: "workspace-base-validation", base,
  factors: [{ ...suggestedMorrisFactorDrafts(base)[0], enabled: true }],
  trajectories: 2, levels: 4, seed: 0, minimumEffects: 2, workerCount: 1,
}).base;

const jsonEqual = (left: unknown, right: unknown): boolean => {
  if (Object.is(left, right)) return true;
  if (Array.isArray(left) && Array.isArray(right)) return left.length === right.length && left.every((item, index) => jsonEqual(item, right[index]));
  if (left === null || right === null || typeof left !== "object" || typeof right !== "object" || Array.isArray(left) || Array.isArray(right)) return false;
  const leftRecord = record(left, "comparison value"); const rightRecord = record(right, "comparison value");
  const keys = Object.keys(leftRecord).sort(); const otherKeys = Object.keys(rightRecord).sort();
  return keys.length === otherKeys.length && keys.every((key, index) => key === otherKeys[index] && jsonEqual(leftRecord[key], rightRecord[key]));
};

const workspaceToWire = (workspace: MorrisWorkspaceDocument) => ({
  schema_id: workspace.schemaId, schema_version: workspace.schemaVersion,
  setup: { export_scope: workspace.setup.exportScope, base: baseToWire(workspace.setup.base), factor_drafts: workspace.setup.factorDrafts.map((draft) => ({ variable_key: draft.variableKey, enabled: draft.enabled, lower: draft.lower, upper: draft.upper, validation_error: draft.validationError })), trajectories: workspace.setup.trajectories, levels: workspace.setup.levels, seed: workspace.setup.seed, minimum_effects: workspace.setup.minimumEffects, worker_count: workspace.setup.workerCount },
  completed_evidence: workspace.completedEvidence === null ? null : { request: workspace.completedEvidence.request, job: morrisJobToDocument(workspace.completedEvidence.job) },
});

const sortedJson = (value: unknown): unknown => Array.isArray(value) ? value.map(sortedJson) : value !== null && typeof value === "object" ? Object.fromEntries(Object.entries(value).sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => [key, sortedJson(item)])) : value;

export function parseMorrisWorkspaceJson(source: string): MorrisWorkspaceDocument {
  if (typeof source !== "string") throw new TypeError("Morris workspace JSON source must be text");
  if (new TextEncoder().encode(source).byteLength > MAX_MORRIS_WORKSPACE_BYTES) throw new RangeError("Morris workspace exceeds the byte limit");
  const parsed = parseUniqueJson(source, "Morris workspace JSON"); inspectJsonBudget(parsed);
  const item = record(parsed, "Morris workspace"); exact(item, ROOT_FIELDS, "Morris workspace");
  if (item.schema_id !== MORRIS_WORKSPACE_SCHEMA_ID || item.schema_version !== MORRIS_WORKSPACE_SCHEMA_VERSION) throw new RangeError("unsupported Morris workspace schema");
  const setup = parseSetup(item.setup); const completedEvidence = parseEvidence(item.completed_evidence, setup);
  return Object.freeze({ schemaId: MORRIS_WORKSPACE_SCHEMA_ID, schemaVersion: MORRIS_WORKSPACE_SCHEMA_VERSION, setup, completedEvidence });
}

const completeDrafts = (base: MorrisAuthorityBase, drafts: readonly MorrisFactorDraft[]): readonly MorrisWorkspaceFactorDraft[] => {
  const suggestions = suggestedMorrisFactorDrafts(base.supportMode === "ground" ? { ...base, supportMode: "tee" } : base);
  const supplied = new Map(drafts.map((draft) => [draft.variableKey, draft]));
  return Object.freeze(suggestions.map((suggestion) => {
    const draft = supplied.get(suggestion.variableKey) ?? { ...suggestion, enabled: false };
    const teeKey = RATE_MORRIS_VARIABLE_KEYS[RATE_MORRIS_VARIABLE_KEYS.length - 1];
    const enabled = base.supportMode === "ground" && draft.variableKey === teeKey ? false : draft.enabled;
    const lower = String(draft.lower ?? ""); const upper = String(draft.upper ?? "");
    const validBounds = Number.isFinite(draft.lower) && Number.isFinite(draft.upper)
      && draft.lower !== null && draft.upper !== null
      && Math.abs(draft.lower) <= MAX_MORRIS_EDITOR_BOUND
      && Math.abs(draft.upper) <= MAX_MORRIS_EDITOR_BOUND
      && draft.lower < draft.upper;
    return Object.freeze({ variableKey: draft.variableKey, enabled, lower, upper,
      validationError: !enabled && !validBounds ? INVALID_MORRIS_BOUNDS_MESSAGE : null });
  }));
};

export function createMorrisWorkspaceDocument(base: MorrisAuthorityBase, drafts: readonly MorrisFactorDraft[], design: MorrisDesignControls, evidence: MorrisCompletedEvidence | null): MorrisWorkspaceDocument {
  const raw = { schema_id: MORRIS_WORKSPACE_SCHEMA_ID, schema_version: 1, setup: { export_scope: MORRIS_WORKSPACE_EXPORT_SCOPE, base: baseToWire(base), factor_drafts: completeDrafts(base, drafts).map((draft) => ({ variable_key: draft.variableKey, enabled: draft.enabled, lower: draft.lower, upper: draft.upper, validation_error: draft.validationError })), trajectories: design.trajectories, levels: design.levels, seed: design.seed, minimum_effects: design.minimumEffects, worker_count: design.workerCount }, completed_evidence: evidence === null ? null : { request: evidence.request, job: morrisJobToDocument(evidence.job) } };
  return parseMorrisWorkspaceJson(JSON.stringify(raw));
}

/** Rebuild a validated document without normalizing imported raw draft text. */
export function createMorrisWorkspaceFromSetup(
  setup: MorrisWorkspaceSetup,
  evidence: MorrisCompletedEvidence | null,
): MorrisWorkspaceDocument {
  const raw = {
    schema_id: MORRIS_WORKSPACE_SCHEMA_ID,
    schema_version: MORRIS_WORKSPACE_SCHEMA_VERSION,
    setup: workspaceToWire({
      schemaId: MORRIS_WORKSPACE_SCHEMA_ID,
      schemaVersion: MORRIS_WORKSPACE_SCHEMA_VERSION,
      setup,
      completedEvidence: null,
    }).setup,
    completed_evidence: evidence === null ? null : {
      request: evidence.request,
      job: morrisJobToDocument(evidence.job),
    },
  };
  return parseMorrisWorkspaceJson(JSON.stringify(raw));
}

export const morrisWorkspaceToJson = (workspace: MorrisWorkspaceDocument): string => `${JSON.stringify(sortedJson(workspaceToWire(workspace)), null, 2)}\n`;
export const morrisWorkspaceMatchesBase = (workspace: MorrisWorkspaceDocument, base: MorrisAuthorityBase): boolean => morrisAuthorityBaseIdentity(workspace.setup.base) === morrisAuthorityBaseIdentity(base);
export const workspaceDraftsForEditor = (setup: MorrisWorkspaceSetup): readonly MorrisFactorDraft[] => {
  const teeKey = RATE_MORRIS_VARIABLE_KEYS[RATE_MORRIS_VARIABLE_KEYS.length - 1];
  const suggestions = new Map(suggestedMorrisFactorDrafts(setup.base).map((draft) => [draft.variableKey, draft]));
  return setup.factorDrafts.filter((draft) => setup.base.supportMode === "tee" || draft.variableKey !== teeKey).map((draft) => {
    const fallback = suggestions.get(draft.variableKey);
    const lower = Number(draft.lower); const upper = Number(draft.upper);
    return Object.freeze({
      variableKey: draft.variableKey,
      enabled: draft.enabled,
      lower: Number.isFinite(lower) ? lower : (fallback?.lower ?? null),
      upper: Number.isFinite(upper) ? upper : (fallback?.upper ?? null),
    });
  });
};

export const morrisCsvCell = (value: unknown): string => {
  let textValue = value === null ? "" : String(value);
  if (typeof value === "string" && /^[=+\-@\t\r]/.test(textValue)) textValue = `'${textValue}`;
  return /[",\r\n]/.test(textValue) ? `"${textValue.replace(/"/g, '""')}"` : textValue;
};
export function morrisWorkspaceReportToCsv(workspace: MorrisWorkspaceDocument): string {
  const evidence = workspace.completedEvidence;
  if (evidence === null || evidence.job.report === null) throw new RangeError("completed archived evidence is required for report CSV");
  const header = ["request_id", "job_id", "evidence_state", "export_scope", "trajectories", "levels", "seed", "total_samples", "normalized_step", "source_spec_id", "source_variable_key", "source_unit", "source_lower", "source_upper", "source_time_start_s", "source_time_end_s", "source_point_ids_json", "target_name", "target_unit", "target_kind", "target_time_s", "target_point_id", "coordinate_frame", "mu", "mu_star", "mu_star_standard_error", "sigma", "availability", "sample_adequacy", "total_pairs", "valid_pairs", "typed_no_impact_pairs", "no_impact_unavailable_pairs", "failed_pairs", "nonfinite_pairs"];
  const report = evidence.job.report; const rows = report.estimates.map((estimate) => [evidence.request.request_id, evidence.job.jobId, "archived-completed-unverified-live", workspace.setup.exportScope, report.design.trajectories, report.design.levels, report.design.seed, report.design.totalSamples, report.design.normalizedStep, estimate.source.specId, estimate.source.variableKey, estimate.source.unit, ...estimate.source.bounds, estimate.source.timeWindowS?.[0] ?? null, estimate.source.timeWindowS?.[1] ?? null, JSON.stringify(estimate.source.pointIds), estimate.target.name, estimate.target.unit, estimate.target.kind, estimate.target.timeS, estimate.target.pointId, estimate.target.coordinateFrame, estimate.effects.mu, estimate.effects.muStar, estimate.effects.muStarStandardError, estimate.effects.sigma, estimate.availability, estimate.sampleAdequacy, estimate.denominator.totalPairs, estimate.denominator.validPairs, estimate.denominator.typedNoImpactPairs, estimate.denominator.noImpactUnavailablePairs, estimate.denominator.failedPairs, estimate.denominator.nonfinitePairs]);
  return `${[header, ...rows].map((row) => row.map(morrisCsvCell).join(",")).join("\n")}\n`;
}
