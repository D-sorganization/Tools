/** UI-neutral factor rows and exact request serialization for the Morris authority. */

import type { SupportMode } from "./ballSetup";
import { CLUB_LIBRARY } from "./club";
import { variableDef } from "./variationRegistry";
import { MORRIS_REQUEST_SCHEMA_ID } from "./morrisAuthorityCapability";
import { MORRIS_AUTHORITY_SCHEMA_VERSION } from "./morrisAuthorityContract";
import { morrisStableId } from "./morrisStableId";

export const RATE_MORRIS_VARIABLE_KEYS = Object.freeze([
  "swing_sim.swing.yaw_deg",
  "swing_sim.swing.side_tilt_deg",
  "swing_sim.swing.forward_tilt_deg",
  "swing_sim.swing.damping_shoulder",
  "swing_sim.swing.damping_wrist",
  "swing_sim.impact.delivery.impact_offset_toe_mm",
  "swing_sim.impact.delivery.impact_offset_high_mm",
  "swing_sim.club.head_mass_kg",
  "swing_sim.club.head_moi_kg_m2",
  "swing_sim.ball_setup.tee_height_m",
] as const);

export const AUTHORITY_FLIGHT_MODELS = Object.freeze([
  "waterloo_penner",
  "macdonald_hanzely",
  "nathan",
  "ballantyne",
  "jcole",
  "rospie_dl",
  "charry_l3",
] as const);

export interface MorrisFactorDraft {
  readonly variableKey: string;
  readonly enabled: boolean;
  readonly lower: number | null;
  readonly upper: number | null;
}

export interface MorrisFactorRow extends MorrisFactorDraft {
  readonly specId: string;
  readonly label: string;
  readonly unit: string;
  readonly guidance: string;
  readonly applicability: "tee_only" | null;
  readonly applicable: boolean;
  readonly validationError: string | null;
}

export interface MorrisFactorValidation {
  readonly valid: boolean;
  readonly rows: readonly MorrisFactorRow[];
  readonly errors: readonly string[];
}

export interface MorrisAuthorityBase {
  readonly clubName: string;
  readonly supportMode: SupportMode;
  readonly teeHeightM: number;
  readonly planeYawDeg: number;
  readonly planeSideTiltDeg: number;
  readonly planeForwardTiltDeg: number;
  readonly pendulumM1Kg: number;
  readonly pendulumL1M: number;
  readonly pendulumLc1M: number;
  readonly pendulumI1KgM2: number;
  readonly pendulumM2Kg: number;
  readonly pendulumL2M: number;
  readonly pendulumLc2M: number;
  readonly pendulumI2KgM2: number;
  readonly dampingShoulder: number;
  readonly dampingWrist: number;
  readonly swingDurationS: number;
  readonly flightModel: string;
  readonly impactOffsetToeMm: number;
  readonly impactOffsetHighMm: number;
}

export interface MorrisAuthorityRequest {
  readonly requestId: string;
  readonly base: MorrisAuthorityBase;
  readonly factors: readonly MorrisFactorDraft[];
  readonly trajectories: number;
  readonly levels: number;
  readonly seed: number;
  readonly minimumEffects: number;
  readonly workerCount: number;
}

interface MorrisFactorDocument {
  readonly spec_id: string;
  readonly variable_key: string;
  readonly lower: number;
  readonly upper: number;
  readonly unit: string;
}

export interface MorrisAuthorityRequestDocument {
  readonly schema_id: typeof MORRIS_REQUEST_SCHEMA_ID;
  readonly schema_version: typeof MORRIS_AUTHORITY_SCHEMA_VERSION;
  readonly request_id: string;
  readonly base: Readonly<Record<string, string | number>>;
  readonly factors: readonly MorrisFactorDocument[];
  readonly trajectories: number;
  readonly levels: number;
  readonly seed: number;
  readonly minimum_effects: number;
  readonly worker_count: number;
}

const supported = new Set<string>(RATE_MORRIS_VARIABLE_KEYS);
const authorityClubNames = new Set(CLUB_LIBRARY.map((club) => club.name));
const authorityFlightModels = new Set<string>(AUTHORITY_FLIGHT_MODELS);
const MAX_MORRIS_SAMPLES = 100_000;
const MAX_MORRIS_OBSERVATION_CELLS = 1_000_000;
const MORRIS_TARGET_COUNT = 17;
const INVALID_SPEC_ID_CHARACTERS = /[^A-Za-z0-9._:-]+/g;
const SPEC_ID_EDGE_CHARACTERS = /^[.-]+|[.-]+$/g;
const ENDPOINT_BOUNDS = new Map<string, readonly [number, number]>([
  ["swing_sim.swing.damping_shoulder", [0, Number.POSITIVE_INFINITY]],
  ["swing_sim.swing.damping_wrist", [0, Number.POSITIVE_INFINITY]],
  ["swing_sim.impact.delivery.impact_offset_toe_mm", [-80, 80]],
  ["swing_sim.impact.delivery.impact_offset_high_mm", [-40, 40]],
  ["swing_sim.club.head_mass_kg", [0.1, 0.5]],
  ["swing_sim.club.head_moi_kg_m2", [5e-5, 2e-3]],
  ["swing_sim.ball_setup.tee_height_m", [0, Number.POSITIVE_INFINITY]],
]);

const stableText = (value: unknown, name: string): string => {
  if (typeof value !== "string" || value === "" || value !== value.trim()) {
    throw new RangeError(`${name} must be a nonempty trimmed string`);
  }
  return value;
};

const integerWithin = (value: unknown, minimum: number, maximum: number, name: string): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be an integer within [${minimum}, ${maximum}]`);
  }
  return value;
};

const errorForDraft = (
  draft: MorrisFactorDraft,
  supportMode: SupportMode,
  duplicateKey: boolean,
): string | null => {
  const definition = variableDef(draft.variableKey);
  const label = definition?.label ?? draft.variableKey;
  if (!draft.enabled) return null;
  if (!supported.has(draft.variableKey)) return "Unsupported Morris factor";
  if (duplicateKey) return "Enabled factor variables must be unique.";
  if (definition?.applicability === "tee_only" && supportMode !== "tee") return "Tee height requires tee support";
  if (draft.lower === null || draft.upper === null
      || !Number.isFinite(draft.lower) || !Number.isFinite(draft.upper)) {
    return `${label} bounds must be finite numbers.`;
  }
  if (draft.lower >= draft.upper) return "Lower bound must be less than upper bound";
  const endpointBounds = ENDPOINT_BOUNDS.get(draft.variableKey);
  if (endpointBounds !== undefined
      && (draft.lower < endpointBounds[0] || draft.upper > endpointBounds[1])) {
    return `${label} bounds exceed the authority endpoint limits.`;
  }
  return null;
};

export function buildMorrisFactorRows(
  drafts: readonly MorrisFactorDraft[], supportMode: SupportMode,
): readonly MorrisFactorRow[] {
  if (!Array.isArray(drafts)) throw new TypeError("Morris factor drafts must be an array");
  if (supportMode !== "ground" && supportMode !== "tee") throw new RangeError("support mode is unsupported");
  drafts.forEach((draft) => {
    if (draft === null || typeof draft !== "object" || Array.isArray(draft)) throw new TypeError("Morris factor draft must be an object");
    if (typeof draft.enabled !== "boolean") throw new TypeError("factor enabled must be boolean");
    stableText(draft.variableKey, "factor variableKey");
    [draft.lower, draft.upper].forEach((value) => {
      if (value !== null && (typeof value !== "number" || !Number.isFinite(value))) {
        throw new TypeError("factor bounds must be finite numbers or null");
      }
    });
  });
  const seenKeys = new Set<string>();
  return Object.freeze(drafts.map((draft) => {
    const definition = variableDef(draft.variableKey);
    const duplicateKey = draft.enabled && seenKeys.has(draft.variableKey);
    if (draft.enabled) seenKeys.add(draft.variableKey);
    return Object.freeze({
      ...draft,
      specId: specIdForKey(draft.variableKey),
      label: definition?.label ?? draft.variableKey,
      unit: definition?.unit ?? "",
      guidance: definition?.guidance ?? "Unsupported authority factor.",
      applicability: definition?.applicability ?? null,
      applicable: definition?.applicability !== "tee_only" || supportMode === "tee",
      validationError: errorForDraft(draft, supportMode, duplicateKey),
    });
  }));
}

export function specIdForKey(variableKey: string): string {
  if (typeof variableKey !== "string") throw new TypeError("variableKey must be a string");
  const specId = variableKey.replace(INVALID_SPEC_ID_CHARACTERS, "-")
    .replace(SPEC_ID_EDGE_CHARACTERS, "");
  if (specId === "" || specId.length > 128) throw new RangeError("variableKey cannot form a stable spec ID");
  return specId;
}

export function validateMorrisFactorDrafts(
  drafts: readonly MorrisFactorDraft[], supportMode: SupportMode,
): MorrisFactorValidation {
  const rows = buildMorrisFactorRows(drafts, supportMode);
  const errors = rows.flatMap((row) => row.validationError === null ? [] : [row.validationError]);
  if (!rows.some((row) => row.enabled)) errors.push("At least one enabled factor is required.");
  return Object.freeze({ valid: errors.length === 0, rows, errors: Object.freeze(errors) });
}

const validateFiniteBaseNumbers = (base: MorrisAuthorityBase): void => {
  const numericFields: ReadonlyArray<readonly [string, unknown]> = [
    ["teeHeightM", base.teeHeightM], ["planeYawDeg", base.planeYawDeg],
    ["planeSideTiltDeg", base.planeSideTiltDeg], ["planeForwardTiltDeg", base.planeForwardTiltDeg],
    ["pendulumM1Kg", base.pendulumM1Kg], ["pendulumL1M", base.pendulumL1M],
    ["pendulumLc1M", base.pendulumLc1M], ["pendulumI1KgM2", base.pendulumI1KgM2],
    ["pendulumM2Kg", base.pendulumM2Kg], ["pendulumL2M", base.pendulumL2M],
    ["pendulumLc2M", base.pendulumLc2M], ["pendulumI2KgM2", base.pendulumI2KgM2],
    ["dampingShoulder", base.dampingShoulder], ["dampingWrist", base.dampingWrist],
    ["swingDurationS", base.swingDurationS], ["impactOffsetToeMm", base.impactOffsetToeMm],
    ["impactOffsetHighMm", base.impactOffsetHighMm],
  ];
  numericFields.forEach(([name, value]) => {
    if (typeof value !== "number" || !Number.isFinite(value)) throw new TypeError(`${name} must be a finite number`);
  });
};

const validateBaseSemantics = (base: MorrisAuthorityBase): readonly [string, string] => {
  if (base.supportMode !== "ground" && base.supportMode !== "tee") throw new RangeError("support mode is unsupported");
  const clubName = stableText(base.clubName, "clubName");
  const flightModel = stableText(base.flightModel, "flightModel");
  if (!authorityClubNames.has(clubName)) throw new RangeError("clubName is not in the authority club library");
  if (!authorityFlightModels.has(flightModel)) throw new RangeError("flightModel is unsupported by the authority");
  if (base.teeHeightM < 0 || (base.supportMode === "ground" && base.teeHeightM !== 0)) {
    throw new RangeError("teeHeightM is inconsistent with support mode");
  }
  const positive = [
    base.pendulumM1Kg, base.pendulumL1M, base.pendulumLc1M, base.pendulumI1KgM2,
    base.pendulumM2Kg, base.pendulumL2M, base.pendulumLc2M, base.pendulumI2KgM2,
  ];
  if (positive.some((value) => value <= 0)) throw new RangeError("pendulum masses, lengths, centers, and inertias must be positive");
  if (base.pendulumLc1M > base.pendulumL1M || base.pendulumLc2M > base.pendulumL2M) {
    throw new RangeError("pendulum center length must not exceed segment length");
  }
  if (base.dampingShoulder < 0 || base.dampingWrist < 0) throw new RangeError("pendulum damping must be nonnegative");
  if (base.swingDurationS <= 0) throw new RangeError("swingDurationS must be positive");
  if (base.impactOffsetToeMm < -80 || base.impactOffsetToeMm > 80) throw new RangeError("impactOffsetToeMm is outside authority bounds");
  if (base.impactOffsetHighMm < -40 || base.impactOffsetHighMm > 40) throw new RangeError("impactOffsetHighMm is outside authority bounds");
  return [clubName, flightModel];
};

const serializeBase = (base: MorrisAuthorityBase): Readonly<Record<string, string | number>> => {
  if (base === null || typeof base !== "object" || Array.isArray(base)) throw new TypeError("Morris base must be an object");
  validateFiniteBaseNumbers(base);
  const [clubName, flightModel] = validateBaseSemantics(base);
  const document = {
    club_name: clubName, support_mode: base.supportMode,
    tee_height_m: base.teeHeightM, plane_yaw_deg: base.planeYawDeg,
    plane_side_tilt_deg: base.planeSideTiltDeg, plane_forward_tilt_deg: base.planeForwardTiltDeg,
    pendulum_m1_kg: base.pendulumM1Kg, pendulum_l1_m: base.pendulumL1M,
    pendulum_lc1_m: base.pendulumLc1M, pendulum_i1_kg_m2: base.pendulumI1KgM2,
    pendulum_m2_kg: base.pendulumM2Kg, pendulum_l2_m: base.pendulumL2M,
    pendulum_lc2_m: base.pendulumLc2M, pendulum_i2_kg_m2: base.pendulumI2KgM2,
    damping_shoulder: base.dampingShoulder, damping_wrist: base.dampingWrist,
    swing_duration_s: base.swingDurationS, flight_model: flightModel,
    impact_offset_toe_mm: base.impactOffsetToeMm, impact_offset_high_mm: base.impactOffsetHighMm,
  };
  return Object.freeze(document);
};

export function serializeMorrisAuthorityRequest(
  request: MorrisAuthorityRequest,
): MorrisAuthorityRequestDocument {
  const supportMode = request.base.supportMode;
  const validation = validateMorrisFactorDrafts(request.factors, supportMode);
  if (!validation.valid) throw new RangeError(validation.errors.join(" "));
  const factors = validation.rows.filter((row) => row.enabled)
    .sort((left, right) => RATE_MORRIS_VARIABLE_KEYS.indexOf(
      left.variableKey as typeof RATE_MORRIS_VARIABLE_KEYS[number],
    ) - RATE_MORRIS_VARIABLE_KEYS.indexOf(
      right.variableKey as typeof RATE_MORRIS_VARIABLE_KEYS[number],
    )).map((row) => Object.freeze({
    spec_id: row.specId, variable_key: row.variableKey,
    lower: row.lower as number, upper: row.upper as number, unit: row.unit,
  }));
  const trajectories = integerWithin(request.trajectories, 1, 2 ** 31 - 1, "trajectories");
  const levels = integerWithin(request.levels, 4, 10_000, "levels");
  if (levels % 2 !== 0) throw new RangeError("levels must be even");
  const minimumEffects = integerWithin(request.minimumEffects, 2, trajectories, "minimumEffects");
  const totalSamples = trajectories * (factors.length + 1);
  if (totalSamples > MAX_MORRIS_SAMPLES
      || totalSamples * MORRIS_TARGET_COUNT > MAX_MORRIS_OBSERVATION_CELLS) {
    throw new RangeError("Morris sample allocation exceeds authority resource limits");
  }
  return Object.freeze({
    schema_id: MORRIS_REQUEST_SCHEMA_ID,
    schema_version: MORRIS_AUTHORITY_SCHEMA_VERSION,
    request_id: morrisStableId(request.requestId, "requestId"),
    base: serializeBase(request.base),
    factors: Object.freeze(factors),
    trajectories,
    levels,
    seed: integerWithin(request.seed, 0, 2 ** 32 - 1, "seed"),
    minimum_effects: minimumEffects,
    worker_count: integerWithin(request.workerCount, 1, 32, "workerCount"),
  });
}
