/**
 * One fully specified putt, evaluated end to end — TypeScript mirror
 * of the deterministic half of
 * `shared/python/swing_sim/putting/variation.py` (#4800 P5, twinned
 * in P7).
 *
 * Scope boundary (the same one `puttingDispersion.ts` declares): the
 * Monte-Carlo **execution** — `PuttVariationPlan` / `runPuttDispersion`
 * and the canonical seeded PCG64 sampler behind them — is
 * Python-authoritative and deliberately not re-implemented here. A
 * second sampler would be a second answer. What *is* twinned is the
 * deterministic evaluation every study is built out of, which is also
 * exactly what the web Putting tab runs for a single putt:
 *
 *   stroke -> `strike` (P1) -> `simulatePuttOnSurface` (P2)
 *          -> `puttingResultDocument` (`putting_result/2`, P5)
 *
 * `evaluatePutt` is the Python function field-for-field. The web
 * binding additionally needs the retained integration samples (the Qt
 * tab reads them from its own integrator call; the React tab has one
 * chokepoint), so `evaluatePuttWithTrajectory` returns the launch and
 * the `PuttResult` alongside the identical document — same call, same
 * numbers, one extra accessor. Nothing is recomputed for it.
 *
 * The registry keys are the shared vocabulary a Python-side plan
 * declares; they are mirrored so a report the web reads back names
 * variables the web can label, and so the two runtimes cannot drift on
 * the spelling of a variable key.
 */

import type { PuttOutcome } from "./puttingDispersion";
import {
  simulatePuttOnSurface,
  type CaptureModel,
  type GreenSurface,
  type PuttLaunch,
  type PuttResult,
} from "./puttingGreen";
import { DEFAULT_SLIDING_MU, strike, type PutterSpec } from "./putting";
import {
  puttingResultDocument,
  type PuttingResultDocument,
  type PuttingResultProvenance,
} from "./puttingResultWire";

/** Variation-registry category this package owns. */
export const CATEGORY_PUTTING = "swing_sim.putting";

export const PUTT_SPEED_KEY = `${CATEGORY_PUTTING}.clubhead_speed_mps`;
export const PUTT_FACE_KEY = `${CATEGORY_PUTTING}.face_angle_deg`;
export const PUTT_PATH_KEY = `${CATEGORY_PUTTING}.path_angle_deg`;
export const PUTT_AIM_KEY = `${CATEGORY_PUTTING}.aim_deg`;
export const PUTT_STRIKE_TOE_KEY = `${CATEGORY_PUTTING}.strike_offset_toe_mm`;

/** Registry keys this package owns; nothing else may enter a plan. */
export const PUTT_VARIABLE_KEYS: readonly string[] = [
  PUTT_AIM_KEY,
  PUTT_FACE_KEY,
  PUTT_PATH_KEY,
  PUTT_SPEED_KEY,
  PUTT_STRIKE_TOE_KEY,
];

/** Stroke-field name each registry key perturbs. */
const KEY_TO_FIELD: Readonly<Record<string, keyof PuttStroke>> = {
  [PUTT_SPEED_KEY]: "clubheadSpeedMps",
  [PUTT_FACE_KEY]: "faceAngleDeg",
  [PUTT_PATH_KEY]: "pathAngleDeg",
  [PUTT_AIM_KEY]: "aimDeg",
  [PUTT_STRIKE_TOE_KEY]: "strikeOffsetToeMm",
};

/**
 * The delivered stroke — P1's `strike` arguments.
 *
 * These are the *declared* nominal values a variation plan perturbs;
 * each field is validated by `strike` itself at evaluation time, so an
 * out-of-envelope value is refused there rather than clamped here.
 */
export interface PuttStroke {
  clubheadSpeedMps: number;
  shaftLeanDeg: number;
  aimDeg: number;
  faceAngleDeg: number;
  pathAngleDeg: number;
  attackAngleDeg: number;
  strikeOffsetToeMm: number;
  strikeOffsetHighMm: number;
}

const STROKE_DEFAULTS: Omit<PuttStroke, "clubheadSpeedMps"> = {
  shaftLeanDeg: 0.0,
  aimDeg: 0.0,
  faceAngleDeg: 0.0,
  pathAngleDeg: 0.0,
  attackAngleDeg: 0.0,
  strikeOffsetToeMm: 0.0,
  strikeOffsetHighMm: 0.0,
};

function requireFinite(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number`);
  }
  return value;
}

/** Validated stroke; every unset field defaults to a square stroke. */
export function puttStroke(
  clubheadSpeedMps: number,
  overrides: Partial<Omit<PuttStroke, "clubheadSpeedMps">> = {},
): PuttStroke {
  const stroke: PuttStroke = {
    ...STROKE_DEFAULTS,
    ...overrides,
    clubheadSpeedMps,
  };
  for (const name of Object.keys(stroke) as (keyof PuttStroke)[]) {
    requireFinite(stroke[name], name);
  }
  return stroke;
}

/** This stroke as the registry-keyed base a plan varies about. */
export function strokeBaseValues(stroke: PuttStroke): Record<string, number> {
  const base: Record<string, number> = {};
  for (const key of PUTT_VARIABLE_KEYS) base[key] = stroke[KEY_TO_FIELD[key]];
  return base;
}

/** One fully specified putt: putter, stroke, green, and hole. */
export interface PuttScenario {
  /** Stable identity carried into every report. */
  scenarioId: string;
  /** The v1 putter spec (build it through `putterHead.ts` for v2). */
  putter: PutterSpec;
  stroke: PuttStroke;
  /** Green geometry (P2). */
  surface: GreenSurface;
  stimpFt: number;
  holeDistanceM: number;
  /** Putter/stroke origin recorded in every result. */
  provenance: PuttingResultProvenance;
  muSlide?: number;
  /** Hole-capture model; P2's published effective-radius by default. */
  captureModel?: CaptureModel;
  /**
   * Putter-head MOI for P1's off-centre effective-mass reduction;
   * `undefined` selects P1's catalogue default. Fill it from
   * `putterHead.ts`'s `headMoiForStrike`.
   */
  headMoiKgM2?: number;
}

/** Fail-closed scenario validation (the Python `__post_init__` twin). */
export function requirePuttScenario(scenario: PuttScenario): PuttScenario {
  if (
    typeof scenario.scenarioId !== "string" ||
    scenario.scenarioId.trim() === ""
  ) {
    throw new Error("scenarioId must be a name");
  }
  if (scenario.surface.kind !== "planar" && scenario.surface.kind !== "grid") {
    throw new Error("surface must be a GreenSurface");
  }
  requireFinite(scenario.stimpFt, "stimpFt");
  requireFinite(scenario.holeDistanceM, "holeDistanceM");
  requireFinite(scenario.muSlide ?? DEFAULT_SLIDING_MU, "muSlide");
  const captureModel = scenario.captureModel ?? "effective_radius";
  if (scenario.provenance.captureModel !== captureModel) {
    throw new Error("provenance captureModel must match the scenario");
  }
  if (scenario.headMoiKgM2 !== undefined) {
    requireFinite(scenario.headMoiKgM2, "headMoiKgM2");
  }
  return scenario;
}

/** One evaluated putt: the launch, the retained samples, the record. */
export interface EvaluatedPutt {
  launch: PuttLaunch;
  result: PuttResult;
  document: PuttingResultDocument;
}

/**
 * Run one putt end to end, keeping the integration samples.
 *
 * Identical arithmetic to `evaluatePutt` — that function is this one
 * with the trajectory dropped, exactly as the Python twin returns only
 * the record. The samples are what the tab's inspector and green view
 * read; they are never re-simulated for presentation.
 */
export function evaluatePuttWithTrajectory(
  scenario: PuttScenario,
): EvaluatedPutt {
  requirePuttScenario(scenario);
  const stroke = scenario.stroke;
  const launch = strike(
    scenario.putter,
    stroke.clubheadSpeedMps,
    stroke.shaftLeanDeg,
    {
      aimDeg: stroke.aimDeg,
      faceAngleDeg: stroke.faceAngleDeg,
      pathAngleDeg: stroke.pathAngleDeg,
      attackAngleDeg: stroke.attackAngleDeg,
      strikeOffsetToeMm: stroke.strikeOffsetToeMm,
      strikeOffsetHighMm: stroke.strikeOffsetHighMm,
      headMoiKgM2: scenario.headMoiKgM2,
    },
  );
  const result = simulatePuttOnSurface(launch, scenario.surface, {
    stimpFt: scenario.stimpFt,
    holeDistanceM: scenario.holeDistanceM,
    muSlide: scenario.muSlide ?? DEFAULT_SLIDING_MU,
    captureModel: scenario.captureModel ?? "effective_radius",
  });
  const document = puttingResultDocument(
    launch,
    result,
    scenario.provenance,
    scenario.holeDistanceM,
  );
  return { launch, result, document };
}

/** Run one putt end to end and return its `putting_result/2` record. */
export function evaluatePutt(scenario: PuttScenario): PuttingResultDocument {
  return evaluatePuttWithTrajectory(scenario).document;
}

/** The dispersion outcome carried by one v2 result record. */
export function puttOutcome(document: PuttingResultDocument): PuttOutcome {
  const miss = document.missDistanceM;
  let leaveDistanceM: number;
  if (document.holed) {
    leaveDistanceM = 0.0;
  } else if (miss === null) {
    throw new Error("a missed putt must report a leave distance");
  } else {
    leaveDistanceM = miss;
  }
  return {
    holed: document.holed,
    startAzimuthDeg: document.startAzimuthDeg,
    leaveDistanceM,
    totalDistanceM: document.totalDistanceM,
    breakM: document.finalBreakM,
    captureMarginM: document.captureMarginM,
  };
}
