/**
 * Variation / Monte-Carlo engine for the web clone (epic #4120, V3).
 *
 * Practical TypeScript mirror of shared/python/swing_sim/variation: the
 * same namespaced variable registry, the same NoiseSpec / VariationPlan
 * JSON schema (snake_case keys, schema_version 1 — plans saved by the
 * desktop Variation tab load here and vice versa), and the same outputs
 * over the existing TS physics (solveImpact + deriveLaunch +
 * simulateFlight).
 *
 * Scope (documented deviations from the Python engine):
 * - Modes: "delivery" and "launch". The pendulum "swing" mode and the
 *   club-parameter category stay desktop-only until the P7 WASM kernels
 *   land (the TS solveImpact has no club-mass/COR/MOI seam).
 * - Runs are worker-less and bounded (MAX_RUNS = 500, UI-capped); the
 *   WASM + web-worker upgrade removes the cap.
 * - RNG: mulberry32 (public-domain 32-bit PRNG) with Box–Muller normals
 *   and inverse-CDF triangular draws, seeded per variable via FNV-1a so
 *   one-at-a-time sensitivity reuses the exact draws of the full study
 *   (same subset-stability property as the Python engine). Exact RNG
 *   parity with numpy's PCG64 is deliberately NOT attempted — the two
 *   engines produce statistically compatible dispersion for the same
 *   plan+seed, pinned loosely by variation.test.ts against a
 *   Python-generated fixture. Bit-identical streams would require
 *   porting PCG64 and numpy's ziggurat/rejection samplers for zero
 *   analytical benefit.
 */

import { deriveLaunch, simulateFlight, type Launch } from "./flight";
import { MPH_PER_MPS, solveImpact, toFlightFrame, type Vec3 } from "./simulation";

import { keysForMode, variableDef, type VariationMode } from "./variationRegistry";

export {
  CATEGORY_DELIVERY,
  CATEGORY_LAUNCH,
  keysForMode,
  variableDef,
  variableLabel,
  VARIABLE_REGISTRY,
  type VariableDefTs,
  type VariationMode,
} from "./variationRegistry";


export const SCHEMA_VERSION = 1;
export const MAX_RUNS = 500;

export type Distribution = "normal" | "uniform" | "triangular";
export interface NoiseSpecTs {
  variableKey: string;
  distribution: Distribution;
  scale: number;
  lower: number | null;
  upper: number | null;
}

export interface VariationPlanTs {
  mode: VariationMode;
  baseVariables: Record<string, number>;
  noise: NoiseSpecTs[];
  nRuns: number;
  seed: number;
  flightModel: string;
}

/** DbC-style validation mirroring spec.py (throws on violations). */
export function validatePlan(plan: VariationPlanTs): void {
  if (plan.mode !== "delivery" && plan.mode !== "launch") {
    throw new Error(
      `mode ${plan.mode} is not supported in the browser (desktop-only)`,
    );
  }
  if (!Number.isInteger(plan.nRuns) || plan.nRuns < 2 || plan.nRuns > MAX_RUNS) {
    throw new Error(`nRuns must be an integer in [2, ${MAX_RUNS}]`);
  }
  if (!Number.isInteger(plan.seed) || plan.seed < 0) {
    throw new Error("seed must be a non-negative integer");
  }
  if (plan.noise.length === 0) {
    throw new Error("plan must vary at least one variable");
  }
  const legal = new Set(keysForMode(plan.mode));
  const seen = new Set<string>();
  for (const spec of plan.noise) {
    if (!legal.has(spec.variableKey)) {
      throw new Error(`noise variable not legal in ${plan.mode} mode: ${spec.variableKey}`);
    }
    if (seen.has(spec.variableKey)) {
      throw new Error(`duplicate noise spec for ${spec.variableKey}`);
    }
    seen.add(spec.variableKey);
    if (!(spec.scale > 0) || !Number.isFinite(spec.scale)) {
      throw new Error(`scale for ${spec.variableKey} must be finite and > 0`);
    }
    if (spec.lower !== null && spec.upper !== null && !(spec.lower < spec.upper)) {
      throw new Error(`truncation bounds for ${spec.variableKey} must be lower < upper`);
    }
  }
  for (const key of Object.keys(plan.baseVariables)) {
    if (!legal.has(key)) {
      throw new Error(`base variable not legal in ${plan.mode} mode: ${key}`);
    }
  }
}

/** Same JSON schema as VariationPlan.to_json_dict() (snake_case). */
export function planToJson(plan: VariationPlanTs): string {
  validatePlan(plan);
  return JSON.stringify(
    {
      schema_version: SCHEMA_VERSION,
      mode: plan.mode,
      base_variables: plan.baseVariables,
      noise: plan.noise.map((s) => ({
        variable_key: s.variableKey,
        distribution: s.distribution,
        scale: s.scale,
        lower: s.lower,
        upper: s.upper,
      })),
      n_runs: plan.nRuns,
      seed: plan.seed,
      flight_model: plan.flightModel,
    },
    null,
    2,
  );
}

export function planFromJson(text: string): VariationPlanTs {
  const data = JSON.parse(text) as Record<string, unknown>;
  const version = Number(data.schema_version ?? SCHEMA_VERSION);
  if (version !== SCHEMA_VERSION) {
    throw new Error(`unsupported schema_version ${version}`);
  }
  const noiseRaw = (data.noise ?? []) as Array<Record<string, unknown>>;
  const plan: VariationPlanTs = {
    mode: String(data.mode) as VariationMode,
    baseVariables: (data.base_variables ?? {}) as Record<string, number>,
    noise: noiseRaw.map((s) => ({
      variableKey: String(s.variable_key),
      distribution: String(s.distribution ?? "normal") as Distribution,
      scale: Number(s.scale ?? 1.0),
      lower: s.lower === null || s.lower === undefined ? null : Number(s.lower),
      upper: s.upper === null || s.upper === undefined ? null : Number(s.upper),
    })),
    nRuns: Number(data.n_runs ?? 200),
    seed: Number(data.seed ?? 0),
    flightModel: String(data.flight_model ?? "waterloo_penner"),
  };
  validatePlan(plan);
  return plan;
}

// --- Seeded RNG -----------------------------------------------------------

/** mulberry32: tiny public-domain 32-bit PRNG (Tommy Ettinger). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** FNV-1a 32-bit string hash (per-variable stream derivation). */
export function fnv1a(text: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

const streamFor = (seed: number, variableKey: string): (() => number) =>
  mulberry32((seed ^ fnv1a(variableKey)) >>> 0);

/** Box–Muller standard normal from a uniform source. */
const normalDraw = (rng: () => number): number => {
  let u = 0;
  while (u === 0) u = rng(); // avoid log(0)
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * rng());
};

/** Inverse-CDF symmetric triangular on [-1, 1] with mode 0. */
const triangularDraw = (rng: () => number): number => {
  const u = rng();
  return u < 0.5 ? Math.sqrt(2.0 * u) - 1.0 : 1.0 - Math.sqrt(2.0 * (1.0 - u));
};

const resolvedBase = (plan: VariationPlanTs): Record<string, number> => {
  const base: Record<string, number> = {};
  for (const key of keysForMode(plan.mode)) {
    base[key] = variableDef(key)!.default;
  }
  return { ...base, ...plan.baseVariables };
};

/** Sample the (nRuns x nSpecs) inputs matrix, subset-stable per variable. */
export function sampleInputs(plan: VariationPlanTs): number[][] {
  validatePlan(plan);
  const base = resolvedBase(plan);
  const columns = plan.noise.map((spec) => {
    const rng = streamFor(plan.seed, spec.variableKey);
    const center = base[spec.variableKey];
    const column: number[] = [];
    for (let i = 0; i < plan.nRuns; i += 1) {
      let value: number;
      if (spec.distribution === "normal") {
        value = center + spec.scale * normalDraw(rng);
      } else if (spec.distribution === "uniform") {
        value = center + spec.scale * (2.0 * rng() - 1.0);
      } else {
        value = center + spec.scale * triangularDraw(rng);
      }
      const lo = spec.lower ?? -Infinity;
      const hi = spec.upper ?? Infinity;
      column.push(Math.min(hi, Math.max(lo, value)));
    }
    return column;
  });
  const rows: number[][] = [];
  for (let i = 0; i < plan.nRuns; i += 1) {
    rows.push(columns.map((column) => column[i]));
  }
  return rows;
}

// --- Pipeline evaluation --------------------------------------------------

export const DELIVERY_OUTPUTS = [
  "club_path_deg",
  "face_angle_deg",
  "attack_angle_deg",
  "dynamic_loft_deg",
] as const;
export const LAUNCH_OUTPUTS = [
  "ball_speed_mph",
  "launch_angle_deg",
  "launch_azimuth_deg",
  "spin_rpm",
  "spin_axis_deg",
] as const;
export const FLIGHT_OUTPUTS = [
  "carry_m",
  "lateral_m",
  "apex_m",
  "landing_angle_deg",
  "flight_time_s",
] as const;

export function outputsForMode(mode: VariationMode): string[] {
  return mode === "launch"
    ? [...LAUNCH_OUTPUTS, ...FLIGHT_OUTPUTS]
    : [...DELIVERY_OUTPUTS, ...LAUNCH_OUTPUTS, ...FLIGHT_OUTPUTS];
}

const RAD = Math.PI / 180.0;
const clampAngle = (v: number): number => Math.max(-89.0, Math.min(89.0, v));
const short = (key: string): string => key.slice(key.lastIndexOf(".") + 1);

/** Imperial launch numbers -> Launch (legacy spin-axis decomposition). */
const launchFromImperial = (v: Record<string, number>): Launch => {
  // Registry azimuth / spin-axis are + right / + fade; the flight frame
  // is + left, so both are negated (mirrors engine.py _evaluate_launch).
  const az = -v.launch_azimuth_deg * RAD;
  const axisAngle = -v.spin_axis_deg * RAD;
  const backspin = Math.cos(axisAngle);
  const sidespin = Math.sin(axisAngle);
  const axisRaw: Vec3 = [
    sidespin * Math.sin(az),
    -backspin,
    sidespin * Math.cos(az),
  ];
  const axisNorm = Math.hypot(axisRaw[0], axisRaw[1], axisRaw[2]) || 1.0;
  return {
    ballSpeedMps: v.ball_speed_mph / MPH_PER_MPS,
    launchAngleRad: clampAngle(v.launch_angle_deg) * RAD,
    azimuthRad: az,
    spinRpm: Math.max(v.spin_rpm, 0.0),
    spinAxis: [
      axisRaw[0] / axisNorm,
      axisRaw[1] / axisNorm,
      axisRaw[2] / axisNorm,
    ],
  };
};

const spinAxisTiltDeg = (spin: Vec3): number => {
  const magnitude = Math.hypot(spin[0], spin[1], spin[2]);
  if (magnitude < 1e-12) return 0.0;
  const axis: Vec3 = [spin[0] / magnitude, spin[1] / magnitude, spin[2] / magnitude];
  const horizontal = Math.hypot(axis[0], axis[2]);
  return Math.atan2(-axis[1], horizontal) / RAD;
};

/** One sampled variable set -> outputs record (throws on bad inputs). */
export function evaluateRun(
  variables: Record<string, number>,
  mode: VariationMode,
): Record<string, number> {
  const v: Record<string, number> = {};
  for (const [key, value] of Object.entries(variables)) v[short(key)] = value;

  if (mode === "launch") {
    if (!(v.ball_speed_mph >= 0)) {
      throw new Error("ball_speed_mph must be >= 0");
    }
    const launch = launchFromImperial(v);
    const flight = simulateFlight(launch);
    return {
      ball_speed_mph: v.ball_speed_mph,
      launch_angle_deg: v.launch_angle_deg,
      launch_azimuth_deg: v.launch_azimuth_deg,
      spin_rpm: v.spin_rpm,
      spin_axis_deg: v.spin_axis_deg,
      carry_m: flight.carryM,
      lateral_m: -flight.lateralM, // flight frame + left -> report + right
      apex_m: flight.maxHeightM,
      landing_angle_deg: flight.landingAngleDeg,
      flight_time_s: flight.flightTimeS,
    };
  }

  const impact = solveImpact({
    clubheadSpeedMps: Math.max(v.clubhead_speed_mps, 1e-3),
    clubPathDeg: clampAngle(v.club_path_deg),
    faceAngleDeg: clampAngle(v.face_angle_deg),
    attackAngleDeg: clampAngle(v.attack_angle_deg),
    dynamicLoftDeg: clampAngle(v.dynamic_loft_deg),
    impactOffsetToeMm: v.impact_offset_toe_mm,
    impactOffsetHighMm: v.impact_offset_high_mm,
  });
  const launch = deriveLaunch(
    toFlightFrame(impact.ballVelocity),
    toFlightFrame(impact.ballAngularVelocity),
  );
  const flight = simulateFlight(launch);
  return {
    club_path_deg: clampAngle(v.club_path_deg),
    face_angle_deg: clampAngle(v.face_angle_deg),
    attack_angle_deg: clampAngle(v.attack_angle_deg),
    dynamic_loft_deg: clampAngle(v.dynamic_loft_deg),
    ball_speed_mph: launch.ballSpeedMps * MPH_PER_MPS,
    launch_angle_deg: launch.launchAngleRad / RAD,
    launch_azimuth_deg: -launch.azimuthRad / RAD, // + = right of target
    spin_rpm: launch.spinRpm,
    // App-frame spin vector, same D-plane convention as objective.py.
    spin_axis_deg: spinAxisTiltDeg(impact.ballAngularVelocity),
    carry_m: flight.carryM,
    lateral_m: -flight.lateralM,
    apex_m: flight.maxHeightM,
    landing_angle_deg: flight.landingAngleDeg,
    flight_time_s: flight.flightTimeS,
  };
}

export interface VariationDatasetTs {
  plan: VariationPlanTs;
  inputNames: string[];
  inputs: number[][];
  outputNames: string[];
  outputs: (number | null)[][]; // null row entries for failed runs
  success: boolean[];
}

/** Execute a plan synchronously (bounded — see module docstring). */
export function runVariation(plan: VariationPlanTs): VariationDatasetTs {
  validatePlan(plan);
  const inputs = sampleInputs(plan);
  const base = resolvedBase(plan);
  const inputNames = plan.noise.map((s) => s.variableKey);
  const outputNames = outputsForMode(plan.mode);
  const outputs: (number | null)[][] = [];
  const success: boolean[] = [];
  for (let i = 0; i < plan.nRuns; i += 1) {
    const variables = { ...base };
    inputNames.forEach((key, j) => {
      variables[key] = inputs[i][j];
    });
    try {
      const result = evaluateRun(variables, plan.mode);
      outputs.push(outputNames.map((name) => result[name]));
      success.push(true);
    } catch {
      outputs.push(outputNames.map(() => null));
      success.push(false);
    }
  }
  return { plan, inputNames, inputs, outputNames, outputs, success };
}
