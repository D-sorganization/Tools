/**
 * Putting vertical â€” TypeScript mirror of
 * `shared/python/swing_sim/putting` (epic #4125 H3, #4800 P2).
 *
 * Same derivations, same constants, same fixed-step RK4 (dt = 2 ms),
 * so the vitest parity suite pins the Python reference putt
 * value-for-value (`tests/rate_of_closure/test_putting.py`).
 *
 * This module carries the impact side (putter-ball strike, the 2-D
 * stroke/impact options of #4800 P1, and the
 * backstroke proxy). The green surface, the 2-D roll integration, and
 * hole capture live in `puttingGreen.ts` (#4800 P2); the closed-form
 * skid and pure-roll relations live in `puttingRoll.ts`. Both are
 * re-exported here so this module is the package façade its Python
 * twin `swing_sim/putting/__init__.py` is, export-for-export — the
 * legacy planar `simulatePutt` and its constants included, so existing
 * imports keep working and the planar results stay bit-identical.
 *
 * Two modules are deliberately **not** re-exported, mirroring the same
 * refusal the Python façade documents: `puttingDispersion.ts` and
 * `puttingScenario.ts`. They carry the P5 study vocabulary, whose
 * sampler is Python-authoritative; importing them from here would
 * suggest this runtime owns a Monte-Carlo engine it does not.
 *
 * Physics summary (full derivations in the Python docstrings):
 * - Impact: 1-D COR impulse along the lofted face normal plus the 2/7
 *   rolling-cap tangential transfer -> launch speed, angle, backspin.
 * - Skid: sliding friction decelerates the ball and spins it up until
 *   v = omega r (pure roll at (5 v0 + 2 omega0 r) / 7).
 * - Green speed: the USGA stimpmeter (36 in ramp, 20 deg release,
 *   ~1.83 m/s release speed) inverts to mu_r = v^2 / (2 g S).
 * - Capture: the ball must fall half its diameter while crossing the
 *   hole mouth -> v_capture = R sqrt(g / 2r) ~= 0.82 m/s.
 */

import { GOLF_BALL_RADIUS_M } from "./puttingGreen";

export {
  captureSpeedMps,
  DEFAULT_SLIDING_MU,
  GOLF_BALL_RADIUS_M,
  GRAVITY_M_S2,
  HOLE_RADIUS_M,
  simulatePutt,
  STIMP_RELEASE_SPEED_MPS,
  stimpToRollingMu,
} from "./puttingGreen";
export type { GreenConditions, PuttLaunch, PuttResult } from "./puttingGreen";

export {
  rollingMuToStimp,
  rollOutDistance,
  rollTimeS,
  solveSkid,
} from "./puttingRoll";
export type { SkidSolution } from "./puttingRoll";

import type { PuttLaunch } from "./puttingGreen";

export const GOLF_BALL_MASS_KG = 0.04593;
export const DEFAULT_PUTTER_COR = 0.78;

/** Typical putter-head MOI about the CG heel-toe axis [kg m^2]. */
export const DEFAULT_PUTTER_MOI_KG_M2 = 4.5e-4;

const ROLLING_CAP = 2.0 / 7.0;

export interface PutterSpec {
  name: string;
  headMassKg: number;
  loftDeg: number;
  cor: number;
}

/** H3-local minimal putters (H1 club-library reconciliation note). */
export const MINIMAL_PUTTERS: PutterSpec[] = [
  {
    name: "Blade Putter",
    headMassKg: 0.35,
    loftDeg: 3.0,
    cor: DEFAULT_PUTTER_COR,
  },
  {
    name: "Mallet Putter",
    headMassKg: 0.36,
    loftDeg: 3.0,
    cor: DEFAULT_PUTTER_COR,
  },
];

/** 2-D stroke/impact parameters for `strike` (#4800 P1); all default 0. */
export interface StrikeOptions {
  /** Start-line aim off the target line [deg]; + = right. */
  aimDeg?: number;
  /** Face angle off the aim line [deg]; + = open. */
  faceAngleDeg?: number;
  /** Putter path off the aim line [deg]; + = in-to-out. */
  pathAngleDeg?: number;
  /** Attack angle [deg]; + = hitting up. */
  attackAngleDeg?: number;
  /** Strike location toward the toe [mm]. */
  strikeOffsetToeMm?: number;
  /** Strike location up the face [mm]. */
  strikeOffsetHighMm?: number;
  /** P3 hook: head MOI about CG [kg m^2]; default catalogue value. */
  headMoiKgM2?: number;
}

/**
 * Putter-ball impact (COR impulse + 2/7 tangential cap).
 *
 * Twin of the Python `strike` op-for-op: the stroke-plane solve is the
 * H3 model at the spin loft (loft - attack) rotated back by the attack
 * angle, the horizontal face-vs-path split follows the 2/7 rolling
 * cap, and off-center strikes reduce the head's effective mass by
 * 1/(1/M + r^2/I). Defaults are bit-identical to the pre-#4800 1-D
 * results.
 */
export function strike(
  putter: PutterSpec,
  clubheadSpeedMps: number,
  shaftLeanDeg = 0.0,
  options: StrikeOptions = {},
): PuttLaunch {
  if (!(clubheadSpeedMps > 0 && clubheadSpeedMps <= 10)) {
    throw new Error("clubheadSpeedMps must be in (0, 10]");
  }
  if (!(Math.abs(shaftLeanDeg) <= 10)) {
    throw new Error("shaft lean must be within +/-10 deg");
  }
  const aimDeg = options.aimDeg ?? 0.0;
  const faceAngleDeg = options.faceAngleDeg ?? 0.0;
  const pathAngleDeg = options.pathAngleDeg ?? 0.0;
  const attackAngleDeg = options.attackAngleDeg ?? 0.0;
  const strikeOffsetToeMm = options.strikeOffsetToeMm ?? 0.0;
  const strikeOffsetHighMm = options.strikeOffsetHighMm ?? 0.0;
  const bounds: Array<[string, number, number]> = [
    ["aimDeg", aimDeg, 45.0],
    ["faceAngleDeg", faceAngleDeg, 20.0],
    ["pathAngleDeg", pathAngleDeg, 20.0],
    ["attackAngleDeg", attackAngleDeg, 10.0],
    ["strikeOffsetToeMm", strikeOffsetToeMm, 40.0],
    ["strikeOffsetHighMm", strikeOffsetHighMm, 20.0],
  ];
  for (const [name, value, bound] of bounds) {
    if (!Number.isFinite(value) || Math.abs(value) > bound) {
      throw new Error(`${name} must be within +/-${bound}`);
    }
  }
  if (options.headMoiKgM2 !== undefined) {
    const moi = options.headMoiKgM2;
    if (!Number.isFinite(moi) || moi < 1e-5 || moi > 1e-2) {
      throw new Error("head MOI must be plausible [kg m^2]");
    }
  }
  const effectiveLoftDeg = putter.loftDeg + shaftLeanDeg;
  if (effectiveLoftDeg < -2 || effectiveLoftDeg > 15) {
    throw new Error("effective loft must stay in [-2, 15] deg");
  }

  // Off-center strike: scalar effective-mass reduction 1/(1/M + r^2/I).
  const offsetRM = Math.hypot(strikeOffsetToeMm, strikeOffsetHighMm) * 1e-3;
  let headMassEff = putter.headMassKg;
  if (offsetRM > 0.0) {
    const moi = options.headMoiKgM2 ?? DEFAULT_PUTTER_MOI_KG_M2;
    headMassEff = 1.0 / (1.0 / putter.headMassKg + (offsetRM * offsetRM) / moi);
  }

  const delta = (effectiveLoftDeg * Math.PI) / 180.0;
  const alpha = (attackAngleDeg * Math.PI) / 180.0;
  // Spin loft: face-normal-to-velocity angle in the stroke plane.
  const beta = delta - alpha;
  const massRatio = headMassEff / (headMassEff + GOLF_BALL_MASS_KG);
  const transfer = (1.0 + putter.cor) * massRatio;

  // Stroke-plane solve (H3 model in the velocity-aligned frame).
  const vNormal = transfer * clubheadSpeedMps * Math.cos(beta);
  const uTangential = -clubheadSpeedMps * Math.sin(beta);
  const vTangential = ROLLING_CAP * uTangential;
  const spinRadS = (-(1.0 - ROLLING_CAP) * uTangential) / GOLF_BALL_RADIUS_M;
  const along = vNormal * Math.cos(beta) - vTangential * Math.sin(beta);
  const lift = vNormal * Math.sin(beta) + vTangential * Math.cos(beta);
  // Rotate by the attack angle back to horizontal/vertical.
  const cosA = Math.cos(alpha);
  const sinA = Math.sin(alpha);
  let horizontal = along * cosA - lift * sinA;
  let vertical = along * sinA + lift * cosA;

  // Horizontal face-vs-path split: normal impulse along the face
  // azimuth, 2/7 tangential impulse toward the path.
  const faceToPath = ((pathAngleDeg - faceAngleDeg) * Math.PI) / 180.0;
  const sinFP = Math.sin(faceToPath);
  const cosFP = Math.cos(faceToPath);
  const deflectionRad = Math.atan2(ROLLING_CAP * sinFP, transfer * cosFP);
  const startAzimuthDeg =
    aimDeg + faceAngleDeg + (deflectionRad * 180.0) / Math.PI;
  // Mismatch trims the normal impulse; exactly 1.0 when square.
  const scale = Math.hypot(transfer * cosFP, ROLLING_CAP * sinFP) / transfer;
  horizontal *= scale;
  vertical *= scale;
  const sidespinRadS =
    ((1.0 - ROLLING_CAP) * clubheadSpeedMps * cosA * sinFP) /
    GOLF_BALL_RADIUS_M;

  return {
    ballSpeedMps: Math.hypot(horizontal, vertical),
    launchAngleDeg: (Math.atan2(vertical, horizontal) * 180.0) / Math.PI,
    horizontalSpeedMps: horizontal,
    spinRadS,
    effectiveLoftDeg,
    startAzimuthDeg,
    sidespinRadS,
  };
}

/** Pendulum backstroke proxy: v = A sqrt(g / L). */
export function clubheadSpeedFromBackstroke(
  backstrokeM: number,
  putterLengthM = 0.889,
): number {
  if (!(backstrokeM > 0 && backstrokeM <= 1.5)) {
    throw new Error("backstrokeM must be in (0, 1.5]");
  }
  return backstrokeM * Math.sqrt(9.80665 / putterLengthM);
}
