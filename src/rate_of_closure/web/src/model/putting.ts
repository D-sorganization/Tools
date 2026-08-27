/**
 * Putting vertical — TypeScript mirror of
 * `shared/python/swing_sim/putting` (epic #4125 H3, #4800 P2).
 *
 * Same derivations, same constants, same fixed-step RK4 (dt = 2 ms),
 * so the vitest parity suite pins the Python reference putt
 * value-for-value (`tests/rate_of_closure/test_putting.py`).
 *
 * This module carries the impact side (putter-ball strike and the
 * backstroke proxy). The green surface, the 2-D roll integration, and
 * hole capture live in `puttingGreen.ts` (#4800 P2); the legacy
 * planar `simulatePutt` and its constants are re-exported from there
 * unchanged, so existing imports keep working and the planar results
 * stay bit-identical.
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

import type { PuttLaunch } from "./puttingGreen";

export const GOLF_BALL_MASS_KG = 0.04593;
export const DEFAULT_PUTTER_COR = 0.78;

const ROLLING_CAP = 2.0 / 7.0;

export interface PutterSpec {
  name: string;
  headMassKg: number;
  loftDeg: number;
  cor: number;
}

/** H3-local minimal putters (H1 club-library reconciliation note). */
export const MINIMAL_PUTTERS: PutterSpec[] = [
  { name: "Blade Putter", headMassKg: 0.35, loftDeg: 3.0, cor: DEFAULT_PUTTER_COR },
  { name: "Mallet Putter", headMassKg: 0.36, loftDeg: 3.0, cor: DEFAULT_PUTTER_COR },
];

/** Putter-ball impact (COR impulse + 2/7 tangential cap). */
export function strike(
  putter: PutterSpec,
  clubheadSpeedMps: number,
  shaftLeanDeg = 0.0,
): PuttLaunch {
  if (!(clubheadSpeedMps > 0 && clubheadSpeedMps <= 10)) {
    throw new Error("clubheadSpeedMps must be in (0, 10]");
  }
  const effectiveLoftDeg = putter.loftDeg + shaftLeanDeg;
  if (effectiveLoftDeg < -2 || effectiveLoftDeg > 15) {
    throw new Error("effective loft must stay in [-2, 15] deg");
  }
  const delta = (effectiveLoftDeg * Math.PI) / 180.0;
  const massRatio = putter.headMassKg / (putter.headMassKg + GOLF_BALL_MASS_KG);
  const transfer = (1.0 + putter.cor) * massRatio;
  const vNormal = transfer * clubheadSpeedMps * Math.cos(delta);
  const uTangential = clubheadSpeedMps * Math.sin(delta);
  const vTangential = ROLLING_CAP * uTangential;
  const spinRadS = (-(1.0 - ROLLING_CAP) * uTangential) / GOLF_BALL_RADIUS_M;
  const horizontal =
    vNormal * Math.cos(delta) - vTangential * Math.sin(delta);
  const vertical = vNormal * Math.sin(delta) + vTangential * Math.cos(delta);
  return {
    ballSpeedMps: Math.hypot(horizontal, vertical),
    launchAngleDeg: (Math.atan2(vertical, horizontal) * 180.0) / Math.PI,
    horizontalSpeedMps: horizontal,
    spinRadS,
    effectiveLoftDeg,
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
