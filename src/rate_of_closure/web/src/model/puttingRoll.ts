/**
 * Closed-form skid and pure-roll relations — TypeScript mirror of
 * `shared/python/swing_sim/putting/roll.py` (#4125 H3, #4800 P7).
 *
 * The green integrator (`puttingGreen.ts`) owns the *numerical* roll:
 * RK4 over the surface, with the sliding/rolling mode switch decided
 * per step. This module carries the analytic companions the Python
 * twin exports and the web runtime had not yet twinned — the
 * flat-green closed forms that the integrator is checked against and
 * that the UI reads for one-line green-reading answers:
 *
 * - `solveSkid` — the skid phase in closed form. Sliding friction
 *   decelerates the ball at `mu_s g` while spinning it up at
 *   `5 mu_s g / (2 r)`; the two meet at `v = omega r` after
 *   `(v0 - omega0 r) / (3.5 mu_s g)`, leaving pure roll at
 *   `(5 v0 + 2 omega0 r) / 7` — the classic 5/7 for a no-spin start.
 * - `rollOutDistance` / `rollTimeS` — the flat-green pure-roll
 *   stopping distance `v^2 / (2 mu_r g)` and time `v / (mu_r g)`.
 * - `rollingMuToStimp` — the exact inverse of `stimpToRollingMu`, so
 *   a measured rolling coefficient reads back as a stimpmeter number.
 *
 * Same constants and same algebra as the Python module, so the vitest
 * suite mirrors `swing_sim/putting/tests/test_roll.py` test-for-test.
 * The stimp -> mu direction, the sliding-friction default, and the
 * gravity constant stay single-sourced in `puttingGreen.ts`; they are
 * re-exported here so a caller reading the roll model has one import.
 */

import {
  DEFAULT_SLIDING_MU,
  GRAVITY_M_S2,
  STIMP_RELEASE_SPEED_MPS,
  stimpToRollingMu,
} from "./puttingGreen";

export {
  DEFAULT_SLIDING_MU,
  GRAVITY_M_S2,
  STIMP_RELEASE_SPEED_MPS,
  stimpToRollingMu,
};

const FOOT_M = 0.3048;

/** Closed-form skid phase (flat green; see the module header). */
export interface SkidSolution {
  /** Time until pure roll begins [s]. */
  durationS: number;
  /** Ground covered while skidding [m]. */
  distanceM: number;
  /** Speed when pure roll begins, `(5 v0 + 2 omega0 r) / 7` [m/s]. */
  exitSpeedMps: number;
}

function requireFinite(value: number, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number`);
  }
  return value;
}

/**
 * Stimpmeter reading for a rolling-resistance coefficient — the exact
 * inverse of `stimpToRollingMu` (round-trip exact).
 */
export function rollingMuToStimp(muR: number): number {
  requireFinite(muR, "muR");
  if (!(muR > 0 && muR < 0.2)) {
    throw new Error("rolling mu must be in (0, 0.2)");
  }
  return (
    (STIMP_RELEASE_SPEED_MPS * STIMP_RELEASE_SPEED_MPS) /
    (2.0 * GRAVITY_M_S2 * muR * FOOT_M)
  );
}

/**
 * Closed-form flat-green skid phase.
 *
 * `spinRadS` is topspin-positive, so a struck putt (backspin) starts
 * negative and skids longer. A ball that is already rolling
 * (`v0 <= omega0 r`) returns the zero-duration solution rather than a
 * negative skid.
 */
export function solveSkid(
  speedMps: number,
  spinRadS: number,
  ballRadiusM: number,
  muSlide: number = DEFAULT_SLIDING_MU,
): SkidSolution {
  requireFinite(speedMps, "speedMps");
  if (!(speedMps > 0)) throw new Error("speed must be positive");
  requireFinite(spinRadS, "spinRadS");
  requireFinite(ballRadiusM, "ballRadiusM");
  if (!(ballRadiusM >= 0.01 && ballRadiusM <= 0.05)) {
    throw new Error("ball radius must be plausible [m]");
  }
  requireFinite(muSlide, "muSlide");
  if (!(muSlide > 0 && muSlide <= 1.5)) {
    throw new Error("muSlide must be in (0, 1.5]");
  }

  const surfaceSpeed = spinRadS * ballRadiusM;
  if (speedMps <= surfaceSpeed) {
    return { durationS: 0.0, distanceM: 0.0, exitSpeedMps: speedMps };
  }
  const durationS = (speedMps - surfaceSpeed) / (3.5 * muSlide * GRAVITY_M_S2);
  const distanceM =
    speedMps * durationS - 0.5 * muSlide * GRAVITY_M_S2 * durationS * durationS;
  const exitSpeedMps = (5.0 * speedMps + 2.0 * surfaceSpeed) / 7.0;
  if (!(exitSpeedMps > 0)) throw new Error("roll must start moving forward");
  if (exitSpeedMps > speedMps) throw new Error("skid cannot speed the ball up");
  if (distanceM < 0) throw new Error("skid distance is non-negative");
  return { durationS, distanceM, exitSpeedMps };
}

function requireRollingMu(muRoll: number): number {
  requireFinite(muRoll, "muRoll");
  if (!(muRoll > 0 && muRoll < 0.2)) {
    throw new Error("muRoll must be in (0, 0.2)");
  }
  return muRoll;
}

function requireRollingSpeed(speedMps: number): number {
  requireFinite(speedMps, "speedMps");
  if (!(speedMps >= 0)) throw new Error("speed must be non-negative");
  return speedMps;
}

/** Flat-green pure-roll stopping distance `v^2 / (2 mu_r g)` [m]. */
export function rollOutDistance(speedMps: number, muRoll: number): number {
  const speed = requireRollingSpeed(speedMps);
  const mu = requireRollingMu(muRoll);
  return (speed * speed) / (2.0 * mu * GRAVITY_M_S2);
}

/** Flat-green pure-roll time to rest `v / (mu_r g)` [s]. */
export function rollTimeS(speedMps: number, muRoll: number): number {
  const speed = requireRollingSpeed(speedMps);
  const mu = requireRollingMu(muRoll);
  return speed / (mu * GRAVITY_M_S2);
}
