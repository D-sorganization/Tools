/**
 * Skid / pure-roll / stimpmeter gates (#4125 H3, #4800 P7).
 *
 * Mirrors `swing_sim/putting/tests/test_roll.py` test-for-test: the
 * stimpmeter round trip (release speed rolled out on a stimp-S green
 * travels exactly S feet), skid transition continuity at `v = omega r`,
 * the classic 5/7 no-spin exit, and the constant-deceleration
 * distance/time identity. Analytic identities only — nothing here is
 * pinned to an implementation-tuned number.
 */

import { describe, expect, it } from "vitest";

import { GOLF_BALL_RADIUS_M } from "./puttingGreen";
import {
  rollOutDistance,
  rollTimeS,
  rollingMuToStimp,
  solveSkid,
  stimpToRollingMu,
  STIMP_RELEASE_SPEED_MPS,
} from "./puttingRoll";

const FOOT_M = 0.3048;
const GRAVITY_M_S2 = 9.80665;

/** Relative closeness matching the Python `pytest.approx(rel=1e-12)`. */
function expectClose(actual: number, expected: number, rel = 1e-12): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(
    rel * Math.abs(expected),
  );
}

describe("stimpmeter", () => {
  it("derives the quoted ~1.83 m/s release speed", () => {
    expect(Math.abs(STIMP_RELEASE_SPEED_MPS - 1.83)).toBeLessThanOrEqual(0.01);
  });

  it("rolls the release speed exactly S feet on a stimp-S green", () => {
    for (const stimp of [6.0, 8.5, 10.0, 12.0, 14.0]) {
      const mu = stimpToRollingMu(stimp);
      const distanceFt = rollOutDistance(STIMP_RELEASE_SPEED_MPS, mu) / FOOT_M;
      expectClose(distanceFt, stimp);
    }
  });

  it("inverts mu back to the stimp reading", () => {
    for (const stimp of [7.0, 10.0, 13.0]) {
      expectClose(rollingMuToStimp(stimpToRollingMu(stimp)), stimp);
    }
  });

  it("gives a faster green the lower rolling coefficient", () => {
    expect(stimpToRollingMu(13.0)).toBeLessThan(stimpToRollingMu(8.0));
  });

  it("puts a stimp-10 green in the published band", () => {
    const mu = stimpToRollingMu(10.0);
    expect(mu).toBeGreaterThanOrEqual(0.05);
    expect(mu).toBeLessThanOrEqual(0.07);
  });

  it("refuses out-of-range readings and coefficients", () => {
    expect(() => stimpToRollingMu(1.0)).toThrow();
    expect(() => rollingMuToStimp(0.5)).toThrow();
    expect(() => rollingMuToStimp(Number.NaN)).toThrow();
  });
});

describe("skid phase", () => {
  it("meets v = omega r exactly at the transition", () => {
    const v0 = 2.0;
    const mu = 0.4;
    const spin0 = -50.0; // backspin [rad/s]
    const solution = solveSkid(v0, spin0, GOLF_BALL_RADIUS_M, mu);
    const vEnd = v0 - mu * GRAVITY_M_S2 * solution.durationS;
    const omegaREnd =
      spin0 * GOLF_BALL_RADIUS_M + 2.5 * mu * GRAVITY_M_S2 * solution.durationS;
    expectClose(vEnd, omegaREnd);
    expectClose(solution.exitSpeedMps, vEnd);
  });

  it("gives the classic five sevenths for a no-spin start", () => {
    const solution = solveSkid(2.1, 0.0, GOLF_BALL_RADIUS_M);
    expectClose(solution.exitSpeedMps, (2.1 * 5.0) / 7.0);
  });

  it("skids zero when the ball is already rolling", () => {
    const omega = 2.0 / GOLF_BALL_RADIUS_M;
    const solution = solveSkid(2.0, omega, GOLF_BALL_RADIUS_M);
    expect(solution.durationS).toBe(0.0);
    expect(solution.distanceM).toBe(0.0);
    expect(solution.exitSpeedMps).toBe(2.0);
  });

  it("extends the skid under backspin", () => {
    const clean = solveSkid(2.0, 0.0, GOLF_BALL_RADIUS_M);
    const spun = solveSkid(2.0, -80.0, GOLF_BALL_RADIUS_M);
    expect(spun.durationS).toBeGreaterThan(clean.durationS);
    expect(spun.exitSpeedMps).toBeLessThan(clean.exitSpeedMps);
  });

  it("refuses bad inputs", () => {
    expect(() => solveSkid(-1.0, 0.0, GOLF_BALL_RADIUS_M)).toThrow();
    expect(() => solveSkid(2.0, 0.0, 0.5)).toThrow();
    expect(() => solveSkid(2.0, 0.0, GOLF_BALL_RADIUS_M, 0.0)).toThrow();
    expect(() => solveSkid(2.0, Number.NaN, GOLF_BALL_RADIUS_M)).toThrow();
  });
});

describe("pure roll", () => {
  it("is quadratic in speed", () => {
    const mu = stimpToRollingMu(10.0);
    expectClose(rollOutDistance(2.0, mu), 4.0 * rollOutDistance(1.0, mu));
  });

  it("keeps d = v t / 2 for constant deceleration to rest", () => {
    const mu = stimpToRollingMu(11.0);
    const v = 1.7;
    expectClose(rollOutDistance(v, mu), 0.5 * v * rollTimeS(v, mu));
  });

  it("refuses negative speeds and out-of-band coefficients", () => {
    expect(() => rollOutDistance(-1.0, 0.06)).toThrow();
    expect(() => rollTimeS(1.0, 0.5)).toThrow();
  });
});
