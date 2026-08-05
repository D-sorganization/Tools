/**
 * Parity pins for the putting vertical (epic #4125, H3).
 *
 * The pinned reference putts mirror
 * `tests/rate_of_closure/test_putting.py::TestReferencePuttPins`
 * value-for-value — both sides use the same closed forms and the same
 * fixed-step RK4 (dt = 2 ms), so tolerances are tight, not banded.
 */

import { describe, expect, it } from "vitest";

import {
  captureSpeedMps,
  clubheadSpeedFromBackstroke,
  GOLF_BALL_RADIUS_M,
  simulatePutt,
  STIMP_RELEASE_SPEED_MPS,
  stimpToRollingMu,
  strike,
} from "./putting";

/** The H1 club-library putter (350 g, 3 deg, COR 0.78). */
const PUTTER = {
  name: "Putter",
  headMassKg: 0.35,
  loftDeg: 3.0,
  cor: 0.78,
};

describe("strike — Python parity", () => {
  it("pins the reference launch (1.8 m/s clubhead)", () => {
    const launch = strike(PUTTER, 1.8);
    expect(launch.ballSpeedMps).toBeCloseTo(2.828565312464848, 10);
    expect(launch.launchAngleDeg).toBeCloseTo(3.5452147542505257, 8);
    expect(launch.horizontalSpeedMps).toBeCloseTo(2.8231523192738344, 10);
    expect(launch.spinRadS).toBeCloseTo(-3.153929533539754, 10);
  });

  it("zero loft gives a pure 1-D COR impulse", () => {
    const launch = strike({ ...PUTTER, loftDeg: 0 }, 2.0);
    const expected = (2.0 * 1.78 * 0.35) / (0.35 + 0.04593);
    expect(launch.ballSpeedMps).toBeCloseTo(expected, 12);
    expect(launch.spinRadS).toBeCloseTo(0, 12);
  });

  it("rejects out-of-range speeds", () => {
    expect(() => strike(PUTTER, 0)).toThrow();
    expect(() => strike(PUTTER, 2, -30)).toThrow();
  });
});

describe("stimpmeter — Python parity", () => {
  it("derives the quoted ~1.83 m/s release speed", () => {
    expect(STIMP_RELEASE_SPEED_MPS).toBeCloseTo(1.8287317526214812, 10);
  });

  it("round-trips stimp through the roll-out formula", () => {
    for (const stimp of [6, 10, 14]) {
      const mu = stimpToRollingMu(stimp);
      const rolloutFt =
        (STIMP_RELEASE_SPEED_MPS * STIMP_RELEASE_SPEED_MPS) /
        (2 * mu * 9.80665) /
        0.3048;
      expect(rolloutFt).toBeCloseTo(stimp, 10);
    }
  });

  it("pins mu at stimp 10", () => {
    expect(stimpToRollingMu(10)).toBeCloseTo(0.05594153480923128, 12);
  });
});

describe("capture bound", () => {
  it("pins R sqrt(g / 2r) ~= 0.82 m/s", () => {
    expect(captureSpeedMps()).toBeCloseTo(0.8186396513958939, 12);
    expect(GOLF_BALL_RADIUS_M).toBeCloseTo(0.021335, 6);
  });
});

describe("simulatePutt — Python parity", () => {
  it("pins the breaking reference putt (stimp 10, 2 %, aspect 90)", () => {
    const launch = strike(PUTTER, 1.8);
    const result = simulatePutt(
      launch,
      { stimpFt: 10, gradePercent: 2, aspectDeg: 90 },
      3.0,
    );
    expect(result.holed).toBe(false);
    expect(result.totalDistanceM).toBeCloseTo(4.417405938785078, 7);
    expect(result.skidDistanceM).toBeCloseTo(0.5103817275162047, 7);
    expect(result.timeS).toBeCloseTo(4.388, 2);
    expect(result.breakM).toBeCloseTo(0.8176068791755766, 7);
    expect(result.missDistanceM).toBeCloseTo(1.4994647284222105, 7);
  });

  it("pins the holed reference putt (1.6 m/s, flat stimp 10)", () => {
    const launch = strike(PUTTER, 1.6);
    const result = simulatePutt(
      launch,
      { stimpFt: 10, gradePercent: 0, aspectDeg: 0 },
      3.0,
    );
    expect(result.holed).toBe(true);
    expect(result.speedAtHoleMps).toBeCloseTo(0.5903262895096224, 7);
    expect(result.marginMps).toBeCloseTo(0.2283133618862715, 7);
  });

  it("mirror aspect mirrors the break", () => {
    const launch = strike(PUTTER, 2.0);
    const left = simulatePutt(
      launch,
      { stimpFt: 10, gradePercent: 2, aspectDeg: 90 },
      10,
    );
    const right = simulatePutt(
      launch,
      { stimpFt: 10, gradePercent: 2, aspectDeg: -90 },
      10,
    );
    expect(left.breakM).toBeCloseTo(-right.breakM, 9);
    expect(left.totalDistanceM).toBeCloseTo(right.totalDistanceM, 9);
  });

  it("speed is monotone non-increasing on a flat green", () => {
    const result = simulatePutt(
      strike(PUTTER, 2.0),
      { stimpFt: 10, gradePercent: 0, aspectDeg: 0 },
      10,
    );
    for (let i = 1; i < result.speedsMps.length; i++) {
      expect(result.speedsMps[i]).toBeLessThanOrEqual(
        result.speedsMps[i - 1] + 1e-12,
      );
    }
  });

  it("backstroke proxy matches the pendulum formula", () => {
    expect(clubheadSpeedFromBackstroke(0.3)).toBeCloseTo(
      0.3 * Math.sqrt(9.80665 / 0.889),
      12,
    );
  });
});
