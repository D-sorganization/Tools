/** Target-region geometry — pinned parity with the Python tests (#4125 H7b). */

import { describe, expect, it } from "vitest";

import { solveGoals } from "./solver";
import {
  CENTERING_WEIGHT,
  contains,
  holdStats,
  residualM,
  signedDistance,
  type TargetRegionTs,
} from "./targets";

const green: TargetRegionTs = {
  kind: "green",
  distanceM: 200,
  radiusM: 10,
  lateralM: 0,
  bandHalfLengthM: 15,
  halfWidthM: 16,
};
const fairway: TargetRegionTs = {
  kind: "fairway",
  distanceM: 230,
  radiusM: 10,
  lateralM: 0,
  bandHalfLengthM: 20,
  halfWidthM: 15,
};

describe("green signed distance (Python parity pins)", () => {
  it("inside / boundary / outside", () => {
    expect(signedDistance(green, 200, 0)).toBeCloseTo(-10);
    expect(signedDistance(green, 210, 0)).toBeCloseTo(0);
    expect(signedDistance(green, 215, 0)).toBeCloseTo(5);
    expect(signedDistance(green, 206, 8)).toBeCloseTo(0); // 6-8-10
    expect(contains(green, 204, 4)).toBe(true);
    expect(contains(green, 212, 0)).toBe(false);
  });

  it("lateral offset moves the circle", () => {
    const offset = { ...green, distanceM: 150, radiusM: 5, lateralM: 20 };
    expect(contains(offset, 150, 20)).toBe(true);
    expect(signedDistance(offset, 150, 0)).toBeCloseTo(15);
  });
});

describe("fairway signed distance (Python parity pins)", () => {
  it("inside / boundary / outside", () => {
    expect(signedDistance(fairway, 230, 0)).toBeCloseTo(-15);
    expect(signedDistance(fairway, 230, 15)).toBeCloseTo(0);
    expect(signedDistance(fairway, 250, 0)).toBeCloseTo(0);
    expect(signedDistance(fairway, 253, 19)).toBeCloseTo(5); // corner 3-4-5
    expect(signedDistance(fairway, 230, 25)).toBeCloseTo(10);
    expect(signedDistance(fairway, 260, 5)).toBeCloseTo(10);
  });

  it("interior distance is the nearest edge", () => {
    const fw = { ...fairway, distanceM: 100, bandHalfLengthM: 30, halfWidthM: 10 };
    expect(signedDistance(fw, 110, 6)).toBeCloseTo(-4);
  });
});

describe("residual + hold stats", () => {
  it("residual is 0 at center, centering-only inside", () => {
    expect(residualM(green, 200, 0)).toBeCloseTo(0);
    expect(residualM(green, 205, 0)).toBeCloseTo(CENTERING_WEIGHT * 5);
    expect(residualM(green, 220, 0)).toBeCloseTo(10 + CENTERING_WEIGHT * 20);
  });

  it("hand-counted hold fixture matches Python (3 of 5, NaN excluded)", () => {
    const { held, total } = holdStats(
      [200, 205, 209, 215, 200, NaN],
      [0, 0, 0, 0, 11, 0],
      green,
    );
    expect(held).toBe(3);
    expect(total).toBe(5);
  });
});

describe("optimize to target (TS solver reuse)", () => {
  it("reaches a reachable green from a cold start", () => {
    const target: TargetRegionTs = {
      kind: "green",
      distanceM: 170,
      radiusM: 15,
      lateralM: 0,
      bandHalfLengthM: 15,
      halfWidthM: 16,
    };
    const result = solveGoals(
      {},
      {
        free: { clubheadSpeedMps: [30, 55], dynamicLoftDeg: [8, 18] },
        fixed: {},
      },
      400,
      target,
    );
    expect(result.achieved.targetDistanceM!).toBeLessThan(1.0);
    expect(
      signedDistance(
        target,
        result.achieved.carryM!,
        result.achieved.landingLateralM!,
      ),
    ).toBeCloseTo(result.achieved.targetDistanceM!);
  });
});
