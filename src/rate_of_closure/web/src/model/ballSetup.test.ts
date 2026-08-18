import { describe, expect, it } from "vitest";

import { getClub } from "./club";
import {
  DRIVER_TEE_HEIGHT_M,
  GOLF_BALL_RADIUS_M,
  ballSetupToJson,
  ballSetupFromJson,
  ballCenterPosition,
  defaultBallSetupForClub,
  resolveBallSetup,
} from "./ballSetup";

describe("ball support contract", () => {
  it("uses the representative editable tee setup for drivers only", () => {
    expect(defaultBallSetupForClub(getClub("Driver 10.5°"))).toEqual({
      supportMode: "tee",
      teeHeightM: DRIVER_TEE_HEIGHT_M,
    });
    expect(defaultBallSetupForClub(getClub("7-Iron"))).toEqual({
      supportMode: "ground",
      teeHeightM: 0,
    });
  });

  it("defines tee height from ground to the bottom of the ball", () => {
    expect(ballCenterPosition({ supportMode: "ground", teeHeightM: 0 })).toEqual([
      0,
      GOLF_BALL_RADIUS_M,
      0,
    ]);
    expect(ballCenterPosition({ supportMode: "tee", teeHeightM: 0.0381 })[1])
      .toBeCloseTo(GOLF_BALL_RADIUS_M + 0.0381, 12);
  });

  it("requires exact zero on Ground and rejects non-finite or negative Tee heights", () => {
    expect(() => resolveBallSetup({ supportMode: "ground", teeHeightM: 0.05 }))
      .toThrow(/Ground support.*exactly 0/i);
    for (const teeHeightM of [-0.001, Number.NaN, Infinity]) {
      expect(() => resolveBallSetup({ supportMode: "tee", teeHeightM }))
        .toThrow(/Tee height.*finite.*non-negative/i);
    }
    expect(resolveBallSetup({ supportMode: "tee", teeHeightM: 0.15 }).teeHeightM)
      .toBe(0.15);
  });

  it("serializes the exact Python geometry contract", () => {
    const encoded = ballSetupToJson({ supportMode: "tee", teeHeightM: 0.0381 });
    expect(encoded).toEqual({
      support_mode: "tee",
      tee_height_m: 0.0381,
      height_reference: "ground_plane_to_ball_bottom",
      ball_center_m: [0, GOLF_BALL_RADIUS_M + 0.0381, 0],
    });
    expect(ballSetupFromJson(encoded)).toEqual({ supportMode: "tee", teeHeightM: 0.0381 });
    expect(ballSetupFromJson({})).toEqual({ supportMode: "ground", teeHeightM: 0 });
    expect(() => ballSetupFromJson({ ...encoded, ball_center_m: [0, 0, 0] }))
      .toThrow(/must match.*derived/i);
  });
});
