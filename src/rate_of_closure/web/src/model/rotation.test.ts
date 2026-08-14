import { describe, expect, it } from "vitest";

import { applyRotation, IDENTITY_ROTATION, rodrigues, slerpRotation } from "./rotation";

describe("rotation interpolation", () => {
  it("uses the shortest proper-rotation arc through an exact half-turn", () => {
    const halfTurn = rodrigues([0, 0, 1], Math.PI);
    const midpoint = slerpRotation(IDENTITY_ROTATION, halfTurn, 0.5);

    expect(applyRotation(midpoint, [1, 0, 0])).toEqual([
      expect.closeTo(0, 12), expect.closeTo(1, 12), expect.closeTo(0, 12),
    ]);
  });

  it("rejects extrapolation", () => {
    expect(() => slerpRotation(IDENTITY_ROTATION, IDENTITY_ROTATION, 1.01))
      .toThrow(/alpha/);
  });
});
