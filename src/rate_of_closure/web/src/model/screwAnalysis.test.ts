import { describe, expect, it } from "vitest";

import {
  analyzeTwist,
  buildScrewGlyph,
  jointMotionAt,
  projectMotion,
  type Vec3,
} from "./screwAnalysis";

const closeVector = (actual: Vec3, expected: Vec3): void => {
  actual.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 10));
};

describe("screw motion analysis", () => {
  it("reconstructs a finite screw at the named reference point", () => {
    const motion = analyzeTwist(
      [0, 0, 4, 0, 4, 0.5],
      [1, 0, 0],
    );

    expect(motion.kind).toBe("finite");
    closeVector(motion.axisDirection, [0, 0, 1]);
    closeVector(motion.axisPointM!, [0, 0, 0]);
    expect(motion.pitchMPerRad).toBeCloseTo(0.125, 12);
    closeVector(
      motion.orbitalVelocityMps.map((value, index) =>
        value + motion.axialVelocityMps[index]) as Vec3,
      motion.referenceVelocityMps,
    );
  });

  it("does not invent a finite axis for translation or rest", () => {
    const translation = analyzeTwist([0, 0, 0, 3, 4, 0], [0, 0, 0]);
    const stationary = analyzeTwist([0, 0, 0, 0, 0, 0], [0, 0, 0]);

    expect(translation.kind).toBe("translation");
    expect(translation.axisPointM).toBeNull();
    expect(stationary.kind).toBe("stationary");
    expect(stationary.axisPointM).toBeNull();
    expect(buildScrewGlyph(translation, 2)).toBeNull();
  });

  it("returns signed target, vertical, and lateral projections", () => {
    const projections = projectMotion(
      analyzeTwist([0, 0, 2, 4, -3, 1.5], [0, 0, 0]),
    );

    expect(projections.target.totalMps).toBeCloseTo(4);
    expect(projections.vertical.totalMps).toBeCloseTo(-3);
    expect(projections.lateral.totalMps).toBeCloseTo(1.5);
  });

  it("builds bounded helix geometry around the finite axis", () => {
    const glyph = buildScrewGlyph(
      analyzeTwist([0, 0, 20, 5, 0, 2], [0, 0, 0]),
      2,
    );

    expect(glyph).not.toBeNull();
    expect(glyph!.axisLineM).toHaveLength(2);
    expect(glyph!.helixM).toHaveLength(96);
    expect(glyph!.handedness).toBe(1);
  });

  it("reconstructs endpoint velocity from planar revolute joints", () => {
    const times = Array.from({ length: 21 }, (_, index) => index * 0.001);
    const positions = times.map((time) => {
      const first = 2 * time;
      const second = -1.25 * time;
      const shoulder: Vec3 = [0, 0, 0];
      const wrist: Vec3 = [0.8 * Math.cos(first), 0.8 * Math.sin(first), 0];
      const head: Vec3 = [
        wrist[0] + 1.1 * Math.cos(second),
        wrist[1] + 1.1 * Math.sin(second),
        0,
      ];
      return [shoulder, wrist, head];
    });

    const result = jointMotionAt(
      times,
      positions,
      ["joint.shoulder", "joint.wrist"],
      10,
    );

    expect(result.contributionVelocityMps).toHaveLength(2);
    expect(result.reconstructionResidualMps).toBeLessThan(2e-5);
  });
});
