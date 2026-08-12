import { describe, expect, it } from "vitest";

import { assessFixedContact, deliveryInspectionOutcome } from "./contact";

const BALL = [0, 0.021335, 0] as const;

describe("contact outcomes", () => {
  it("labels forced alignment as inspection rather than collision detection", () => {
    const outcome = deliveryInspectionOutcome(0.25, BALL, BALL[1]);
    expect(outcome.status).toBe("hit");
    expect(outcome.geometryModel).toBe("forced_reference_point_alignment");
    expect(outcome.geometryLimitations).toContain("not geometric contact detection");
  });

  it("reports the sampled instant of closest approach for a miss", () => {
    const outcome = assessFixedContact(
      [
        { t: 0, position: [1, 0, 0] },
        { t: 0.1, position: [0.5, 0, 0] },
        { t: 0.2, position: [0.8, 0, 0] },
      ],
      BALL,
      BALL[1],
    );
    expect(outcome.status).toBe("miss");
    expect(outcome.candidateTimeS).toBe(0.1);
    expect(outcome.contactMarginM).toBeLessThan(0);
    expect(outcome.geometryLimitations).toContain("swept contact");
  });

  it("accepts a point on the spherical threshold as a hit", () => {
    const outcome = assessFixedContact(
      [{ t: 0.5, position: [0, 0, 0] }],
      BALL,
      BALL[1],
    );
    expect(outcome.status).toBe("hit");
    expect(outcome.contactMarginM).toBeCloseTo(0, 12);
  });

  it("rejects malformed contact inputs", () => {
    expect(() => assessFixedContact([], BALL, BALL[1])).toThrow(/at least one/);
    expect(() => assessFixedContact([{ t: Number.NaN, position: [0, 0, 0] }], BALL, BALL[1])).toThrow(
      /finite/,
    );
    expect(() => deliveryInspectionOutcome(-1, BALL, BALL[1])).toThrow(/non-negative/);
  });
});
