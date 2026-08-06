import { describe, expect, it } from "vitest";

import {
  FIRM_FAIRWAY_TURF,
  simulateReducedTurfContact,
} from "./turfContact";

describe("reduced passive turf contact", () => {
  it("matches the shared Python firm-fairway parity case", () => {
    const result = simulateReducedTurfContact(
      FIRM_FAIRWAY_TURF,
      [13, -2, 0.5],
      0.3,
    );

    expect(result.status).toBe("separated");
    expect(result.durationS).toBeCloseTo(0.00476, 12);
    expect(result.stepCount).toBe(952);
    expect(result.peakPenetrationM).toBeCloseTo(0.001861886589858214, 12);
    expect(result.normalImpulseNs).toBeCloseTo(0.7043813775238607, 10);
    expect(result.dissipatedEnergyJ).toBeCloseTo(3.68903042074591, 9);
  });

  it("returns typed no-contact evidence for an upward-moving sole", () => {
    const result = simulateReducedTurfContact(
      FIRM_FAIRWAY_TURF,
      [2, 0.1, 0],
      0.3,
    );

    expect(result.status).toBe("no_contact");
    expect(result.stepCount).toBe(0);
    expect(result.peakPenetrationM).toBe(0);
  });

  it.each([
    { normalStiffnessNM: -1 },
    { normalDampingNsM: -1 },
    { frictionCoefficient: -0.1 },
    { frictionCoefficient: 1.1 },
    { frictionRegularizationMps: 0 },
    { maxPenetrationM: 0 },
    { maxPenetrationM: 0.251 },
  ])("rejects nonphysical profile fields: %o", (override) => {
    expect(() => simulateReducedTurfContact(
      { ...FIRM_FAIRWAY_TURF, ...override },
      [13, -2, 0.5],
      0.3,
    )).toThrow(RangeError);
  });

  it("rejects malformed profile identity, status, and velocity shape", () => {
    expect(() => simulateReducedTurfContact(
      { ...FIRM_FAIRWAY_TURF, profileId: "  " },
      [13, -2, 0.5],
      0.3,
    )).toThrow(/profileId/);
    expect(() => simulateReducedTurfContact(
      { ...FIRM_FAIRWAY_TURF, calibrationStatus: "unknown" as "calibrated" },
      [13, -2, 0.5],
      0.3,
    )).toThrow(/calibrationStatus/);
    expect(() => simulateReducedTurfContact(
      FIRM_FAIRWAY_TURF,
      [13, -2] as unknown as [number, number, number],
      0.3,
    )).toThrow(/three finite components/);
  });
});
