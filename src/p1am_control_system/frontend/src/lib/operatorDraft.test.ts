import { describe, it, expect } from "vitest";
import { seedDraftText } from "./operatorDraft";

/**
 * Shared "seed a setpoint entry from the device, but never fight the operator"
 * rule (#4013, #4020).
 *
 * Two opposite defects motivated this:
 *  - The Alicat inspector re-seeded its entry from live telemetry on EVERY
 *    frame, so on real hardware (where mass_flow/pressure/temperature change
 *    each scan) the field was overwritten between keystrokes. Pressing Set
 *    could commit a partially-reverted value — 2 instead of 25.
 *  - The power-supply entry was NEVER seeded, so it read "0" after a kiosk
 *    reload while the supply ran at 30 A.
 *
 * One rule serves both: seed while untouched, never once touched.
 */

describe("seedDraftText", () => {
  it("formats the device value while the operator has not typed", () => {
    expect(seedDraftText(30, false, 2)).toBe("30.00");
    expect(seedDraftText(25, false, 1)).toBe("25.0");
    expect(seedDraftText(0, false, 2)).toBe("0.00");
  });

  it("returns null once the operator has touched the field", () => {
    // This is the anti-keystroke-clobber guarantee: a live frame arriving
    // mid-entry must produce NO new text for the controlled input.
    expect(seedDraftText(30, true, 2)).toBeNull();
    expect(seedDraftText(0, true, 2)).toBeNull();
  });

  it("returns null when the device value is absent or non-finite", () => {
    expect(seedDraftText(undefined, false, 2)).toBeNull();
    expect(seedDraftText(null, false, 2)).toBeNull();
    expect(seedDraftText(Number.NaN, false, 2)).toBeNull();
    expect(seedDraftText(Number.POSITIVE_INFINITY, false, 2)).toBeNull();
  });

  it("defaults to one decimal", () => {
    expect(seedDraftText(37.25, false)).toBe("37.3");
  });

  it("rejects a non-boolean touched flag (DbC — a truthy object would disable the guard)", () => {
    expect(() =>
      seedDraftText(30, "yes" as unknown as boolean, 2),
    ).toThrow(TypeError);
  });

  it("rejects a nonsensical digit count", () => {
    expect(() => seedDraftText(30, false, -1)).toThrow(RangeError);
    expect(() => seedDraftText(30, false, 1.5)).toThrow(RangeError);
  });
});
