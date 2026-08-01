/**
 * Seeding rule for operator-owned setpoint entry fields (#4013, #4020).
 *
 * A controlled input that commands hardware has exactly one safe policy: show
 * the device's value until the operator starts entering their own, then never
 * touch it again until the entry is re-scoped (a different device, a fresh
 * mount). Both halves of that policy were broken, in opposite directions:
 *
 *  - `AlicatInspector`'s draft was re-seeded from an effect keyed on the whole
 *    `alicats` list. On real hardware `mass_flow`, `pressure` and `temperature`
 *    change every scan, so the list got a new reference on essentially every
 *    frame and the field fought each keystroke. Pressing Set could commit a
 *    partially-reverted value — 2 instead of 25 — which passed the in-range
 *    check and became a tenfold gas-flow error with no indication.
 *  - `PowerSupplyControl`'s staged setpoint was never seeded at all: it read
 *    "0" after a kiosk reload while telemetry correctly showed 30 A, so one tap
 *    of "+" commanded 1 A and collapsed the output by 29 A.
 *
 * `TemperatureControl` already had the right shape (a pure decision helper plus
 * an `operatorTouchedRef` latch); this is that decision, shared.
 *
 * `TemperatureControl.recallSetpointText` now delegates here: it kept its own
 * copy of this logic only because the 1975-line file could not be edited without
 * failing the file-size guardrail, which the split of that component resolved.
 */

/**
 * Decide what a setpoint entry should be seeded to, if anything.
 *
 * @param deviceValue - the device's current setpoint (null/undefined until the
 *   first status frame arrives).
 * @param operatorTouched - true once the operator has typed or nudged the
 *   field this session. While true the entry is operator-owned and this always
 *   returns null, so live frames cannot overwrite an in-progress edit.
 * @param digits - decimal places for the formatted text.
 * @returns the text to seed the entry with, or `null` to leave it alone.
 * @throws TypeError if `operatorTouched` is not a boolean — a truthy object
 *   would silently disable the anti-clobber guard.
 * @throws RangeError if `digits` is not a non-negative integer.
 */
export function seedDraftText(
  deviceValue: number | null | undefined,
  operatorTouched: boolean,
  digits: number = 1,
): string | null {
  if (typeof operatorTouched !== "boolean") {
    throw new TypeError("operatorTouched must be a boolean");
  }
  if (!Number.isInteger(digits) || digits < 0) {
    throw new RangeError("digits must be a non-negative integer");
  }
  if (operatorTouched) return null;
  if (typeof deviceValue !== "number" || !Number.isFinite(deviceValue)) return null;
  return deviceValue.toFixed(digits);
}
