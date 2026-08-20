import { useCallback, useRef, useState } from "react";
import { seedDraftText } from "../lib/operatorDraft";

/**
 * A setpoint entry that the operator owns the moment they touch it (#4013, #4020).
 *
 * Every setpoint field in this HMI has the same two-sided requirement:
 *
 *  1. It must START at the device's real value, so a kiosk reload never leaves
 *     a stale "0" sitting under a +/- button next to a supply running at 30 A.
 *  2. It must STOP tracking the device the instant the operator types, because
 *     live telemetry arrives ~10 Hz and would otherwise overwrite the entry
 *     between keystrokes — committing "2" when the operator meant "25".
 *
 * `TemperatureControl` solved this locally with an `operatorTouchedRef` latch.
 * This hook is that pattern, once, for every entry that needs it.
 *
 * @param deviceValue - the device's current setpoint; undefined until the first
 *   status frame arrives.
 * @param options.scope - identity of the thing being edited (a device id, or a
 *   constant for a single-device panel). Changing it is a NEW entry: the draft
 *   re-seeds and ownership returns to the device. Note this must be a scalar —
 *   keying re-seeding on an object that changes reference every frame is the
 *   original defect.
 * @param options.digits - decimal places used when seeding.
 */
export interface SetpointDraft {
  /** Current entry text — render this in the controlled input. */
  text: string;
  /** Set the entry text and mark it operator-owned. */
  setText: (next: string) => void;
  /** True once the operator has edited this entry within the current scope. */
  touched: boolean;
  /** Discard the operator's edit and return to the device value. */
  reseed: () => void;
}

export function useSetpointDraft(
  deviceValue: number | null | undefined,
  options: { scope: string; digits?: number },
): SetpointDraft {
  const { scope, digits = 1 } = options;
  const [text, setTextState] = useState<string>("");
  const [touched, setTouched] = useState<boolean>(false);
  const scopeRef = useRef<string>(scope);

  // Derive during render rather than in an effect: an effect keyed on live
  // telemetry is exactly what made the old Alicat field fight the operator,
  // and it also renders one frame of stale text before correcting itself.
  let effectiveTouched = touched;
  if (scopeRef.current !== scope) {
    // The entry now refers to a different device — a genuinely new entry, so
    // ownership resets. Safe during render: this only ever moves state forward
    // in response to a prop change, never in a loop.
    scopeRef.current = scope;
    effectiveTouched = false;
    if (touched) setTouched(false);
  }

  const seeded = seedDraftText(deviceValue, effectiveTouched, digits);
  const effectiveText = seeded ?? (effectiveTouched ? text : "");
  if (!effectiveTouched && effectiveText !== text) setTextState(effectiveText);

  const setText = useCallback((next: string) => {
    setTouched(true);
    setTextState(next);
  }, []);

  const reseed = useCallback(() => {
    setTouched(false);
  }, []);

  return { text: effectiveText, setText, touched: effectiveTouched, reseed };
}
