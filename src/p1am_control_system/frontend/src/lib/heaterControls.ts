/**
 * Pure decision helpers for the heater control panel.
 *
 * The operator-facing controls used to speak in "permissive" terms, which is
 * misleading. These helpers express the same controller state machine as a plain
 * Start/Stop model and decide what a committed setpoint should do, kept pure and
 * React-free so the behavior is unit-testable (the panel just renders the result).
 */

/** The minimal live-controller status the heater UI decisions depend on. */
export interface HeaterStatusView {
  state: "idle" | "armed" | "running" | "tripped";
  /** Master enable (the old "permissive"): true once the heater is started. */
  permissive: boolean;
}

/** The command the Start/Stop button issues. */
export type HeaterCommand = "start" | "stop";

export interface StartStopView {
  /** Operator-facing button label ("Start" or "Stop"). */
  label: string;
  /** Which command the button issues when clicked. */
  command: HeaterCommand;
  /** True when the button must be disabled (no status yet, or tripped). */
  disabled: boolean;
}

/**
 * Decide the Start/Stop button presentation from the live controller status.
 *
 * - Stopped (idle) → a **Start** button (arms the heater and heats to target).
 * - Started (armed/running) → a **Stop** button (opens the relay immediately).
 * - Tripped → **Start**, disabled: the trip must be acknowledged first.
 * - No status yet (`undefined`) → **Start**, disabled: nothing to command safely.
 */
export function startStopView(status: HeaterStatusView | undefined): StartStopView {
  if (!status || status.state === "tripped") {
    return { label: "Start", command: "start", disabled: true };
  }
  if (status.permissive) {
    return { label: "Stop", command: "stop", disabled: false };
  }
  return { label: "Start", command: "start", disabled: false };
}

/** What committing a setpoint (Enter / nudge) should do given the status. */
export type SetpointOutcome = "apply" | "stage" | "blocked";

/**
 * Decide whether a committed setpoint is:
 * - `apply`   — sent now (heater is started, so it takes effect immediately);
 * - `stage`   — kept as the target only (heater is stopped; it will be applied
 *               when the operator presses Start);
 * - `blocked` — refused (tripped, or no live status to confirm against).
 */
export function setpointOutcome(status: HeaterStatusView | undefined): SetpointOutcome {
  if (!status || status.state === "tripped") return "blocked";
  if (status.permissive) return "apply";
  return "stage";
}

/**
 * Resolve the effective target temperature (°C) a Start click should apply.
 *
 * The operator's typed/nudged value wins whenever it is a positive finite
 * number. Otherwise we fall back to the recalled setpoint the controller
 * already holds (seeded at boot from persistence, surfaced as
 * `last_setpoint_c` / `setpoint_c`). This closes the startup race where the
 * entry box still reads its default "0" because the first telemetry frame
 * hasn't pre-filled it yet: without the fallback, Start would only arm and the
 * operator had to click Start a second time once the box populated. Returns
 * `null` when there is no positive target at all, so Start simply arms.
 *
 * @param stagedText - the current setpoint entry-box text.
 * @param fallbackSetpointC - the recalled/held setpoint to use when the box has
 *   no positive value (e.g. `status.last_setpoint_c ?? status.setpoint_c`).
 * @throws TypeError if `stagedText` is not a string.
 */
export function resolveStartTarget(
  stagedText: string,
  fallbackSetpointC: number | null | undefined,
): number | null {
  if (typeof stagedText !== "string") {
    throw new TypeError("stagedText must be a string");
  }
  const typed = Number.parseFloat(stagedText);
  if (Number.isFinite(typed) && typed > 0) return typed;
  if (
    typeof fallbackSetpointC === "number" &&
    Number.isFinite(fallbackSetpointC) &&
    fallbackSetpointC > 0
  ) {
    return fallbackSetpointC;
  }
  return null;
}

/**
 * True when any trip name looks like a high-high temperature cutoff.
 *
 * Kept here with the other pure heater decisions rather than beside the banner
 * it drives: the same predicate decides both the HH banner and whether the
 * generic trips banner is suppressed, and a trip name is server-supplied text.
 */
export function hasHighHighTrip(trips: string[]): boolean {
  return trips.some((t) => /(hh|high.?high|over.?temp)/i.test(t));
}
