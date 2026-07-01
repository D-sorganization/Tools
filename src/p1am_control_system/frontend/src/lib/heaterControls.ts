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
