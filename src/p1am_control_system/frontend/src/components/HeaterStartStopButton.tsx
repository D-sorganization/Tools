import React from "react";
import type { StartStopView } from "../lib/heaterControls";

/**
 * The heater Start/Stop command button.
 *
 * The controller screen shows this control twice — once in the status header and
 * once beside the setpoint entry — and both copies were byte-identical markup
 * apart from one extra class. That duplication is a hazard on a control that
 * energizes a heater: a change to the disabled logic or the busy label applied to
 * one copy and not the other would leave two buttons on the same screen
 * disagreeing about whether a command is safe to send. There is now one.
 *
 * @param view - decision from the pure `startStopView()` helper (which command
 *   the button issues, and whether it is inhibited).
 * @param busy - a command is in flight; the button locks and reads "APPLYING…".
 */
interface Props {
  view: StartStopView;
  busy: boolean;
  onStart: () => void;
  onStop: () => void;
  /** Extra classes appended to the shared `tc-permissive` base class. */
  className?: string;
}

export const HeaterStartStopButton: React.FC<Props> = ({
  view,
  busy,
  onStart,
  onStop,
  className,
}) => (
  <button
    className={className ? `tc-permissive ${className}` : "tc-permissive"}
    onClick={view.command === "start" ? onStart : onStop}
    disabled={busy || view.disabled}
    style={{
      background:
        view.command === "stop" ? "var(--color-error)" : "var(--color-success)",
      color: "#04141b",
      borderColor: "transparent",
    }}
    title={
      busy
        ? "Applying — please wait…"
        : view.command === "start"
          ? "Start the heater — arm and heat to the target"
          : "Stop the heater — the relay opens immediately"
    }
  >
    <span className="dot" />
    {busy ? "APPLYING…" : view.command === "stop" ? "■ STOP" : "▶ START"}
  </button>
);
