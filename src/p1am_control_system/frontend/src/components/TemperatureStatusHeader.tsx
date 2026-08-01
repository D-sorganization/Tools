import React from "react";
import { EditableValue } from "./EditableValue";
import { HeaterStartStopButton } from "./HeaterStartStopButton";
import type { StartStopView } from "../lib/heaterControls";
import type { TemperatureConfig, TemperatureStatus } from "../types";

/**
 * Status header for the temperature controller: state badge, the four live
 * metrics (measured / setpoint / HH limit / thermocouple) and the command
 * buttons.
 *
 * Split out of TemperatureControl.tsx unchanged. Per the TuningPanel (#4053)
 * shape, every piece of state and all REST calls stay in the parent — this
 * component receives resolved values and callbacks, so the layer that talks to
 * the heater is not also the layer that lays out a badge.
 */

const STATE_LABELS: Record<TemperatureStatus["state"], string> = {
  idle: "IDLE",
  armed: "ARMED",
  running: "RUNNING",
  tripped: "TRIPPED",
};

const STATE_HINTS: Record<TemperatureStatus["state"], string> = {
  idle: "stopped · press Start",
  armed: "started · set a target",
  running: "heating to setpoint",
  tripped: "latched · acknowledge",
};

interface Props {
  /** Live status; undefined while waiting for the first frame. */
  status: TemperatureStatus | undefined;
  config: TemperatureConfig;
  state: TemperatureStatus["state"];
  relayOn: boolean;
  /** Live HH cutoff (status value when present, else the configured one). */
  hhLimit: number;
  /** Measured temperature is within ~90 % of the HH limit. */
  tempWarn: boolean;
  tripped: boolean;
  busy: boolean;
  startStop: StartStopView;
  onCommitHhLimit: (value: number) => void;
  onAcknowledgeTrip: () => void;
  onStart: () => void;
  onStop: () => void;
}

export const TemperatureStatusHeader: React.FC<Props> = ({
  status: s,
  config,
  state,
  relayOn,
  hhLimit,
  tempWarn,
  tripped,
  busy,
  startStop,
  onCommitHhLimit,
  onAcknowledgeTrip,
  onStart,
  onStop,
}) => (
  <div className={`tc-status is-${state}`}>
    <div className="tc-state">
      <span className="tc-state-badge">{STATE_LABELS[state]}</span>
      <span className="tc-state-sub">{STATE_HINTS[state]}</span>
    </div>

    <div className="tc-status-metrics">
      <div className="tc-metric">
        <span className="tc-metric-label">Measured</span>
        <span className={`tc-metric-value ${tempWarn ? "is-warning" : ""}`}>
          {s ? s.measured_temp_c.toFixed(1) : "—"} °C
        </span>
      </div>
      <div className="tc-metric">
        <span className="tc-metric-label">Setpoint</span>
        <span className="tc-metric-value">
          {s ? s.setpoint_c.toFixed(1) : "—"} °C
        </span>
      </div>
      <div className="tc-metric">
        <span className="tc-metric-label">HH limit</span>
        <EditableValue
          className="tc-metric-value is-warning"
          value={hhLimit}
          label="High-high cutoff"
          unit="°C"
          format={(v) => v.toFixed(0)}
          min={0}
          max={config.temp_full_scale_c}
          step={10}
          title="High-high cutoff — click to edit"
          onCommit={onCommitHhLimit}
        />
      </div>
      <div className="tc-metric">
        <span className="tc-metric-label">Thermocouple</span>
        <span className="tc-metric-value">
          {s?.active_tc_label ?? config.active_tc_label ?? "—"}
        </span>
      </div>
    </div>

    <div className="tc-status-actions">
      {/* Prominent heater relay indicator */}
      <div
        className={`tc-relay ${relayOn ? "is-on" : ""}`}
        title={`${config.heater_label} relay`}
        aria-label={`Heater relay ${relayOn ? "on" : "off"}`}
      >
        <span className="tc-relay-coil" />
        <span className="tc-relay-text">HEATER {relayOn ? "ON" : "OFF"}</span>
      </div>

      {tripped && (
        <button
          className={`btn tc-btn-danger ${busy ? "tc-disabled" : ""}`}
          onClick={onAcknowledgeTrip}
          disabled={busy}
        >
          Acknowledge Trip
        </button>
      )}
      <HeaterStartStopButton
        view={startStop}
        busy={busy}
        onStart={onStart}
        onStop={onStop}
      />
    </div>
  </div>
);
