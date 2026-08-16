import React from "react";
import type { AlicatMFCState } from "../api/schemas";
import type { TriggerNotification } from "../types";

export const AlicatInspector: React.FC<{
  deviceId: string;
  alicats: AlicatMFCState[];
  setpointValue: string;
  onSetpointValueChange: (value: string) => void;
  onSetpoint: (deviceId: string, value: number) => void;
  onGasChange: (deviceId: string, gas: string) => void;
  triggerNotification: TriggerNotification;
}> = ({
  deviceId,
  alicats,
  setpointValue,
  onSetpointValueChange,
  onSetpoint,
  onGasChange,
  triggerNotification,
}) => {
  const mfc = alicats.find((item) => item.device_id === deviceId);
  if (!mfc) {
    return (
      <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
        Alicat MFC {deviceId} not found.
      </div>
    );
  }

  // A staged entry that disagrees with the device is called out explicitly.
  // Tolerance matches the entry's own 0.1 SLPM step so trailing-zero text
  // ("25" vs "25.00") never reads as a pending change.
  const draftValue = Number.parseFloat(setpointValue);
  const draftDiffersFromDevice =
    Number.isFinite(draftValue) && Math.abs(draftValue - mfc.setpoint) > 0.05;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <div>
        <h3
          style={{
            fontSize: "1rem",
            fontWeight: 700,
            color: "var(--color-warning)",
            textTransform: "uppercase",
          }}
        >
          Inspect {mfc.name}
        </h3>
        <div
          style={{
            fontSize: "0.75rem",
            color: "var(--text-secondary)",
            marginTop: "0.2rem",
          }}
        >
          Device ID: {mfc.device_id} | State: {mfc.connection_state}
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "0.5rem",
          background: "var(--input-bg)",
          padding: "0.75rem",
          borderRadius: "4px",
          border: "1px solid var(--panel-border)",
        }}
      >
        <MfcReadout label="Mass Flow" value={mfc.mass_flow.toFixed(2)} unit="SLPM" />
        <MfcReadout
          label="Vol. Flow"
          value={mfc.volumetric_flow.toFixed(2)}
          unit="LPM"
          muted
        />
        <MfcReadout label="Pressure" value={mfc.pressure.toFixed(2)} unit="PSIA" />
        <MfcReadout
          label="Temperature"
          value={mfc.temperature.toFixed(1)}
          unit={"\u00b0C"}
        />
      </div>

      <div
        style={{
          borderTop: "1px solid var(--panel-border)",
          paddingTop: "0.75rem",
        }}
      >
        {/* The DEVICE's live setpoint, as its own read-only readout. The entry
            below is the operator's draft and no longer tracks telemetry
            (#4013), so both numbers have to be visible at once — otherwise a
            draft that disagrees with the device is invisible until Set. */}
        <div
          data-testid="alicat-device-setpoint"
          style={{
            display: "flex",
            alignItems: "baseline",
            gap: "0.4rem",
            marginBottom: "0.5rem",
          }}
        >
          <span
            style={{
              fontSize: "0.65rem",
              color: "var(--text-muted)",
              textTransform: "uppercase",
            }}
          >
            Device setpoint
          </span>
          <span className="mono-text" style={{ fontWeight: 700 }}>
            {mfc.setpoint.toFixed(2)}
          </span>
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>SLPM</span>
        </div>

        <label
          className="input-label"
          htmlFor={`alicat-setpoint-${mfc.device_id}`}
          style={{ fontWeight: 700, marginBottom: "0.4rem", display: "block" }}
        >
          Flow Setpoint Command (SLPM)
        </label>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <input
            id={`alicat-setpoint-${mfc.device_id}`}
            type="number"
            step="0.1"
            min="0"
            max={mfc.max_flow}
            className="form-input"
            style={{ flex: 1 }}
            value={setpointValue}
            onChange={(e) => onSetpointValueChange(e.target.value)}
          />
          <button
            type="button"
            className="btn btn-primary"
            style={{
              padding: "0.45rem 1rem",
              fontSize: "0.8rem",
              whiteSpace: "nowrap",
            }}
            onClick={() => {
              const parsed = parseFloat(setpointValue);
              if (!isNaN(parsed) && parsed >= 0 && parsed <= mfc.max_flow) {
                onSetpoint(mfc.device_id, parsed);
              } else {
                triggerNotification(
                  `Please enter a valid setpoint between 0 and ${mfc.max_flow}.`,
                  "error",
                );
              }
            }}
          >
            Set
          </button>
        </div>
        {draftDiffersFromDevice && (
          <div
            data-testid="alicat-setpoint-pending"
            style={{
              fontSize: "0.68rem",
              color: "var(--color-warning)",
              marginTop: "0.3rem",
              fontWeight: 600,
            }}
          >
            Pending — entry differs from the device setpoint ({mfc.setpoint.toFixed(2)}{" "}
            SLPM). Press Set to command it.
          </div>
        )}
        <div
          style={{
            fontSize: "0.65rem",
            color: "var(--text-muted)",
            marginTop: "0.25rem",
          }}
        >
          Maximum flow limit: {mfc.max_flow} SLPM
        </div>
      </div>

      <div
        style={{
          borderTop: "1px solid var(--panel-border)",
          paddingTop: "0.75rem",
        }}
      >
        <label
          className="input-label"
          style={{ fontWeight: 700, marginBottom: "0.4rem", display: "block" }}
        >
          Active Gas Calibration
        </label>
        <select
          className="form-input"
          value={mfc.gas}
          onChange={(e) => onGasChange(mfc.device_id, e.target.value)}
        >
          {["O2", "N2", "CO2", "He", "H2", "Air"].map((species) => (
            <option key={species} value={species}>
              {species} ({gasLabel(species)})
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};

const MfcReadout: React.FC<{
  label: string;
  value: string;
  unit: string;
  muted?: boolean;
}> = ({ label, value, unit, muted }) => (
  <div style={{ marginTop: label === "Mass Flow" || label === "Vol. Flow" ? 0 : "0.35rem" }}>
    <div
      style={{
        fontSize: "0.65rem",
        color: "var(--text-muted)",
        textTransform: "uppercase",
      }}
    >
      {label}
    </div>
    <div
      className="mono-text"
      style={{
        fontSize: "1.05rem",
        fontWeight: 700,
        color: muted ? "var(--text-secondary)" : "var(--text-primary)",
      }}
    >
      {value} <span style={{ fontSize: "0.7rem", fontWeight: 500 }}>{unit}</span>
    </div>
  </div>
);

function gasLabel(species: string): string {
  if (species === "O2") return "Oxygen";
  if (species === "N2") return "Nitrogen";
  if (species === "CO2") return "Carbon Dioxide";
  if (species === "He") return "Helium";
  if (species === "H2") return "Hydrogen";
  return "Clean Air";
}
