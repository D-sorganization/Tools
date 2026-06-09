import React, { useEffect, useState, useCallback, useRef } from "react";

/**
 * Power-supply control tab.
 *
 * Talks to the backend's /api/power_supply/* endpoints and consumes the
 * `power_supply` field on the /api/stream WebSocket payload (broadcast each
 * scan tick by the polling loop in main.py).
 *
 * Safety patterns implemented:
 *  - Setpoint clamping happens server-side, so the displayed value reflects
 *    what was actually applied (not what was typed).
 *  - "Apply" button submits the staged value — typing in the textbox alone
 *    doesn't command anything. Fat-finger protection.
 *  - Up/down nudges step by a configurable increment, also clamped server-side.
 *  - Permissive must be ON before any setpoint takes effect (server enforces).
 *  - Trip indicator + acknowledge are first-class UI.
 *  - Power-mode and current-mode are visually distinct; switching modes is
 *    explicit.
 */

export interface PowerSupplyConfig {
  command_tag: string;
  current_feedback_tag: string;
  voltage_feedback_tag: string;
  temp_tag: string;
  current_full_scale_a: number;
  voltage_full_scale_v: number;
  current_setpoint_min_a: number;
  current_setpoint_max_a: number;
  power_alarm_max_w: number;
  temp_alarm_max_c: number;
  /**
   * Max rate at which the AO command percent may INCREASE (decreases pass
   * through instantly). Default 5 %/s gives a slow-start ramp from 0 to
   * 100 % in 20 s.
   */
  setpoint_ramp_rate_pct_per_s: number;
}

export interface PowerSupplyStatus {
  state: "idle" | "armed" | "running" | "tripped";
  mode: "current" | "power";
  permissive: boolean;
  setpoint_a: number;
  setpoint_w: number | null;
  measured_current_a: number;
  measured_voltage_v: number;
  measured_power_w: number;
  measured_temp_c: number;
  commanded_output_percent: number;
  trips: string[];
}

interface Props {
  /** Status object pushed each scan via the parent's WebSocket; undefined while waiting. */
  liveStatus?: PowerSupplyStatus;
}

const STATE_COLORS: Record<PowerSupplyStatus["state"], string> = {
  idle: "var(--text-secondary)",
  armed: "var(--accent-cyan)",
  running: "var(--color-success, #4ade80)",
  tripped: "var(--color-danger, #ef4444)",
};

const STATE_LABELS: Record<PowerSupplyStatus["state"], string> = {
  idle: "IDLE",
  armed: "ARMED",
  running: "RUNNING",
  tripped: "TRIPPED",
};

export const PowerSupplyControl: React.FC<Props> = ({ liveStatus }) => {
  const [config, setConfig] = useState<PowerSupplyConfig | null>(null);
  const [configDraft, setConfigDraft] = useState<PowerSupplyConfig | null>(null);
  const [mode, setMode] = useState<"current" | "power">("current");
  const [stagedSetpointText, setStagedSetpointText] = useState<string>("0");
  const [setpointStep, setSetpointStep] = useState<number>(1.0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const lastAppliedRef = useRef<number>(0);

  // Load config once.
  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch("/api/power_supply/config");
        if (!res.ok) throw new Error(`config GET ${res.status}`);
        const cfg = (await res.json()) as PowerSupplyConfig;
        setConfig(cfg);
        setConfigDraft(cfg);
      } catch (e) {
        setError(`Load config failed: ${(e as Error).message}`);
      }
    };
    load();
  }, []);

  // Sync mode display with whatever the server most recently applied.
  useEffect(() => {
    if (liveStatus) setMode(liveStatus.mode);
  }, [liveStatus?.mode]);

  const flash = useCallback((msg: string, kind: "info" | "error" = "info") => {
    if (kind === "info") {
      setInfo(msg);
      setError(null);
    } else {
      setError(msg);
      setInfo(null);
    }
    setTimeout(() => {
      setInfo(null);
      setError(null);
    }, 4000);
  }, []);

  const applySetpoint = useCallback(
    async (rawValue: number) => {
      if (!Number.isFinite(rawValue)) {
        flash("Setpoint must be a finite number", "error");
        return;
      }
      if (!config) {
        flash("Config not loaded", "error");
        return;
      }
      setBusy(true);
      try {
        const body =
          mode === "current"
            ? { mode: "current", value_a: rawValue }
            : { mode: "power", value_w: rawValue };
        const res = await fetch("/api/power_supply/setpoint", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
        });
        const json = await res.json();
        if (!res.ok) {
          flash(`Setpoint rejected: ${json.detail ?? res.statusText}`, "error");
          return;
        }
        const applied = mode === "current" ? json.applied_a : json.achievable_w;
        lastAppliedRef.current = applied;
        flash(
          mode === "current"
            ? `Applied ${applied.toFixed(2)} A`
            : `Achievable ${applied.toFixed(1)} W (clamped if needed)`,
        );
      } catch (e) {
        flash(`Setpoint failed: ${(e as Error).message}`, "error");
      } finally {
        setBusy(false);
      }
    },
    [config, mode, flash],
  );

  const nudgeSetpoint = useCallback(
    (delta: number) => {
      const current = Number.parseFloat(stagedSetpointText);
      const next = (Number.isFinite(current) ? current : 0) + delta;
      setStagedSetpointText(next.toFixed(2));
      applySetpoint(next);
    },
    [stagedSetpointText, applySetpoint],
  );

  const handleApplyClick = useCallback(() => {
    const v = Number.parseFloat(stagedSetpointText);
    if (!Number.isFinite(v)) {
      flash("Enter a number first", "error");
      return;
    }
    applySetpoint(v);
  }, [stagedSetpointText, applySetpoint, flash]);

  const setPermissive = useCallback(
    async (enabled: boolean) => {
      if (
        !enabled &&
        liveStatus?.state === "running" &&
        !window.confirm(
          "Disabling permissive will drop output to 0 immediately. Continue?",
        )
      )
        return;
      setBusy(true);
      try {
        const res = await fetch("/api/power_supply/permissive", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ enabled }),
        });
        if (!res.ok) throw new Error(await res.text());
        flash(enabled ? "Permissive ON" : "Permissive OFF");
      } catch (e) {
        flash(`Permissive set failed: ${(e as Error).message}`, "error");
      } finally {
        setBusy(false);
      }
    },
    [liveStatus?.state, flash],
  );

  const acknowledgeTrip = useCallback(async () => {
    setBusy(true);
    try {
      const res = await fetch("/api/power_supply/acknowledge_trip", {
        method: "POST",
      });
      if (!res.ok) throw new Error(await res.text());
      flash("Trip acknowledged");
    } catch (e) {
      flash(`Ack failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [flash]);

  const saveConfig = useCallback(async () => {
    if (!configDraft) return;
    setBusy(true);
    try {
      const res = await fetch("/api/power_supply/config", {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(configDraft),
      });
      const json = await res.json();
      if (!res.ok) {
        flash(
          `Config rejected: ${json.detail ? JSON.stringify(json.detail) : res.statusText}`,
          "error",
        );
        return;
      }
      setConfig(json);
      setConfigDraft(json);
      flash("Config saved");
    } catch (e) {
      flash(`Config save failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [configDraft, flash]);

  if (!config || !configDraft) {
    return (
      <div style={{ padding: 16, color: "var(--text-secondary)" }}>
        Loading power-supply config…
      </div>
    );
  }

  const s = liveStatus;
  const stateColor = s ? STATE_COLORS[s.state] : "var(--text-secondary)";
  const stateLabel = s ? STATE_LABELS[s.state] : "—";

  return (
    <div style={{ display: "grid", gap: 16, padding: 16 }}>
      {/* State strip */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "auto 1fr auto auto",
          gap: 16,
          alignItems: "center",
          padding: "12px 16px",
          background: "rgba(255,255,255,0.03)",
          border: `1px solid ${stateColor}`,
          borderRadius: 8,
        }}
      >
        <span
          style={{
            fontWeight: 700,
            fontSize: "1.25rem",
            color: stateColor,
            padding: "4px 12px",
            background: "rgba(255,255,255,0.05)",
            borderRadius: 4,
          }}
        >
          {stateLabel}
        </span>
        <span style={{ color: "var(--text-secondary)" }}>
          Mode: <strong style={{ color: "var(--text-primary)" }}>{s?.mode ?? "—"}</strong>
          &nbsp;|&nbsp; Output cmd:{" "}
          <strong style={{ color: "var(--text-primary)" }}>
            {s ? s.commanded_output_percent.toFixed(2) : "—"} %
          </strong>
        </span>
        <button
          onClick={() => setPermissive(!s?.permissive)}
          disabled={busy}
          style={{
            background: s?.permissive
              ? "var(--color-success, #4ade80)"
              : "var(--text-secondary)",
            color: "#000",
            border: "none",
            borderRadius: 4,
            padding: "8px 14px",
            cursor: "pointer",
            fontWeight: 600,
          }}
        >
          Permissive: {s?.permissive ? "ON" : "OFF"}
        </button>
        {s?.state === "tripped" && (
          <button
            onClick={acknowledgeTrip}
            disabled={busy}
            style={{
              background: "var(--color-danger, #ef4444)",
              color: "#fff",
              border: "none",
              borderRadius: 4,
              padding: "8px 14px",
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            Acknowledge Trip
          </button>
        )}
      </div>

      {/* Trips banner */}
      {s && s.trips.length > 0 && (
        <div
          style={{
            padding: "10px 14px",
            background: "rgba(239,68,68,0.12)",
            border: "1px solid rgba(239,68,68,0.5)",
            borderRadius: 6,
            color: "var(--color-danger, #ef4444)",
            fontWeight: 600,
          }}
        >
          Active trips: {s.trips.join(", ")}
        </div>
      )}

      {/* Setpoint card */}
      <div
        style={{
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 8,
          padding: 16,
        }}
      >
        <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12 }}>
          <h3 style={{ margin: 0, color: "var(--text-primary)" }}>Setpoint</h3>
          <div style={{ display: "flex", gap: 4 }}>
            {(["current", "power"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                style={{
                  padding: "4px 12px",
                  background:
                    mode === m ? "var(--accent-cyan)" : "rgba(255,255,255,0.05)",
                  color: mode === m ? "#000" : "var(--text-secondary)",
                  border: "none",
                  borderRadius: 4,
                  cursor: "pointer",
                  fontWeight: 600,
                }}
              >
                {m === "current" ? "Current (A)" : "Power (W)"}
              </button>
            ))}
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "auto 1fr auto auto auto",
            gap: 8,
            alignItems: "center",
          }}
        >
          <button
            onClick={() => nudgeSetpoint(-setpointStep)}
            disabled={busy}
            style={{
              padding: "8px 14px",
              fontSize: "1.25rem",
              background: "rgba(255,255,255,0.05)",
              color: "var(--text-primary)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 4,
              cursor: "pointer",
            }}
            title={`Decrement by ${setpointStep}`}
          >
            ▼
          </button>
          <input
            type="text"
            inputMode="decimal"
            value={stagedSetpointText}
            onChange={(e) => setStagedSetpointText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleApplyClick();
            }}
            style={{
              padding: "10px 12px",
              fontSize: "1.25rem",
              background: "rgba(0,0,0,0.4)",
              color: "var(--text-primary)",
              border: "1px solid rgba(255,255,255,0.15)",
              borderRadius: 4,
              fontFamily: "monospace",
              textAlign: "right",
            }}
          />
          <button
            onClick={() => nudgeSetpoint(setpointStep)}
            disabled={busy}
            style={{
              padding: "8px 14px",
              fontSize: "1.25rem",
              background: "rgba(255,255,255,0.05)",
              color: "var(--text-primary)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 4,
              cursor: "pointer",
            }}
            title={`Increment by ${setpointStep}`}
          >
            ▲
          </button>
          <input
            type="number"
            value={setpointStep}
            min={0.01}
            step={0.1}
            onChange={(e) =>
              setSetpointStep(Math.max(0.01, Number.parseFloat(e.target.value) || 0.01))
            }
            style={{
              width: 80,
              padding: "8px",
              background: "rgba(0,0,0,0.4)",
              color: "var(--text-primary)",
              border: "1px solid rgba(255,255,255,0.15)",
              borderRadius: 4,
              fontFamily: "monospace",
            }}
            title="Step size for ▲/▼"
          />
          <button
            onClick={handleApplyClick}
            disabled={busy}
            style={{
              padding: "10px 16px",
              background: "var(--accent-cyan)",
              color: "#000",
              border: "none",
              borderRadius: 4,
              cursor: "pointer",
              fontWeight: 700,
              fontSize: "1rem",
            }}
          >
            Apply
          </button>
        </div>
        <p style={{ color: "var(--text-secondary)", marginTop: 8, fontSize: "0.85rem" }}>
          {mode === "current"
            ? `Range ${configDraft.current_setpoint_min_a} – ${configDraft.current_setpoint_max_a} A (server clamps any value outside this band).`
            : `Power is converted to current via measured V. Server clamps resulting current to [${configDraft.current_setpoint_min_a}, ${configDraft.current_setpoint_max_a}] A.`}
        </p>
      </div>

      {/* Live readings card */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: 12,
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 8,
          padding: 16,
        }}
      >
        <Reading label="Setpoint" value={`${s?.setpoint_a.toFixed(2) ?? "—"} A`} />
        <Reading
          label="Measured I"
          value={`${s?.measured_current_a.toFixed(2) ?? "—"} A`}
        />
        <Reading
          label="Measured V"
          value={`${s?.measured_voltage_v.toFixed(2) ?? "—"} V`}
        />
        <Reading
          label="Power (V × I)"
          value={`${s?.measured_power_w.toFixed(1) ?? "—"} W`}
          warning={
            s ? s.measured_power_w >= 0.9 * configDraft.power_alarm_max_w : false
          }
        />
        <Reading
          label="Temp"
          value={`${s?.measured_temp_c.toFixed(1) ?? "—"} °C`}
          warning={
            s ? s.measured_temp_c >= 0.9 * configDraft.temp_alarm_max_c : false
          }
        />
        <Reading
          label="Commanded Output"
          value={`${s?.commanded_output_percent.toFixed(2) ?? "—"} %`}
        />
        {s?.mode === "power" && (
          <Reading
            label="Power Setpoint"
            value={s.setpoint_w != null ? `${s.setpoint_w.toFixed(1)} W` : "—"}
          />
        )}
      </div>

      {/* Config card */}
      <details
        style={{
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 8,
          padding: 16,
        }}
      >
        <summary
          style={{
            cursor: "pointer",
            fontWeight: 600,
            color: "var(--text-primary)",
            userSelect: "none",
          }}
        >
          Configuration (scaling, bounds, alarm thresholds, tag map)
        </summary>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 12,
            marginTop: 14,
          }}
        >
          <ConfigField
            label="Current full scale (A at 100 %)"
            value={configDraft.current_full_scale_a}
            onChange={(v) =>
              setConfigDraft({ ...configDraft, current_full_scale_a: v })
            }
          />
          <ConfigField
            label="Voltage full scale (V at 100 %)"
            value={configDraft.voltage_full_scale_v}
            onChange={(v) =>
              setConfigDraft({ ...configDraft, voltage_full_scale_v: v })
            }
          />
          <ConfigField
            label="Min current setpoint (A)"
            value={configDraft.current_setpoint_min_a}
            onChange={(v) =>
              setConfigDraft({ ...configDraft, current_setpoint_min_a: v })
            }
          />
          <ConfigField
            label="Max current setpoint (A)"
            value={configDraft.current_setpoint_max_a}
            onChange={(v) =>
              setConfigDraft({ ...configDraft, current_setpoint_max_a: v })
            }
          />
          <ConfigField
            label="HH Power alarm (W)"
            value={configDraft.power_alarm_max_w}
            onChange={(v) =>
              setConfigDraft({ ...configDraft, power_alarm_max_w: v })
            }
          />
          <ConfigField
            label="HH Temperature alarm (°C)"
            value={configDraft.temp_alarm_max_c}
            onChange={(v) =>
              setConfigDraft({ ...configDraft, temp_alarm_max_c: v })
            }
          />
          <ConfigField
            label="Ramp rate on INCREASE (%/s; decreases are instant)"
            value={configDraft.setpoint_ramp_rate_pct_per_s}
            onChange={(v) =>
              setConfigDraft({
                ...configDraft,
                setpoint_ramp_rate_pct_per_s: v,
              })
            }
          />
          <ConfigField
            label="Command tag"
            value={configDraft.command_tag}
            stringMode
            onChange={(v) =>
              setConfigDraft({ ...configDraft, command_tag: v as unknown as string })
            }
          />
          <ConfigField
            label="Current feedback tag"
            value={configDraft.current_feedback_tag}
            stringMode
            onChange={(v) =>
              setConfigDraft({
                ...configDraft,
                current_feedback_tag: v as unknown as string,
              })
            }
          />
          <ConfigField
            label="Voltage feedback tag"
            value={configDraft.voltage_feedback_tag}
            stringMode
            onChange={(v) =>
              setConfigDraft({
                ...configDraft,
                voltage_feedback_tag: v as unknown as string,
              })
            }
          />
          <ConfigField
            label="Temperature tag"
            value={configDraft.temp_tag}
            stringMode
            onChange={(v) =>
              setConfigDraft({ ...configDraft, temp_tag: v as unknown as string })
            }
          />
        </div>
        <div style={{ marginTop: 14 }}>
          <button
            onClick={saveConfig}
            disabled={busy}
            style={{
              padding: "8px 16px",
              background: "var(--accent-purple, #a78bfa)",
              color: "#000",
              border: "none",
              borderRadius: 4,
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            Save config
          </button>
        </div>
      </details>

      {info && (
        <div
          style={{
            padding: "8px 12px",
            background: "rgba(74,222,128,0.12)",
            border: "1px solid rgba(74,222,128,0.5)",
            color: "var(--color-success, #4ade80)",
            borderRadius: 6,
          }}
        >
          {info}
        </div>
      )}
      {error && (
        <div
          style={{
            padding: "8px 12px",
            background: "rgba(239,68,68,0.12)",
            border: "1px solid rgba(239,68,68,0.5)",
            color: "var(--color-danger, #ef4444)",
            borderRadius: 6,
          }}
        >
          {error}
        </div>
      )}
    </div>
  );
};

const Reading: React.FC<{ label: string; value: string; warning?: boolean }> = ({
  label,
  value,
  warning,
}) => (
  <div
    style={{
      padding: "8px 12px",
      borderRadius: 4,
      background: warning ? "rgba(245,158,11,0.12)" : "rgba(0,0,0,0.25)",
      border: warning
        ? "1px solid rgba(245,158,11,0.5)"
        : "1px solid rgba(255,255,255,0.05)",
    }}
  >
    <div
      style={{
        color: "var(--text-secondary)",
        fontSize: "0.75rem",
        textTransform: "uppercase",
        letterSpacing: 0.5,
      }}
    >
      {label}
    </div>
    <div
      style={{
        color: warning ? "var(--color-warning, #f59e0b)" : "var(--text-primary)",
        fontSize: "1.25rem",
        fontFamily: "monospace",
        fontWeight: 600,
      }}
    >
      {value}
    </div>
  </div>
);

interface ConfigFieldProps {
  label: string;
  value: number | string;
  stringMode?: boolean;
  onChange: (v: number) => void;
}
const ConfigField: React.FC<ConfigFieldProps> = ({
  label,
  value,
  stringMode,
  onChange,
}) => (
  <label style={{ display: "grid", gap: 4 }}>
    <span style={{ color: "var(--text-secondary)", fontSize: "0.8rem" }}>{label}</span>
    <input
      type={stringMode ? "text" : "number"}
      value={value}
      onChange={(e) =>
        onChange(
          (stringMode ? e.target.value : Number.parseFloat(e.target.value)) as number,
        )
      }
      style={{
        padding: "6px 8px",
        background: "rgba(0,0,0,0.4)",
        color: "var(--text-primary)",
        border: "1px solid rgba(255,255,255,0.15)",
        borderRadius: 4,
        fontFamily: "monospace",
      }}
    />
  </label>
);
