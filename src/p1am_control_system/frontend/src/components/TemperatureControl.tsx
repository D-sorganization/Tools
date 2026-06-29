import React, { useEffect, useState, useCallback } from "react";
import { fetchWithTimeout } from "../lib/fetchWithTimeout";
import {
  type AxisRange,
  defaultAxisRange,
  resolveRange,
  axisTicks,
} from "../lib/trendAxis";
import {
  MAX_TREND_SAMPLES,
  windowSamples as windowSampleCount,
  downsample,
  formatWindow,
} from "../lib/trendTime";
import { TrendAxisControls } from "./TrendAxisControls";
import { TrendTimeControls } from "./TrendTimeControls";
import "./TemperatureControl.css";

// Rolling trend buffer: deep enough for the longest selectable window
// (5 min @ ~10 Hz); the plot slices/downsamples to the chosen window.
const TREND_MAX_POINTS = MAX_TREND_SAMPLES;
const DEFAULT_WINDOW_SECONDS = 30;

/**
 * Temperature controller tab.
 *
 * Drives a resistive heater through an on/off relay using a thermocouple
 * temperature reading (0–1400 °C). Talks to the backend's /api/temperature/*
 * endpoints and consumes the `temperature` field on the /api/stream WebSocket
 * payload (broadcast each scan tick by the parent), exactly like
 * PowerSupplyControl.
 *
 * Safety patterns implemented:
 *  - On/off (bang-bang) control: the relay closes below setpoint − deadband and
 *    opens above setpoint. A prominent HEATER relay indicator shows real state.
 *  - High-high cutoff: a hard limit that latches the heater OFF and trips the
 *    controller; the operator must acknowledge to reset (server enforces).
 *  - Setpoint clamping happens server-side, so the displayed value reflects what
 *    was actually applied (not what was typed).
 *  - "Apply Setpoint" commits the staged value — typing alone commands nothing.
 *  - Permissive must be ON before any setpoint takes effect (server enforces).
 */

export interface TemperatureConfig {
  temp_tag: string;
  temp_full_scale_c: number;
  setpoint_min_c: number;
  setpoint_max_c: number;
  deadband_c: number;
  hh_limit_c: number;
  heater_label: string;
}

export interface TemperatureStatus {
  state: "idle" | "armed" | "running" | "tripped";
  permissive: boolean;
  setpoint_c: number;
  measured_temp_c: number;
  relay_on: boolean;
  trips: string[];
  hh_limit_c: number;
  deadband_c: number;
}

interface Props {
  /** Status pushed each scan via the parent's WebSocket; undefined while waiting. */
  liveStatus?: TemperatureStatus;
}

const STATE_LABELS: Record<TemperatureStatus["state"], string> = {
  idle: "IDLE",
  armed: "ARMED",
  running: "RUNNING",
  tripped: "TRIPPED",
};

const STATE_HINTS: Record<TemperatureStatus["state"], string> = {
  idle: "permissive off",
  armed: "ready · set a target",
  running: "regulating heater",
  tripped: "latched · acknowledge",
};

/** True when any trip name looks like a high-high temperature cutoff. */
function hasHighHighTrip(trips: string[]): boolean {
  return trips.some((t) => /(hh|high.?high|over.?temp)/i.test(t));
}

export const TemperatureControl: React.FC<Props> = ({ liveStatus }) => {
  const [config, setConfig] = useState<TemperatureConfig | null>(null);
  const [configDraft, setConfigDraft] = useState<TemperatureConfig | null>(null);
  const [stagedSetpointText, setStagedSetpointText] = useState<string>("0");
  const [setpointStep, setSetpointStep] = useState<number>(10);
  const [trend, setTrend] = useState<number[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  // Load config once.
  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetchWithTimeout("/api/temperature/config");
        if (!res.ok) throw new Error(`config GET ${res.status}`);
        const cfg = (await res.json()) as TemperatureConfig;
        setConfig(cfg);
        setConfigDraft(cfg);
      } catch (e) {
        setError(`Load config failed: ${(e as Error).message}`);
      }
    };
    load();
  }, []);

  // Accumulate a rolling trend buffer from the live status broadcasts.
  useEffect(() => {
    if (!liveStatus) return;
    setTrend((prev) => {
      const next = [...prev, liveStatus.measured_temp_c];
      return next.length > TREND_MAX_POINTS
        ? next.slice(next.length - TREND_MAX_POINTS)
        : next;
    });
  }, [liveStatus]);

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
      // Guard rail: the server silently ignores setpoints unless the controller
      // is armed, so tell the operator instead of pretending it was applied.
      // Without live status we can't verify permissive/trip — refuse.
      if (!liveStatus) {
        flash("Live status unavailable — cannot confirm permissive/trip.", "error");
        return;
      }
      if (liveStatus.state === "tripped") {
        flash("Controller is TRIPPED — acknowledge the trip first.", "error");
        return;
      }
      if (!liveStatus.permissive) {
        flash("Permissive is OFF — enable it before commanding the heater.", "error");
        return;
      }
      setBusy(true);
      try {
        const res = await fetchWithTimeout("/api/temperature/setpoint", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ value_c: rawValue }),
        });
        const json = await res.json();
        if (!res.ok) {
          flash(`Setpoint rejected: ${json.detail ?? res.statusText}`, "error");
          return;
        }
        const applied = json.applied_c as number;
        setStagedSetpointText(applied.toFixed(1));
        flash(`Applied ${applied.toFixed(1)} °C`);
      } catch (e) {
        flash(`Setpoint failed: ${(e as Error).message}`, "error");
      } finally {
        setBusy(false);
      }
    },
    [config, flash, liveStatus],
  );

  const nudgeSetpoint = useCallback(
    (delta: number) => {
      const current = Number.parseFloat(stagedSetpointText);
      const next = Math.max(0, (Number.isFinite(current) ? current : 0) + delta);
      setStagedSetpointText(next.toFixed(1));
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
        liveStatus?.relay_on &&
        !window.confirm(
          "Disabling permissive will open the heater relay immediately. Continue?",
        )
      )
        return;
      setBusy(true);
      try {
        const res = await fetchWithTimeout("/api/temperature/permissive", {
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
    [liveStatus?.relay_on, flash],
  );

  const acknowledgeTrip = useCallback(async () => {
    setBusy(true);
    try {
      const res = await fetchWithTimeout("/api/temperature/acknowledge_trip", {
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
      const res = await fetchWithTimeout("/api/temperature/config", {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(configDraft),
      });
      const json = await res.json();
      if (!res.ok) {
        flash(
          `Config rejected: ${
            json.detail ? JSON.stringify(json.detail) : res.statusText
          }`,
          "error",
        );
        return;
      }
      setConfig(json);
      setConfigDraft(json);
      flash("Configuration saved");
    } catch (e) {
      flash(`Config save failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [configDraft, flash]);

  if (!config || !configDraft) {
    return (
      <div className="tc">
        <div className="tc-card" style={{ color: "var(--text-muted)" }}>
          Loading temperature config…
        </div>
      </div>
    );
  }

  const s = liveStatus;
  const state = s?.state ?? "idle";
  const relayOn = s?.relay_on ?? false;
  const hhLimit = s?.hh_limit_c ?? config.hh_limit_c;
  const deadband = s?.deadband_c ?? config.deadband_c;
  const tripped = state === "tripped";
  const hhTripped = tripped || (s ? hasHighHighTrip(s.trips) : false);

  // Highlight measured temperature amber within ~90 % of the HH limit.
  const tempWarn = s ? s.measured_temp_c >= 0.9 * hhLimit : false;

  // Live setpoint warning (shown under the entry; does not block typing).
  const stagedValue = Number.parseFloat(stagedSetpointText);
  let setpointWarning: string | null = null;
  if (Number.isFinite(stagedValue)) {
    if (stagedValue > config.setpoint_max_c) {
      setpointWarning = `Above max setpoint (${config.setpoint_max_c} °C) — will be clamped.`;
    } else if (stagedValue < config.setpoint_min_c) {
      setpointWarning = `Below min setpoint (${config.setpoint_min_c} °C) — will be clamped.`;
    } else if (stagedValue >= hhLimit) {
      setpointWarning = `At or above the high-high cutoff (${hhLimit} °C) — the heater would latch OFF.`;
    }
  }

  return (
    <div className="tc">
      {/* ---- Status header ---- */}
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
            <span className="tc-metric-value is-warning">{hhLimit.toFixed(0)} °C</span>
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
            <span className="tc-relay-text">
              HEATER {relayOn ? "ON" : "OFF"}
            </span>
          </div>

          {tripped && (
            <button
              className={`btn tc-btn-danger ${busy ? "tc-disabled" : ""}`}
              onClick={acknowledgeTrip}
              disabled={busy}
            >
              Acknowledge Trip
            </button>
          )}
          <button
            className={`tc-permissive ${s?.permissive ? "is-on" : ""}`}
            onClick={() => setPermissive(!s?.permissive)}
            disabled={busy}
            title={
              busy
                ? "Applying — please wait…"
                : "Master enable — the heater relay is forced open while OFF"
            }
          >
            <span className="dot" />
            {busy
              ? "APPLYING…"
              : s?.permissive
                ? "PERMISSIVE ON"
                : "PERMISSIVE OFF"}
          </button>
        </div>
      </div>

      {/* ---- High-high cutoff banner ---- */}
      {hhTripped && (
        <div className="tc-trip-banner">
          ⚠ High-high temperature cutoff — heater latched OFF. Acknowledge to
          reset.
        </div>
      )}

      {/* ---- Generic trips banner (non-HH trips still surface) ---- */}
      {s && s.trips.length > 0 && !hhTripped && (
        <div className="tc-trip-banner">⚠ Active trips: {s.trips.join(", ")}</div>
      )}

      {/* ---- Live trend ---- */}
      <div className="tc-card">
        <div className="tc-card-title">
          <span>Live temperature — {config.heater_label}</span>
        </div>
        <TempTrend
          samples={trend}
          fullScale={config.temp_full_scale_c}
          setpoint={s?.setpoint_c ?? config.setpoint_min_c}
          hhLimit={hhLimit}
        />
      </div>

      {/* ---- Setpoint ---- */}
      <div className="tc-card">
        <div className="tc-card-title">
          <span>Setpoint</span>
        </div>

        <div className="tc-setpoint-row">
          <button
            className="tc-step-btn"
            onClick={() => nudgeSetpoint(-setpointStep)}
            disabled={busy}
            title={`Decrease by ${setpointStep} °C`}
          >
            −
          </button>
          <input
            className="tc-setpoint-input"
            type="text"
            inputMode="decimal"
            value={stagedSetpointText}
            onChange={(e) => setStagedSetpointText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleApplyClick();
            }}
          />
          <button
            className="tc-step-btn"
            onClick={() => nudgeSetpoint(setpointStep)}
            disabled={busy}
            title={`Increase by ${setpointStep} °C`}
          >
            +
          </button>
        </div>

        <div className="tc-setpoint-controls">
          <span className="tc-step-field">
            step&nbsp;
            <input
              type="number"
              min={1}
              step={1}
              value={setpointStep}
              onChange={(e) =>
                setSetpointStep(Math.max(1, Number.parseFloat(e.target.value) || 1))
              }
            />
            °C
          </span>
          <button
            className={`btn btn-primary ${busy ? "tc-disabled" : ""}`}
            onClick={handleApplyClick}
            disabled={busy}
          >
            Apply Setpoint
          </button>
        </div>

        {s && s.state === "tripped" ? (
          <p className="tc-setpoint-warn">
            ⚠ Controller is TRIPPED — acknowledge the trip before commanding.
          </p>
        ) : s && !s.permissive ? (
          <p className="tc-setpoint-warn">
            ⚠ Permissive is OFF — enable it (switch, top-right) before commanding
            the heater.
          </p>
        ) : !s ? (
          <p className="tc-setpoint-warn">
            ⚠ Live status unavailable — commands are blocked until the controller
            reports in.
          </p>
        ) : (
          setpointWarning && <p className="tc-setpoint-warn">⚠ {setpointWarning}</p>
        )}

        <p className="tc-hint">
          Allowed setpoint {config.setpoint_min_c}–{config.setpoint_max_c} °C. The
          relay closes below <strong>setpoint − {deadband.toFixed(0)} °C</strong>{" "}
          and opens at setpoint (bang-bang control). The high-high cutoff at{" "}
          <strong>{hhLimit.toFixed(0)} °C</strong> latches the heater OFF
          regardless of setpoint.
        </p>
      </div>

      {/* ---- Live telemetry ---- */}
      <div className="tc-card">
        <div className="tc-card-title">Live telemetry</div>
        <div className="tc-readouts">
          <Readout label="Setpoint" value={`${s?.setpoint_c.toFixed(1) ?? "—"} °C`} />
          <Readout
            label="Measured temp"
            value={`${s?.measured_temp_c.toFixed(1) ?? "—"} °C`}
            warning={tempWarn}
          />
          <Readout
            label="Relay state"
            value={s ? (relayOn ? "ON" : "OFF") : "—"}
            warning={relayOn}
          />
          <Readout label="HH limit" value={`${hhLimit.toFixed(0)} °C`} />
        </div>
      </div>

      {/* ---- Advanced config ---- */}
      <details className="tc-card tc-config">
        <summary>Calibration &amp; limits</summary>

        <p className="tc-hint" style={{ marginTop: "0.8rem" }}>
          The thermocouple module reports temperature directly. These limits set
          the safe operating envelope — the server clamps setpoints to the band and
          latches the heater OFF at the high-high cutoff.
        </p>

        <div className="tc-config-grid">
          <ConfigField
            label="Deadband (°C)"
            value={configDraft.deadband_c}
            onChange={(v) => setConfigDraft({ ...configDraft, deadband_c: v })}
          />
          <ConfigField
            label="High-high cutoff (°C)"
            value={configDraft.hh_limit_c}
            onChange={(v) => setConfigDraft({ ...configDraft, hh_limit_c: v })}
          />
          <ConfigField
            label="Temperature full scale (°C)"
            value={configDraft.temp_full_scale_c}
            onChange={(v) => setConfigDraft({ ...configDraft, temp_full_scale_c: v })}
          />
          <ConfigField
            label="Max setpoint (°C)"
            value={configDraft.setpoint_max_c}
            onChange={(v) => setConfigDraft({ ...configDraft, setpoint_max_c: v })}
          />
          <ConfigField
            label="Temperature tag"
            value={configDraft.temp_tag}
            stringMode
            onChange={(v) =>
              setConfigDraft({ ...configDraft, temp_tag: v as unknown as string })
            }
          />
          <ConfigField
            label="Heater label"
            value={configDraft.heater_label}
            stringMode
            onChange={(v) =>
              setConfigDraft({ ...configDraft, heater_label: v as unknown as string })
            }
          />
        </div>

        <div style={{ marginTop: "1rem" }}>
          <button
            className={`btn ${busy ? "tc-disabled" : ""}`}
            onClick={saveConfig}
            disabled={busy}
          >
            Save Configuration
          </button>
        </div>
      </details>

      {info && <div className="tc-toast is-info">{info}</div>}
      {error && <div className="tc-toast is-error">{error}</div>}
    </div>
  );
};

/**
 * Compact single-trace temperature trend for the controller screen: measured
 * temperature plotted against a 0–full-scale °C Y axis, with the setpoint and
 * high-high cutoff drawn as horizontal reference lines.
 *
 * Self-contained SVG (no chart lib): the parent accumulates a rolling sample
 * buffer from the live status and passes it in.
 */
const TREND_W = 600;
const TREND_H = 180;
const TREND_PAD_L = 40;
const TREND_PAD_R = 10;
const TREND_PAD_T = 10;
const TREND_PAD_B = 18;
const TREND_PLOT_W = TREND_W - TREND_PAD_L - TREND_PAD_R;
const TREND_PLOT_H = TREND_H - TREND_PAD_T - TREND_PAD_B;

const TEMP_COLOR = "var(--color-error)";
const SETPOINT_COLOR = "var(--accent-cyan)";
const HH_COLOR = "var(--color-warning)";

function tempPath(values: number[], min: number, max: number): string {
  if (values.length < 2 || max <= min) return "";
  const n = values.length;
  return values
    .map((val, idx) => {
      const x = TREND_PAD_L + (idx / (n - 1)) * TREND_PLOT_W;
      const frac = Math.max(0, Math.min(1, (val - min) / (max - min)));
      const y = TREND_PAD_T + (1 - frac) * TREND_PLOT_H;
      return `${idx === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

interface TrendProps {
  samples: number[];
  fullScale: number;
  setpoint: number;
  hhLimit: number;
}

const TempTrend: React.FC<TrendProps> = ({
  samples,
  fullScale,
  setpoint,
  hhLimit,
}) => {
  const [axis, setAxis] = useState<AxisRange>(defaultAxisRange(0, fullScale));
  const [windowSeconds, setWindowSeconds] =
    useState<number>(DEFAULT_WINDOW_SECONDS);

  const windowed = samples.slice(-windowSampleCount(windowSeconds));
  const plotted = downsample(windowed);
  const { min, max } = resolveRange(axis, plotted, { min: 0, max: fullScale });
  const path = tempPath(plotted, min, max);
  const last = windowed[windowed.length - 1];

  const refY = (value: number): number => {
    const frac = Math.max(0, Math.min(1, (value - min) / (max - min)));
    return TREND_PAD_T + (1 - frac) * TREND_PLOT_H;
  };

  return (
    <div className="tc-trend">
      <div className="tc-trend-legend">
        <span className="tc-trend-key">
          <span className="swatch" style={{ background: TEMP_COLOR }} />
          Measured
          <strong>{last != null ? `${last.toFixed(1)} °C` : "—"}</strong>
        </span>
        <span className="tc-trend-key">
          <span className="swatch" style={{ background: SETPOINT_COLOR }} />
          Setpoint
          <strong>{setpoint.toFixed(1)} °C</strong>
        </span>
        <span className="tc-trend-key">
          <span className="swatch" style={{ background: HH_COLOR }} />
          HH cutoff
          <strong>{hhLimit.toFixed(0)} °C</strong>
        </span>
        <span className="tc-trend-window">
          last {formatWindow(windowSeconds)} · {min.toFixed(0)}–{max.toFixed(0)} °C
        </span>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          margin: "0.1rem 0 0.3rem",
          flexWrap: "wrap",
        }}
      >
        <TrendTimeControls value={windowSeconds} onChange={setWindowSeconds} />
        <TrendAxisControls value={axis} onChange={setAxis} unit="°C" />
      </div>

      <svg
        className="tc-trend-svg"
        viewBox={`0 0 ${TREND_W} ${TREND_H}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="Temperature trend"
      >
        {/* gridlines + °C axis labels across the resolved [min, max] range */}
        {axisTicks(min, max, 4).map((tick, i) => {
          const y = TREND_PAD_T + (1 - (tick - min) / (max - min)) * TREND_PLOT_H;
          return (
            <g key={i}>
              <line
                x1={TREND_PAD_L}
                y1={y}
                x2={TREND_W - TREND_PAD_R}
                y2={y}
                className="tc-trend-grid"
              />
              <text
                x={TREND_PAD_L - 6}
                y={y + 3}
                className="tc-trend-axis"
                textAnchor="end"
              >
                {tick.toFixed(0)}
              </text>
            </g>
          );
        })}

        {/* setpoint + HH reference lines (only when within the visible range) */}
        {max > min && (
          <>
            <line
              x1={TREND_PAD_L}
              y1={refY(setpoint)}
              x2={TREND_W - TREND_PAD_R}
              y2={refY(setpoint)}
              className="tc-trend-ref"
              stroke={SETPOINT_COLOR}
            />
            <line
              x1={TREND_PAD_L}
              y1={refY(hhLimit)}
              x2={TREND_W - TREND_PAD_R}
              y2={refY(hhLimit)}
              className="tc-trend-ref"
              stroke={HH_COLOR}
            />
          </>
        )}

        {plotted.length < 2 ? (
          <text
            x={TREND_W / 2}
            y={TREND_H / 2}
            className="tc-trend-empty"
            textAnchor="middle"
          >
            waiting for live data…
          </text>
        ) : (
          <path d={path} className="tc-trend-line" stroke={TEMP_COLOR} />
        )}
      </svg>
    </div>
  );
};

const Readout: React.FC<{ label: string; value: string; warning?: boolean }> = ({
  label,
  value,
  warning,
}) => (
  <div className={`tc-readout ${warning ? "is-warning" : ""}`}>
    <div className="tc-readout-label">{label}</div>
    <div className="tc-readout-value">{value}</div>
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
  <label className="tc-field">
    <span>{label}</span>
    <input
      type={stringMode ? "text" : "number"}
      value={value}
      onChange={(e) =>
        onChange(
          (stringMode ? e.target.value : Number.parseFloat(e.target.value)) as number,
        )
      }
    />
  </label>
);
