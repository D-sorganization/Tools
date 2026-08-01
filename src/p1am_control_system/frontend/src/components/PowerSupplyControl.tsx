import React, { useEffect, useState, useCallback, useRef } from "react";
import { Database } from "lucide-react";
import { PowerSupplyTrend, type TrendSample } from "./PowerSupplyTrend";
import { ExportButton } from "./ExportButton";
import { CollapsibleSection } from "./CollapsibleSection";
import { EditableValue } from "./EditableValue";
import { fetchWithTimeout } from "../lib/fetchWithTimeout";
import { useSetpointDraft } from "../hooks/useSetpointDraft";
import { MAX_TREND_SAMPLES } from "../lib/trendTime";
import "./PowerSupplyControl.css";

// Tags carrying the unit's current/voltage feedback (P1-4ADL2DAL AI0/AI1).
const PS_EXPORT_TAGS = [12, 13] as const;

// Rolling trend buffer: deep enough for the longest selectable window (5 min
// @ ~10 Hz); the plot itself slices/downsamples to the chosen window.
const TREND_MAX_POINTS = MAX_TREND_SAMPLES;

/**
 * Power-supply control tab.
 *
 * Talks to the backend's /api/power_supply/* endpoints and consumes the
 * `power_supply` field on the /api/stream WebSocket payload (broadcast each
 * scan tick by the polling loop in main.py).
 *
 * Safety patterns implemented:
 *  - Output clamp: a hard upper limit (% of full output) the controller never
 *    commands above, regardless of setpoint. Front-and-center, adjustable,
 *    default 20 % — the operator's guard rail for live-current testing.
 *  - Setpoint clamping happens server-side, so the displayed value reflects
 *    what was actually applied (not what was typed).
 *  - "Apply" commits the staged value — typing alone commands nothing.
 *  - Permissive must be ON before any setpoint takes effect (server enforces).
 *  - Trip indicator + acknowledge are first-class UI.
 */

export interface PowerSupplyConfig {
  command_tag: string;
  current_feedback_tag: string;
  voltage_feedback_tag: string;
  temp_tag: string;
  command_label: string;
  aux_command_label: string;
  current_feedback_label: string;
  voltage_feedback_label: string;
  temp_label: string;
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
  /**
   * Hard upper clamp on the commanded AO output, as a percent of full output
   * (0-100]. The controller never commands above this even when the setpoint
   * would scale higher. Operator safety limit for live-current testing.
   */
  output_clamp_percent: number;
  /** Rolling window (sample count) used to quantify feedback noise. */
  noise_window: number;
  /** Which noise metric the arc thresholds are compared against. */
  noise_metric: NoiseMetric;
  /** Arc-detect threshold for current-feedback noise (null = disabled). */
  current_arc_threshold: number | null;
  /** Arc-detect threshold for voltage-feedback noise (null = disabled). */
  voltage_arc_threshold: number | null;
}

export type NoiseMetric = "std" | "peak_to_peak" | "rms" | "cv";

export interface NoiseStats {
  sample_count: number;
  mean: number;
  std: number;
  peak_to_peak: number;
  rms_about_mean: number;
  coeff_of_variation: number;
  metric: NoiseMetric;
  metric_value: number;
  threshold: number | null;
  arcing: boolean;
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
  /** Active hard clamp on commanded output (% of full). */
  output_clamp_percent: number;
  /** True when the clamp is actively limiting the command this tick. */
  output_clamped: boolean;
  /** Real deliverable max current given the clamp (clamp% × full-scale). */
  effective_max_current_a: number;
  /** Rolling noise/variability stats for the current feedback. */
  current_noise: NoiseStats;
  /** Rolling noise/variability stats for the voltage feedback. */
  voltage_noise: NoiseStats;
  /** True when either channel's noise metric exceeds its arc threshold. */
  arcing: boolean;
}

interface Props {
  /** Status pushed each scan via the parent's WebSocket; undefined while waiting. */
  liveStatus?: PowerSupplyStatus;
  /** Opens the capture status / historian-management drawer. */
  onOpenCapture?: () => void;
}

const STATE_LABELS: Record<PowerSupplyStatus["state"], string> = {
  idle: "IDLE",
  armed: "ARMED",
  running: "RUNNING",
  tripped: "TRIPPED",
};

const STATE_HINTS: Record<PowerSupplyStatus["state"], string> = {
  idle: "permissive off",
  armed: "ready · set a target",
  running: "commanding output",
  tripped: "latched · acknowledge",
};

/** Format a wattage as kW (the supply runs in the kW range). */
function fmtKW(watts: number | null | undefined): string {
  if (watts == null) return "—";
  const kw = watts / 1000;
  return `${kw.toFixed(kw >= 10 ? 1 : 2)} kW`;
}

const NOISE_METRIC_LABELS: Record<NoiseMetric, string> = {
  std: "Std deviation",
  peak_to_peak: "Peak-to-peak",
  rms: "AC RMS",
  cv: "Coeff. of variation",
};

const NOISE_METRIC_HINTS: Record<NoiseMetric, string> = {
  std: "sample standard deviation, in engineering units (A / V)",
  peak_to_peak: "max − min over the window, in engineering units (A / V)",
  rms: "RMS of the AC content about the mean, in engineering units (A / V)",
  cv: "std ÷ |mean|, a dimensionless ratio (noise relative to DC level)",
};

/**
 * Is a measurement within `fraction` of its alarm limit?
 *
 * Deliberately fails LOUD (#4042b). The pre-alarm cue used to be `x >= 0.9 *
 * limit` computed against the LOCAL UNCOMMITTED config draft; clearing the
 * limit field stored `NaN`, and `x >= 0.9 * NaN` is false, so the cue switched
 * OFF while the supply was still climbing. An unusable limit is a reason to
 * warn, never a reason to reassure.
 *
 * @param measured - the live measurement (undefined before the first frame).
 * @param limit - the server-enforced alarm limit.
 * @param fraction - proportion of the limit at which to start warning.
 * @returns true when the operator should see the approaching-alarm cue.
 */
// Pure helper co-located with the component it serves so it can be unit-tested;
// not a component, so it never participates in fast refresh.
// eslint-disable-next-line react-refresh/only-export-components
export function approachingLimit(
  measured: number | undefined,
  limit: number | undefined,
  fraction = 0.9,
): boolean {
  if (measured === undefined || !Number.isFinite(measured)) return false;
  if (limit === undefined || !Number.isFinite(limit)) return true;
  return measured >= fraction * limit;
}

/**
 * Parse a numeric config entry, rejecting anything that is not finite.
 *
 * `Number.parseFloat(e.target.value)` written straight into the draft stored
 * `NaN` for an empty field, which then silently disabled the approaching-alarm
 * cue and would have been PUT to the backend on save (#4042b).
 *
 * @param text - raw input text.
 * @returns the finite number, or `null` when the text is not usable.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function parseConfigNumber(text: string): number | null {
  if (typeof text !== "string" || text.trim() === "") return null;
  const parsed = Number.parseFloat(text);
  return Number.isFinite(parsed) ? parsed : null;
}

/** Pull the value of the selected metric out of a NoiseStats snapshot. */
function noiseMetricValue(stats: NoiseStats | undefined): number | null {
  if (!stats) return null;
  switch (stats.metric) {
    case "std":
      return stats.std;
    case "peak_to_peak":
      return stats.peak_to_peak;
    case "rms":
      return stats.rms_about_mean;
    case "cv":
      return stats.coeff_of_variation;
    default:
      return stats.metric_value;
  }
}

const PowerSupplyControlImpl: React.FC<Props> = ({ liveStatus, onOpenCapture }) => {
  const [config, setConfig] = useState<PowerSupplyConfig | null>(null);
  const [configDraft, setConfigDraft] = useState<PowerSupplyConfig | null>(null);
  const [mode, setMode] = useState<"current" | "power">("current");
  // Staged setpoint entry, seeded from the supply's REAL setpoint and then
  // operator-owned (#4020). It used to be hard-initialised to "0" and never
  // seeded, so after a kiosk reload of a supply running at 30 A the box read 0
  // next to a telemetry readout of 30.00 A.
  const stagedSetpoint = useSetpointDraft(liveStatus?.setpoint_a, {
    scope: "power-supply",
    digits: 2,
  });
  const stagedSetpointText = stagedSetpoint.text;
  const setStagedSetpointText = stagedSetpoint.setText;
  const [setpointStep, setSetpointStep] = useState<number>(1.0);
  const [clampDraft, setClampDraft] = useState<number>(20);
  const [trend, setTrend] = useState<TrendSample[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const lastAppliedRef = useRef<number>(0);

  // Load config once.
  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetchWithTimeout("/api/power_supply/config");
        if (!res.ok) throw new Error(`config GET ${res.status}`);
        const cfg = (await res.json()) as PowerSupplyConfig;
        setConfig(cfg);
        setConfigDraft(cfg);
        setClampDraft(cfg.output_clamp_percent);
      } catch (e) {
        setError(`Load config failed: ${(e as Error).message}`);
      }
    };
    load();
  }, []);

  // Adopt the server's mode ONCE, on the first status frame. After that the
  // toggle is operator-owned: continuously syncing to liveStatus.mode would
  // yank the toggle back mid-selection (the server only changes mode on an
  // applied setpoint), which could lead to commanding the wrong quantity.
  const modeInitialized = useRef(false);
  useEffect(() => {
    if (liveStatus && !modeInitialized.current) {
      modeInitialized.current = true;
      setMode(liveStatus.mode);
    }
  }, [liveStatus]);

  // Accumulate a rolling trend buffer from the live status broadcasts.
  useEffect(() => {
    if (!liveStatus) return;
    setTrend((prev) => {
      const next = [
        ...prev,
        {
          t: Date.now(),
          i: liveStatus.measured_current_a,
          v: liveStatus.measured_voltage_v,
          p: liveStatus.measured_power_w,
        },
      ];
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
      // If we have no live status, we can't verify permissive/trip — refuse.
      if (!liveStatus) {
        flash("Live status unavailable — cannot confirm permissive/trip.", "error");
        return;
      }
      if (liveStatus.state === "tripped") {
        flash("Controller is TRIPPED — acknowledge the trip first.", "error");
        return;
      }
      if (!liveStatus.permissive) {
        flash("Permissive is OFF — enable it before commanding output.", "error");
        return;
      }
      setBusy(true);
      try {
        const body =
          mode === "current"
            ? { mode: "current", value_a: rawValue }
            : { mode: "power", value_w: rawValue };
        const res = await fetchWithTimeout("/api/power_supply/setpoint", {
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
    [config, mode, flash, liveStatus],
  );

  // +/- STAGE a value; they never command (#4020). Nudging used to call
  // applySetpoint immediately from a staged value that was never seeded, so on
  // a supply running at 30 A one tap of "+" commanded 1 A and collapsed the
  // output by 29 A with no confirmation — and it contradicted this file's own
  // documented contract, "Apply commits the staged value — typing alone
  // commands nothing". Apply is now the sole write path.
  const nudgeSetpoint = useCallback(
    (delta: number) => {
      const current = Number.parseFloat(stagedSetpointText);
      const base = Number.isFinite(current) ? current : (liveStatus?.setpoint_a ?? 0);
      setStagedSetpointText(Math.max(0, base + delta).toFixed(2));
    },
    [stagedSetpointText, liveStatus?.setpoint_a, setStagedSetpointText],
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
        const res = await fetchWithTimeout("/api/power_supply/permissive", {
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
      const res = await fetchWithTimeout("/api/power_supply/acknowledge_trip", {
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

  // PUT the full config with one or more fields overridden. Used by both the
  // clamp control (override output_clamp_percent only) and the advanced editor.
  const putConfig = useCallback(
    async (next: PowerSupplyConfig, okMessage: string) => {
      setBusy(true);
      try {
        const res = await fetchWithTimeout("/api/power_supply/config", {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(next),
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
        setClampDraft(json.output_clamp_percent);
        flash(okMessage);
      } catch (e) {
        flash(`Config save failed: ${(e as Error).message}`, "error");
      } finally {
        setBusy(false);
      }
    },
    [flash],
  );

  const applyClamp = useCallback(() => {
    if (!config) return;
    const clamped = Math.min(100, Math.max(0.1, clampDraft));
    putConfig(
      { ...config, output_clamp_percent: clamped },
      `Output limit set to ${clamped.toFixed(0)} %`,
    );
  }, [config, clampDraft, putConfig]);

  // Inline commit for the live "Limit" readout (shared EditableValue): applies
  // the output limit at once via the same config PUT the clamp control uses.
  const commitClamp = useCallback(
    (value: number) => {
      if (!config) return;
      putConfig(
        { ...config, output_clamp_percent: value },
        `Output limit set to ${value.toFixed(0)} %`,
      );
    },
    [config, putConfig],
  );

  const saveConfig = useCallback(() => {
    if (!configDraft) return;
    putConfig(configDraft, "Configuration saved");
  }, [configDraft, putConfig]);

  if (!config || !configDraft) {
    return (
      <div className="ps">
        <div className="ps-card" style={{ color: "var(--text-muted)" }}>
          Loading power-supply config…
        </div>
      </div>
    );
  }

  const s = liveStatus;
  const state = s?.state ?? "idle";
  const activeClamp = s?.output_clamp_percent ?? config.output_clamp_percent;
  const commanded = s?.commanded_output_percent ?? 0;
  const isClamped = s?.output_clamped ?? false;
  const unit = mode === "current" ? "A" : "W";
  const clampDirty = Math.abs(clampDraft - config.output_clamp_percent) > 1e-6;

  // Effective deliverable max current = clamp% × full-scale. The setpoint band
  // may go higher, but the output limit caps real output here. Prefer the live
  // value from the server (single source) and fall back to the local formula.
  const effectiveMaxA =
    s?.effective_max_current_a ??
    (config.output_clamp_percent / 100) * config.current_full_scale_a;

  // Live setpoint warning (shown under the entry; does not block typing).
  const stagedValue = Number.parseFloat(stagedSetpointText);
  let setpointWarning: string | null = null;
  if (mode === "current" && Number.isFinite(stagedValue)) {
    if (stagedValue > config.current_setpoint_max_a) {
      setpointWarning = `Above max setpoint (${config.current_setpoint_max_a} A) — will be clamped.`;
    } else if (stagedValue < config.current_setpoint_min_a) {
      setpointWarning = `Below min setpoint (${config.current_setpoint_min_a} A) — will be clamped.`;
    } else if (stagedValue > effectiveMaxA + 1e-6) {
      setpointWarning = `Output limit (${activeClamp.toFixed(0)} %) caps delivery at ${effectiveMaxA.toFixed(1)} A — raise the limit to go higher.`;
    }
  }

  // Approaching-alarm cues come from the SERVER-ENFORCED config, never from the
  // local uncommitted draft (#4042b): the draft is whatever the operator has
  // half-typed into the advanced editor, and an in-progress edit must not
  // change what the live readouts warn about.
  const powerWarn = approachingLimit(s?.measured_power_w, config.power_alarm_max_w);
  const tempWarn = approachingLimit(s?.measured_temp_c, config.temp_alarm_max_c);

  return (
    <div className="ps">
      {/* ---- Status header ---- */}
      <div className={`ps-status is-${state}`}>
        <div className="ps-state">
          <span className="ps-state-badge">{STATE_LABELS[state]}</span>
          <span className="ps-state-sub">{STATE_HINTS[state]}</span>
        </div>

        <div className="ps-status-metrics">
          <div className="ps-metric">
            <span className="ps-metric-label">Output cmd</span>
            <span className="ps-metric-value">{commanded.toFixed(1)} %</span>
          </div>
          <div className="ps-metric">
            <span className="ps-metric-label">Measured I</span>
            <span className="ps-metric-value">
              {s ? s.measured_current_a.toFixed(2) : "—"} A
            </span>
          </div>
          <div className="ps-metric">
            <span className="ps-metric-label">Measured V</span>
            <span className="ps-metric-value">
              {s ? s.measured_voltage_v.toFixed(2) : "—"} V
            </span>
          </div>
          <div className="ps-metric">
            <span className="ps-metric-label">Limit</span>
            <EditableValue
              className="ps-metric-value is-warning"
              value={activeClamp}
              label="Output limit"
              unit="%"
              format={(v) => v.toFixed(0)}
              min={0.1}
              max={100}
              step={5}
              title="Output limit — click to edit"
              onCommit={commitClamp}
            />
          </div>
        </div>

        <div className="ps-status-actions">
          {state === "tripped" && (
            <button
              className={`btn ps-btn-danger ${busy ? "ps-disabled" : ""}`}
              onClick={acknowledgeTrip}
              disabled={busy}
            >
              Acknowledge Trip
            </button>
          )}
          <button
            className={`ps-permissive ${s?.permissive ? "is-on" : ""}`}
            onClick={() => setPermissive(!s?.permissive)}
            disabled={busy}
            title={
              busy
                ? "Applying — please wait…"
                : "Master enable — output is forced to 0 while OFF"
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

      {/* ---- Trips banner ---- */}
      {s && s.trips.length > 0 && (
        <div className="ps-trip-banner">⚠ Active trips: {s.trips.join(", ")}</div>
      )}

      {/* ---- Live trend (current + voltage from the unit) ---- */}
      <CollapsibleSection
        className="ps-card"
        title={<span>Live signals — current, voltage &amp; power feedback</span>}
        headerExtra={
          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
            <ExportButton tags={PS_EXPORT_TAGS} label="Export" />
            {onOpenCapture && (
              <button
                className="btn ps-export-btn"
                onClick={onOpenCapture}
                title="Capture status & historian management"
              >
                <Database size={13} /> Capture
              </button>
            )}
          </div>
        }
      >
        <PowerSupplyTrend
          samples={trend}
          currentFullScale={config.current_full_scale_a}
          voltageFullScale={config.voltage_full_scale_v}
          powerFullScale={config.power_alarm_max_w}
          currentLabel={config.current_feedback_label}
          voltageLabel={config.voltage_feedback_label}
        />
      </CollapsibleSection>

      <div className="ps-grid">
        {/* ---- Output clamp (safety) ---- */}
        <div className="ps-card ps-clamp">
          <div className="ps-card-title">
            <span>Output limit</span>
            {isClamped && <span className="ps-clamp-badge">● clamping</span>}
          </div>

          <div className="ps-clamp-row">
            <input
              type="range"
              className="ps-slider"
              min={1}
              max={100}
              step={1}
              value={clampDraft}
              style={{ ["--ps-fill" as string]: `${clampDraft}%` }}
              onChange={(e) => setClampDraft(Number.parseFloat(e.target.value))}
              disabled={busy}
              aria-label="Output current limit percent"
            />
            <span className="ps-clamp-value">{clampDraft.toFixed(0)}%</span>
          </div>

          <div className="ps-setpoint-controls">
            <span className="ps-step-field">
              <input
                type="number"
                min={1}
                max={100}
                step={1}
                value={clampDraft}
                onChange={(e) =>
                  setClampDraft(
                    Math.min(100, Math.max(1, Number.parseFloat(e.target.value) || 1)),
                  )
                }
                disabled={busy}
              />
              %
            </span>
            <button
              className={`btn ${clampDirty ? "btn-primary" : ""} ${
                busy ? "ps-disabled" : ""
              }`}
              onClick={applyClamp}
              disabled={busy || !clampDirty}
            >
              {clampDirty ? "Set Limit" : "Limit Set"}
            </button>
          </div>

          <p className="ps-clamp-hint">
            Hard cap on commanded output. The supply will never drive above this
            percent of full scale even if the setpoint asks for more — your guard
            rail for live-current testing. Lowering it takes effect immediately.
          </p>
        </div>

        {/* ---- Setpoint ---- */}
        <div className="ps-card">
          <div className="ps-card-title">
            <span>Setpoint</span>
            <span className="ps-segment">
              {(["current", "power"] as const).map((m) => (
                <button
                  key={m}
                  className={mode === m ? "is-active" : ""}
                  onClick={() => setMode(m)}
                >
                  {m === "current" ? "Current (A)" : "Power (W)"}
                </button>
              ))}
            </span>
          </div>

          <div className="ps-setpoint-row">
            <button
              className="ps-step-btn"
              onClick={() => nudgeSetpoint(-setpointStep)}
              title={`Decrease by ${setpointStep} — stages only, press Apply to command`}
            >
              −
            </button>
            <input
              className="ps-setpoint-input"
              type="text"
              inputMode="decimal"
              value={stagedSetpointText}
              onChange={(e) => setStagedSetpointText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleApplyClick();
              }}
            />
            <button
              className="ps-step-btn"
              onClick={() => nudgeSetpoint(setpointStep)}
              title={`Increase by ${setpointStep} — stages only, press Apply to command`}
            >
              +
            </button>
          </div>

          <div className="ps-setpoint-controls">
            <span className="ps-step-field">
              step&nbsp;
              <input
                type="number"
                min={0.01}
                step={0.1}
                value={setpointStep}
                onChange={(e) =>
                  setSetpointStep(
                    Math.max(0.01, Number.parseFloat(e.target.value) || 0.01),
                  )
                }
              />
              {unit}
            </span>
            <button
              className={`btn btn-primary ${busy ? "ps-disabled" : ""}`}
              onClick={handleApplyClick}
              disabled={busy}
            >
              Apply Setpoint
            </button>
          </div>

          {s && s.state === "tripped" ? (
            <p className="ps-setpoint-warn">
              ⚠ Controller is TRIPPED — acknowledge the trip before commanding.
            </p>
          ) : s && !s.permissive ? (
            <p className="ps-setpoint-warn">
              ⚠ Permissive is OFF — enable it (switch, top-right) before commanding
              output.
            </p>
          ) : (
            setpointWarning && <p className="ps-setpoint-warn">⚠ {setpointWarning}</p>
          )}

          <p className="ps-clamp-hint">
            {mode === "current" ? (
              <>
                Allowed setpoint {config.current_setpoint_min_a}–
                {config.current_setpoint_max_a} A. With the {activeClamp.toFixed(0)} %
                output limit, the supply delivers at most{" "}
                <strong>{effectiveMaxA.toFixed(1)} A</strong> — raise the limit on
                the Output limit tile to deliver more.
              </>
            ) : (
              `Power → current via measured V; resulting current clamped to ${config.current_setpoint_min_a}–${config.current_setpoint_max_a} A, then to the ${activeClamp.toFixed(0)} % output limit.`
            )}
          </p>
        </div>
      </div>

      {/* ---- Live telemetry ---- */}
      <CollapsibleSection className="ps-card" title="Live telemetry">
        <div className="ps-readouts">
          <Readout label="Setpoint" value={`${s?.setpoint_a.toFixed(2) ?? "—"} A`} />
          <Readout
            label={config.current_feedback_label}
            value={`${s?.measured_current_a.toFixed(2) ?? "—"} A`}
          />
          <Readout
            label={config.voltage_feedback_label}
            value={`${s?.measured_voltage_v.toFixed(2) ?? "—"} V`}
          />
          <Readout
            label="Power (V × I)"
            value={fmtKW(s?.measured_power_w)}
            warning={powerWarn}
          />
          <Readout
            label={config.temp_label}
            value={`${s?.measured_temp_c.toFixed(1) ?? "—"} °C`}
            warning={tempWarn}
          />
          {s?.mode === "power" && (
            <Readout
              label="Power setpoint"
              value={s.setpoint_w != null ? fmtKW(s.setpoint_w) : "—"}
            />
          )}
        </div>

        {/* Output-vs-limit bar */}
        <div className="ps-outbar-wrap">
          <div className="ps-outbar-head">
            <span>Commanded output</span>
            <span>
              {commanded.toFixed(1)} % / {activeClamp.toFixed(0)} % limit
            </span>
          </div>
          <div className="ps-outbar">
            <div
              className={`ps-outbar-fill ${isClamped ? "is-clamped" : ""}`}
              style={{ width: `${Math.min(100, commanded)}%` }}
            />
            <div
              className="ps-outbar-marker"
              style={{ left: `${Math.min(100, activeClamp)}%` }}
            />
          </div>
        </div>
      </CollapsibleSection>

      {/* ---- Signal noise / arc detection ---- */}
      <div className={`ps-card ps-noise ${s?.arcing ? "is-arcing" : ""}`}>
        <div className="ps-card-title">
          <span>Signal noise · arc detection</span>
          <span className={`ps-arc-badge ${s?.arcing ? "is-arcing" : ""}`}>
            {s?.arcing ? "⚡ POSSIBLE ARCING" : "stable"}
          </span>
        </div>
        <p className="ps-noise-hint">
          A DC arc rides as AC noise on the feedback. Watch the{" "}
          <strong>{NOISE_METRIC_LABELS[config.noise_metric]}</strong> — when it
          jumps while the setpoint is steady, the arc is active. Set a threshold
          below to flag it automatically. Window:{" "}
          <strong>{config.noise_window}</strong> samples (~
          {(config.noise_window / 10).toFixed(0)} s @ 10 Hz) ·{" "}
          {NOISE_METRIC_HINTS[config.noise_metric]}.
        </p>
        <div className="ps-noise-grid">
          <NoiseChannel
            title={config.current_feedback_label}
            unit="A"
            stats={s?.current_noise}
            threshold={config.current_arc_threshold}
          />
          <NoiseChannel
            title={config.voltage_feedback_label}
            unit="V"
            stats={s?.voltage_noise}
            threshold={config.voltage_arc_threshold}
          />
        </div>
      </div>

      {/* ---- Advanced config ---- */}
      {/* ---- Calibration & signal names ---- */}
      <details className="ps-card ps-config">
        <summary>Calibration &amp; signal names</summary>

        <p className="ps-clamp-hint" style={{ marginTop: "0.8rem" }}>
          The PLC speaks 0–100 % (4–20 mA / 0–5 V). Calibration tells the HMI what
          full scale means in real units — set these to what the supply's own
          meters read at full output so the readings match.
        </p>

        <div className="ps-config-grid">
          <ConfigField
            label="Current full scale — A at 5 V / 20 mA"
            value={configDraft.current_full_scale_a}
            onChange={(v) => setConfigDraft({ ...configDraft, current_full_scale_a: v })}
          />
          <ConfigField
            label="Voltage full scale — V at 5 V / 20 mA"
            value={configDraft.voltage_full_scale_v}
            onChange={(v) => setConfigDraft({ ...configDraft, voltage_full_scale_v: v })}
          />
        </div>

        <div className="ps-card-title" style={{ marginTop: "1.1rem" }}>
          Signal names (HMI labels)
        </div>
        <div className="ps-config-grid">
          <ConfigField
            label="Current command (AO0)"
            value={configDraft.command_label}
            stringMode
            onChange={(v) =>
              setConfigDraft({ ...configDraft, command_label: v as unknown as string })
            }
          />
          <ConfigField
            label="Aux command (AO1)"
            value={configDraft.aux_command_label}
            stringMode
            onChange={(v) =>
              setConfigDraft({
                ...configDraft,
                aux_command_label: v as unknown as string,
              })
            }
          />
          <ConfigField
            label="Current feedback (AI0)"
            value={configDraft.current_feedback_label}
            stringMode
            onChange={(v) =>
              setConfigDraft({
                ...configDraft,
                current_feedback_label: v as unknown as string,
              })
            }
          />
          <ConfigField
            label="Voltage feedback (AI1)"
            value={configDraft.voltage_feedback_label}
            stringMode
            onChange={(v) =>
              setConfigDraft({
                ...configDraft,
                voltage_feedback_label: v as unknown as string,
              })
            }
          />
          <ConfigField
            label="Temperature (TC0)"
            value={configDraft.temp_label}
            stringMode
            onChange={(v) =>
              setConfigDraft({ ...configDraft, temp_label: v as unknown as string })
            }
          />
        </div>

        <div className="ps-card-title" style={{ marginTop: "1.1rem" }}>
          Wiring guide
        </div>
        <WiringGuide config={configDraft} />

        <div style={{ marginTop: "1rem" }}>
          <button
            className={`btn btn-primary ${busy ? "ps-disabled" : ""}`}
            onClick={saveConfig}
            disabled={busy}
          >
            Save Calibration
          </button>
        </div>
      </details>

      <details className="ps-card ps-config">
        <summary>Advanced — scaling, bounds, alarms, ramp, tag map</summary>
        <div className="ps-config-grid">
          <ConfigField
            label="Current full scale (A at 100 %)"
            value={configDraft.current_full_scale_a}
            onChange={(v) => setConfigDraft({ ...configDraft, current_full_scale_a: v })}
          />
          <ConfigField
            label="Voltage full scale (V at 100 %)"
            value={configDraft.voltage_full_scale_v}
            onChange={(v) => setConfigDraft({ ...configDraft, voltage_full_scale_v: v })}
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
            label="Output limit (% of full)"
            value={configDraft.output_clamp_percent}
            onChange={(v) =>
              setConfigDraft({ ...configDraft, output_clamp_percent: v })
            }
          />
          <ConfigField
            label="Ramp rate on INCREASE (%/s)"
            value={configDraft.setpoint_ramp_rate_pct_per_s}
            onChange={(v) =>
              setConfigDraft({ ...configDraft, setpoint_ramp_rate_pct_per_s: v })
            }
          />
          <ConfigField
            label="HH Power alarm (W)"
            value={configDraft.power_alarm_max_w}
            onChange={(v) => setConfigDraft({ ...configDraft, power_alarm_max_w: v })}
          />
          <ConfigField
            label="HH Temperature alarm (°C)"
            value={configDraft.temp_alarm_max_c}
            onChange={(v) => setConfigDraft({ ...configDraft, temp_alarm_max_c: v })}
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
        <div style={{ marginTop: "1rem" }}>
          <button
            className={`btn ${busy ? "ps-disabled" : ""}`}
            onClick={saveConfig}
            disabled={busy}
          >
            Save Configuration
          </button>
        </div>
      </details>

      <details className="ps-card ps-config">
        <summary>Arc / noise detection — window, metric, thresholds</summary>
        <p className="ps-noise-hint">
          Tune how the system decides it&apos;s arcing. Pick the metric that best
          separates a quiet DC output from a noisy arc, size the averaging window,
          then set per-channel thresholds (leave blank to disable that channel).
        </p>
        <div className="ps-config-grid">
          <ConfigField
            label="Noise window (samples)"
            value={configDraft.noise_window}
            onChange={(v) =>
              setConfigDraft({ ...configDraft, noise_window: Math.round(v) })
            }
          />
          <label className="ps-field">
            <span>Noise metric</span>
            <select
              value={configDraft.noise_metric}
              onChange={(e) =>
                setConfigDraft({
                  ...configDraft,
                  noise_metric: e.target.value as NoiseMetric,
                })
              }
            >
              <option value="std">Std deviation (A / V)</option>
              <option value="peak_to_peak">Peak-to-peak (A / V)</option>
              <option value="rms">AC RMS (A / V)</option>
              <option value="cv">Coeff. of variation (ratio)</option>
            </select>
          </label>
          <label className="ps-field">
            <span>Current arc threshold</span>
            <input
              type="number"
              placeholder="disabled"
              value={configDraft.current_arc_threshold ?? ""}
              onChange={(e) =>
                setConfigDraft({
                  ...configDraft,
                  current_arc_threshold:
                    e.target.value === ""
                      ? null
                      : Number.parseFloat(e.target.value),
                })
              }
            />
          </label>
          <label className="ps-field">
            <span>Voltage arc threshold</span>
            <input
              type="number"
              placeholder="disabled"
              value={configDraft.voltage_arc_threshold ?? ""}
              onChange={(e) =>
                setConfigDraft({
                  ...configDraft,
                  voltage_arc_threshold:
                    e.target.value === ""
                      ? null
                      : Number.parseFloat(e.target.value),
                })
              }
            />
          </label>
        </div>
        <div style={{ marginTop: "1rem" }}>
          <button
            className={`btn ${busy ? "ps-disabled" : ""}`}
            onClick={saveConfig}
            disabled={busy}
          >
            Save Detection
          </button>
        </div>
      </details>

      {info && <div className="ps-toast is-info">{info}</div>}
      {error && <div className="ps-toast is-error">{error}</div>}
    </div>
  );
};

/**
 * Memoized so this always-mounted panel only re-renders when `liveStatus`
 * (ref-stable from useTelemetryStream — only a real status change bumps the
 * reference) or `onOpenCapture` change, not on every ~10 Hz App re-render.
 */
export const PowerSupplyControl = React.memo(PowerSupplyControlImpl);

/**
 * Static reference of how the power-supply signals map to the P1AM analog
 * terminals. Labels and tags come from the config (DRY) so renaming a signal
 * updates this table too.
 */
const WiringGuide: React.FC<{ config: PowerSupplyConfig }> = ({ config }) => {
  const rows: { name: string; tag: string; terminal: string; ps: string }[] = [
    {
      name: config.command_label,
      tag: config.command_tag,
      terminal: "Slot 2 · AO0 (4–20 mA out)",
      ps: "Remote current-setpoint input",
    },
    {
      name: config.aux_command_label,
      tag: "TAG_11",
      terminal: "Slot 2 · AO1 (4–20 mA out)",
      ps: "Spare (e.g. voltage-setpoint input)",
    },
    {
      name: config.current_feedback_label,
      tag: config.current_feedback_tag,
      terminal: "Slot 2 · AI0 (4–20 mA in)",
      ps: "Current-monitor output",
    },
    {
      name: config.voltage_feedback_label,
      tag: config.voltage_feedback_tag,
      terminal: "Slot 2 · AI1 (4–20 mA in)",
      ps: "Voltage-monitor output",
    },
    {
      name: config.temp_label,
      tag: config.temp_tag,
      terminal: "Slot 1 · TC0 (thermocouple)",
      ps: "Supply / load thermocouple",
    },
  ];
  return (
    <table className="ps-wire">
      <thead>
        <tr>
          <th>Signal</th>
          <th>Tag</th>
          <th>P1AM terminal</th>
          <th>Power-supply connection</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.tag + r.terminal}>
            <td>{r.name}</td>
            <td className="mono">{r.tag}</td>
            <td>{r.terminal}</td>
            <td>{r.ps}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

const Readout: React.FC<{ label: string; value: string; warning?: boolean }> = ({
  label,
  value,
  warning,
}) => (
  <div className={`ps-readout ${warning ? "is-warning" : ""}`}>
    <div className="ps-readout-label">{label}</div>
    <div className="ps-readout-value">{value}</div>
  </div>
);

const NoiseChannel: React.FC<{
  title: string;
  unit: string;
  stats?: NoiseStats;
  threshold: number | null;
}> = ({ title, unit, stats, threshold }) => {
  const metric = stats?.metric ?? "std";
  const u = metric === "cv" ? "" : unit;
  const value = noiseMetricValue(stats);
  const arcing = stats?.arcing ?? false;
  const fmt = (n: number | null | undefined, digits = 3) =>
    n == null ? "—" : n.toFixed(digits);
  return (
    <div className={`ps-noise-ch ${arcing ? "is-arcing" : ""}`}>
      <div className="ps-noise-ch-head">
        <span>{title}</span>
        {arcing && <span className="ps-noise-flag">arcing</span>}
      </div>
      <div className="ps-noise-metric">
        {fmt(value)} <span className="ps-noise-unit">{u || "ratio"}</span>
      </div>
      <div className="ps-noise-sub">
        {NOISE_METRIC_LABELS[metric]}
        {threshold != null
          ? ` · threshold ${threshold} ${u}`.trimEnd()
          : " · no threshold set"}
      </div>
      <div className="ps-noise-stats">
        <span>σ {fmt(stats?.std)}</span>
        <span>p-p {fmt(stats?.peak_to_peak)}</span>
        <span>rms {fmt(stats?.rms_about_mean)}</span>
        <span>cv {fmt(stats?.coeff_of_variation)}</span>
        <span>mean {fmt(stats?.mean, 2)}</span>
        <span>n {stats?.sample_count ?? 0}</span>
      </div>
    </div>
  );
};

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
  <label className="ps-field">
    <span>{label}</span>
    <input
      type={stringMode ? "text" : "number"}
      value={value}
      onChange={(e) => {
        if (stringMode) {
          onChange(e.target.value as unknown as number);
          return;
        }
        // Reject non-finite input rather than writing it into the draft
        // (#4042b): `Number.parseFloat("")` is NaN, and a NaN alarm limit
        // silently switched the pre-alarm cue OFF while the supply climbed.
        // An unparseable keystroke simply leaves the last good value in place.
        const parsed = parseConfigNumber(e.target.value);
        if (parsed !== null) onChange(parsed);
      }}
    />
  </label>
);
