import React, { useEffect, useState, useCallback, useRef } from "react";
import { fetchWithTimeout } from "../lib/fetchWithTimeout";
import {
  startStopView,
  setpointOutcome,
  resolveStartTarget,
  hasHighHighTrip,
} from "../lib/heaterControls";
import { MAX_TREND_SAMPLES } from "../lib/trendTime";
import { seedDraftText } from "../lib/operatorDraft";
import type { TempSample } from "../lib/temperatureTrend";
import { ExportButton } from "./ExportButton";
import { CollapsibleSection } from "./CollapsibleSection";
import { HeaterStartStopButton } from "./HeaterStartStopButton";
import { TemperatureConfigPanel } from "./TemperatureConfigPanel";
import { TemperatureStatusHeader } from "./TemperatureStatusHeader";
import { TempTrend } from "./TemperatureTrend";
import { ThermocoupleSelector } from "./ThermocoupleSelector";
import type { TcType, TemperatureConfig, TemperatureStatus } from "../types";
import "./TemperatureControl.css";

// Re-exported so existing importers of these types keep working now that they
// live in the shared types module (type-only: erased at build time).
export type {
  TcType,
  ThermocoupleChannel,
  TemperatureConfig,
  TemperatureStatus,
} from "../types";

// Rolling trend buffer: deep enough for the longest selectable window
// (up to 1 h); the plot slices to the chosen window by real time.
const TREND_MAX_POINTS = MAX_TREND_SAMPLES;

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
 *  - Operator model is Start/Stop: Start arms the heater (and applies the shown
 *    target); Stop opens the relay immediately. A setpoint is committed by
 *    pressing Enter (or the ± steps) — there is no separate apply step. While
 *    stopped a typed target is staged and takes effect on Start (server enforces
 *    that a setpoint only applies once the controller is started).
 *
 * This file owns the controller's state and every REST call. The screen's
 * sections — status header, thermocouple selector, trend and config editor —
 * are prop-driven components (see the TuningPanel split, #4053); the file was
 * 1975 lines, 475 over the repo's source budget, which made it uneditable
 * without tripping the file-size guardrail.
 */

/**
 * Decide the text to pre-fill the setpoint entry with on recall.
 *
 * Pure so it can be unit-tested without rendering. Returns the formatted
 * (one-decimal) string when a persisted last setpoint should be shown, or
 * `null` when nothing should change — i.e. the operator has already typed this
 * session (`operatorTouched`) or the recalled value is not a finite number.
 *
 * Thin wrapper over the shared {@link seedDraftText} rule (#4013, #4020): this
 * entry is one of several that must show the device's value until the operator
 * takes ownership, and the policy now lives in exactly one place.
 *
 * @param lastSetpointC - recalled last-session setpoint (may be null/undefined).
 * @param operatorTouched - true once the operator has edited the field.
 * @throws TypeError if `operatorTouched` is not a boolean.
 */
// Small pure helper co-located with the component it serves so it can be
// unit-tested; not a component, so it never participates in fast refresh.
// eslint-disable-next-line react-refresh/only-export-components
export function recallSetpointText(
  lastSetpointC: number | null | undefined,
  operatorTouched: boolean,
): string | null {
  return seedDraftText(lastSetpointC, operatorTouched, 1);
}

interface Props {
  /** Status pushed each scan via the parent's WebSocket; undefined while waiting. */
  liveStatus?: TemperatureStatus;
}

const TemperatureControlImpl: React.FC<Props> = ({ liveStatus }) => {
  const [config, setConfig] = useState<TemperatureConfig | null>(null);
  const [configDraft, setConfigDraft] = useState<TemperatureConfig | null>(null);
  const [stagedSetpointText, setStagedSetpointText] = useState<string>("0");
  const [setpointStep, setSetpointStep] = useState<number>(10);
  const [trend, setTrend] = useState<TempSample[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  // Recall guard: true once the operator has typed/nudged a setpoint this
  // session (so live status updates never clobber their edit), and a one-shot
  // latch so the recalled last_setpoint_c only pre-fills the field a single
  // time even though liveStatus arrives on every scan tick.
  const operatorTouchedRef = useRef(false);
  const recalledRef = useRef(false);

  // Mark the setpoint entry as operator-owned once they type or nudge it.
  const touchSetpoint = useCallback((text: string) => {
    operatorTouchedRef.current = true;
    setStagedSetpointText(text);
  }, []);

  // Recall the last-session setpoint ONCE, before any operator edit, so after a
  // backend restart the previous target is shown ready to Start. Guarded by a
  // pure helper + a latch so live updates don't overwrite operator input.
  useEffect(() => {
    if (recalledRef.current) return;
    const recalled = recallSetpointText(
      liveStatus?.last_setpoint_c,
      operatorTouchedRef.current,
    );
    if (recalled !== null) {
      recalledRef.current = true;
      setStagedSetpointText(recalled);
    }
  }, [liveStatus?.last_setpoint_c]);

  // Load config (reused after a thermocouple switch re-clamps the limits).
  const loadConfig = useCallback(async () => {
    try {
      const res = await fetchWithTimeout("/api/temperature/config");
      if (!res.ok) throw new Error(`config GET ${res.status}`);
      const cfg = (await res.json()) as TemperatureConfig;
      setConfig(cfg);
      setConfigDraft(cfg);
    } catch (e) {
      setError(`Load config failed: ${(e as Error).message}`);
    }
  }, []);

  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  // Accumulate a rolling trend buffer from the live status broadcasts. Capture
  // BOTH thermocouple readings (and the relay state) each scan so the trend can
  // plot K and R together — the active channel's reading is `measured_temp_c`,
  // but type_k_temp_c / type_r_temp_c are present every scan regardless.
  useEffect(() => {
    if (!liveStatus) return;
    setTrend((prev) => {
      const next = [
        ...prev,
        {
          t: Date.now(),
          k: liveStatus.type_k_temp_c ?? null,
          r: liveStatus.type_r_temp_c ?? null,
          relayOn: liveStatus.relay_on,
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

  // Raw POST helpers (DRY): used by both the Start/Stop button and the setpoint
  // commit. They don't consult the (possibly stale) live status — the caller
  // decides when it is safe to send.
  const postPermissive = useCallback(async (enabled: boolean): Promise<void> => {
    const res = await fetchWithTimeout("/api/temperature/permissive", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    if (!res.ok) throw new Error(await res.text());
  }, []);

  const postSetpoint = useCallback(async (value: number): Promise<number> => {
    const res = await fetchWithTimeout("/api/temperature/setpoint", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ value_c: value }),
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail ?? res.statusText);
    return json.applied_c as number;
  }, []);

  // Commit a setpoint. When the heater is started it applies now; when stopped
  // it is staged as the target (applied on Start); when tripped / status-unknown
  // it is refused — decided by the pure setpointOutcome() helper.
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
      const outcome = setpointOutcome(liveStatus);
      if (outcome === "blocked") {
        flash(
          liveStatus?.state === "tripped"
            ? "Controller is TRIPPED — acknowledge the trip first."
            : "Live status unavailable — commands are blocked.",
          "error",
        );
        return;
      }
      if (outcome === "stage") {
        setStagedSetpointText(rawValue.toFixed(1));
        flash(`Target set to ${rawValue.toFixed(1)} °C — press Start to heat`);
        return;
      }
      setBusy(true);
      try {
        const applied = await postSetpoint(rawValue);
        setStagedSetpointText(applied.toFixed(1));
        flash(`Applied ${applied.toFixed(1)} °C`);
      } catch (e) {
        flash(`Setpoint failed: ${(e as Error).message}`, "error");
      } finally {
        setBusy(false);
      }
    },
    [config, flash, liveStatus, postSetpoint],
  );

  const nudgeSetpoint = useCallback(
    (delta: number) => {
      const current = Number.parseFloat(stagedSetpointText);
      const next = Math.max(0, (Number.isFinite(current) ? current : 0) + delta);
      operatorTouchedRef.current = true;
      setStagedSetpointText(next.toFixed(1));
      applySetpoint(next);
    },
    [stagedSetpointText, applySetpoint],
  );

  // Commit the typed setpoint (Enter key). No separate "apply" button.
  const commitSetpoint = useCallback(() => {
    const v = Number.parseFloat(stagedSetpointText);
    if (!Number.isFinite(v)) {
      flash("Enter a number first", "error");
      return;
    }
    applySetpoint(v);
  }, [stagedSetpointText, applySetpoint, flash]);

  // Start: arm the heater, then heat to the shown target if one is set. Sends
  // the setpoint after arming (server-side ordering) so it isn't rejected.
  const handleStart = useCallback(async () => {
    setBusy(true);
    try {
      await postPermissive(true);
      // Fall back to the recalled/held setpoint when the entry box hasn't been
      // pre-filled yet (startup race), so a single Start applies the intended
      // target instead of only arming. The backend seeds setpoint_c at boot.
      const target = resolveStartTarget(
        stagedSetpointText,
        liveStatus?.last_setpoint_c ?? liveStatus?.setpoint_c,
      );
      if (target !== null) {
        const applied = await postSetpoint(target);
        setStagedSetpointText(applied.toFixed(1));
        flash(`Heater started — heating to ${applied.toFixed(1)} °C`);
      } else {
        flash("Heater started — enter a target to begin heating");
      }
    } catch (e) {
      flash(`Start failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [
    postPermissive,
    postSetpoint,
    stagedSetpointText,
    liveStatus?.last_setpoint_c,
    liveStatus?.setpoint_c,
    flash,
  ]);

  // Stop: open the relay immediately (confirm if it is currently energized).
  const handleStop = useCallback(async () => {
    if (
      liveStatus?.relay_on &&
      !window.confirm("Stop the heater? The relay will open immediately.")
    ) {
      return;
    }
    setBusy(true);
    try {
      await postPermissive(false);
      flash("Heater stopped");
    } catch (e) {
      flash(`Stop failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [postPermissive, liveStatus?.relay_on, flash]);

  const setActiveTcType = useCallback(
    async (tcType: TcType) => {
      setBusy(true);
      try {
        const res = await fetchWithTimeout("/api/temperature/tc_type", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ active_tc_type: tcType }),
        });
        if (!res.ok) throw new Error(await res.text());
        flash(`Heater now reading the Type ${tcType} thermocouple`);
        await loadConfig(); // limits may have re-clamped to the new channel
      } catch (e) {
        flash(`Thermocouple switch failed: ${(e as Error).message}`, "error");
      } finally {
        setBusy(false);
      }
    },
    [flash, loadConfig],
  );

  // Select the P1-04THM open-circuit fail direction. High-side = an open reads
  // full scale so the heater shuts off (fail-safe); low-side = an open reads cold.
  const setBurnoutMode = useCallback(
    async (highSide: boolean) => {
      setBusy(true);
      try {
        const res = await fetchWithTimeout("/api/temperature/burnout_mode", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ high_side: highSide }),
        });
        if (!res.ok) throw new Error(await res.text());
        flash(
          highSide
            ? "Open circuit now reads HOT — heater shuts off (fail-safe)"
            : "Open circuit now reads COLD — low-side burnout",
        );
      } catch (e) {
        flash(`Burnout mode change failed: ${(e as Error).message}`, "error");
      } finally {
        setBusy(false);
      }
    },
    [flash],
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

  // Commit a single config field via the same PUT the config editor uses, so an
  // inline edit (e.g. the HH cutoff shown in the live section) applies at once
  // without a separate Save step. The server re-validates + clamps and returns
  // the authoritative config, which we adopt. Best-effort: a reject/failure is
  // surfaced and the displayed value stays at the server's last-known value.
  const commitConfigField = useCallback(
    async <K extends keyof TemperatureConfig>(
      field: K,
      value: TemperatureConfig[K],
      describe: (cfg: TemperatureConfig) => string,
    ): Promise<void> => {
      if (!config) return;
      try {
        const res = await fetchWithTimeout("/api/temperature/config", {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ ...config, [field]: value }),
        });
        const json = await res.json();
        if (!res.ok) {
          flash(
            `Change rejected: ${
              json.detail ? JSON.stringify(json.detail) : res.statusText
            }`,
            "error",
          );
          return;
        }
        setConfig(json);
        setConfigDraft(json);
        flash(describe(json as TemperatureConfig));
      } catch (e) {
        flash(`Change failed: ${(e as Error).message}`, "error");
      }
    },
    [config, flash],
  );

  const commitHhLimit = useCallback(
    (value: number) =>
      commitConfigField(
        "hh_limit_c",
        value,
        (cfg) => `High-high cutoff set to ${cfg.hh_limit_c.toFixed(0)} °C`,
      ),
    [commitConfigField],
  );

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
  const activeTcType: TcType = s?.active_tc_type ?? config.active_tc_type;
  const hhLimit = s?.hh_limit_c ?? config.hh_limit_c;
  const deadband = s?.deadband_c ?? config.deadband_c;
  const tripped = state === "tripped";
  const hhTripped = tripped || (s ? hasHighHighTrip(s.trips) : false);
  // Operator-facing Start/Stop button state (pure helper — see lib/heaterControls).
  const startStop = startStopView(s);

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
      <TemperatureStatusHeader
        status={s}
        config={config}
        state={state}
        relayOn={relayOn}
        hhLimit={hhLimit}
        tempWarn={tempWarn}
        tripped={tripped}
        busy={busy}
        startStop={startStop}
        onCommitHhLimit={commitHhLimit}
        onAcknowledgeTrip={acknowledgeTrip}
        onStart={handleStart}
        onStop={handleStop}
      />

      {/* ---- Setpoint (prominent, always-visible primary control) ---- */}
      <div className="tc-card tc-setpoint-card">
        <div className="tc-setpoint-card-head">
          <span className="tc-card-title">Setpoint</span>
          <span className="tc-setpoint-card-current">
            target {s ? s.setpoint_c.toFixed(1) : "—"} °C
          </span>
        </div>

        <div className="tc-setpoint-primary">
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
              onChange={(e) => touchSetpoint(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") commitSetpoint();
              }}
              aria-label="Setpoint target (°C) — press Enter to apply"
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

          <HeaterStartStopButton
            view={startStop}
            busy={busy}
            onStart={handleStart}
            onStop={handleStop}
            className="tc-setpoint-startstop"
          />
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
          <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
            Type a target and press <strong>Enter</strong> to apply — or use ± .
          </span>
        </div>

        {s && s.state === "tripped" ? (
          <p className="tc-setpoint-warn">
            ⚠ Controller is TRIPPED — acknowledge the trip before commanding.
          </p>
        ) : s && !s.permissive ? (
          <p className="tc-setpoint-warn">
            Heater is stopped — set a target, then press <strong>Start</strong> to
            begin heating.
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

      {/* ---- Live trend (kept high, just under the setpoint, for at-a-glance
             monitoring during a heat run) ---- */}
      <CollapsibleSection
        className="tc-card"
        title={<span>Live temperature — {config.heater_label}</span>}
        headerExtra={
          <ExportButton
            tags={[Number.parseInt(config.temp_tag.replace(/\D/g, ""), 10) || 0]}
            label="Export"
            onError={setError}
          />
        }
      >
        <TempTrend
          samples={trend}
          tagId={Number.parseInt(config.temp_tag.replace(/\D/g, ""), 10) || 0}
          fullScale={config.temp_full_scale_c}
          setpoint={s?.setpoint_c ?? config.setpoint_min_c}
          hhLimit={hhLimit}
          activeTcType={activeTcType}
        />
      </CollapsibleSection>

      {/* ---- Thermocouple selector (Type K / Type R) ---- */}
      <ThermocoupleSelector
        config={config}
        status={s}
        activeTcType={activeTcType}
        busy={busy}
        onSelectTcType={setActiveTcType}
        onSetBurnoutMode={setBurnoutMode}
      />

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

      {/* ---- Control-sensor deglitch hold (intermittent dropout in progress) ---- */}
      {s?.control_sensor_holding && s.trips.length === 0 && (
        <div className="tc-hold-banner">
          ⚠ Control thermocouple glitching — holding last-good reading. Control is
          protected; check the {activeTcType === "K" ? "Type K" : "Type R"} probe
          wiring/connections. A sustained fault will trip the heater.
        </div>
      )}

      {/* ---- Live telemetry ---- */}
      <CollapsibleSection className="tc-card" title="Live telemetry">
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
      </CollapsibleSection>

      {/* ---- Advanced config ---- */}
      <TemperatureConfigPanel
        draft={configDraft}
        busy={busy}
        onDraftChange={setConfigDraft}
        onSave={saveConfig}
      />

      {info && <div className="tc-toast is-info">{info}</div>}
      {error && <div className="tc-toast is-error">{error}</div>}
    </div>
  );
};

/**
 * Memoized so this always-mounted panel only re-renders when `liveStatus`
 * (ref-stable from useTelemetryStream) changes, not on every ~10 Hz App frame.
 */
export const TemperatureControl = React.memo(TemperatureControlImpl);

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
