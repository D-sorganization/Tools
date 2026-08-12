import React, { useEffect, useState, useCallback, useRef } from "react";
import { fetchWithTimeout } from "../lib/fetchWithTimeout";
import { sourceLabel, TEMP_SOURCES } from "../lib/tempSource";
import type { TcType, TcPath } from "../lib/tempSource";
import {
  startStopView,
  setpointOutcome,
  resolveStartTarget,
} from "../lib/heaterControls";
import {
  type AxisRange,
  defaultAxisRange,
  resolveRange,
  axisTicks,
} from "../lib/trendAxis";
import {
  MAX_TREND_SAMPLES,
  MAX_WINDOW_SECONDS,
  TREND_BACKFILL_MAX_POINTS,
  downsample,
  formatWindow,
  formatClock,
  timeSeriesPath,
  timeToX,
} from "../lib/trendTime";
import { nearestIndexByX } from "../lib/plotCursor";
import {
  fitSeries,
  NO_FIT_ID,
  pointsInLastWindow,
  heatUpRateFromFit,
  formatHeatUpRate,
  type FitPoint,
} from "../lib/curveFit";
import { useTrendBackfill } from "../hooks/useTrendBackfill";
import { useTrendViewport } from "../hooks/useTrendViewport";
import { useNonPassiveWheel } from "../hooks/useNonPassiveWheel";
import { TrendAxisControls } from "./TrendAxisControls";
import { TrendTimeControls } from "./TrendTimeControls";
import { TrendFitControls } from "./TrendFitControls";
import {
  TrendTimeAxis,
  TrendFitOverlay,
  TrendCrosshair,
  type CrosshairSeries,
} from "./TrendPlotOverlays";
import { ExportButton } from "./ExportButton";
import { SnapshotButton } from "./SnapshotButton";
import { CollapsibleSection } from "./CollapsibleSection";
import { EditableValue } from "./EditableValue";
import "./TemperatureControl.css";

// Rolling trend buffer: deep enough for the longest selectable window
// (up to 1 h); the plot slices to the chosen window by real time.
const TREND_MAX_POINTS = MAX_TREND_SAMPLES;
const DEFAULT_WINDOW_SECONDS = 3600; // 60 minutes

/** One temperature sample: epoch-ms timestamp + BOTH thermocouple readings (°C,
 * null when a channel has not reported) + the heater relay state at that instant.
 * Timestamping the buffer makes the window, axis, and fit slope accurate at any
 * poll rate; keeping both channels lets the trend plot K and R together (and spot
 * a dead sensor) even while only one is controlling. */
interface TempSample {
  t: number;
  /** Latest Type-K reading (°C), or null when the channel has not reported. */
  k: number | null;
  /** Latest Type-R reading (°C), or null when the channel has not reported. */
  r: number | null;
  /** Heater relay closed at this sample (for the heater-status band). */
  relayOn: boolean;
}

/** Pick the reading of one thermocouple channel from a sample (LOD: the trend
 * math never reaches into the sample shape directly). */
function tcSampleValue(sample: TempSample, tcType: TcType): number | null {
  return tcType === "K" ? sample.k : sample.r;
}

/**
 * Format a single live thermocouple reading for the selector readout.
 *
 * Pure so it can be unit-tested without rendering. Returns the value to one
 * decimal with a "°C" suffix, or an em-dash placeholder when the channel has
 * not reported (null/undefined) or is not a finite number (e.g. NaN from a
 * stuck/broken sensor).
 */
// Small pure helper co-located with the component it serves so it can be
// unit-tested; not a component, so it never participates in fast refresh.
// eslint-disable-next-line react-refresh/only-export-components
export function formatTcReadout(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${value.toFixed(1)} °C`;
}

/** The text a heat-up-rate readout box should show for the current selection. */
export interface HeatUpRateReadout {
  /** Ramp rate line, e.g. "+12.3 °C/min · +740 °C/hr" or "—" when no fit. */
  rate: string;
  /** Fit-quality line, e.g. "R² = 0.987", or "" when there is no fit. */
  r2: string;
  /** True when a linear fit was found over the windowed active-TC series. */
  hasFit: boolean;
}

/**
 * Build the heat-up-rate readout from the rolling buffer + operator choices.
 *
 * Pure so it can be unit-tested without rendering. It projects the ACTIVE
 * thermocouple's readings into {@link FitPoint}s (x = epoch ms, y = °C), keeps
 * only the last `fitWindowMin` minutes via {@link pointsInLastWindow}, fits with
 * the chosen method via {@link fitSeries}, then formats the ramp rate and R²
 * using the shared curveFit helpers (DRY — no rate/window math is duplicated
 * here). Null readings are dropped so a gap never poisons the regression.
 *
 * Returns a neutral placeholder (`rate: "—"`, empty `r2`, `hasFit: false`) when
 * no method is selected, there are too few points, or the fit is not linear.
 *
 * @param samples - the rolling trend buffer (ascending timestamps).
 * @param activeTcType - which channel is controlling ("K" | "R").
 * @param fitMethodId - selected curveFit method id (NO_FIT_ID for none).
 * @param fitWindowMin - regression look-back in MINUTES (<= 0 means "no window").
 * @throws TypeError if `fitWindowMin` is not a finite number.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function heatUpRateReadout(
  samples: TempSample[],
  activeTcType: TcType,
  fitMethodId: string,
  fitWindowMin: number,
): HeatUpRateReadout {
  if (typeof fitWindowMin !== "number" || !Number.isFinite(fitWindowMin)) {
    throw new TypeError("heatUpRateReadout: fitWindowMin must be a finite number");
  }
  const points: FitPoint[] = samples
    .map((sample) => ({ x: sample.t, y: tcSampleValue(sample, activeTcType) }))
    .filter((p): p is FitPoint => typeof p.y === "number" && Number.isFinite(p.y));
  const windowed = pointsInLastWindow(points, fitWindowMin * 60000);
  const fit = fitSeries(windowed, fitMethodId);
  if (!fit) {
    return { rate: "—", r2: "", hasFit: false };
  }
  // x is in ms (Date.now()), so msPerXUnit = 1.
  return {
    rate: formatHeatUpRate(heatUpRateFromFit(fit, 1)),
    r2: `R² = ${fit.r2.toFixed(3)}`,
    hasFit: true,
  };
}

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
 */

// Source domain types + labels live in lib/tempSource (shared with tests; keeps
// this file exporting only components for fast-refresh). Re-export the types so
// existing consumers of `TcType` from this module keep working.
export type { TcType, TcPath } from "../lib/tempSource";

export interface ThermocoupleChannel {
  tag: string;
  /** deg C at 0 % of the tag (4 mA for a 4-20 mA conditioner). Optional for
   *  back-compat; absent means 0 (TC-card channels are zero-based). */
  range_min_c?: number;
  full_scale_c: number;
  label: string;
}

export interface TemperatureConfig {
  type_k: ThermocoupleChannel;
  type_r: ThermocoupleChannel;
  analog_k: ThermocoupleChannel;
  analog_r: ThermocoupleChannel;
  active_tc_type: TcType;
  active_tc_path: TcPath;
  /** Derived (read-only) from the active channel — see backend computed fields. */
  temp_tag: string;
  temp_full_scale_c: number;
  active_tc_label: string;
  setpoint_min_c: number;
  setpoint_max_c: number;
  deadband_c: number;
  min_on_time_s: number;
  min_off_time_s: number;
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
  min_on_time_s: number;
  min_off_time_s: number;
  active_tc_type: TcType;
  active_tc_path: TcPath;
  active_tc_label: string;
  /** Operator's setpoint from the last session, recalled by the backend on
   * restart (null when none was ever persisted). Used to pre-fill the entry. */
  last_setpoint_c?: number | null;
  /** Latest type-K reading (deg C) on the ACTIVE path, regardless of which TC is
   * controlling, so the HMI can show/plot both channels of the selected path.
   * null/undefined before the first scan. */
  type_k_temp_c?: number | null;
  /** Latest type-R reading (deg C) on the ACTIVE path, regardless of which TC is
   * controlling. null/undefined before the first scan. */
  type_r_temp_c?: number | null;
  /** True while the deglitch filter is holding the control thermocouple's
   * last-good value through a live dropout — a hint the control sensor is
   * intermittently faulting (a sustained fault escalates to a TC_FAULT trip). */
  control_sensor_holding?: boolean;
  /** P1-04THM open-circuit fail direction. True = high-side (an open reads full
   * scale -> heater shuts off, fail-safe); false = low-side (an open reads cold). */
  burnout_high_side?: boolean;
}

/**
 * Decide the text to pre-fill the setpoint entry with on recall.
 *
 * Pure so it can be unit-tested without rendering. Returns the formatted
 * (one-decimal) string when a persisted last setpoint should be shown, or
 * `null` when nothing should change — i.e. the operator has already typed this
 * session (`operatorTouched`) or the recalled value is not a finite number.
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
  if (typeof operatorTouched !== "boolean") {
    throw new TypeError("operatorTouched must be a boolean");
  }
  if (operatorTouched) return null;
  if (typeof lastSetpointC !== "number" || !Number.isFinite(lastSetpointC)) {
    return null;
  }
  return lastSetpointC.toFixed(1);
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
  idle: "stopped · press Start",
  armed: "started · set a target",
  running: "heating to setpoint",
  tripped: "latched · acknowledge",
};

/** True when any trip name looks like a high-high temperature cutoff. */
function hasHighHighTrip(trips: string[]): boolean {
  return trips.some((t) => /(hh|high.?high|over.?temp)/i.test(t));
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
    // A Stop is the SAFE direction and must be immediate and UNCONDITIONAL —
    // never gate it behind a confirmation dialog. Kiosk / screen-shared browsers
    // can suppress native window.confirm() (it returns false with no prompt),
    // which would silently swallow the Stop while the heater keeps running.
    // Confirmation belongs on Start, never Stop.
    setBusy(true);
    try {
      await postPermissive(false);
      flash("Heater stopped");
    } catch (e) {
      flash(`Stop failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [postPermissive, flash]);

  const setActiveSource = useCallback(
    async (tcType: TcType, tcPath: TcPath) => {
      setBusy(true);
      try {
        const res = await fetchWithTimeout("/api/temperature/tc_type", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            active_tc_type: tcType,
            active_tc_path: tcPath,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        flash(`Heater now reading ${sourceLabel(tcType, tcPath)}`);
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
  const activeTcPath: TcPath = s?.active_tc_path ?? config.active_tc_path;
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
              onCommit={commitHhLimit}
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
            className="tc-permissive"
            onClick={startStop.command === "start" ? handleStart : handleStop}
            disabled={busy || startStop.disabled}
            style={{
              background:
                startStop.command === "stop"
                  ? "var(--color-error)"
                  : "var(--color-success)",
              color: "#04141b",
              borderColor: "transparent",
            }}
            title={
              busy
                ? "Applying — please wait…"
                : startStop.command === "start"
                  ? "Start the heater — arm and heat to the target"
                  : "Stop the heater — the relay opens immediately"
            }
          >
            <span className="dot" />
            {busy
              ? "APPLYING…"
              : startStop.command === "stop"
                ? "■ STOP"
                : "▶ START"}
          </button>
        </div>
      </div>

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

          <button
            className="tc-permissive tc-setpoint-startstop"
            onClick={startStop.command === "start" ? handleStart : handleStop}
            disabled={busy || startStop.disabled}
            style={{
              background:
                startStop.command === "stop"
                  ? "var(--color-error)"
                  : "var(--color-success)",
              color: "#04141b",
              borderColor: "transparent",
            }}
            title={
              busy
                ? "Applying — please wait…"
                : startStop.command === "start"
                  ? "Start the heater — arm and heat to the target"
                  : "Stop the heater — the relay opens immediately"
            }
          >
            <span className="dot" />
            {busy
              ? "APPLYING…"
              : startStop.command === "stop"
                ? "■ STOP"
                : "▶ START"}
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

      {/* ---- Thermocouple source selector (K/R x TC card / analog) ---- */}
      <CollapsibleSection
        className="tc-card"
        title="Thermocouple — heater temperature source"
      >
        <div
          role="group"
          aria-label="Thermocouple source"
          style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}
        >
          {TEMP_SOURCES.map((src) => {
            const ch =
              src.path === "analog"
                ? src.type === "K"
                  ? config.analog_k
                  : config.analog_r
                : src.type === "K"
                  ? config.type_k
                  : config.type_r;
            const isActive =
              activeTcType === src.type && activeTcPath === src.path;
            // The status publishes the ACTIVE path's K/R pair, so a live reading
            // is available only for the two buttons on the current path.
            const reading =
              activeTcPath === src.path
                ? src.type === "K"
                  ? s?.type_k_temp_c
                  : s?.type_r_temp_c
                : undefined;
            const label = sourceLabel(src.type, src.path);
            return (
              <button
                key={`${src.type}-${src.path}`}
                type="button"
                onClick={() =>
                  !isActive && setActiveSource(src.type, src.path)
                }
                disabled={busy || isActive}
                aria-pressed={isActive}
                title={`${ch.label} on ${ch.tag} — live ${formatTcReadout(reading)}`}
                style={{
                  flex: "1 1 0",
                  minWidth: "8rem",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "flex-start",
                  gap: "0.15rem",
                  padding: "0.5rem 0.75rem",
                  borderRadius: "6px",
                  cursor: isActive ? "default" : "pointer",
                  border: "2px solid",
                  borderColor: isActive ? "var(--accent-cyan)" : "var(--panel-border)",
                  background: isActive ? "var(--cell-hover-bg)" : "var(--input-bg)",
                  color: isActive ? "var(--accent-cyan)" : "var(--text-secondary)",
                }}
              >
                <span
                  style={{
                    display: "flex",
                    alignItems: "baseline",
                    justifyContent: "space-between",
                    gap: "0.5rem",
                    width: "100%",
                  }}
                >
                  <strong style={{ fontSize: "0.95rem" }}>
                    {label}
                    {isActive ? " ✓" : ""}
                  </strong>
                  <span
                    className="tc-tc-reading"
                    aria-label={`${label} live reading`}
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "1.05rem",
                      fontWeight: 700,
                      // Emphasize the active channel; dim the other so a dead/stuck
                      // sensor is still visible but reads as secondary.
                      color: isActive ? "var(--accent-cyan)" : "var(--text-primary)",
                      opacity: isActive ? 1 : 0.75,
                    }}
                  >
                    {formatTcReadout(reading)}
                  </span>
                </span>
                <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                  {ch.label} · {ch.tag}
                  {isActive ? " · controlling" : ""}
                </span>
              </button>
            );
          })}
        </div>
        <p
          style={{
            fontSize: "0.72rem",
            color: "var(--text-secondary)",
            margin: "0.5rem 0 0",
            lineHeight: 1.5,
          }}
        >
          The selected source drives <strong>all</strong> heater control —
          setpoint band, HH cutoff and trends. Each thermocouple (K/R) can be read
          off the P1-04THM card or through a 4-20 mA signal conditioner on the
          analog card; switching re-clamps the limits to the chosen channel's
          range. Live values are shown for the two sources on the active path.
        </p>

        {/* ---- Open-circuit (burnout) fail direction ---- */}
        <div
          style={{
            marginTop: "0.85rem",
            paddingTop: "0.75rem",
            borderTop: "1px solid var(--panel-border)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "0.5rem",
              flexWrap: "wrap",
            }}
          >
            <span style={{ fontSize: "0.78rem", fontWeight: 600 }}>
              On a broken thermocouple, read:
            </span>
            <div
              style={{
                display: "inline-flex",
                borderRadius: "6px",
                overflow: "hidden",
                border: "1px solid var(--panel-border)",
              }}
            >
              {(
                [
                  { high: true, label: "Hot (fail-safe)" },
                  { high: false, label: "Cold" },
                ] as const
              ).map(({ high, label }) => {
                const active = (s?.burnout_high_side ?? true) === high;
                return (
                  <button
                    key={label}
                    type="button"
                    disabled={busy || active}
                    onClick={() => setBurnoutMode(high)}
                    aria-pressed={active}
                    style={{
                      border: "none",
                      padding: "0.3rem 0.75rem",
                      fontSize: "0.75rem",
                      fontWeight: 600,
                      cursor: active ? "default" : "pointer",
                      background: active
                        ? high
                          ? "var(--accent-cyan)"
                          : "var(--accent-amber, #f59e0b)"
                        : "var(--input-bg)",
                      color: active ? "#0f172a" : "var(--text-secondary)",
                    }}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          </div>
          <p
            style={{
              fontSize: "0.7rem",
              color: "var(--text-muted)",
              margin: "0.45rem 0 0",
              lineHeight: 1.5,
            }}
          >
            Sets the P1-04THM open-circuit (burnout) direction. <strong>Hot</strong>{" "}
            makes an open sensor read full-scale so the heater shuts off
            (recommended). <strong>Cold</strong> makes it read 0&nbsp;°C. The deglitch
            filter rides out brief dropouts either way and trips on a sustained fault.
          </p>
        </div>
      </CollapsibleSection>

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
      <details className="tc-card tc-config">
        <summary>Calibration &amp; limits</summary>

        <p className="tc-hint" style={{ marginTop: "0.8rem" }}>
          The thermocouple module reports temperature directly. These limits set
          the safe operating envelope — the server clamps setpoints to the band and
          latches the heater OFF at the high-high cutoff.
        </p>

        <div className="tc-config-grid">
          <ConfigField
            label="Deadband (°C, ± half-band)"
            value={configDraft.deadband_c}
            onChange={(v) => setConfigDraft({ ...configDraft, deadband_c: v })}
          />
          <ConfigField
            label="Min ON time (s)"
            value={configDraft.min_on_time_s}
            onChange={(v) => setConfigDraft({ ...configDraft, min_on_time_s: v })}
          />
          <ConfigField
            label="Min OFF time (s)"
            value={configDraft.min_off_time_s}
            onChange={(v) => setConfigDraft({ ...configDraft, min_off_time_s: v })}
          />
          <ConfigField
            label="High-high cutoff (°C)"
            value={configDraft.hh_limit_c}
            onChange={(v) => setConfigDraft({ ...configDraft, hh_limit_c: v })}
          />
          <ConfigField
            label="Max setpoint (°C)"
            value={configDraft.setpoint_max_c}
            onChange={(v) => setConfigDraft({ ...configDraft, setpoint_max_c: v })}
          />
          <ConfigField
            label="Type-K tag"
            value={configDraft.type_k.tag}
            stringMode
            onChange={(v) =>
              setConfigDraft({
                ...configDraft,
                type_k: { ...configDraft.type_k, tag: v as unknown as string },
              })
            }
          />
          <ConfigField
            label="Type-K full scale (°C)"
            value={configDraft.type_k.full_scale_c}
            onChange={(v) =>
              setConfigDraft({
                ...configDraft,
                type_k: { ...configDraft.type_k, full_scale_c: v },
              })
            }
          />
          <ConfigField
            label="Type-R tag"
            value={configDraft.type_r.tag}
            stringMode
            onChange={(v) =>
              setConfigDraft({
                ...configDraft,
                type_r: { ...configDraft.type_r, tag: v as unknown as string },
              })
            }
          />
          <ConfigField
            label="Type-R full scale (°C)"
            value={configDraft.type_r.full_scale_c}
            onChange={(v) =>
              setConfigDraft({
                ...configDraft,
                type_r: { ...configDraft.type_r, full_scale_c: v },
              })
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
 * Memoized so this always-mounted panel only re-renders when `liveStatus`
 * (ref-stable from useTelemetryStream) changes, not on every ~10 Hz App frame.
 */
export const TemperatureControl = React.memo(TemperatureControlImpl);

/**
 * Compact single-trace temperature trend for the controller screen: measured
 * temperature plotted against a 0–full-scale °C Y axis, with the setpoint and
 * high-high cutoff drawn as horizontal reference lines.
 *
 * Self-contained SVG (no chart lib): the parent accumulates a rolling sample
 * buffer from the live status and passes it in.
 */
const TREND_W = 600;
const TREND_H = 188;
const TREND_PAD_L = 40;
const TREND_PAD_R = 10;
const TREND_PAD_T = 10;
const TREND_PAD_B = 26; // room for the X-axis time labels
const TREND_PLOT_H = TREND_H - TREND_PAD_T - TREND_PAD_B;

const K_COLOR = "var(--color-error)"; // Type-K trace (also the historian backfill)
const R_COLOR = "var(--accent-magenta)"; // Type-R trace
const SETPOINT_COLOR = "var(--accent-cyan)";
const HH_COLOR = "var(--color-warning)";
const RELAY_BAND_COLOR = "var(--color-error)";

/** Default look-back (minutes) for the heat-up-rate regression. */
const DEFAULT_FIT_WINDOW_MIN = 5;

/**
 * Split a series into contiguous runs of finite (non-null) points, so a trace is
 * drawn as separate path segments across gaps instead of a straight line through
 * a dead-sensor hole. Pure and generic over the timed sample shape.
 */
function timedSegments(
  points: { t: number; v: number | null }[],
): { t: number; v: number }[][] {
  const runs: { t: number; v: number }[][] = [];
  let run: { t: number; v: number }[] = [];
  for (const p of points) {
    if (typeof p.v === "number" && Number.isFinite(p.v)) {
      run.push({ t: p.t, v: p.v });
    } else if (run.length) {
      runs.push(run);
      run = [];
    }
  }
  if (run.length) runs.push(run);
  return runs;
}

/**
 * Contiguous [start, end] time spans (epoch ms) where `on(sample)` holds — used
 * to draw the heater-status band as shaded rectangles. Each span extends to the
 * next sample's timestamp so a single-scan pulse is still visible.
 */
function activeSpans(
  samples: TempSample[],
  on: (s: TempSample) => boolean,
): { start: number; end: number }[] {
  const spans: { start: number; end: number }[] = [];
  let start: number | null = null;
  for (let i = 0; i < samples.length; i++) {
    if (on(samples[i])) {
      if (start === null) start = samples[i].t;
    } else if (start !== null) {
      spans.push({ start, end: samples[i].t });
      start = null;
    }
  }
  if (start !== null) {
    const lastT = samples[samples.length - 1].t;
    spans.push({ start, end: lastT });
  }
  return spans;
}

/**
 * Map a plot-area pixel (0…plotW) to a time within the resolved [t0,t1] window.
 * Pure so the wheel / drag-zoom mapping is unit-testable without a DOM, and the
 * inverse of the SVG's X placement. Linear because the trend SVG uses
 * preserveAspectRatio="none" (x scales with the rendered width). Degenerates to
 * t0 for a zero-width plot (DbC — callers get a valid time, never NaN).
 */
// Pure helper, not a component, so it never participates in fast refresh.
// eslint-disable-next-line react-refresh/only-export-components
export function plotPxToTime(
  px: number,
  plotW: number,
  t0: number,
  t1: number,
): number {
  if (plotW <= 0) return t0;
  return t0 + (px / plotW) * (t1 - t0);
}

interface TrendProps {
  samples: TempSample[];
  /** Tag index backing the temperature (for historian backfill). */
  tagId: number;
  fullScale: number;
  setpoint: number;
  hhLimit: number;
  /** Which thermocouple is controlling — emphasized and used for the fit. */
  activeTcType: TcType;
}

const TempTrend: React.FC<TrendProps> = ({
  samples,
  tagId,
  fullScale,
  setpoint,
  hhLimit,
  activeTcType,
}) => {
  const [axis, setAxis] = useState<AxisRange>(defaultAxisRange(0, fullScale));
  const [fitMethodId, setFitMethodId] = useState<string>(NO_FIT_ID);
  const [fitWindowMin, setFitWindowMin] = useState<number>(DEFAULT_FIT_WINDOW_MIN);

  // Signal picker: default every trace + the heater-status band visible.
  const [showK, setShowK] = useState(true);
  const [showR, setShowR] = useState(true);
  const [showRelay, setShowRelay] = useState(true);

  // Pan / zoom / pause come from the shared, tested viewport model (DRY). The
  // domain is epoch ms; the visible window (seconds) is DERIVED from its span so
  // it still drives the historian backfill depth and the "last …" label.
  const view = useTrendViewport({
    defaultSpan: DEFAULT_WINDOW_SECONDS * 1000,
    minSpan: 1000,
    maxSpan: MAX_WINDOW_SECONDS * 1000,
  });
  const windowSeconds = view.viewport.span / 1000;

  // While paused, plot a frozen snapshot of the samples captured at pause time
  // so incoming live data never slides the view out from under the operator.
  const [frozen, setFrozen] = useState<TempSample[]>([]);
  // Plot pixel X of the hover crosshair, or null when not hovering.
  const [hoverPx, setHoverPx] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Backfill from the historian so widening the window immediately shows past
  // data (stored tag is a 0–100 %, so scale it to °C). The historian carries the
  // ACTIVE channel's tag only, so merge it into that channel of the buffer
  // (anything older than the live buffer) ahead of the live samples.
  const backfill = useTrendBackfill(
    tagId,
    windowSeconds,
    fullScale / 100,
    TREND_BACKFILL_MAX_POINTS,
  );
  // Freeze swaps the live prop for the snapshot captured at pause time; either
  // way backfill merges in ahead of it, so windowing/backfill stay unchanged.
  const source = view.paused ? frozen : samples;
  const liveStart = source.length ? source[0].t : Infinity;
  const older = backfill.filter((b) => b.t < liveStart);
  const series: TempSample[] = older.length
    ? [
        ...older.map((b) => ({
          t: b.t,
          k: activeTcType === "K" ? b.v : null,
          r: activeTcType === "R" ? b.v : null,
          relayOn: false,
        })),
        ...source,
      ]
    : source;

  // The viewport decides which slice of real wall-clock time is on screen. Its
  // bounds are the full data extent; resolve() clamps the visible [t0,t1] for
  // the current span (zoom) and offset (pan), then we window the samples to it.
  const bounds = {
    min: series[0]?.t ?? 0,
    max: series[series.length - 1]?.t ?? Date.now(),
  };
  const { start: t0, end: t1 } = view.resolve(bounds);
  const windowed = series.filter((s) => s.t >= t0 && s.t <= t1);
  const plotted = downsample(windowed);

  // Plot-pixel ↔ time mapping for wheel / drag-zoom. preserveAspectRatio="none"
  // means x scales with the rendered width, so convert client px → viewBox px →
  // plot px, and (via the pure helper) plot px → time within the window.
  const plotW = TREND_W - TREND_PAD_L - TREND_PAD_R;
  const plotPx = (clientX: number): number => {
    const el = svgRef.current;
    if (!el) return 0;
    const r = el.getBoundingClientRect();
    if (!r.width) return 0;
    const x = ((clientX - r.left) / r.width) * TREND_W;
    return Math.max(0, Math.min(plotW, x - TREND_PAD_L));
  };
  const pxToUnit = (px: number): number => plotPxToTime(px, plotW, t0, t1);

  // Wheel zoom about the cursor, attached as a NON-passive native listener (see
  // useNonPassiveWheel) so preventDefault suppresses page scroll — React's
  // onWheel is passive under React 18.
  const handleWheel = (e: WheelEvent): void => {
    e.preventDefault();
    view.zoomBy(e.deltaY > 0 ? 1.15 : 0.87, pxToUnit(plotPx(e.clientX)), bounds);
  };
  useNonPassiveWheel(svgRef, handleWheel);

  // Resolve the Y range against BOTH channels so neither trace clips.
  const plottedC = plotted
    .flatMap((s) => [s.k, s.r])
    .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const { min, max } = resolveRange(axis, plottedC, { min: 0, max: fullScale });

  // Latest reading of each channel (for the legend), and the active reading.
  const lastSample = windowed.length ? windowed[windowed.length - 1] : undefined;
  const lastK = lastSample?.k ?? null;
  const lastR = lastSample?.r ?? null;

  const refY = (value: number): number => {
    const frac = Math.max(0, Math.min(1, (value - min) / (max - min)));
    return TREND_PAD_T + (1 - frac) * TREND_PLOT_H;
  };

  // Position traces by real timestamp (not array index) so they are time-accurate
  // even when sparse historian backfill is merged with dense live samples. The X
  // axis spans the viewport's resolved [t0,t1] window, so zoom / pan / window
  // changes rescale the axis immediately.
  const geom = {
    t0,
    t1,
    min,
    max,
    x0: TREND_PAD_L,
    x1: TREND_W - TREND_PAD_R,
    yTop: TREND_PAD_T,
    plotH: TREND_PLOT_H,
  } as const;

  // Build each channel as gap-aware path segments (a null skips the line).
  const kSegments = timedSegments(plotted.map((s) => ({ t: s.t, v: s.k })));
  const rSegments = timedSegments(plotted.map((s) => ({ t: s.t, v: s.r })));
  const kPaths = kSegments.map((seg) => timeSeriesPath(seg, geom));
  const rPaths = rSegments.map((seg) => timeSeriesPath(seg, geom));

  // Heater-status band: shaded spans where the relay was closed.
  const relaySpans = activeSpans(plotted, (s) => s.relayOn);

  // Resolve the hover crosshair: snap to the nearest sample by time and read
  // each visible channel's °C there, placing markers with the same refY mapping
  // the traces use (DRY). Channels that are null at that sample are skipped.
  const hover: { px: number; series: CrosshairSeries[]; xLabel: string } | null =
    (() => {
      if (hoverPx === null || plotted.length < 2) return null;
      const idx = nearestIndexByX(
        plotted.map((s) => s.t),
        pxToUnit(hoverPx),
      );
      if (idx === null) return null;
      const s = plotted[idx];
      const series: CrosshairSeries[] = [];
      if (showK && typeof s.k === "number" && Number.isFinite(s.k)) {
        series.push({ label: "K", color: K_COLOR, text: `${s.k.toFixed(1)} °C`, py: refY(s.k) });
      }
      if (showR && typeof s.r === "number" && Number.isFinite(s.r)) {
        series.push({ label: "R", color: R_COLOR, text: `${s.r.toFixed(1)} °C`, py: refY(s.r) });
      }
      if (series.length === 0) return null;
      return {
        px: timeToX(s.t, t0, t1, TREND_PAD_L, TREND_W - TREND_PAD_R),
        series,
        xLabel: formatClock(s.t),
      };
    })();

  // Curve fit runs over the ACTIVE thermocouple series, restricted to the last
  // `fitWindowMin` minutes (the operator-chosen regression window). x is epoch
  // ms so the shared heat-up-rate helpers (called with msPerXUnit = 1) convert
  // the slope to °C/min and °C/hr — no rate math is duplicated here (DRY).
  const activePoints: FitPoint[] = plotted
    .map((s) => ({ x: s.t, y: tcSampleValue(s, activeTcType) }))
    .filter((p): p is FitPoint => typeof p.y === "number" && Number.isFinite(p.y));
  const windowedFitPoints = pointsInLastWindow(activePoints, fitWindowMin * 60000);
  const fit = fitSeries(windowedFitPoints, fitMethodId);
  const rateReadout = heatUpRateReadout(
    plotted,
    activeTcType,
    fitMethodId,
    fitWindowMin,
  );
  // The fit's x IS the epoch-ms timestamp, so both the X position (p.t) and the
  // value fed to fit.predict (p.x) are that same timestamp.
  const fitOverlayPoints = windowedFitPoints.map((p) => ({ t: p.x, x: p.x }));

  const activeColor = activeTcType === "K" ? K_COLOR : R_COLOR;

  // CSV export of the plotted K/R samples (empty cell for a channel gap; relay
  // as 0/1). Hidden (undefined) until the window holds at least one point.
  const snapshotCsv = plotted.length
    ? {
        headers: ["t_ms", "type_k_c", "type_r_c", "relay_on"],
        rows: plotted.map((s) => [
          s.t,
          s.k ?? "",
          s.r ?? "",
          s.relayOn ? 1 : 0,
        ]),
      }
    : undefined;

  return (
    <div className="tc-trend">
      <div className="tc-trend-legend">
        <label
          className="tc-trend-key tc-trend-toggle"
          title="Show / hide the Type-K trace"
        >
          <input
            type="checkbox"
            checked={showK}
            onChange={(e) => setShowK(e.target.checked)}
            aria-label="Show Type K trace"
          />
          <span
            className="swatch"
            style={{
              background: K_COLOR,
              opacity: activeTcType === "K" ? 1 : 0.55,
            }}
          />
          Type K{activeTcType === "K" ? " (active)" : ""}
          <strong>{formatTcReadout(lastK)}</strong>
        </label>
        <label
          className="tc-trend-key tc-trend-toggle"
          title="Show / hide the Type-R trace"
        >
          <input
            type="checkbox"
            checked={showR}
            onChange={(e) => setShowR(e.target.checked)}
            aria-label="Show Type R trace"
          />
          <span
            className="swatch"
            style={{
              background: R_COLOR,
              opacity: activeTcType === "R" ? 1 : 0.55,
            }}
          />
          Type R{activeTcType === "R" ? " (active)" : ""}
          <strong>{formatTcReadout(lastR)}</strong>
        </label>
        <label
          className="tc-trend-key tc-trend-toggle"
          title="Shade the periods when the heater relay was ON"
        >
          <input
            type="checkbox"
            checked={showRelay}
            onChange={(e) => setShowRelay(e.target.checked)}
            aria-label="Show heater ON band"
          />
          <span
            className="swatch"
            style={{ background: RELAY_BAND_COLOR, opacity: 0.25 }}
          />
          Heater ON
        </label>
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
        {!view.live && (
          <span
            className="tc-trend-key"
            style={{
              color: "var(--color-warning)",
              border: "1px solid var(--color-warning)",
              borderRadius: "3px",
              padding: "0.05rem 0.3rem",
              fontSize: "0.66rem",
              fontWeight: 600,
              letterSpacing: "0.5px",
              textTransform: "uppercase",
            }}
            title={
              view.paused
                ? "Plot frozen — live updates paused"
                : "Scrolled back in time — not following the live edge"
            }
          >
            {view.paused ? "Frozen" : "Panned"}
          </span>
        )}
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
        <div
          style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}
          role="group"
          aria-label="Trend pan, zoom and pause controls"
        >
          <button
            type="button"
            className="btn"
            style={{ padding: "0.2rem 0.45rem", fontSize: "0.72rem" }}
            onClick={() => {
              if (!view.paused) setFrozen(samples);
              view.togglePause();
            }}
            title={view.paused ? "Resume live streaming" : "Pause / freeze the plot"}
            aria-label={view.paused ? "Resume live" : "Pause"}
          >
            {view.paused ? "Live" : "Pause"}
          </button>
          <button
            type="button"
            className="btn"
            style={{ padding: "0.2rem 0.45rem", fontSize: "0.72rem" }}
            onClick={() => view.panBy((t1 - t0) * 0.3, bounds)}
            title="Scroll back in time"
            aria-label="Pan back in time"
          >
            ◀
          </button>
          <button
            type="button"
            className="btn"
            style={{ padding: "0.2rem 0.45rem", fontSize: "0.72rem" }}
            onClick={() => view.panBy(-(t1 - t0) * 0.3, bounds)}
            title="Scroll forward in time"
            aria-label="Pan forward in time"
          >
            ▶
          </button>
          <button
            type="button"
            className="btn"
            style={{ padding: "0.2rem 0.45rem", fontSize: "0.72rem" }}
            onClick={() => view.zoomBy(0.7, (t0 + t1) / 2, bounds)}
            title="Zoom in (narrow the time window)"
            aria-label="Zoom in"
          >
            +
          </button>
          <button
            type="button"
            className="btn"
            style={{ padding: "0.2rem 0.45rem", fontSize: "0.72rem" }}
            onClick={() => view.zoomBy(1.43, (t0 + t1) / 2, bounds)}
            title="Zoom out (widen the time window)"
            aria-label="Zoom out"
          >
            −
          </button>
          <button
            type="button"
            className="btn"
            style={{ padding: "0.2rem 0.45rem", fontSize: "0.72rem" }}
            onClick={() => view.reset()}
            title="Reset zoom / pan and resume live"
            aria-label="Reset view"
          >
            Reset
          </button>
        </div>
        <TrendTimeControls
          value={windowSeconds}
          onChange={(seconds) => view.setSpan(seconds * 1000)}
        />
        <TrendAxisControls value={axis} onChange={setAxis} unit="°C" />
        <TrendFitControls value={fitMethodId} onChange={setFitMethodId} />
        <label
          className="tc-trend-fitwindow"
          title="How many recent minutes the linear regression covers"
        >
          <span>Fit window (min)</span>
          <input
            type="number"
            min={0.1}
            step={0.5}
            value={fitWindowMin}
            onChange={(e) => {
              const v = Number.parseFloat(e.target.value);
              setFitWindowMin(Number.isFinite(v) && v > 0 ? v : DEFAULT_FIT_WINDOW_MIN);
            }}
            aria-label="Fit window in minutes"
          />
        </label>
        <SnapshotButton
          targetRef={svgRef}
          filename="temperature_trend"
          csv={snapshotCsv}
          label="Export temperature trend snapshot"
        />
      </div>

      {/* Heat-up-rate readout box: ramp rate (°/min · °/hr) + fit quality. */}
      <div
        className={`tc-rate-readout ${rateReadout.hasFit ? "has-fit" : ""}`}
        aria-label="Heat-up rate readout"
      >
        <span className="tc-rate-label">
          Heat-up rate (Type {activeTcType}, last {fitWindowMin} min)
        </span>
        <span className="tc-rate-value">{rateReadout.rate}</span>
        <span className="tc-rate-r2">
          {rateReadout.hasFit
            ? rateReadout.r2
            : "select a fit method to read the ramp rate"}
        </span>
      </div>

      <svg
        ref={svgRef}
        className="tc-trend-svg"
        viewBox={`0 0 ${TREND_W} ${TREND_H}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="Temperature trend"
        style={{
          cursor: view.selectionPx ? "ew-resize" : "crosshair",
          touchAction: "none",
        }}
        onPointerDown={(e) => {
          setHoverPx(null); // a drag-zoom gesture supersedes the hover readout
          view.startSelect(plotPx(e.clientX));
        }}
        onPointerMove={(e) => {
          if (view.selectionPx) {
            view.moveSelect(plotPx(e.clientX));
            return;
          }
          setHoverPx(plotPx(e.clientX)); // hover: value readout at the cursor
        }}
        onPointerUp={() => view.endSelect(bounds, pxToUnit)}
        onPointerLeave={() => {
          view.cancelSelect();
          setHoverPx(null);
        }}
      >
        {/* heater-ON status band behind everything else */}
        {showRelay &&
          relaySpans.map((span, i) => {
            const xa = timeToX(span.start, t0, t1, TREND_PAD_L, TREND_W - TREND_PAD_R);
            const xb = timeToX(span.end, t0, t1, TREND_PAD_L, TREND_W - TREND_PAD_R);
            return (
              <rect
                key={i}
                x={xa}
                y={TREND_PAD_T}
                width={Math.max(0.5, xb - xa)}
                height={TREND_PLOT_H}
                className="tc-trend-relay-band"
                fill={RELAY_BAND_COLOR}
              />
            );
          })}

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

        <TrendTimeAxis
          x0={TREND_PAD_L}
          x1={TREND_W - TREND_PAD_R}
          yBottom={TREND_PAD_T + TREND_PLOT_H}
          spanSeconds={windowSeconds}
        />

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
          <>
            {/* Non-active channel dimmer; active channel bold/solid. Draw the
                inactive one first so the active trace sits on top. */}
            {showR &&
              rPaths.map((d, i) => (
                <path
                  key={`r-${i}`}
                  d={d}
                  className="tc-trend-line"
                  stroke={R_COLOR}
                  opacity={activeTcType === "R" ? 1 : 0.5}
                  strokeWidth={activeTcType === "R" ? 2.4 : 1.4}
                />
              ))}
            {showK &&
              kPaths.map((d, i) => (
                <path
                  key={`k-${i}`}
                  d={d}
                  className="tc-trend-line"
                  stroke={K_COLOR}
                  opacity={activeTcType === "K" ? 1 : 0.5}
                  strokeWidth={activeTcType === "K" ? 2.4 : 1.4}
                />
              ))}
            {fit && (
              <TrendFitOverlay
                fit={fit}
                points={fitOverlayPoints}
                t0={t0}
                t1={t1}
                x0={TREND_PAD_L}
                x1={TREND_W - TREND_PAD_R}
                yScale={refY}
                color={activeColor}
              />
            )}
          </>
        )}

        {/* Hover crosshair + value tooltip (shared overlay). */}
        {hover && (
          <TrendCrosshair
            px={hover.px}
            yTop={TREND_PAD_T}
            yBottom={TREND_PAD_T + TREND_PLOT_H}
            plotLeft={TREND_PAD_L}
            plotRight={TREND_W - TREND_PAD_R}
            series={hover.series}
            xLabel={hover.xLabel}
          />
        )}

        {/* Drag-to-zoom selection: translucent band over the plot area. */}
        {view.selectionPx && (
          <rect
            x={TREND_PAD_L + Math.min(view.selectionPx.fromPx, view.selectionPx.toPx)}
            y={TREND_PAD_T}
            width={Math.abs(view.selectionPx.toPx - view.selectionPx.fromPx)}
            height={TREND_PLOT_H}
            fill="var(--accent-cyan)"
            fillOpacity={0.18}
            stroke="var(--accent-cyan)"
            strokeOpacity={0.6}
          />
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
