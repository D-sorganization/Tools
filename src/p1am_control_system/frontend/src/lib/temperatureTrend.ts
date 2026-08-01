import {
  fitSeries,
  pointsInLastWindow,
  heatUpRateFromFit,
  formatHeatUpRate,
  type FitPoint,
} from "./curveFit";
import type { TcType } from "../types";

/**
 * Pure sample/readout math behind the heater temperature trend.
 *
 * Split out of TemperatureControl.tsx (1975 lines — 475 over the repo's source
 * budget, so the file could not be edited at all without failing the
 * fleet-fast-guardrails hook). Nothing here renders, so the trend's windowing,
 * gap handling and ramp-rate arithmetic are unit-testable on their own and the
 * component files stay presentation-only.
 */

/** One temperature sample: epoch-ms timestamp + BOTH thermocouple readings (°C,
 * null when a channel has not reported) + the heater relay state at that instant.
 * Timestamping the buffer makes the window, axis, and fit slope accurate at any
 * poll rate; keeping both channels lets the trend plot K and R together (and spot
 * a dead sensor) even while only one is controlling. */
export interface TempSample {
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
export function tcSampleValue(sample: TempSample, tcType: TcType): number | null {
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
 * Split a series into contiguous runs of finite (non-null) points, so a trace is
 * drawn as separate path segments across gaps instead of a straight line through
 * a dead-sensor hole. Pure and generic over the timed sample shape.
 */
export function timedSegments(
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
export function activeSpans(
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
export function plotPxToTime(
  px: number,
  plotW: number,
  t0: number,
  t1: number,
): number {
  if (plotW <= 0) return t0;
  return t0 + (px / plotW) * (t1 - t0);
}
