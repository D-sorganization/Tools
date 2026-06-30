import React, { useState } from "react";
import {
  type AxisRange,
  defaultAxisRange,
  resolveRange,
  axisTicks,
} from "../lib/trendAxis";
import {
  downsample,
  formatWindow,
  fixedWindowRange,
  windowStartIndex,
  timeSeriesPath,
} from "../lib/trendTime";
import { TrendAxisControls } from "./TrendAxisControls";
import { TrendTimeControls } from "./TrendTimeControls";
import { TrendTimeAxis } from "./TrendPlotOverlays";
import { ExportButton } from "./ExportButton";

/**
 * Signal Diagnostics plot — the raw, unscaled 0-5 V of every analog channel on
 * the P1-4ADL2DAL card (4 analog inputs + 2 analog outputs), straight from the
 * firmware diagnostic tags (TAG_20..25). No process scaling / calibration is
 * applied, so an operator can compare each trace against a multimeter at the
 * terminal and troubleshoot the monitor card independently of the engineering
 * (A / V) readouts on the Power Supply tab.
 *
 * Both axes are adjustable: the Y axis (Auto-fit or manual 0-5 V via
 * TrendAxisControls) and the time window (TrendTimeControls). Long windows are
 * downsampled before drawing so the SVG stays light on the Pi.
 *
 * Self-contained SVG (no chart lib): the parent passes the rolling per-scan tag
 * history; this slices the last `windowSeconds` of tag columns 20..25.
 */

export interface DiagChannel {
  tag: number; // tag index in the history rows
  label: string;
  color: string;
}

const CHANNELS: DiagChannel[] = [
  { tag: 20, label: "AI0 — current monitor", color: "var(--color-success)" },
  { tag: 21, label: "AI1 — voltage monitor", color: "var(--accent-purple)" },
  { tag: 22, label: "AI2 — spare", color: "var(--accent-cyan)" },
  { tag: 23, label: "AI3 — spare", color: "var(--accent-magenta)" },
  { tag: 24, label: "AO0 — current cmd", color: "var(--color-warning)" },
  { tag: 25, label: "AO1 — aux cmd", color: "var(--text-secondary)" },
];

const FULL_SCALE_V = 5;
const W = 600;
const H = 232;
const PAD_L = 34;
const PAD_R = 10;
const PAD_T = 10;
const PAD_B = 28; // room for the X-axis time labels
const PLOT_H = H - PAD_T - PAD_B;

interface Props {
  /** Rolling per-scan tag history: each row is the full tag array for a scan. */
  history: number[][];
  /** Epoch-ms timestamp of each history row (same length as `history`). */
  historyTimes: number[];
}

export const SignalDiagnostics: React.FC<Props> = ({ history, historyTimes }) => {
  const [axis, setAxis] = useState<AxisRange>(
    defaultAxisRange(0, FULL_SCALE_V),
  );
  const [windowSeconds, setWindowSeconds] = useState<number>(3600); // 60 minutes

  // Window by real wall-clock time so the span is correct regardless of the
  // actual poll rate (the Pi runs below the nominal 10 Hz under load).
  const start = windowStartIndex(historyTimes, windowSeconds);
  const windowRows = history.slice(start);
  const windowTimes = historyTimes.slice(start);
  // Downsample the rows (and their timestamps in lockstep) once, then read each
  // channel — far cheaper than downsampling six series over a long window. Each
  // point carries its timestamp so the trace is positioned by time, not index.
  const downRows = downsample(windowRows);
  const downTimes = downsample(windowTimes);
  const series = CHANNELS.map((ch) => ({
    ...ch,
    points: downRows.map((row, i) => ({ t: downTimes[i], v: row[ch.tag] ?? 0 })),
    last: windowRows.length
      ? (windowRows[windowRows.length - 1][ch.tag] ?? 0)
      : null,
  }));
  const { min, max } = resolveRange(
    axis,
    series.flatMap((s) => s.points.map((p) => p.v)),
    { min: 0, max: FULL_SCALE_V },
  );
  const latestMs = downTimes.length ? downTimes[downTimes.length - 1] : Date.now();
  const { t0, t1 } = fixedWindowRange(latestMs, windowSeconds);
  const geom = {
    t0,
    t1,
    min,
    max,
    x0: PAD_L,
    x1: W - PAD_R,
    yTop: PAD_T,
    plotH: PLOT_H,
  };

  return (
    <div className="ps-card">
      <div className="ps-card-title">Signal Diagnostics — raw 0–5 V (unscaled)</div>
      <p style={{ fontSize: "0.72rem", color: "var(--text-secondary)", margin: "0 0 0.6rem", lineHeight: 1.5 }}>
        The actual 0–5 V on every analog channel of the P1-4ADL2DAL card, with
        <strong> no scaling or calibration</strong> — for troubleshooting the
        supply's monitor card. Inputs (AI) show the voltage at the terminal;
        outputs (AO) show the commanded level as 0–5 V. Compare against a
        multimeter at the terminal.
      </p>

      <div className="ps-trend-legend">
        {series.map((s) => (
          <span className="ps-trend-key" key={s.tag}>
            <span className="swatch" style={{ background: s.color }} />
            {s.label}
            <strong>{s.last != null ? `${s.last.toFixed(3)} V` : "—"}</strong>
          </span>
        ))}
        <span className="ps-trend-window">last {formatWindow(windowSeconds)}</span>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          margin: "0.2rem 0 0.4rem",
          flexWrap: "wrap",
        }}
      >
        <TrendTimeControls value={windowSeconds} onChange={setWindowSeconds} />
        <TrendAxisControls value={axis} onChange={setAxis} unit="V" />
        <ExportButton tags={CHANNELS.map((c) => c.tag)} label="Export" />
      </div>

      <svg
        className="ps-trend-svg"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="Raw 0-5 V analog signal diagnostics"
      >
        {axisTicks(min, max, 5).map((volt, i) => {
          const frac = (volt - min) / (max - min);
          const y = PAD_T + (1 - frac) * PLOT_H;
          return (
            <g key={i}>
              <line x1={PAD_L} y1={y} x2={W - PAD_R} y2={y} className="ps-trend-grid" />
              <text x={PAD_L - 5} y={y + 3} className="ps-trend-axis" textAnchor="end">
                {volt.toFixed(volt >= 10 || Number.isInteger(volt) ? 0 : 1)}
              </text>
            </g>
          );
        })}

        <TrendTimeAxis
          x0={PAD_L}
          x1={W - PAD_R}
          yBottom={PAD_T + PLOT_H}
          spanSeconds={windowSeconds}
        />

        {windowRows.length < 2 ? (
          <text x={W / 2} y={H / 2} className="ps-trend-empty" textAnchor="middle">
            waiting for live data…
          </text>
        ) : (
          series.map((s) => (
            <path
              key={s.tag}
              d={timeSeriesPath(s.points, geom)}
              className="ps-trend-line"
              stroke={s.color}
            />
          ))
        )}
      </svg>
    </div>
  );
};
