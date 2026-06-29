import React, { useState } from "react";
import {
  type AxisRange,
  defaultAxisRange,
  resolveRange,
  axisTicks,
} from "../lib/trendAxis";
import { TrendAxisControls } from "./TrendAxisControls";

/**
 * Signal Diagnostics plot — the raw, unscaled 0-5 V of every analog channel on
 * the P1-4ADL2DAL card (4 analog inputs + 2 analog outputs), straight from the
 * firmware diagnostic tags (TAG_20..25). No process scaling / calibration is
 * applied, so an operator can compare each trace against a multimeter at the
 * terminal and troubleshoot the monitor card independently of the engineering
 * (A / V) readouts on the Power Supply tab.
 *
 * The Y axis defaults to 0-5 V but is rescalable (Auto-fit or manual min/max)
 * via the shared TrendAxisControls so a small signal can be zoomed in.
 *
 * Self-contained SVG (no chart lib): the parent passes the rolling per-scan tag
 * history; this slices the last `windowSamples` of tag columns 20..25.
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
const H = 220;
const PAD_L = 34;
const PAD_R = 10;
const PAD_T = 10;
const PAD_B = 18;
const PLOT_W = W - PAD_L - PAD_R;
const PLOT_H = H - PAD_T - PAD_B;

function buildPath(values: number[], min: number, max: number): string {
  if (values.length < 2 || max <= min) return "";
  const n = values.length;
  return values
    .map((val, idx) => {
      const x = PAD_L + (idx / (n - 1)) * PLOT_W;
      const frac = Math.max(0, Math.min(1, (val - min) / (max - min)));
      const y = PAD_T + (1 - frac) * PLOT_H;
      return `${idx === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

interface Props {
  /** Rolling per-scan tag history: each row is the full tag array for a scan. */
  history: number[][];
  /** How many trailing samples to plot (≈ samples × 0.1 s at 10 Hz). */
  windowSamples?: number;
}

export const SignalDiagnostics: React.FC<Props> = ({
  history,
  windowSamples = 300,
}) => {
  const [axis, setAxis] = useState<AxisRange>(
    defaultAxisRange(0, FULL_SCALE_V),
  );
  const window = history.slice(-windowSamples);
  const series = CHANNELS.map((ch) => ({
    ...ch,
    values: window.map((row) => row[ch.tag] ?? 0),
    last: window.length ? (window[window.length - 1][ch.tag] ?? 0) : null,
  }));
  const seconds = Math.round((windowSamples * 0.1) / 1) || 30;
  const { min, max } = resolveRange(
    axis,
    series.flatMap((s) => s.values),
    { min: 0, max: FULL_SCALE_V },
  );

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
        <span className="ps-trend-window">last {seconds}s</span>
      </div>

      <div style={{ margin: "0.2rem 0 0.4rem" }}>
        <TrendAxisControls value={axis} onChange={setAxis} unit="V" />
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

        {window.length < 2 ? (
          <text x={W / 2} y={H / 2} className="ps-trend-empty" textAnchor="middle">
            waiting for live data…
          </text>
        ) : (
          series.map((s) => (
            <path
              key={s.tag}
              d={buildPath(s.values, min, max)}
              className="ps-trend-line"
              stroke={s.color}
            />
          ))
        )}
      </svg>
    </div>
  );
};
