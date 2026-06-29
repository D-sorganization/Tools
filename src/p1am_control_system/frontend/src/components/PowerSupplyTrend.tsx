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
  elapsedSeconds,
  windowStartIndex,
} from "../lib/trendTime";
import { TrendAxisControls } from "./TrendAxisControls";
import { TrendTimeControls } from "./TrendTimeControls";
import { TrendTimeAxis } from "./TrendPlotOverlays";

/**
 * Compact dual-trace trend for the power-supply screen: measured current and
 * voltage coming back from the unit, plotted as a percent of their respective
 * full scales on a shared axis so both fit one clean chart. Live engineering
 * values are shown in the legend.
 *
 * The shared axis is in % of full scale and defaults to 0-100 %, but is
 * rescalable (Auto-fit or manual min/max %) via the shared TrendAxisControls so
 * small signals can be zoomed in.
 *
 * Self-contained SVG (no chart lib): the parent accumulates a rolling sample
 * buffer from the live status and passes it in.
 */

export interface TrendSample {
  t: number; // epoch ms when sampled
  i: number; // measured current (A)
  v: number; // measured voltage (V)
  p: number; // measured power (W)
}

interface Props {
  samples: TrendSample[];
  currentFullScale: number;
  voltageFullScale: number;
  powerFullScale: number;
  currentLabel?: string;
  voltageLabel?: string;
}

const W = 600;
const H = 192;
const PAD_L = 34;
const PAD_R = 10;
const PAD_T = 10;
const PAD_B = 28; // room for the X-axis time labels
const PLOT_W = W - PAD_L - PAD_R;
const PLOT_H = H - PAD_T - PAD_B;

const CURRENT_COLOR = "var(--color-success)";
const VOLTAGE_COLOR = "var(--accent-purple)";
const POWER_COLOR = "var(--accent-cyan)";

/** Convert a measured value to percent of its full scale (0 when no scale). */
function toPct(value: number, fullScale: number): number {
  return fullScale > 0 ? (value / fullScale) * 100 : 0;
}

/** Plot a series already expressed in % against the resolved [min, max] % axis. */
function buildPctPath(pctValues: number[], min: number, max: number): string {
  if (pctValues.length < 2 || max <= min) return "";
  const n = pctValues.length;
  return pctValues
    .map((pct, idx) => {
      const x = PAD_L + (idx / (n - 1)) * PLOT_W;
      const frac = Math.max(0, Math.min(1, (pct - min) / (max - min)));
      const y = PAD_T + (1 - frac) * PLOT_H;
      return `${idx === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export const PowerSupplyTrend: React.FC<Props> = ({
  samples,
  currentFullScale,
  voltageFullScale,
  powerFullScale,
  currentLabel = "Current",
  voltageLabel = "Voltage",
}) => {
  const [axis, setAxis] = useState<AxisRange>(defaultAxisRange(0, 100));
  const [windowSeconds, setWindowSeconds] = useState<number>(120);

  // Window by real wall-clock time (rate-independent).
  const windowed = samples.slice(
    windowStartIndex(
      samples.map((s) => s.t),
      windowSeconds,
    ),
  );
  const iPct = downsample(windowed.map((s) => toPct(s.i, currentFullScale)));
  const vPct = downsample(windowed.map((s) => toPct(s.v, voltageFullScale)));
  const pPct = downsample(windowed.map((s) => toPct(s.p, powerFullScale)));
  const { min, max } = resolveRange(axis, [...iPct, ...vPct, ...pPct], {
    min: 0,
    max: 100,
  });

  const currentPath = buildPctPath(iPct, min, max);
  const voltagePath = buildPctPath(vPct, min, max);
  const powerPath = buildPctPath(pPct, min, max);
  const last = windowed[windowed.length - 1];

  return (
    <div className="ps-trend">
      <div className="ps-trend-legend">
        <span className="ps-trend-key">
          <span className="swatch" style={{ background: CURRENT_COLOR }} />
          {currentLabel}
          <strong>{last ? `${last.i.toFixed(2)} A` : "—"}</strong>
        </span>
        <span className="ps-trend-key">
          <span className="swatch" style={{ background: VOLTAGE_COLOR }} />
          {voltageLabel}
          <strong>{last ? `${last.v.toFixed(2)} V` : "—"}</strong>
        </span>
        <span className="ps-trend-key">
          <span className="swatch" style={{ background: POWER_COLOR }} />
          Power
          <strong>
            {last ? `${(last.p / 1000).toFixed(last.p / 1000 >= 10 ? 1 : 2)} kW` : "—"}
          </strong>
        </span>
        <span className="ps-trend-window">
          last {formatWindow(windowSeconds)} · % of full scale
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
        <TrendAxisControls value={axis} onChange={setAxis} unit="%" />
      </div>

      <svg
        className="ps-trend-svg"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="Current and voltage trend"
      >
        {/* gridlines + % axis labels across the resolved [min, max] range */}
        {axisTicks(min, max, 4).map((tick, i) => {
          const y = PAD_T + (1 - (tick - min) / (max - min)) * PLOT_H;
          return (
            <g key={i}>
              <line x1={PAD_L} y1={y} x2={W - PAD_R} y2={y} className="ps-trend-grid" />
              <text x={PAD_L - 6} y={y + 3} className="ps-trend-axis" textAnchor="end">
                {tick.toFixed(Number.isInteger(tick) ? 0 : 1)}
              </text>
            </g>
          );
        })}

        <TrendTimeAxis
          x0={PAD_L}
          x1={W - PAD_R}
          yBottom={PAD_T + PLOT_H}
          spanSeconds={elapsedSeconds(windowed.map((s) => s.t))}
        />

        {windowed.length < 2 ? (
          <text x={W / 2} y={H / 2} className="ps-trend-empty" textAnchor="middle">
            waiting for live data…
          </text>
        ) : (
          <>
            <path d={powerPath} className="ps-trend-line" stroke={POWER_COLOR} />
            <path d={voltagePath} className="ps-trend-line" stroke={VOLTAGE_COLOR} />
            <path d={currentPath} className="ps-trend-line" stroke={CURRENT_COLOR} />
          </>
        )}
      </svg>
    </div>
  );
};
