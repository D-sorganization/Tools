import React from "react";
import { timeAxisTicks, timeToX } from "../lib/trendTime";
import { placeTooltip } from "../lib/plotCursor";
import type { FitResult } from "../lib/curveFit";

/** A plotted sample's timestamp (for X) plus its fit-space x (for predict). */
export interface FitOverlayPoint {
  t: number;
  x: number;
}

/** One trace's value at the hovered x, plus the pixel Y of its marker. */
export interface CrosshairSeries {
  label: string;
  color: string;
  /** Preformatted value text with units, e.g. "72.5" or "1.23 A". Each chart
   *  formats its own so per-trace units (A / V / W / °C) are respected. */
  text: string;
  /** Pixel Y of this series' point at the hovered x. */
  py: number;
}

const CH_FONT = 10;
const CH_LINE_H = 13;
const CH_PAD_X = 6;
const CH_PAD_Y = 5;
const CH_SWATCH = 8;
// Monospace advance at CH_FONT px — used to size the tooltip box without
// measuring the DOM (keeps the component pure + testable under jsdom).
const CH_CHAR_W = 6.2;

/**
 * Shared hover crosshair + value tooltip for the live trend plots.
 *
 * Draws a vertical guide at the hovered (snapped) x, a marker per series at its
 * point, and a tooltip box listing the x label and every series' value. The box
 * is positioned with the shared `placeTooltip` so it always stays inside the
 * plot rectangle. Purely presentational: each chart resolves the nearest sample
 * (via `nearestIndexByX` over its own domain) and hands the pixel geometry here,
 * so the same overlay serves the sample-index chart and the time-domain charts.
 */
export const TrendCrosshair: React.FC<{
  /** Absolute SVG x of the crosshair guide (already snapped to a sample). */
  px: number;
  yTop: number;
  yBottom: number;
  plotLeft: number;
  plotRight: number;
  series: readonly CrosshairSeries[];
  /** Formatted x label (e.g. a clock time or "-3.2s"). */
  xLabel: string;
}> = ({ px, yTop, yBottom, plotLeft, plotRight, series, xLabel }) => {
  if (series.length === 0) return null;

  // OPTIMIZATION: Use a single-pass loop instead of chained .map() and .reduce()
  // array operations to eliminate intermediate array allocations and closure
  // overhead during high-frequency pointer move rendering.
  const lines: string[] = [xLabel];
  let maxChars = xLabel.length;
  let anchorY = yBottom;

  for (let i = 0; i < series.length; i++) {
    const s = series[i];
    const text = `${s.label}  ${s.text}`;
    lines.push(text);
    if (text.length > maxChars) maxChars = text.length;
    if (s.py < anchorY) anchorY = s.py;
  }

  const boxW = CH_PAD_X * 2 + CH_SWATCH + 4 + maxChars * CH_CHAR_W;
  const boxH = CH_PAD_Y * 2 + lines.length * CH_LINE_H;
  const pos = placeTooltip(
    { x: px, y: anchorY },
    { w: boxW, h: boxH },
    { x0: plotLeft, y0: yTop, x1: plotRight, y1: yBottom },
  );
  return (
    <g aria-hidden pointerEvents="none">
      <line
        x1={px}
        y1={yTop}
        x2={px}
        y2={yBottom}
        stroke="var(--text-muted)"
        strokeDasharray="3 3"
      />
      {series.map((s, i) => (
        <circle
          key={i}
          cx={px}
          cy={s.py}
          r={3}
          fill={s.color}
          stroke="var(--panel-bg)"
          strokeWidth={1}
        />
      ))}
      <g transform={`translate(${pos.x.toFixed(1)},${pos.y.toFixed(1)})`}>
        <rect
          width={boxW}
          height={boxH}
          rx={3}
          fill="var(--panel-bg)"
          stroke="var(--panel-border)"
          opacity={0.96}
        />
        <text
          x={CH_PAD_X}
          y={CH_PAD_Y + CH_FONT}
          fontSize={CH_FONT}
          fontFamily="var(--font-mono)"
          fill="var(--text-muted)"
        >
          {xLabel}
        </text>
        {series.map((s, i) => (
          <g
            key={i}
            transform={`translate(${CH_PAD_X},${CH_PAD_Y + (i + 1) * CH_LINE_H})`}
          >
            <rect x={0} y={2} width={CH_SWATCH} height={CH_SWATCH} rx={1} fill={s.color} />
            <text
              x={CH_SWATCH + 4}
              y={CH_FONT}
              fontSize={CH_FONT}
              fontFamily="var(--font-mono)"
              fill="var(--text-primary)"
            >
              {`${s.label}  ${s.text}`}
            </text>
          </g>
        ))}
      </g>
    </g>
  );
};

/**
 * Reusable SVG decorations shared by the trend plots. Each takes the plot's
 * geometry (and, for the fit, the caller's value→Y scale) so it can sit inside
 * any trend's <svg> without coupling to that trend's units or layout.
 */

/** X-axis time ticks: "-5m" … "now" across the visible span. */
export const TrendTimeAxis: React.FC<{
  x0: number;
  x1: number;
  yBottom: number;
  spanSeconds: number;
  count?: number;
}> = ({ x0, x1, yBottom, spanSeconds, count = 4 }) => (
  <g aria-hidden>
    {timeAxisTicks(spanSeconds, count).map((t, i) => {
      const x = x0 + t.frac * (x1 - x0);
      const anchor = i === 0 ? "start" : i === count ? "end" : "middle";
      return (
        <g key={i}>
          <line x1={x} y1={yBottom} x2={x} y2={yBottom + 3} stroke="var(--panel-border)" />
          <text
            x={x}
            y={yBottom + 13}
            fill="var(--text-muted)"
            fontSize="9"
            fontFamily="var(--font-mono)"
            textAnchor={anchor}
          >
            {t.label}
          </text>
        </g>
      );
    })}
  </g>
);

/**
 * Dashed fitted-curve overlay, positioned by time so it tracks the data line
 * exactly (each point's X comes from its timestamp via [t0,t1], not its index).
 * `point.x` is the fit-space x (e.g. elapsed minutes) fed to `fit.predict`;
 * `yScale` maps the predicted value to an SVG Y. Per-point sampling means a
 * future non-linear method renders as a curve with no changes here.
 */
export const TrendFitOverlay: React.FC<{
  fit: FitResult;
  points: readonly FitOverlayPoint[];
  t0: number;
  t1: number;
  x0: number;
  x1: number;
  yScale: (value: number) => number;
  color?: string;
}> = ({ fit, points, t0, t1, x0, x1, yScale, color = "var(--text-primary)" }) => {
  if (points.length < 2 || !(t1 > t0)) return null;

  // ⚡ Bolt Optimization: Build the SVG path string in a single pass instead of
  // allocating an intermediate array of segment strings with .map().join(" ").
  let d = "";
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const px = timeToX(p.t, t0, t1, x0, x1);
    const py = yScale(fit.predict(p.x));
    if (i > 0) d += " ";
    d += (i === 0 ? "M" : "L") + px.toFixed(1) + "," + py.toFixed(1);
  }
  return <path d={d} fill="none" stroke={color} strokeWidth={1.5} strokeDasharray="5 4" />;
};
