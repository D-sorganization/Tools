import React from "react";
import { timeAxisTicks, timeToX } from "../lib/trendTime";
import type { FitResult } from "../lib/curveFit";

/** A plotted sample's timestamp (for X) plus its fit-space x (for predict). */
export interface FitOverlayPoint {
  t: number;
  x: number;
}

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
  const d = points
    .map((p, i) => {
      const px = timeToX(p.t, t0, t1, x0, x1);
      const py = yScale(fit.predict(p.x));
      return `${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
    })
    .join(" ");
  return <path d={d} fill="none" stroke={color} strokeWidth={1.5} strokeDasharray="5 4" />;
};
