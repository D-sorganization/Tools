import React from "react";
import { timeAxisTicks } from "../lib/trendTime";
import type { FitResult } from "../lib/curveFit";

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
 * Dashed fitted-curve overlay. `xs` is the fit-space x for each plotted sample
 * (e.g. elapsed minutes); `yScale` maps a value to an SVG Y. Sampling per point
 * means a future non-linear method renders as a curve with no changes here.
 */
export const TrendFitOverlay: React.FC<{
  fit: FitResult;
  xs: number[];
  yScale: (value: number) => number;
  x0: number;
  x1: number;
  color?: string;
}> = ({ fit, xs, yScale, x0, x1, color = "var(--text-primary)" }) => {
  if (xs.length < 2) return null;
  const n = xs.length;
  const d = xs
    .map((x, i) => {
      const px = x0 + (i / (n - 1)) * (x1 - x0);
      const py = yScale(fit.predict(x));
      return `${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
    })
    .join(" ");
  return <path d={d} fill="none" stroke={color} strokeWidth={1.5} strokeDasharray="5 4" />;
};
