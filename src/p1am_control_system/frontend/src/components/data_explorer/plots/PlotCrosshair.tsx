/**
 * Presentational hover overlays for the Data Explorer SVG plots.
 *
 * {@link PlotTooltip} draws a small background box with one text line per entry,
 * kept inside the plot rectangle via {@link placeTooltip} (shared by the
 * crosshair, the histogram bin hover and the heatmap cell hover — DRY).
 * {@link PlotCrosshair} draws a vertical guide line, a marker per series and a
 * tooltip from a pre-computed {@link CrosshairModel} (built by the pure
 * {@link buildCrosshairModel}, so this component stays render-only).
 *
 * All overlays are decorative: they carry `aria-hidden` and disable pointer
 * events so they never intercept the cursor or confuse assistive tech.
 */

import React from "react";

import type { CrosshairModel } from "./projection";
import { placeTooltip, type PlotRect, type TooltipPos } from "../../../lib/plotCursor";
import { fmtNumber } from "../../../lib/format";

// Rough SVG text metrics at `TOOLTIP_FONT`px — good enough to size the box so
// `placeTooltip` can keep it inside the plot; exact glyph widths don't matter.
const TOOLTIP_FONT = 10;
const TOOLTIP_LINE_H = 13;
const TOOLTIP_PAD = 5;
const TOOLTIP_CHAR_W = 6;

export interface PlotTooltipProps {
  /** One text line per row (already formatted). */
  lines: string[];
  /** Where the tooltip points (inner-plot pixels). */
  anchor: TooltipPos;
  /** Inner-plot rectangle the box must stay within. */
  bounds: PlotRect;
}

/** Background box + text lines, clamped inside `bounds`. Decorative only. */
export function PlotTooltip({
  lines,
  anchor,
  bounds,
}: PlotTooltipProps): React.ReactElement | null {
  if (lines.length === 0) return null;
  const longest = lines.reduce((m, l) => Math.max(m, l.length), 0);
  const w = longest * TOOLTIP_CHAR_W + TOOLTIP_PAD * 2;
  const h = lines.length * TOOLTIP_LINE_H + TOOLTIP_PAD * 2;
  const pos = placeTooltip(anchor, { w, h }, bounds);
  return (
    <g className="plot-tooltip" aria-hidden="true" pointerEvents="none">
      <rect
        x={pos.x}
        y={pos.y}
        width={w}
        height={h}
        rx={3}
        fill="var(--bg-color)"
        fillOpacity={0.95}
        stroke="var(--panel-border)"
        strokeWidth={1}
      />
      {lines.map((line, i) => (
        <text
          key={i}
          x={pos.x + TOOLTIP_PAD}
          y={pos.y + TOOLTIP_PAD + TOOLTIP_LINE_H * (i + 1) - 3}
          fontSize={TOOLTIP_FONT}
          fill="var(--text-primary)"
        >
          {line}
        </text>
      ))}
    </g>
  );
}

export interface PlotCrosshairProps {
  /** Resolved geometry for the current hover position. */
  model: CrosshairModel;
  /** Inner plot-area size in pixels (for the line span + tooltip bounds). */
  innerWidth: number;
  innerHeight: number;
}

/**
 * Vertical crosshair guide + per-series markers + value tooltip.
 *
 * Renders inside the frame's translated inner-plot `<g>`, so all coordinates
 * are inner-plot pixels. Values are formatted with the shared {@link fmtNumber}.
 */
export function PlotCrosshair({
  model,
  innerWidth,
  innerHeight,
}: PlotCrosshairProps): React.ReactElement {
  const lines = [
    `x: ${fmtNumber(model.dataX)}`,
    ...model.markers.map((m) => `${m.label}: ${fmtNumber(m.value)}`),
  ];
  const anchor: TooltipPos = {
    x: model.lineX,
    y: model.markers[0]?.py ?? 0,
  };
  const bounds: PlotRect = { x0: 0, y0: 0, x1: innerWidth, y1: innerHeight };
  return (
    <g className="plot-crosshair" aria-hidden="true" pointerEvents="none">
      <line
        className="plot-crosshair-line"
        x1={model.lineX}
        x2={model.lineX}
        y1={0}
        y2={innerHeight}
        stroke="var(--accent-cyan)"
        strokeWidth={1}
        strokeDasharray="3 3"
      />
      {model.markers.map((m, i) => (
        <circle
          key={i}
          className="plot-crosshair-marker"
          cx={m.px}
          cy={m.py}
          r={3.5}
          fill={m.color}
          stroke="var(--bg-color)"
          strokeWidth={1}
        />
      ))}
      <PlotTooltip lines={lines} anchor={anchor} bounds={bounds} />
    </g>
  );
}
