/**
 * Multi-series line plot for the Data Explorer.
 *
 * {@link LinePlot} draws one `<path>` per series inside a shared
 * {@link PlotFrame}, with optional dash patterns, per-series colors (cycling
 * {@link colorForIndex} when unset) and an optional legend. Domains are derived
 * from the supplied points unless every series is empty.
 *
 * Presentational only: no API calls, no app state. Forwards a ref to the root
 * `<svg>` (via {@link PlotFrame}) so a container can export it. Theme-aware via
 * CSS variables.
 */

import React from "react";
import { PlotFrame } from "./PlotFrame";
import { makeProjector } from "./projection";
import { colorForIndex } from "../../../lib/explorer/palette";

export interface LineSeries {
  name: string;
  color?: string;
  /** SVG `stroke-dasharray` value, e.g. "4 2". */
  dash?: string;
  /** Stroke width in pixels (default 1.5). */
  width?: number;
  points: [number, number][];
}

export interface LinePlotProps {
  width: number;
  height: number;
  series: LineSeries[];
  xLabel?: string;
  yLabel?: string;
  logX?: boolean;
  logY?: boolean;
  grid?: boolean;
  legend?: boolean;
}

/** [min, max] over a single axis across every series; `[0, 1]` if no data. */
function axisExtent(
  series: LineSeries[],
  index: 0 | 1,
): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const s of series) {
    for (const p of s.points) {
      const v = p[index];
      if (!Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  if (min === Infinity) return [0, 1];
  if (min === max) return [min - 1, max + 1];
  return [min, max];
}

/** Build an SVG path `d` string, breaking the line at non-finite points. */
function buildPath(
  points: [number, number][],
  px: (v: number) => number,
  py: (v: number) => number,
): string {
  const segments: string[] = [];
  let penDown = false;
  for (const [dx, dy] of points) {
    if (!Number.isFinite(dx) || !Number.isFinite(dy)) {
      penDown = false;
      continue;
    }
    const cmd = penDown ? "L" : "M";
    segments.push(`${cmd}${px(dx)},${py(dy)}`);
    penDown = true;
  }
  return segments.join(" ");
}

/** Multi-series line plot. Forwards a ref to the root `<svg>`. */
export const LinePlot = React.forwardRef<SVGSVGElement, LinePlotProps>(
  function LinePlot(props, ref) {
    const { width, height, series, legend = false } = props;
    const xDomain = axisExtent(series, 0);
    const yDomain = axisExtent(series, 1);
    const { x, y } = makeProjector({ ...props, xDomain, yDomain });

    return (
      <PlotFrame
        ref={ref}
        width={width}
        height={height}
        xDomain={xDomain}
        yDomain={yDomain}
        xLabel={props.xLabel}
        yLabel={props.yLabel}
        logX={props.logX}
        logY={props.logY}
        grid={props.grid}
      >
        {series.map((s, i) => (
          <path
            key={`line-${i}-${s.name}`}
            className="plot-line"
            d={buildPath(s.points, x, y)}
            fill="none"
            stroke={s.color ?? colorForIndex(i)}
            strokeWidth={s.width ?? 1.5}
            strokeDasharray={s.dash}
            data-series={s.name}
          />
        ))}

        {legend &&
          series.map((s, i) => (
            <g
              key={`legend-${i}-${s.name}`}
              className="plot-legend-item"
              transform={`translate(8, ${8 + i * 16})`}
            >
              <line
                x1={0}
                x2={16}
                y1={0}
                y2={0}
                stroke={s.color ?? colorForIndex(i)}
                strokeWidth={s.width ?? 1.5}
                strokeDasharray={s.dash}
              />
              <text x={20} y={3} fontSize={10} fill="var(--text-primary)">
                {s.name}
              </text>
            </g>
          ))}
      </PlotFrame>
    );
  },
);
