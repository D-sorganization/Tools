/**
 * Scatter plot with optional trendline overlay for the Data Explorer.
 *
 * {@link ScatterPlot} draws one marker (`circle`, `square` or `triangle`) per
 * point of each series inside a shared {@link PlotFrame}, with configurable
 * size and opacity. An optional `trendline` series is drawn as a `<path>` over
 * the markers. Domains span both the scatter points and the trendline.
 *
 * Presentational only: no API calls, no app state. Forwards a ref to the root
 * `<svg>` (via {@link PlotFrame}). Theme-aware via CSS variables.
 */

import React from "react";
import { PlotFrame } from "./PlotFrame";
import { finitePairs, makeProjector, type HoverSeries } from "./projection";
import { colorForIndex } from "../../../lib/explorer/palette";
import { buildPolylinePath } from "./polylinePath";

export type MarkerShape = "circle" | "square" | "triangle";

export interface ScatterSeries {
  name: string;
  color?: string;
  shape?: MarkerShape;
  /** Marker radius / half-size in pixels (default 3). */
  size?: number;
  /** Marker fill opacity in `[0, 1]` (default 0.8). */
  opacity?: number;
  points: [number, number][];
}

export interface ScatterPlotProps {
  width: number;
  height: number;
  series: ScatterSeries[];
  /** Optional trendline overlay drawn as a connected path. */
  trendline?: { points: [number, number][]; color?: string; width?: number };
  xLabel?: string;
  yLabel?: string;
  logX?: boolean;
  logY?: boolean;
  grid?: boolean;
  legend?: boolean;
}

/** Collect every (x, y) pair across scatter series + optional trendline. */
function allPoints(props: ScatterPlotProps): [number, number][] {
  const out: [number, number][] = [];
  for (const s of props.series) out.push(...s.points);
  if (props.trendline) out.push(...props.trendline.points);
  return out;
}

/** [min, max] over one axis; `[0, 1]` if no data, padded if degenerate. */
function axisExtent(points: [number, number][], index: 0 | 1): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const p of points) {
    const v = p[index];
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (min === Infinity) return [0, 1];
  if (min === max) return [min - 1, max + 1];
  return [min, max];
}

/** Render a single marker of the requested shape centred at (cx, cy). */
function Marker(props: {
  shape: MarkerShape;
  cx: number;
  cy: number;
  size: number;
  fill: string;
  opacity: number;
}): React.ReactElement {
  const { shape, cx, cy, size, fill, opacity } = props;
  if (shape === "square") {
    return (
      <rect
        className="plot-marker"
        x={cx - size}
        y={cy - size}
        width={size * 2}
        height={size * 2}
        fill={fill}
        fillOpacity={opacity}
      />
    );
  }
  if (shape === "triangle") {
    const pts = `${cx},${cy - size} ${cx - size},${cy + size} ${cx + size},${
      cy + size
    }`;
    return (
      <polygon
        className="plot-marker"
        points={pts}
        fill={fill}
        fillOpacity={opacity}
      />
    );
  }
  return (
    <circle
      className="plot-marker"
      cx={cx}
      cy={cy}
      r={size}
      fill={fill}
      fillOpacity={opacity}
    />
  );
}

/** Scatter plot with optional trendline. Forwards a ref to the root `<svg>`. */
export const ScatterPlot = React.forwardRef<SVGSVGElement, ScatterPlotProps>(
  function ScatterPlot(props, ref) {
    const { width, height, series, trendline, legend = false } = props;
    const pts = allPoints(props);
    const xDomain = axisExtent(pts, 0);
    const yDomain = axisExtent(pts, 1);
    const { x, y } = makeProjector({ ...props, xDomain, yDomain });

    // Hover crosshair series: scatter x is unordered, so sort each series'
    // finite pairs ascending in x (ys in tandem) for a meaningful nearest-by-x.
    const hoverSeries: HoverSeries[] = series
      .map((s, i) => {
        const { xs, ys } = finitePairs(s.points);
        const order = xs.map((_, k) => k).sort((a, b) => xs[a] - xs[b]);
        return {
          label: s.name,
          color: s.color ?? colorForIndex(i),
          xs: order.map((k) => xs[k]),
          ys: order.map((k) => ys[k]),
        };
      })
      .filter((s) => s.xs.length > 0);

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
        snapshotName="scatter_plot"
        hoverSeries={hoverSeries}
      >
        {series.map((s, i) => {
          const fill = s.color ?? colorForIndex(i);
          const shape = s.shape ?? "circle";
          const size = s.size ?? 3;
          const opacity = s.opacity ?? 0.8;
          return (
            <g
              key={`scatter-${i}-${s.name}`}
              className="plot-scatter"
              data-series={s.name}
            >
              {s.points.map((p, j) => {
                if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) {
                  return null;
                }
                return (
                  <Marker
                    key={j}
                    shape={shape}
                    cx={x(p[0])}
                    cy={y(p[1])}
                    size={size}
                    fill={fill}
                    opacity={opacity}
                  />
                );
              })}
            </g>
          );
        })}

        {trendline && trendline.points.length > 0 && (
          <path
            className="plot-trendline"
            d={buildPolylinePath(trendline.points, x, y)}
            fill="none"
            stroke={trendline.color ?? "var(--accent-cyan)"}
            strokeWidth={trendline.width ?? 2}
          />
        )}

        {legend &&
          series.map((s, i) => (
            <g
              key={`legend-${i}-${s.name}`}
              className="plot-legend-item"
              transform={`translate(8, ${8 + i * 16})`}
            >
              <circle
                cx={4}
                cy={0}
                r={4}
                fill={s.color ?? colorForIndex(i)}
                fillOpacity={s.opacity ?? 0.8}
              />
              <text x={14} y={3} fontSize={10} fill="var(--text-primary)">
                {s.name}
              </text>
            </g>
          ))}
      </PlotFrame>
    );
  },
);
