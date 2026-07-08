/**
 * Histogram (binned bar chart) for the Data Explorer.
 *
 * {@link Histogram} draws one `<rect>` per bin inside a shared
 * {@link PlotFrame}. Bin extents come from `binEdges` (length = `counts.length
 * + 1`) and bar heights from `counts`. Bars touch the x-axis baseline. The
 * x-domain spans the full edge range; the y-domain spans `[0, max(counts)]`.
 *
 * Presentational only: no API calls, no app state. Forwards a ref to the root
 * `<svg>` (via {@link PlotFrame}). Theme-aware via CSS variables.
 */

import React from "react";
import { PlotFrame } from "./PlotFrame";
import { makeProjector } from "./projection";

export interface HistogramProps {
  width: number;
  height: number;
  /** Bin boundaries; length must be `counts.length + 1`. */
  binEdges: number[];
  /** Per-bin counts (or densities). */
  counts: number[];
  color?: string;
  xLabel?: string;
  yLabel?: string;
}

/** Max of an array, treating non-finite entries as absent; `1` if none. */
function safeMax(values: number[]): number {
  let max = -Infinity;
  for (const v of values) {
    if (Number.isFinite(v) && v > max) max = v;
  }
  return max === -Infinity || max <= 0 ? 1 : max;
}

/** Histogram bar chart. Forwards a ref to the root `<svg>`. */
export const Histogram = React.forwardRef<SVGSVGElement, HistogramProps>(
  function Histogram(props, ref) {
    const { width, height, binEdges, counts, color } = props;
    const usable = Math.max(0, binEdges.length - 1);
    const nBins = Math.min(usable, counts.length);

    const xDomain: [number, number] =
      binEdges.length >= 2
        ? [binEdges[0], binEdges[binEdges.length - 1]]
        : [0, 1];
    const yDomain: [number, number] = [0, safeMax(counts)];

    const { x, y } = makeProjector({ ...props, xDomain, yDomain });
    const baseline = y(0);
    const fill = color ?? "var(--accent-cyan)";

    return (
      <PlotFrame
        ref={ref}
        width={width}
        height={height}
        xDomain={xDomain}
        yDomain={yDomain}
        xLabel={props.xLabel}
        yLabel={props.yLabel}
        snapshotName="histogram"
      >
        {Array.from({ length: nBins }, (_, i) => {
          const left = x(binEdges[i]);
          const right = x(binEdges[i + 1]);
          const count = counts[i];
          if (!Number.isFinite(count)) return null;
          const top = y(Math.max(0, count));
          const barWidth = Math.max(0, right - left);
          const barHeight = Math.max(0, baseline - top);
          return (
            <rect
              key={`bin-${i}`}
              className="plot-bar"
              x={left}
              y={top}
              width={barWidth}
              height={barHeight}
              fill={fill}
              fillOpacity={0.75}
              stroke="var(--panel-border)"
              strokeWidth={0.5}
            />
          );
        })}
      </PlotFrame>
    );
  },
);
