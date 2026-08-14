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

import React, { useState } from "react";
import { PlotFrame } from "./PlotFrame";
import { PlotTooltip } from "./PlotCrosshair";
import { makeProjector } from "./projection";
import { fmtNumber } from "../../../lib/format";

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

    const { x, y, innerWidth, innerHeight } = makeProjector({
      ...props,
      xDomain,
      yDomain,
    });
    const baseline = y(0);
    const fill = color ?? "var(--accent-cyan)";

    // Bin hover: track which bar the pointer is over and show its range + count.
    // A categorical hover (not the shared nearest-x crosshair) reads cleanly for
    // bars, so this reuses only the shared tooltip primitive.
    const [hoverBin, setHoverBin] = useState<number | null>(null);
    const activeBin =
      hoverBin !== null && hoverBin >= 0 && hoverBin < nBins ? hoverBin : null;
    const binTooltip =
      activeBin !== null && Number.isFinite(counts[activeBin]) ? (
        <PlotTooltip
          lines={[
            `[${fmtNumber(binEdges[activeBin])}, ${fmtNumber(
              binEdges[activeBin + 1],
            )})`,
            `count: ${fmtNumber(counts[activeBin])}`,
          ]}
          anchor={{
            x: (x(binEdges[activeBin]) + x(binEdges[activeBin + 1])) / 2,
            y: y(Math.max(0, counts[activeBin])),
          }}
          bounds={{ x0: 0, y0: 0, x1: innerWidth, y1: innerHeight }}
        />
      ) : null;

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
        {(() => {
          // ⚡ Bolt Optimization: Replace Array.from() with a pre-allocated array
          // and a standard for-loop to prevent intermediate garbage collection pressure.
          const rects = new Array(nBins);
          for (let i = 0; i < nBins; i++) {
            const left = x(binEdges[i]);
            const right = x(binEdges[i + 1]);
            const count = counts[i];
            if (!Number.isFinite(count)) {
              rects[i] = null;
              continue;
            }
            const top = y(Math.max(0, count));
            const barWidth = Math.max(0, right - left);
            const barHeight = Math.max(0, baseline - top);
            rects[i] = (
              <rect
                key={`bin-${i}`}
                className="plot-bar"
                x={left}
                y={top}
                width={barWidth}
                height={barHeight}
                fill={fill}
                fillOpacity={activeBin === i ? 0.95 : 0.75}
                stroke="var(--panel-border)"
                strokeWidth={0.5}
                data-bin={i}
                onPointerEnter={() => setHoverBin(i)}
                onPointerLeave={() => setHoverBin(null)}
              />
            );
          }
          return rects;
        })()}

        {binTooltip}
      </PlotFrame>
    );
  },
);
