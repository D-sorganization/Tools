/**
 * Shared axes/grid/labels frame for the Data Explorer SVG plots.
 *
 * {@link PlotFrame} draws the plot box, gridlines, tick marks, tick labels and
 * axis titles for a linear or log10 axis, and renders its `children` inside the
 * inner plot area (translated past the margins). The pure helper
 * {@link makeProjector} builds a data → pixel mapping from the same props so
 * data layers (lines, points, bars) can position themselves identically.
 *
 * Purely presentational: no API calls, no app state. The root `<svg>` is
 * forwarded via `React.forwardRef` so a container can serialize/export it.
 * Theme-aware through CSS variables (`--text-primary`, `--panel-border`,
 * `--bg-color`, `--accent-cyan`).
 */

import React, { useCallback, useRef, useState } from "react";

import {
  buildCrosshairModel,
  makeProjector,
  type HoverSeries,
  type PlotMargin,
} from "./projection";
import { PlotCrosshair } from "./PlotCrosshair";
import { SnapshotButton } from "../../SnapshotButton";

export type { HoverSeries, PlotMargin } from "./projection";

export interface PlotFrameProps {
  width: number;
  height: number;
  xDomain: [number, number];
  yDomain: [number, number];
  xLabel?: string;
  yLabel?: string;
  logX?: boolean;
  logY?: boolean;
  grid?: boolean;
  margin?: Partial<PlotMargin>;
  /** Filename prefix for the shared PNG/SVG snapshot control (default "plot"). */
  snapshotName?: string;
  /**
   * When present, enables the shared hover crosshair + value tooltip: on
   * pointer move the frame snaps a vertical guide to the nearest x-sample of
   * each series and shows a marker and per-series value. Omit to keep a plain,
   * non-interactive frame (the default; existing consumers are unaffected).
   */
  hoverSeries?: HoverSeries[];
  children?: React.ReactNode;
}

/** Format a tick value compactly (trims noisy float tails). */
function formatTick(value: number): string {
  if (!Number.isFinite(value)) return "";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1e5 || abs < 1e-3) return value.toExponential(1);
  return String(Number(value.toPrecision(6)));
}

/**
 * Axes/grid/labels frame; renders `children` inside the inner plot area.
 *
 * @remarks Presentational only. Forwards a ref to the root `<svg>`.
 */
export const PlotFrame = React.forwardRef<SVGSVGElement, PlotFrameProps>(
  function PlotFrame(props, ref) {
    const { width, height, xLabel, yLabel, grid = true, snapshotName, hoverSeries } =
      props;
    const { x, y, innerWidth, innerHeight, margin } = makeProjector(props);

    // The snapshot control needs the live `<svg>` node, and callers (and their
    // tests) still expect the forwarded ref. Keep a private ref for export and
    // mirror the node into the forwarded ref (DRY: one <svg>, two consumers).
    const innerRef = useRef<SVGSVGElement | null>(null);
    const setSvgRef = useCallback(
      (node: SVGSVGElement | null) => {
        innerRef.current = node;
        if (typeof ref === "function") ref(node);
        else if (ref) ref.current = node;
      },
      [ref],
    );

    // Hover crosshair state: the cursor's x within the inner plot area (pixels),
    // or `null` when the pointer is away. Only the geometry needed to place the
    // guide lives in state; the per-series markers are derived at render time.
    const [hoverX, setHoverX] = useState<number | null>(null);
    const hoverEnabled = hoverSeries !== undefined && hoverSeries.length > 0;

    // Stable handlers: they read only primitive geometry (size + left margin),
    // so the cursor→inner-pixel conversion doesn't churn on every render. The
    // <svg> uses viewBox="0 0 width height" with a responsive rendered size, so
    // scale the client offset back into viewBox space before removing the left
    // margin (mirrors the live TrendChart's pixel mapping).
    const handlePointerMove = useCallback(
      (e: React.PointerEvent<SVGSVGElement>) => {
        const svg = innerRef.current;
        if (!svg) return;
        const rect = svg.getBoundingClientRect();
        if (rect.width === 0) return; // unlaid-out / jsdom: avoid a NaN cursor
        const svgX = ((e.clientX - rect.left) / rect.width) * width;
        const innerX = svgX - margin.left;
        setHoverX(innerX < 0 || innerX > innerWidth ? null : innerX);
      },
      [width, margin.left, innerWidth],
    );
    const handlePointerLeave = useCallback(() => setHoverX(null), []);

    const crosshair =
      hoverEnabled && hoverX !== null
        ? buildCrosshairModel(hoverX, hoverSeries, x, y)
        : null;

    return (
      <div style={{ position: "relative", display: "inline-block", maxWidth: "100%" }}>
      <svg
        ref={setSvgRef}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        style={{
          background: "var(--bg-color)",
          cursor: hoverEnabled ? "crosshair" : undefined,
        }}
        onPointerMove={hoverEnabled ? handlePointerMove : undefined}
        onPointerLeave={hoverEnabled ? handlePointerLeave : undefined}
      >
        <g transform={`translate(${margin.left},${margin.top})`}>
          {/* Gridlines */}
          {grid &&
            x.ticks.map((t, i) => {
              const px = x(t);
              return (
                <line
                  key={`gx-${i}`}
                  className="plot-grid"
                  x1={px}
                  x2={px}
                  y1={0}
                  y2={innerHeight}
                  stroke="var(--panel-border)"
                  strokeOpacity={0.35}
                  strokeWidth={1}
                />
              );
            })}
          {grid &&
            y.ticks.map((t, i) => {
              const py = y(t);
              return (
                <line
                  key={`gy-${i}`}
                  className="plot-grid"
                  x1={0}
                  x2={innerWidth}
                  y1={py}
                  y2={py}
                  stroke="var(--panel-border)"
                  strokeOpacity={0.35}
                  strokeWidth={1}
                />
              );
            })}

          {/* Plot-area border */}
          <rect
            className="plot-border"
            x={0}
            y={0}
            width={innerWidth}
            height={innerHeight}
            fill="none"
            stroke="var(--panel-border)"
            strokeWidth={1}
          />

          {/* X ticks + labels */}
          {x.ticks.map((t, i) => {
            const px = x(t);
            return (
              <g key={`tx-${i}`}>
                <line
                  x1={px}
                  x2={px}
                  y1={innerHeight}
                  y2={innerHeight + 5}
                  stroke="var(--text-primary)"
                  strokeWidth={1}
                />
                <text
                  x={px}
                  y={innerHeight + 18}
                  textAnchor="middle"
                  fontSize={10}
                  fill="var(--text-primary)"
                >
                  {formatTick(t)}
                </text>
              </g>
            );
          })}

          {/* Y ticks + labels */}
          {y.ticks.map((t, i) => {
            const py = y(t);
            return (
              <g key={`ty-${i}`}>
                <line
                  x1={-5}
                  x2={0}
                  y1={py}
                  y2={py}
                  stroke="var(--text-primary)"
                  strokeWidth={1}
                />
                <text
                  x={-8}
                  y={py + 3}
                  textAnchor="end"
                  fontSize={10}
                  fill="var(--text-primary)"
                >
                  {formatTick(t)}
                </text>
              </g>
            );
          })}

          {/* Data layers */}
          {props.children}

          {/* Hover crosshair (drawn over the data, inside the inner area) */}
          {crosshair && (
            <PlotCrosshair
              model={crosshair}
              innerWidth={innerWidth}
              innerHeight={innerHeight}
            />
          )}
        </g>

        {/* Axis titles */}
        {xLabel && (
          <text
            className="plot-axis-label"
            x={margin.left + innerWidth / 2}
            y={height - 6}
            textAnchor="middle"
            fontSize={11}
            fill="var(--text-primary)"
          >
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text
            className="plot-axis-label"
            x={12}
            y={margin.top + innerHeight / 2}
            textAnchor="middle"
            fontSize={11}
            fill="var(--text-primary)"
            transform={`rotate(-90 12 ${margin.top + innerHeight / 2})`}
          >
            {yLabel}
          </text>
        )}
      </svg>
      <div style={{ position: "absolute", top: 6, right: 6 }}>
        <SnapshotButton
          targetRef={innerRef}
          filename={snapshotName ?? "plot"}
          label="Export plot snapshot"
        />
      </div>
      </div>
    );
  },
);
