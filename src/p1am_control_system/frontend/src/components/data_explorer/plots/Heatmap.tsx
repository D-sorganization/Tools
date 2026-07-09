/**
 * Correlation-style heatmap for the Data Explorer.
 *
 * {@link Heatmap} draws an NxN grid of `<rect>` cells coloured by
 * `colorFor(value)` (defaulting to the diverging blue↔white↔red ramp
 * {@link divergingColor}, suited to correlation values in `[-1, 1]`), with row
 * and column labels and optional in-cell value annotations.
 *
 * Presentational only: no API calls, no app state. Forwards a ref to the root
 * `<svg>` so a container can export it. Theme-aware via CSS variables.
 */

import React, { useCallback, useRef, useState } from "react";
import { divergingColor } from "../../../lib/explorer/palette";
import { PlotTooltip } from "./PlotCrosshair";
import { SnapshotButton } from "../../SnapshotButton";
import { fmtNumber } from "../../../lib/format";

export interface HeatmapProps {
  width: number;
  height: number;
  /** Row/column labels; `matrix` is expected to be `labels.length` square. */
  labels: string[];
  /** Square value matrix, `matrix[row][col]`. */
  matrix: number[][];
  /** Annotate each cell with its numeric value. */
  showValues?: boolean;
  /** Map a cell value to a fill color (default {@link divergingColor}). */
  colorFor?: (t: number) => string;
}

/** Format a cell value for annotation, trimming float noise. */
function formatCell(value: number): string {
  if (!Number.isFinite(value)) return "";
  return value.toFixed(2);
}

/** Correlation-style heatmap. Forwards a ref to the root `<svg>`. */
export const Heatmap = React.forwardRef<SVGSVGElement, HeatmapProps>(
  function Heatmap(props, ref) {
    const {
      width,
      height,
      labels,
      matrix,
      showValues = false,
      colorFor = divergingColor,
    } = props;

    const n = labels.length;
    const marginLeft = 64;
    const marginTop = 56;
    const marginRight = 12;
    const marginBottom = 12;
    const gridW = Math.max(0, width - marginLeft - marginRight);
    const gridH = Math.max(0, height - marginTop - marginBottom);
    const cellW = n > 0 ? gridW / n : 0;
    const cellH = n > 0 ? gridH / n : 0;

    // Mirror the <svg> node into both a private ref (for the snapshot control)
    // and the forwarded ref that callers/tests expect (DRY: one <svg> element).
    const innerRef = useRef<SVGSVGElement | null>(null);
    const setSvgRef = useCallback(
      (node: SVGSVGElement | null) => {
        innerRef.current = node;
        if (typeof ref === "function") ref(node);
        else if (ref) ref.current = node;
      },
      [ref],
    );

    // Cell hover: track the row/col under the pointer and show its value. A
    // categorical grid needs no crosshair line, so this reuses only the shared
    // tooltip primitive, anchored at the hovered cell's centre.
    const [hoverCell, setHoverCell] = useState<{ r: number; c: number } | null>(
      null,
    );
    const cellTooltip =
      hoverCell !== null &&
      hoverCell.r < matrix.length &&
      hoverCell.c < (matrix[hoverCell.r]?.length ?? 0) ? (
        <PlotTooltip
          lines={[
            `row: ${labels[hoverCell.r] ?? hoverCell.r}`,
            `col: ${labels[hoverCell.c] ?? hoverCell.c}`,
            `value: ${fmtNumber(matrix[hoverCell.r][hoverCell.c])}`,
          ]}
          anchor={{
            x: hoverCell.c * cellW + cellW / 2,
            y: hoverCell.r * cellH + cellH / 2,
          }}
          bounds={{ x0: 0, y0: 0, x1: gridW, y1: gridH }}
        />
      ) : null;

    return (
      <div style={{ position: "relative", display: "inline-block", maxWidth: "100%" }}>
      <svg
        ref={setSvgRef}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        style={{ background: "var(--bg-color)" }}
      >
        <g transform={`translate(${marginLeft},${marginTop})`}>
          {/* Cells */}
          {matrix.map((row, r) =>
            row.map((value, c) => {
              const cx = c * cellW;
              const cy = r * cellH;
              const fill = Number.isFinite(value)
                ? colorFor(value)
                : "var(--bg-color)";
              return (
                <g key={`cell-${r}-${c}`}>
                  <rect
                    className="heatmap-cell"
                    x={cx}
                    y={cy}
                    width={cellW}
                    height={cellH}
                    fill={fill}
                    stroke="var(--panel-border)"
                    strokeWidth={0.5}
                    data-row={r}
                    data-col={c}
                    onPointerEnter={() => setHoverCell({ r, c })}
                    onPointerLeave={() => setHoverCell(null)}
                  />
                  {showValues && (
                    <text
                      x={cx + cellW / 2}
                      y={cy + cellH / 2 + 3}
                      textAnchor="middle"
                      fontSize={9}
                      fill="var(--text-primary)"
                    >
                      {formatCell(value)}
                    </text>
                  )}
                </g>
              );
            }),
          )}

          {/* Column labels (rotated) */}
          {labels.map((label, c) => {
            const cx = c * cellW + cellW / 2;
            return (
              <text
                key={`col-${c}`}
                className="heatmap-col-label"
                x={cx}
                y={-6}
                textAnchor="start"
                fontSize={10}
                fill="var(--text-primary)"
                transform={`rotate(-45 ${cx} -6)`}
              >
                {label}
              </text>
            );
          })}

          {/* Row labels */}
          {labels.map((label, r) => (
            <text
              key={`row-${r}`}
              className="heatmap-row-label"
              x={-8}
              y={r * cellH + cellH / 2 + 3}
              textAnchor="end"
              fontSize={10}
              fill="var(--text-primary)"
            >
              {label}
            </text>
          ))}

          {/* Hover cell tooltip */}
          {cellTooltip}
        </g>
      </svg>
      <div style={{ position: "absolute", top: 6, right: 6 }}>
        <SnapshotButton
          targetRef={innerRef}
          filename="correlation_heatmap"
          label="Export heatmap snapshot"
        />
      </div>
      </div>
    );
  },
);
