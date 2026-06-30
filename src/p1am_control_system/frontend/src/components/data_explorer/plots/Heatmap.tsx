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

import React from "react";
import { divergingColor } from "../../../lib/explorer/palette";

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

    return (
      <svg
        ref={ref}
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
        </g>
      </svg>
    );
  },
);
