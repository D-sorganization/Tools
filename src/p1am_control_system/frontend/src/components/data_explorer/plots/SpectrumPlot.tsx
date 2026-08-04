/**
 * Frequency-spectrum plot for the Data Explorer.
 *
 * {@link SpectrumPlot} draws a single power-versus-frequency `<path>` inside a
 * shared {@link PlotFrame}, with optional log10 frequency and/or power axes
 * (the common presentation for FFT/Welch output). Domains are derived from the
 * supplied `freqs`/`power`; on a log axis non-positive samples are dropped from
 * the domain calculation.
 *
 * Presentational only: no API calls, no app state. Forwards a ref to the root
 * `<svg>` (via {@link PlotFrame}). Theme-aware via CSS variables.
 */

import React from "react";
import { PlotFrame } from "./PlotFrame";
import { makeProjector, type HoverSeries } from "./projection";

export interface SpectrumPlotProps {
  width: number;
  height: number;
  freqs: number[];
  power: number[];
  logX?: boolean;
  logY?: boolean;
  xLabel?: string;
  yLabel?: string;
}

/** [min, max] over finite samples, honouring a log axis (positives only). */
function axisExtent(values: number[], log: boolean): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (log && v <= 0) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (min === Infinity) return log ? [1, 10] : [0, 1];
  if (min === max) return log ? [min / 2 || 1, max * 2 || 10] : [min, max + 1];
  return [min, max];
}

/** Build the spectrum path, breaking at non-finite or (log) non-positive. */
function buildPath(
  freqs: number[],
  power: number[],
  logX: boolean,
  logY: boolean,
  px: (v: number) => number,
  py: (v: number) => number,
): string {
  const n = Math.min(freqs.length, power.length);
  // ⚡ Bolt Optimization: Use a single string with concatenation to build SVG path directly
  // avoiding intermediate array allocation and .join() overhead on each re-render
  let d = "";
  let penDown = false;
  for (let i = 0; i < n; i += 1) {
    const f = freqs[i];
    const p = power[i];
    const bad =
      !Number.isFinite(f) ||
      !Number.isFinite(p) ||
      (logX && f <= 0) ||
      (logY && p <= 0);
    if (bad) {
      penDown = false;
      continue;
    }
    if (d.length > 0) d += " ";
    d += `${penDown ? "L" : "M"}${px(f)},${py(p)}`;
    penDown = true;
  }
  return d;
}

/** Frequency-spectrum plot. Forwards a ref to the root `<svg>`. */
export const SpectrumPlot = React.forwardRef<SVGSVGElement, SpectrumPlotProps>(
  function SpectrumPlot(props, ref) {
    const { width, height, freqs, power } = props;
    const logX = props.logX ?? false;
    const logY = props.logY ?? false;
    const xDomain = axisExtent(freqs, logX);
    const yDomain = axisExtent(power, logY);
    const { x, y } = makeProjector({ ...props, xDomain, yDomain });

    // Hover crosshair series: finite (freq, power) samples, dropping any that a
    // log axis would exclude (freqs are already ascending — no sort needed).
    const nHover = Math.min(freqs.length, power.length);
    const hxs: number[] = [];
    const hys: number[] = [];
    for (let i = 0; i < nHover; i += 1) {
      const f = freqs[i];
      const p = power[i];
      if (!Number.isFinite(f) || !Number.isFinite(p)) continue;
      if ((logX && f <= 0) || (logY && p <= 0)) continue;
      hxs.push(f);
      hys.push(p);
    }
    const hoverSeries: HoverSeries[] =
      hxs.length > 0
        ? [
            {
              label: props.yLabel ?? "power",
              color: "var(--accent-cyan)",
              xs: hxs,
              ys: hys,
            },
          ]
        : [];

    return (
      <PlotFrame
        ref={ref}
        width={width}
        height={height}
        xDomain={xDomain}
        yDomain={yDomain}
        xLabel={props.xLabel}
        yLabel={props.yLabel}
        logX={logX}
        logY={logY}
        snapshotName="spectrum"
        hoverSeries={hoverSeries}
      >
        <path
          className="plot-spectrum"
          d={buildPath(freqs, power, logX, logY, x, y)}
          fill="none"
          stroke="var(--accent-cyan)"
          strokeWidth={1.5}
        />
      </PlotFrame>
    );
  },
);
