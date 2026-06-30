/**
 * Pure data → pixel projection for the Data Explorer SVG plots.
 *
 * Split out of {@link PlotFrame} so the frame component and every data layer
 * (lines, points, bars) share one mapping, and so the plot modules export only
 * components (keeps React Fast Refresh happy). No React, no DOM — unit-testable.
 */

import { niceTicks } from "../../../lib/explorer/scale";

export interface PlotMargin {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

/** Minimal geometry a projector needs; {@link PlotFrameProps} is a superset. */
export interface ProjectorInput {
  width: number;
  height: number;
  xDomain: [number, number];
  yDomain: [number, number];
  logX?: boolean;
  logY?: boolean;
  margin?: Partial<PlotMargin>;
}

/** Maps a single data coordinate to a pixel coordinate on one axis. */
export interface AxisProjector {
  /** Project a data value to a pixel along this axis. */
  (value: number): number;
  /** Tick values (in data space) chosen for this axis. */
  readonly ticks: number[];
  /** `true` when the axis uses a log10 transform. */
  readonly log: boolean;
}

/** A pair of axis projectors mapping data → inner-plot pixels. */
export interface PlotProjector {
  x: AxisProjector;
  y: AxisProjector;
  /** Inner plot-area width in pixels (excludes margins). */
  innerWidth: number;
  /** Inner plot-area height in pixels (excludes margins). */
  innerHeight: number;
  /** Resolved margins used to derive the inner area. */
  margin: PlotMargin;
}

const DEFAULT_MARGIN: PlotMargin = { top: 16, right: 16, bottom: 40, left: 52 };

/** Resolve a partial margin against the frame defaults. */
function resolveMargin(margin?: Partial<PlotMargin>): PlotMargin {
  return {
    top: margin?.top ?? DEFAULT_MARGIN.top,
    right: margin?.right ?? DEFAULT_MARGIN.right,
    bottom: margin?.bottom ?? DEFAULT_MARGIN.bottom,
    left: margin?.left ?? DEFAULT_MARGIN.left,
  };
}

/** Safe log10 transform: non-positive values fall back to a tiny epsilon. */
function safeLog10(value: number): number {
  return Math.log10(value > 0 ? value : Number.MIN_VALUE);
}

/**
 * Build an axis projector mapping `domain` → `[pixelMin, pixelMax]`.
 *
 * For a log axis the transform is applied in log10 space; ticks are still the
 * "nice" linear ticks over the (log-transformed) domain so the frame stays
 * dependency-light. A zero-width (transformed) domain maps to the pixel mid.
 */
function makeAxis(
  domain: [number, number],
  pixelMin: number,
  pixelMax: number,
  log: boolean,
  tickCount: number,
): AxisProjector {
  const t = (v: number): number => (log ? safeLog10(v) : v);
  const d0 = t(domain[0]);
  const d1 = t(domain[1]);
  const span = d1 - d0;
  const pSpan = pixelMax - pixelMin;
  const pMid = (pixelMin + pixelMax) / 2;

  const project = ((value: number): number => {
    if (span === 0) return pMid;
    return pixelMin + ((t(value) - d0) / span) * pSpan;
  }) as AxisProjector;

  const lo = Math.min(d0, d1);
  const hi = Math.max(d0, d1);
  const transformedTicks =
    lo === hi ? [lo] : niceTicks(lo, hi, Math.max(1, tickCount));
  const ticks = log
    ? transformedTicks.map((v) => Math.pow(10, v))
    : transformedTicks;

  Object.defineProperties(project, {
    ticks: { value: ticks, enumerable: true },
    log: { value: log, enumerable: true },
  });
  return project;
}

/**
 * Build a data → inner-pixel projector from a {@link ProjectorInput}.
 *
 * The y-axis pixel range is inverted (data max at the top) as is conventional
 * for screen coordinates. Shared so data layers reuse the exact frame mapping.
 */
export function makeProjector(props: ProjectorInput): PlotProjector {
  const margin = resolveMargin(props.margin);
  const innerWidth = Math.max(0, props.width - margin.left - margin.right);
  const innerHeight = Math.max(0, props.height - margin.top - margin.bottom);
  const x = makeAxis(props.xDomain, 0, innerWidth, props.logX ?? false, 6);
  const y = makeAxis(props.yDomain, innerHeight, 0, props.logY ?? false, 5);
  return { x, y, innerWidth, innerHeight, margin };
}
