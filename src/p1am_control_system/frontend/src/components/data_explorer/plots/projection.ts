/**
 * Pure data → pixel projection for the Data Explorer SVG plots.
 *
 * Split out of {@link PlotFrame} so the frame component and every data layer
 * (lines, points, bars) share one mapping, and so the plot modules export only
 * components (keeps React Fast Refresh happy). No React, no DOM — unit-testable.
 */

import { niceTicks } from "../../../lib/explorer/scale";
import { nearestIndexByX } from "../../../lib/plotCursor";

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
  /**
   * Inverse of the projection: the data value at an inner-pixel coordinate.
   *
   * Mirrors `linearScale.invert`, honouring a log10 axis (the result is mapped
   * back out of log space). Like the forward projector it does not validate per
   * call — callers pass finite cursor pixels.
   */
  invert(pixel: number): number;
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

  // Inverse mapping (pixel → data), mirroring `linearScale.invert`. A zero-width
  // pixel range can't be inverted, so fall back to the domain midpoint.
  const invert = (pixel: number): number => {
    if (pSpan === 0) {
      const mid = (d0 + d1) / 2;
      return log ? Math.pow(10, mid) : mid;
    }
    const transformed = d0 + ((pixel - pixelMin) / pSpan) * span;
    return log ? Math.pow(10, transformed) : transformed;
  };

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
    invert: { value: invert, enumerable: true },
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

/**
 * One hoverable series handed to {@link PlotFrame}: ascending `xs` paired with
 * matching `ys`, plus the label/color used in the crosshair tooltip and marker.
 */
export interface HoverSeries {
  label: string;
  color: string;
  xs: number[];
  ys: number[];
}

/**
 * Split `[x, y]` points into parallel `xs`/`ys` arrays, dropping any pair with a
 * non-finite coordinate. Shared by the point-based plots when they build their
 * {@link HoverSeries} (DRY).
 *
 * @throws TypeError if `points` is not an array.
 */
export function finitePairs(points: readonly [number, number][]): {
  xs: number[];
  ys: number[];
} {
  if (!Array.isArray(points)) {
    throw new TypeError("finitePairs: points must be an array");
  }
  const xs: number[] = [];
  const ys: number[] = [];
  for (const [px, py] of points) {
    if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
    xs.push(px);
    ys.push(py);
  }
  return { xs, ys };
}

/** A resolved crosshair marker: one series' nearest sample, in inner pixels. */
export interface CrosshairMarker {
  label: string;
  color: string;
  /** Inner-pixel x of the sample. */
  px: number;
  /** Inner-pixel y of the sample. */
  py: number;
  /** The sample's data-y value (for the tooltip line). */
  value: number;
}

/** Geometry a crosshair renderer needs for a single hover position. */
export interface CrosshairModel {
  /** Inner-pixel x of the vertical crosshair line (snapped to a sample). */
  lineX: number;
  /** Snapped data-x value shown in the tooltip. */
  dataX: number;
  markers: CrosshairMarker[];
}

/**
 * Resolve the crosshair geometry for a cursor at `innerX` (inner-plot pixels).
 *
 * The cursor pixel is inverted to a data-x via {@link AxisProjector.invert};
 * for each series the sample nearest that data-x is located with
 * {@link nearestIndexByX} and projected back to inner pixels. The vertical line
 * snaps to the first series' nearest sample. Non-finite samples are skipped;
 * `null` is returned when no series yields a finite sample (e.g. all empty).
 *
 * Precondition: `innerX` is a finite number; `series` is an array.
 * @throws TypeError if `innerX` is not finite or `series` is not an array.
 */
export function buildCrosshairModel(
  innerX: number,
  series: readonly HoverSeries[],
  x: AxisProjector,
  y: AxisProjector,
): CrosshairModel | null {
  if (typeof innerX !== "number" || !Number.isFinite(innerX)) {
    throw new TypeError("buildCrosshairModel: innerX must be a finite number");
  }
  if (!Array.isArray(series)) {
    throw new TypeError("buildCrosshairModel: series must be an array");
  }
  const dataX = x.invert(innerX);
  const markers: CrosshairMarker[] = [];
  let snapDataX: number | null = null;
  for (const s of series) {
    const idx = nearestIndexByX(s.xs, dataX);
    if (idx === null) continue;
    const sx = s.xs[idx];
    const sy = s.ys[idx];
    if (!Number.isFinite(sx) || !Number.isFinite(sy)) continue;
    markers.push({
      label: s.label,
      color: s.color,
      px: x(sx),
      py: y(sy),
      value: sy,
    });
    if (snapDataX === null) snapDataX = sx;
  }
  if (snapDataX === null || markers.length === 0) return null;
  return { lineX: x(snapDataX), dataX: snapDataX, markers };
}
