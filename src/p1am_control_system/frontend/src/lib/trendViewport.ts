/**
 * Pan / zoom / pause viewport model for the trend plots.
 *
 * A single, chart-agnostic model of "which slice of the data is on screen" so
 * every trend gets identical scroll-back-in-time, zoom, and freeze behavior from
 * ONE tested source (DRY). It works in an abstract numeric *domain* — the caller
 * decides the units, so an index-based chart passes sample indices and a
 * time-based chart passes epoch-ms; the math is the same either way.
 *
 * A viewport is three numbers:
 *   - `span`   : visible width in domain units (the zoom level),
 *   - `offset` : how far the right edge sits BEHIND the latest data edge
 *                (0 = following the live edge; >0 = panned back into history),
 *   - `paused` : the chart should stop consuming new data (snapshot its buffer).
 *
 * Pure functions with explicit precondition checks (DbC); no React/DOM here so
 * the behavior is unit-testable and reused by every chart via `useTrendViewport`.
 */

export interface TrendViewport {
  paused: boolean;
  span: number;
  offset: number;
}

/** Oldest (`min`) and newest (`max`) available data positions, in domain units. */
export interface DomainBounds {
  min: number;
  max: number;
}

/** Allowed zoom range: the smallest and largest visible span, in domain units. */
export interface SpanLimits {
  minSpan: number;
  maxSpan: number;
}

/** The resolved on-screen window, in domain units. */
export interface VisibleRange {
  start: number;
  end: number;
}

function assertFinite(name: string, ...vals: number[]): void {
  for (const v of vals) {
    if (typeof v !== "number" || !Number.isFinite(v)) {
      throw new TypeError(`${name}: expected finite numbers`);
    }
  }
}

function clampSpan(span: number, limits: SpanLimits): number {
  return Math.max(limits.minSpan, Math.min(limits.maxSpan, span));
}

/** Farthest the right edge can sit behind `max` and still show data. */
function maxOffset(span: number, bounds: DomainBounds): number {
  return Math.max(0, bounds.max - bounds.min - span);
}

/**
 * A live viewport (following the edge) with the given visible span.
 *
 * @throws TypeError if `span` is not a positive finite number.
 */
export function createViewport(span: number): TrendViewport {
  assertFinite("createViewport", span);
  if (span <= 0) throw new RangeError("createViewport: span must be > 0");
  return { paused: false, span, offset: 0 };
}

/** True when the viewport tracks the newest data (not paused, not panned back). */
export function isLive(vp: TrendViewport): boolean {
  return !vp.paused && vp.offset === 0;
}

/**
 * Resolve the on-screen `[start, end]` window, clamped to the data and span
 * limits. `end = max - offset`, `start = end - span`, then nudged so the window
 * never runs past the oldest sample.
 *
 * @throws TypeError on non-finite bounds/limits.
 */
export function resolveVisible(
  vp: TrendViewport,
  bounds: DomainBounds,
  limits: SpanLimits,
): VisibleRange {
  assertFinite("resolveVisible", bounds.min, bounds.max, limits.minSpan, limits.maxSpan);
  if (bounds.max < bounds.min) {
    throw new RangeError("resolveVisible: bounds.max must be >= bounds.min");
  }
  const span = clampSpan(vp.span, limits);
  const offset = Math.max(0, Math.min(vp.offset, maxOffset(span, bounds)));
  let end = bounds.max - offset;
  let start = end - span;
  if (start < bounds.min) {
    start = bounds.min;
    end = start + span;
  }
  if (end > bounds.max) {
    end = bounds.max;
    start = end - span;
  }
  return { start, end };
}

/**
 * Pan by `deltaUnits` domain units — positive pans BACK into history, negative
 * toward the live edge. Clamps to `[0, maxOffset]`; reaching 0 resumes following.
 *
 * @throws TypeError if `deltaUnits` is not finite.
 */
export function panBy(
  vp: TrendViewport,
  deltaUnits: number,
  bounds: DomainBounds,
  limits: SpanLimits,
): TrendViewport {
  assertFinite("panBy", deltaUnits);
  const span = clampSpan(vp.span, limits);
  const offset = Math.max(
    0,
    Math.min(vp.offset + deltaUnits, maxOffset(span, bounds)),
  );
  return { ...vp, span, offset };
}

/**
 * Zoom by `factor` (<1 zooms IN / narrows the span, >1 zooms OUT) keeping the
 * `focus` position stationary on screen. Span is clamped to the limits; the
 * offset is recomputed so `focus` stays put.
 *
 * @throws TypeError on non-finite factor/focus; RangeError if factor <= 0.
 */
export function zoomBy(
  vp: TrendViewport,
  factor: number,
  focus: number,
  bounds: DomainBounds,
  limits: SpanLimits,
): TrendViewport {
  assertFinite("zoomBy", factor, focus);
  if (factor <= 0) throw new RangeError("zoomBy: factor must be > 0");
  const before = resolveVisible(vp, bounds, limits);
  const newSpan = clampSpan(vp.span * factor, limits);
  // Keep `focus` at the same fraction of the window after the zoom.
  const frac =
    before.end > before.start
      ? (focus - before.start) / (before.end - before.start)
      : 1;
  const newEnd = focus + (1 - frac) * newSpan;
  const offset = Math.max(
    0,
    Math.min(bounds.max - newEnd, maxOffset(newSpan, bounds)),
  );
  return { ...vp, span: newSpan, offset };
}

/**
 * Zoom to an explicit `[a, b]` domain range (a click-drag selection). Span
 * becomes `|b - a|` (clamped) and the right edge is placed at `max(a, b)`.
 *
 * @throws TypeError on non-finite a/b.
 */
export function zoomToRange(
  vp: TrendViewport,
  a: number,
  b: number,
  bounds: DomainBounds,
  limits: SpanLimits,
): TrendViewport {
  assertFinite("zoomToRange", a, b);
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  const span = clampSpan(hi - lo, limits);
  const offset = Math.max(
    0,
    Math.min(bounds.max - hi, maxOffset(span, bounds)),
  );
  return { ...vp, span, offset };
}

/** Reset to a live viewport at `span` (clears pause and pan). */
export function reset(span: number): TrendViewport {
  return createViewport(span);
}

/** Return a copy with the paused flag set. */
export function setPaused(vp: TrendViewport, paused: boolean): TrendViewport {
  if (typeof paused !== "boolean") {
    throw new TypeError("setPaused: paused must be a boolean");
  }
  return { ...vp, paused };
}
