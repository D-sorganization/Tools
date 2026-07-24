/**
 * Pure, DOM-free helpers for the shared plot crosshair / hover tooltip.
 *
 * Every chart in the HMI (the live trends and the Data Explorer plots) draws
 * bespoke SVG over different domains — sample-index, epoch-milliseconds, or
 * arbitrary data units. Rather than duplicate "which point is under the cursor
 * and where does the tooltip go" in each chart, these two primitives capture
 * exactly that, kept React-free so they are trivially unit-testable. Each chart
 * converts the cursor pixel to its own domain-x (it already has a `pxToUnit`
 * converter for pan/zoom) and hands the ascending x-values here.
 */

/** Pixel bounds of a plot's drawable area (inclusive edges). */
export interface PlotRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** A tooltip top-left position in pixels. */
export interface TooltipPos {
  x: number;
  y: number;
}

function assertFiniteNumber(v: unknown, name: string): asserts v is number {
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new TypeError(`${name} must be a finite number`);
  }
}

/**
 * Index of the sample whose x-coordinate is nearest to `targetX`.
 *
 * `xs` MUST be sorted ascending (the natural order of a time/index series), so
 * the search is O(log n) — cheap even on hour-long buffers. Values at or beyond
 * the ends clamp to the first/last index. An exact midpoint tie resolves to the
 * lower index for determinism. Returns `null` for an empty array.
 *
 * Precondition: `xs` is an array; `targetX` is a finite number.
 * @throws TypeError if `xs` is not an array or `targetX` is not finite.
 */
export function nearestIndexByX(
  xs: readonly number[],
  targetX: number,
): number | null {
  if (!Array.isArray(xs)) {
    throw new TypeError("xs must be an array");
  }
  assertFiniteNumber(targetX, "targetX");
  const n = xs.length;
  if (n === 0) return null;
  if (targetX <= xs[0]) return 0;
  if (targetX >= xs[n - 1]) return n - 1;

  let lo = 0;
  let hi = n - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const v = xs[mid];
    if (v === targetX) return mid;
    if (v < targetX) lo = mid + 1;
    else hi = mid - 1;
  }
  // Loop exits with hi === lo - 1: xs[hi] < targetX < xs[lo]. Pick the closer,
  // preferring the lower index on an exact tie.
  const lower = hi;
  const upper = lo;
  return targetX - xs[lower] <= xs[upper] - targetX ? lower : upper;
}

/**
 * Top-left pixel for a tooltip box of `size`, anchored near `anchor` and kept
 * fully inside `bounds`.
 *
 * Prefers up-and-to-the-right of the anchor (the conventional cursor tooltip
 * placement); flips to the left when it would overflow the right edge and below
 * when it would overflow the top, then clamps as a final guarantee so the box
 * never leaves the plot rectangle.
 *
 * Precondition: `anchor` coordinates are finite; `size` is non-negative.
 * @throws TypeError if `anchor` coordinates are not finite.
 * @throws RangeError if `size` has a negative dimension.
 */
export function placeTooltip(
  anchor: TooltipPos,
  size: { w: number; h: number },
  bounds: PlotRect,
  offset = 8,
): TooltipPos {
  assertFiniteNumber(anchor.x, "anchor.x");
  assertFiniteNumber(anchor.y, "anchor.y");
  if (size.w < 0 || size.h < 0) {
    throw new RangeError("size dimensions must be non-negative");
  }

  let x = anchor.x + offset;
  if (x + size.w > bounds.x1) {
    x = anchor.x - offset - size.w; // flip to the left of the anchor
  }
  x = Math.max(bounds.x0, Math.min(x, bounds.x1 - size.w));

  let y = anchor.y - size.h - offset; // prefer above the anchor
  if (y < bounds.y0) {
    y = anchor.y + offset; // flip below
  }
  y = Math.max(bounds.y0, Math.min(y, bounds.y1 - size.h));

  return { x, y };
}
