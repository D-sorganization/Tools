/**
 * Color palette for the Data Explorer charts.
 *
 * A fixed, colorblind-friendly categorical palette ({@link SERIES_COLORS}) with
 * {@link colorForIndex} cycling through it for any series count, plus
 * {@link divergingColor} for a blue↔grey↔red diverging ramp (e.g. correlation
 * heatmaps) parameterized on `t ∈ [-1, 1]`.
 *
 * Pure, dependency-free, and unit-testable. Inputs are validated (DbC).
 */

/**
 * Categorical series colors (Okabe–Ito-derived, distinguishable for common
 * forms of color-vision deficiency). Cycled by {@link colorForIndex}.
 */
export const SERIES_COLORS: readonly string[] = [
  "#1f77b4", // blue
  "#ff7f0e", // orange
  "#2ca02c", // green
  "#d62728", // red
  "#9467bd", // purple
  "#8c564b", // brown
  "#e377c2", // pink
  "#17becf", // cyan
  "#bcbd22", // olive
  "#7f7f7f", // grey
] as const;

/**
 * Color for the i-th series, cycling through {@link SERIES_COLORS}.
 *
 * @throws TypeError if `i` is not a non-negative integer.
 */
export function colorForIndex(i: number): string {
  if (!Number.isInteger(i) || i < 0) {
    throw new TypeError("colorForIndex: i must be a non-negative integer");
  }
  return SERIES_COLORS[i % SERIES_COLORS.length];
}

/** Clamp + format an RGB channel to a two-digit hex byte. */
function hexByte(channel: number): string {
  const v = Math.round(channel < 0 ? 0 : channel > 255 ? 255 : channel);
  return v.toString(16).padStart(2, "0");
}

/** Linear interpolation between two RGB triples. */
function lerpRgb(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  t: number,
): string {
  const r = a[0] + (b[0] - a[0]) * t;
  const g = a[1] + (b[1] - a[1]) * t;
  const bch = a[2] + (b[2] - a[2]) * t;
  return `#${hexByte(r)}${hexByte(g)}${hexByte(bch)}`;
}

const NEG: readonly [number, number, number] = [33, 102, 172]; // blue
const MID: readonly [number, number, number] = [247, 247, 247]; // near-white
const POS: readonly [number, number, number] = [178, 24, 43]; // red

/**
 * Diverging blue↔white↔red color for `t ∈ [-1, 1]`.
 *
 * `t = -1` is blue, `t = 0` is the near-white midpoint, `t = +1` is red.
 * Values outside `[-1, 1]` are clamped.
 *
 * @throws TypeError if `t` is not a finite number.
 */
export function divergingColor(t: number): string {
  if (!Number.isFinite(t)) {
    throw new TypeError("divergingColor: t must be a finite number");
  }
  const clamped = t < -1 ? -1 : t > 1 ? 1 : t;
  if (clamped < 0) {
    // -1..0 -> blue..mid
    return lerpRgb(NEG, MID, clamped + 1);
  }
  // 0..1 -> mid..red
  return lerpRgb(MID, POS, clamped);
}
