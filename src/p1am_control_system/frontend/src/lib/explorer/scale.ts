/**
 * Axis-scaling primitives for the Data Explorer charts.
 *
 * Small, pure helpers that map data values to pixel space and pick
 * human-friendly tick marks: {@link linearScale} (domain → range mapper),
 * {@link niceTicks} (a "nice" 1/2/5×10ⁿ tick sequence), {@link extent}
 * (min/max of a column, ignoring `null`/non-finite), and {@link clamp}.
 *
 * Dependency-free and unit-testable. Public functions validate their inputs
 * (DbC): non-finite bounds or a non-positive tick count throw.
 */

/** Maps a value from a numeric domain onto a numeric range (e.g. pixels). */
export interface LinearScale {
  (value: number): number;
  /** [domainMin, domainMax] supplied at construction. */
  readonly domain: readonly [number, number];
  /** [rangeMin, rangeMax] supplied at construction. */
  readonly range: readonly [number, number];
  /** Map a range value back to the domain. */
  invert(value: number): number;
}

/** Restrict `value` to the inclusive `[min, max]` interval. */
export function clamp(value: number, min: number, max: number): number {
  if (
    !Number.isFinite(value) ||
    !Number.isFinite(min) ||
    !Number.isFinite(max)
  ) {
    throw new TypeError("clamp: value, min and max must be finite numbers");
  }
  if (min > max) {
    throw new RangeError("clamp: min must be <= max");
  }
  return value < min ? min : value > max ? max : value;
}

/**
 * Build a linear scale mapping `domain` → `range`.
 *
 * A degenerate (zero-width) domain maps every value to the range midpoint so
 * callers never divide by zero.
 *
 * @throws TypeError if any bound is non-finite.
 */
export function linearScale(
  domain: readonly [number, number],
  range: readonly [number, number],
): LinearScale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  if (![d0, d1, r0, r1].every(Number.isFinite)) {
    throw new TypeError("linearScale: domain and range bounds must be finite");
  }
  const dSpan = d1 - d0;
  const rSpan = r1 - r0;
  const mid = (r0 + r1) / 2;

  const scale = ((value: number): number => {
    if (dSpan === 0) return mid;
    return r0 + ((value - d0) / dSpan) * rSpan;
  }) as LinearScale;

  Object.defineProperties(scale, {
    domain: { value: [d0, d1] as const, enumerable: true },
    range: { value: [r0, r1] as const, enumerable: true },
    invert: {
      value: (value: number): number => {
        if (rSpan === 0) return (d0 + d1) / 2;
        return d0 + ((value - r0) / rSpan) * dSpan;
      },
      enumerable: true,
    },
  });
  return scale;
}

/** Round a raw step up to the nearest "nice" 1/2/5/10 × 10ⁿ value. */
function niceStep(rawStep: number): number {
  const exponent = Math.floor(Math.log10(rawStep));
  const magnitude = Math.pow(10, exponent);
  const fraction = rawStep / magnitude;
  let niceFraction: number;
  if (fraction <= 1) niceFraction = 1;
  else if (fraction <= 2) niceFraction = 2;
  else if (fraction <= 5) niceFraction = 5;
  else niceFraction = 10;
  return niceFraction * magnitude;
}

/**
 * Produce up to ~`count` "nice" tick values spanning `[min, max]`.
 *
 * Ticks fall on multiples of a 1/2/5×10ⁿ step and include the rounded-out
 * endpoints. When `min === max`, a single tick at that value is returned.
 *
 * @throws TypeError if `min`/`max` are non-finite.
 * @throws RangeError if `count < 1` or `min > max`.
 */
export function niceTicks(min: number, max: number, count: number): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    throw new TypeError("niceTicks: min and max must be finite numbers");
  }
  if (!Number.isInteger(count) || count < 1) {
    throw new RangeError("niceTicks: count must be a positive integer");
  }
  if (min > max) {
    throw new RangeError("niceTicks: min must be <= max");
  }
  if (min === max) return [min];

  const step = niceStep((max - min) / count);
  const niceMin = Math.floor(min / step) * step;
  const niceMax = Math.ceil(max / step) * step;
  const ticks: number[] = [];
  // Guard against float drift accumulating; round to the step's decimals.
  const decimals = Math.max(0, -Math.floor(Math.log10(step)));
  for (let v = niceMin; v <= niceMax + step / 2; v += step) {
    ticks.push(Number(v.toFixed(decimals)));
  }
  return ticks;
}

/**
 * Min/max of a column, ignoring `null` and non-finite entries.
 *
 * Returns `[NaN, NaN]` when no finite value is present, so callers can detect
 * an empty/all-gap column.
 *
 * @throws TypeError if `values` is not an array.
 */
export function extent(values: (number | null)[]): [number, number] {
  if (!Array.isArray(values)) {
    throw new TypeError("extent: values must be an array");
  }
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v === null || !Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (min === Infinity) return [NaN, NaN];
  return [min, max];
}
