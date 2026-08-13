/**
 * Shared Y-axis range model for the trend plots.
 *
 * Each trend keeps an {@link AxisRange} in state and resolves it against the
 * live data via {@link resolveRange}: "auto" fits the visible samples (with a
 * little headroom) so a small signal isn't squashed against a fixed full-scale
 * axis, while "manual" honors the operator's min/max. The matching control row
 * is rendered by `TrendAxisControls`.
 */

export interface AxisRange {
  /** When true, fit the axis to the visible data; otherwise use min/max. */
  auto: boolean;
  /** Manual lower bound (used when auto is false). */
  min: number;
  /** Manual upper bound (used when auto is false). */
  max: number;
}

/** A sensible starting range for a trend (auto on, manual bounds = the fixed
 * full-scale the plot used before this feature, so toggling to manual is a
 * no-op surprise-free starting point). */
export function defaultAxisRange(min: number, max: number): AxisRange {
  return { auto: true, min, max };
}

/**
 * Resolve the effective [min, max] for a trend's Y axis.
 *
 * - manual: returns the operator's bounds, guarding against max <= min.
 * - auto with data: fits min/max of the samples with ~8% padding; a flat trace
 *   gets a small ±1 band so it doesn't collapse to a zero-height line.
 * - auto with no data: falls back to the provided defaults.
 */
export function resolveRange(
  range: AxisRange,
  values: number[],
  defaults: { min: number; max: number },
): { min: number; max: number } {
  if (!range.auto) {
    return {
      min: range.min,
      max: range.max > range.min ? range.max : range.min + 1,
    };
  }
  if (values.length === 0) return defaults;
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (hi - lo < 1e-9) {
    lo -= 1;
    hi += 1;
  } else {
    const pad = (hi - lo) * 0.08;
    lo -= pad;
    hi += pad;
  }
  return { min: lo, max: hi };
}

/** Evenly spaced tick values across [min, max] (inclusive), `count` segments. */
export function axisTicks(min: number, max: number, count = 4): number[] {
  if (!(max > min)) return [min];
  return Array.from({ length: count + 1 }, (_, i) => min + ((max - min) * i) / count);
}
