/**
 * Shared time-window (X-axis) model for the trend plots.
 *
 * The trends buffer samples at the ~10 Hz broadcast rate; a trend keeps a
 * `windowSeconds` selection and slices the trailing `windowSeconds * 10`
 * samples to plot. `downsample` then caps the points actually drawn so a long
 * window stays cheap to render on slow hardware (the Pi). The matching button
 * row is rendered by `TrendTimeControls`.
 */

export const SAMPLES_PER_SECOND = 10;

/** Selectable windows (seconds): 30 s, 1 m, 2 m, 5 m. */
export const TIME_WINDOW_PRESETS = [30, 60, 120, 300] as const;

/** Deepest buffer any trend needs to satisfy the longest preset. */
export const MAX_TREND_SAMPLES = 300 * SAMPLES_PER_SECOND; // 5 min @ 10 Hz

/** Max points actually drawn per trace — keeps SVG paths light on the Pi. */
export const RENDER_MAX_POINTS = 600;

/** "30s" / "2m" label for a window in seconds. */
export function formatWindow(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = seconds / 60;
  return `${Number.isInteger(m) ? m : m.toFixed(1)}m`;
}

/** Trailing sample count for a window (clamped to what the buffer holds). */
export function windowSamples(seconds: number): number {
  return Math.max(2, Math.round(seconds * SAMPLES_PER_SECOND));
}

/**
 * Stride-sample `values` down to at most `maxPoints`, always keeping the final
 * sample so the live edge stays accurate. Returns the input untouched when it's
 * already short enough. Preserves the X (time) proportions of the trace.
 */
export function downsample(values: number[], maxPoints = RENDER_MAX_POINTS): number[] {
  if (values.length <= maxPoints || maxPoints < 2) return values;
  const stride = values.length / maxPoints;
  const out: number[] = [];
  for (let i = 0; i < maxPoints; i++) {
    out.push(values[Math.floor(i * stride)]);
  }
  out[out.length - 1] = values[values.length - 1];
  return out;
}
