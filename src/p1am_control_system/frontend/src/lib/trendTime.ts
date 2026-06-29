/**
 * Shared time-window (X-axis) model for the trend plots.
 *
 * The trends buffer samples at the ~10 Hz broadcast rate. A trend keeps a
 * `windowSeconds` selection, slices the trailing samples for that window, and
 * `downsample`s to a render-friendly point count so a long window stays cheap
 * to draw on the Pi. {@link timeAxisTicks} turns the visible span into labelled
 * X-axis ticks. The matching value+unit input is `TrendTimeControls`.
 *
 * Design notes: pure functions with explicit precondition checks (DbC); no
 * React or DOM here so the math is unit-testable and reusable across every
 * trend (DRY).
 */

export const SAMPLES_PER_SECOND = 10;

/** Deepest buffer/window the trends retain: 1 hour @ 10 Hz. */
export const MAX_TREND_SAMPLES = 60 * 60 * SAMPLES_PER_SECOND;
export const MAX_WINDOW_SECONDS = MAX_TREND_SAMPLES / SAMPLES_PER_SECOND;
export const MIN_WINDOW_SECONDS = 1;

/** Max points actually drawn per trace — keeps SVG paths light on the Pi. */
export const RENDER_MAX_POINTS = 600;

export type TimeUnit = "s" | "m" | "h";

export const TIME_UNITS: readonly TimeUnit[] = ["s", "m", "h"] as const;

/** Units offered in the window dropdown — seconds is excluded by design. */
export const SELECTABLE_TIME_UNITS: readonly TimeUnit[] = ["m", "h"] as const;

const UNIT_SECONDS: Record<TimeUnit, number> = { s: 1, m: 60, h: 3600 };
const UNIT_LABEL: Record<TimeUnit, string> = { s: "sec", m: "min", h: "hr" };

/** Long label for a unit, e.g. for a <select>. */
export function unitLabel(unit: TimeUnit): string {
  return UNIT_LABEL[unit];
}

/**
 * Convert a value in `unit` to seconds.
 * @throws TypeError if value is not finite, RangeError if negative.
 */
export function toSeconds(value: number, unit: TimeUnit): number {
  if (!Number.isFinite(value)) {
    throw new TypeError("toSeconds: value must be a finite number");
  }
  if (value < 0) {
    throw new RangeError("toSeconds: value must be >= 0");
  }
  return value * UNIT_SECONDS[unit];
}

/** Express a window (seconds) in `unit`. Inverse of {@link toSeconds}. */
export function fromSeconds(seconds: number, unit: TimeUnit): number {
  return seconds / UNIT_SECONDS[unit];
}

/** Clamp a window to the supported [MIN, MAX] range. */
export function clampWindow(seconds: number): number {
  if (!Number.isFinite(seconds)) return MIN_WINDOW_SECONDS;
  return Math.max(MIN_WINDOW_SECONDS, Math.min(MAX_WINDOW_SECONDS, seconds));
}

/** Trailing sample count for a window (≥2 so a path can be drawn). */
export function windowSamples(seconds: number): number {
  return Math.max(2, Math.round(seconds * SAMPLES_PER_SECOND));
}

/** The natural unit to display a window in (h ≥ 1 h, m ≥ 1 m, else s). */
export function naturalUnit(seconds: number): TimeUnit {
  if (seconds >= 3600) return "h";
  if (seconds >= 60) return "m";
  return "s";
}

/** Default dropdown unit for a window — minutes, or hours at ≥ 1 h. */
export function windowUnit(seconds: number): TimeUnit {
  return seconds >= 3600 ? "h" : "m";
}

/** Compact window label: "30s" / "5m" / "1.5h". */
export function formatWindow(seconds: number): string {
  const u = naturalUnit(seconds);
  const v = fromSeconds(seconds, u);
  const txt = Number.isInteger(v) ? `${v}` : v.toFixed(1);
  return `${txt}${u}`;
}

/** Clock-style elapsed label: "0s","45s","1m30s","2m","1h05m". */
export function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) {
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return rem ? `${m}m${String(rem).padStart(2, "0")}s` : `${m}m`;
  }
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return m ? `${h}h${String(m).padStart(2, "0")}m` : `${h}h`;
}

export interface TimeTick {
  /** 0 = left edge (oldest), 1 = right edge (now). */
  frac: number;
  /** "-5m" … "now". */
  label: string;
}

/**
 * Evenly spaced X-axis ticks across a visible span. The right edge is "now";
 * earlier ticks are labelled with how long ago they were (e.g. "-2m30s").
 */
export function timeAxisTicks(spanSeconds: number, count = 4): TimeTick[] {
  const n = Math.max(1, count);
  return Array.from({ length: n + 1 }, (_, i) => {
    const frac = i / n;
    const secondsAgo = spanSeconds * (1 - frac);
    return {
      frac,
      label: secondsAgo < 0.5 ? "now" : `-${formatDuration(secondsAgo)}`,
    };
  });
}

/** Visible time span (seconds) for a trailing slice of `sampleCount` samples.
 * Approximate (assumes the nominal rate); prefer {@link elapsedSeconds} when
 * real per-sample timestamps are available. */
export function spanSeconds(sampleCount: number): number {
  return Math.max(0, (sampleCount - 1) / SAMPLES_PER_SECOND);
}

/** Exact visible span (seconds) from ascending per-sample timestamps (ms). */
export function elapsedSeconds(times: number[]): number {
  if (times.length < 2) return 0;
  return Math.max(0, (times[times.length - 1] - times[0]) / 1000);
}

/**
 * First index whose timestamp falls within `windowSeconds` of the latest
 * sample — i.e. the start of the trailing time window. `times` must be
 * ascending (append-only buffers are). Binary search; returns 0 when the whole
 * buffer fits the window. This makes windowing rate-independent: a "5 min"
 * window is exactly 5 min of wall-clock even when the Pi polls below 10 Hz.
 */
export function windowStartIndex(times: number[], windowSeconds: number): number {
  const n = times.length;
  if (n === 0) return 0;
  const cutoff = times[n - 1] - windowSeconds * 1000;
  let lo = 0;
  let hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < cutoff) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * Stride-sample `values` down to at most `maxPoints`, always keeping the final
 * element so the live edge stays accurate. Returns the input untouched when
 * already short enough. Generic so it works on scalar series or history rows.
 * Preserves the X (time) proportions of the data.
 */
export function downsample<T>(values: T[], maxPoints = RENDER_MAX_POINTS): T[] {
  if (values.length <= maxPoints || maxPoints < 2) return values;
  const stride = values.length / maxPoints;
  const out: T[] = [];
  for (let i = 0; i < maxPoints; i++) {
    out.push(values[Math.floor(i * stride)]);
  }
  out[out.length - 1] = values[values.length - 1];
  return out;
}
