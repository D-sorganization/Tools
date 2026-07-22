/**
 * Pure helpers for the live-link diagnostics feed.
 *
 * The operator needs to troubleshoot the data path (PLC → backend → HMI) at a
 * glance: is the stream live, how fast is it actually arriving, how stale is the
 * last frame, and what are the most recent raw tag values coming off the PLC.
 * These are React/DOM-free so the rate/staleness/formatting logic is unit-tested
 * and reused wherever the diagnostics feed is shown (DRY).
 */

/**
 * Effective sample rate (Hz) from the last up-to-`window` frame timestamps
 * (epoch ms, ascending). Returns 0 for fewer than two frames or a zero span.
 * Uses a trailing window so a temporary slow-down shows immediately rather than
 * being averaged away over the whole session.
 *
 * @throws TypeError if `times` is not an array.
 */
export function effectiveHz(times: readonly number[], window = 20): number {
  if (!Array.isArray(times)) throw new TypeError("times must be an array");
  const n = times.length;
  if (n < 2) return 0;
  const start = Math.max(0, n - Math.max(2, Math.floor(window)));
  const spanS = (times[n - 1] - times[start]) / 1000;
  return spanS > 0 ? (n - 1 - start) / spanS : 0;
}

/**
 * Milliseconds since the most recent frame (stream staleness). `Infinity` when
 * there are no frames yet. Clamped to ≥ 0 so a mildly skewed clock never reads
 * negative.
 *
 * @throws TypeError if `times` is not an array or `now` is not finite.
 */
export function stalenessMs(times: readonly number[], now: number): number {
  if (!Array.isArray(times)) throw new TypeError("times must be an array");
  if (typeof now !== "number" || !Number.isFinite(now)) {
    throw new TypeError("now must be a finite number");
  }
  if (times.length === 0) return Number.POSITIVE_INFINITY;
  return Math.max(0, now - times[times.length - 1]);
}

/** Compact staleness label: "0.1s", "3.4s", "12s", or "—" when never seen. */
export function formatStaleness(ms: number): string {
  if (!Number.isFinite(ms)) return "—";
  const s = ms / 1000;
  if (s < 10) return `${s.toFixed(1)}s`;
  return `${Math.round(s)}s`;
}

/** Local wall clock with milliseconds ("HH:MM:SS.mmm") for a diagnostics line. */
export function formatClockMillis(ms: number): string {
  const d = new Date(ms);
  const p = (n: number, w = 2): string => String(n).padStart(w, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}.${p(
    d.getMilliseconds(),
    3,
  )}`;
}

/**
 * Format one raw feed line: the frame clock time plus the given tag indices as
 * fixed-decimal `Ti=value` fields. A `0` reads as a possible dropped sample, so
 * they are shown verbatim (never hidden) — this is the troubleshooting view.
 *
 * @throws TypeError if `tags` or `indices` is not an array.
 */
export function formatFeedLine(
  timeMs: number,
  tags: readonly number[],
  indices: readonly number[],
  decimals = 1,
): string {
  if (!Array.isArray(tags)) throw new TypeError("tags must be an array");
  if (!Array.isArray(indices)) throw new TypeError("indices must be an array");
  const fields = indices
    .map((i) => `T${i}=${Number.isFinite(tags[i]) ? tags[i].toFixed(decimals) : "·"}`)
    .join("  ");
  return `${formatClockMillis(timeMs)}  ${fields}`;
}
