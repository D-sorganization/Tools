import { useEffect, useState } from "react";
import { fetchWithTimeout } from "../lib/fetchWithTimeout";

/**
 * Backfill a trend from the backend historian so changing the time window shows
 * data from *before* the page was opened — instead of only what has streamed in
 * since. Fetches the last `windowSeconds` of one tag from `/api/trends` and
 * re-fetches whenever the window (or tag/scale) changes.
 *
 * Timezone care: the historian stores **naive UTC** timestamps, so request
 * bounds are sent without a zone and responses are parsed as UTC (not the
 * browser's local zone) to line up with the live, Date.now()-stamped buffer.
 */

export interface BackfillPoint {
  /** Epoch ms. */
  t: number;
  /** Value, already multiplied by `scale`. */
  v: number;
}

/** Naive-UTC ISO (drop the trailing "Z") to match the historian's format. */
export function naiveUtcIso(epochMs: number): string {
  return new Date(epochMs).toISOString().replace("Z", "");
}

/** Parse a historian timestamp as UTC even when it carries no zone suffix. */
export function parseHistorianTs(iso: string): number {
  return Date.parse(/([zZ]|[+-]\d\d:?\d\d)$/.test(iso) ? iso : `${iso}Z`);
}

export function useTrendBackfill(
  tagId: number,
  windowSeconds: number,
  scale = 1,
  maxPoints?: number,
): BackfillPoint[] {
  const [points, setPoints] = useState<BackfillPoint[]>([]);

  useEffect(() => {
    if (!Number.isFinite(tagId) || tagId < 0) return;
    let cancelled = false;
    const now = Date.now();
    // Bound the historian response so a multi-hour window returns a light,
    // whole-span decimated series (the server spreads max_points across the
    // full [start,end] instead of clipping to the newest rows).
    const cap =
      typeof maxPoints === "number" && Number.isFinite(maxPoints) && maxPoints > 0
        ? `&max_points=${Math.floor(maxPoints)}`
        : "";
    const url =
      `/api/trends?tag_id=${tagId}` +
      `&start_time=${encodeURIComponent(naiveUtcIso(now - windowSeconds * 1000))}` +
      `&end_time=${encodeURIComponent(naiveUtcIso(now))}` +
      cap;

    fetchWithTimeout(url)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !d?.timestamps) return;
        const ts: string[] = d.timestamps;
        const vs: number[] = d.values ?? [];
        // ⚡ Bolt Optimization: Replace chained .map().filter() passes with a single-pass for loop
        // to avoid intermediate array allocations and closure overhead on large historian backfill arrays.
        const pts: BackfillPoint[] = [];
        for (let i = 0; i < ts.length; i++) {
          const t = parseHistorianTs(ts[i]);
          if (Number.isFinite(t)) {
            pts.push({ t, v: (vs[i] ?? 0) * scale });
          }
        }
        setPoints(pts);
      })
      .catch(() => {
        if (!cancelled) setPoints([]);
      });

    return () => {
      cancelled = true;
    };
  }, [tagId, windowSeconds, scale, maxPoints]);

  return points;
}
