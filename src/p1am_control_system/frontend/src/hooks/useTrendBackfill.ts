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
): BackfillPoint[] {
  const [points, setPoints] = useState<BackfillPoint[]>([]);

  useEffect(() => {
    if (!Number.isFinite(tagId) || tagId < 0) return;
    let cancelled = false;
    const now = Date.now();
    const url =
      `/api/trends?tag_id=${tagId}` +
      `&start_time=${encodeURIComponent(naiveUtcIso(now - windowSeconds * 1000))}` +
      `&end_time=${encodeURIComponent(naiveUtcIso(now))}`;

    fetchWithTimeout(url)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !d?.timestamps) return;
        const ts: string[] = d.timestamps;
        const vs: number[] = d.values ?? [];
        setPoints(
          ts
            .map((iso, i) => ({ t: parseHistorianTs(iso), v: (vs[i] ?? 0) * scale }))
            .filter((p) => Number.isFinite(p.t)),
        );
      })
      .catch(() => {
        if (!cancelled) setPoints([]);
      });

    return () => {
      cancelled = true;
    };
  }, [tagId, windowSeconds, scale]);

  return points;
}
