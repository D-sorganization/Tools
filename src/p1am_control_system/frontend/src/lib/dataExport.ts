/**
 * Shared CSV-export model for the historian (`/api/export`).
 *
 * One place builds the export URL for any panel in the GUI: pick a tag set and
 * a time range, get a `/api/export?tag_ids=&start_time=&end_time=` URL the
 * browser can download. Pure functions with explicit precondition checks (DbC)
 * and no React/DOM, so the URL/range math is unit-testable and reused by both
 * the per-panel ExportButton and the sidebar CsvExporter (DRY).
 */

export interface ExportRange {
  /** Inclusive window start (epoch ms). */
  startMs: number;
  /** Inclusive window end (epoch ms). */
  endMs: number;
}

export type RangePresetId = "15m" | "1h" | "6h" | "today" | "all";

export interface RangePreset {
  id: RangePresetId;
  label: string;
}

export const RANGE_PRESETS: readonly RangePreset[] = [
  { id: "15m", label: "Last 15 min" },
  { id: "1h", label: "Last hour" },
  { id: "6h", label: "Last 6 h" },
  { id: "today", label: "Today" },
  { id: "all", label: "All captured" },
] as const;

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;

/**
 * Resolve a preset to an absolute [startMs, endMs] window ending at `nowMs`.
 *
 * @throws TypeError if `nowMs` is not finite.
 * @throws RangeError if `preset` is not a known preset id.
 */
export function resolveRange(preset: RangePresetId, nowMs: number): ExportRange {
  if (!Number.isFinite(nowMs)) {
    throw new TypeError("resolveRange: nowMs must be a finite number");
  }
  switch (preset) {
    case "15m":
      return { startMs: nowMs - 15 * MINUTE, endMs: nowMs };
    case "1h":
      return { startMs: nowMs - HOUR, endMs: nowMs };
    case "6h":
      return { startMs: nowMs - 6 * HOUR, endMs: nowMs };
    case "today": {
      const midnight = new Date(nowMs);
      midnight.setHours(0, 0, 0, 0);
      return { startMs: midnight.getTime(), endMs: nowMs };
    }
    case "all":
      return { startMs: 0, endMs: nowMs };
    default:
      throw new RangeError(`resolveRange: unknown preset ${preset}`);
  }
}

/** Comma-joined, trimmed, de-duplicated (order-preserving) tag-id list. */
export function formatTagIds(tags: ReadonlyArray<number | string>): string {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const tag of tags) {
    const s = String(tag).trim();
    if (s && !seen.has(s)) {
      seen.add(s);
      out.push(s);
    }
  }
  return out.join(",");
}

/**
 * Build the `/api/export` URL for a tag set + time range.
 *
 * @throws RangeError if no usable tags are given or the range is empty/inverted.
 */
export function buildExportUrl(
  tags: ReadonlyArray<number | string>,
  range: ExportRange,
): string {
  const tagList = formatTagIds(tags);
  if (!tagList) {
    throw new RangeError("buildExportUrl: at least one tag is required");
  }
  if (!(range.endMs > range.startMs)) {
    throw new RangeError("buildExportUrl: endMs must be after startMs");
  }
  const start = new Date(range.startMs).toISOString();
  const end = new Date(range.endMs).toISOString();
  return (
    `/api/export?tag_ids=${encodeURIComponent(tagList)}` +
    `&start_time=${encodeURIComponent(start)}` +
    `&end_time=${encodeURIComponent(end)}`
  );
}
