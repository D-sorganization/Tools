import {
  finiteLaunchMonitorScalar,
  type LaunchMonitorRow,
} from "./launchMonitorAnalysisTypes";

export const MAX_LINKED_SCATTER_POINTS = 2_000;
export const MAX_LINKED_SCATTER_ROWS = 250_000;

export interface LinkedScatterPoint {
  readonly rawIndex: number;
  readonly x: number;
  readonly y: number;
  readonly shotId: string | null;
  readonly sessionId: string | null;
  readonly monitorVendor: string | null;
}

export interface LinkedScatterPlan {
  readonly xField: string;
  readonly yField: string;
  readonly rawCount: number;
  readonly finiteCount: number;
  readonly displayedCount: number;
  readonly selectedRawIndex: number | null;
  readonly points: readonly LinkedScatterPoint[];
}

export interface PlotAxisProjection {
  readonly coordinates: readonly number[];
  readonly scale: number;
}

export function projectPlotAxis(values: readonly number[]): PlotAxisProjection {
  if (!Array.isArray(values) || values.length === 0) {
    throw new RangeError("plot axis values must be a nonempty finite number sequence");
  }

  let low = values[0];
  let high = values[0];
  let maxAbs = Math.abs(values[0]);

  if (!Number.isFinite(low)) {
    throw new RangeError("plot axis values must be a nonempty finite number sequence");
  }

  // ⚡ Bolt Optimization: Replace multiple spread operators Math.max(...values) and chained map calls
  // with a single-pass loop to avoid massive call stack expansions and heavy garbage collection pressure.
  for (let i = 1; i < values.length; i++) {
    const val = values[i];
    if (!Number.isFinite(val)) {
      throw new RangeError("plot axis values must be a nonempty finite number sequence");
    }
    if (val < low) low = val;
    if (val > high) high = val;
    const absVal = Math.abs(val);
    if (absVal > maxAbs) maxAbs = absVal;
  }

  const scale = maxAbs;

  if (scale === 0 || low === high) {
    const coords = new Array(values.length);
    for (let i = 0; i < values.length; i++) coords[i] = 0;
    return Object.freeze({ coordinates: Object.freeze(coords), scale: scale === 0 ? 1 : scale });
  }

  const useScaledBasis = low < 0 && high > 0;
  const basisLow = useScaledBasis ? low / scale : low;
  const basisHigh = useScaledBasis ? high / scale : high;
  const span = basisHigh - basisLow;

  const coordinates = new Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const basisVal = useScaledBasis ? values[i] / scale : values[i];
    coordinates[i] = 2 * ((basisVal - basisLow) / span) - 1;
  }

  return Object.freeze({
    coordinates: Object.freeze(coordinates), scale,
  });
}

const text = (value: unknown): string | null =>
  typeof value === "string" && value.length > 0 ? value : null;

export function planLinkedScatter(
  rows: readonly LaunchMonitorRow[], xField: string, yField: string,
  selectedRawIndex: number | null = null, cap = MAX_LINKED_SCATTER_POINTS,
): LinkedScatterPlan {
  if (typeof xField !== "string" || typeof yField !== "string" ||
      !xField || !yField || xField === yField) {
    throw new RangeError("linked scatter requires two distinct field names");
  }
  if (!Array.isArray(rows)) {
    throw new RangeError("linked scatter rows must be a retained record sequence");
  }
  if (!Number.isSafeInteger(cap) || cap < 2 || cap > MAX_LINKED_SCATTER_POINTS) {
    throw new RangeError("linked scatter cap must be an integer from 2 through 2000");
  }
  if (rows.length > MAX_LINKED_SCATTER_ROWS) {
    throw new RangeError(`linked scatter retains at most ${MAX_LINKED_SCATTER_ROWS} rows`);
  }
  if (selectedRawIndex !== null && (!Number.isSafeInteger(selectedRawIndex) ||
      selectedRawIndex < 0 || selectedRawIndex >= rows.length)) {
    throw new RangeError("selected raw row index is outside the retained records");
  }
  let finiteCount = 0;
  let selectedIsFinite = false;
  let selectedBucket: number | null = null;
  const candidates = new Map<number, LinkedScatterPoint>();
  for (let rawIndex = 0; rawIndex < rows.length; rawIndex += 1) {
    const row = rows[rawIndex];
    if (typeof row !== "object" || row === null || Array.isArray(row)) {
      throw new RangeError("each linked scatter row must be a record");
    }
    const x = finiteLaunchMonitorScalar(row[xField]);
    const y = finiteLaunchMonitorScalar(row[yField]);
    if (x === null || y === null) continue;
    finiteCount += 1;
    const bucket = rows.length <= cap ? rawIndex : Math.floor(rawIndex * cap / rows.length);
    if (rawIndex === selectedRawIndex) {
      candidates.set(bucket, {
        rawIndex, x, y, shotId: text(row.shot_id), sessionId: text(row.session_id),
        monitorVendor: text(row.monitor_vendor),
      });
      selectedBucket = bucket;
      selectedIsFinite = true;
    } else if (!candidates.has(bucket) && bucket !== selectedBucket) {
      candidates.set(bucket, {
        rawIndex, x, y, shotId: text(row.shot_id), sessionId: text(row.session_id),
        monitorVendor: text(row.monitor_vendor),
      });
    }
  }
  const points = [...candidates.entries()].sort(([left], [right]) => left - right)
    .map(([, point]) => point);
  return Object.freeze({
    xField, yField, rawCount: rows.length, finiteCount, displayedCount: points.length,
    selectedRawIndex: selectedIsFinite ? selectedRawIndex : null,
    points: Object.freeze(points.map((point) => Object.freeze(point))),
  });
}

export type LinkedScatterNavigation = "previous" | "next" | "home" | "end" | "clear";

export function navigateLinkedScatter(
  plan: LinkedScatterPlan, currentRawIndex: number | null, command: LinkedScatterNavigation,
): number | null {
  if (!["previous", "next", "home", "end", "clear"].includes(command)) {
    throw new RangeError("unknown linked scatter navigation command");
  }
  const indices = plan.points.map((point) => point.rawIndex);
  if (command === "clear" || indices.length === 0) return null;
  if (command === "home") return indices[0];
  if (command === "end") return indices[indices.length - 1];
  const position = indices.indexOf(currentRawIndex ?? -1);
  if (position < 0) return command === "next" ? indices[0] : indices[indices.length - 1];
  const offset = command === "next" ? 1 : -1;
  return indices[(position + offset + indices.length) % indices.length];
}
