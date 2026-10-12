/**
 * Data freshness for live process values (#4010).
 *
 * The HMI used to express liveness as a boolean latched true by any successful
 * frame parse. Because every telemetry field is optional, a dead backend's
 * `{}` parsed cleanly, so the header stayed green and the trend appended the
 * same frozen value forever — which reads to an operator as a beautifully
 * stable process. A boolean simply cannot distinguish "stable" from "frozen".
 *
 * Liveness is therefore an AGE: milliseconds since the last frame that actually
 * carried recognised telemetry. These helpers are pure so the mapping from age
 * to what the operator is shown can be unit-tested without rendering.
 */

/**
 * Age at which live process values stop being trustworthy and are rendered
 * stale (greyed + cross-hatched). The backend broadcasts at ~10 Hz and the HTTP
 * snapshot fallback polls every 1.5 s, so 5 s is several missed opportunities —
 * long enough not to flicker on a loaded Pi, short enough that a frozen value
 * cannot be misread as a steady one.
 */
export const DATA_STALE_MS = 5000;

/** How the operator is told to read the current process values. */
export type DataFreshness =
  /** Fresh enough to act on. */
  | "live"
  /** Still displayed, but visibly marked as not updating. */
  | "stale"
  /** Nothing has ever arrived (or the age is unusable). */
  | "offline";

/**
 * Classify a data age.
 *
 * @param ageMs - milliseconds since the last recognised frame, or `undefined`
 *   when no frame has ever been applied.
 * @param staleMs - threshold at/after which data counts as stale.
 * @returns the freshness band to render.
 * @throws RangeError if `ageMs` is negative — an age cannot run backwards, and
 *   silently clamping one would resurrect the "always live" defect.
 */
export function dataFreshness(
  ageMs: number | undefined,
  staleMs: number = DATA_STALE_MS,
): DataFreshness {
  if (ageMs === undefined) return "offline";
  if (typeof ageMs !== "number") {
    throw new TypeError("ageMs must be a number or undefined");
  }
  if (ageMs < 0) throw new RangeError("ageMs must not be negative");
  // A non-finite age means the clock source is broken; fail toward "no data"
  // rather than toward "everything is fine".
  if (!Number.isFinite(ageMs)) return "offline";
  return ageMs < staleMs ? "live" : "stale";
}

/**
 * Render a data age for the status bar.
 *
 * Minutes are spelled out past 60 s so a twenty-minute freeze cannot be
 * mistaken for a twenty-second one at a glance on the Pi's touchscreen.
 *
 * @param ageMs - milliseconds since the last recognised frame, or `undefined`.
 * @returns a short human-readable age (`"—"` when nothing has ever arrived).
 */
export function formatDataAge(ageMs: number | undefined): string {
  if (ageMs === undefined || !Number.isFinite(ageMs) || ageMs < 0) return "—";
  const totalSeconds = Math.floor(ageMs / 1000);
  if (totalSeconds < 60) return `${totalSeconds} s`;
  const minutes = Math.floor(totalSeconds / 60);
  return `${minutes} m ${totalSeconds - minutes * 60} s`;
}

/**
 * Scan provenance reported by the backend in every telemetry frame (#5525).
 *
 * Single frontend source for the wire values; they mirror
 * `DataSource` in `backend/models.py` (locked by a contract test). Only `live`
 * is a plain measurement: `simulated`, `held` and `fault` scans can arrive on
 * time yet carry no real reading, so frame age alone must never read as healthy.
 */
export const DATA_SOURCES = ["live", "simulated", "held", "fault"] as const;
export type DataSource = (typeof DATA_SOURCES)[number];

/** What the status pill shows: freshness bands plus every non-live source. */
export type DataStatus =
  | "live"
  | "stale"
  | "offline"
  | Exclude<DataSource, "live">
  /** Fresh frame whose `data_source` is missing or unrecognised (fail safe). */
  | "unknown";

function isDataSource(value: unknown): value is DataSource {
  return (DATA_SOURCES as readonly unknown[]).includes(value);
}

/**
 * Combine frame age with the scan's reported source.
 *
 * Age gates first: a stale or offline stream is stale/offline whatever source
 * it last reported. Only a fresh frame is classified by source, and anything
 * but a recognised value is `unknown`, never `live` — the HMI must not infer
 * health from the absence of a fault flag.
 *
 * @param freshness - the age band from {@link dataFreshness}.
 * @param source - raw `data_source` from the latest frame, if any.
 */
export function dataStatus(
  freshness: DataFreshness,
  source: string | undefined,
): DataStatus {
  if (freshness !== "live") return freshness;
  return isDataSource(source) ? source : "unknown";
}
