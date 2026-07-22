/**
 * Bridge isolated dropout samples in a trend series before plotting.
 *
 * The PLC occasionally returns a spurious 0 for a live tag — a dropped Modbus
 * read — which the raw historian stores and the trends would otherwise draw as a
 * misleading instantaneous dip to zero and back (the vessel cannot physically go
 * 475 °C → 0 → 475 in 100 ms). The backend deglitch filter already protects the
 * CONTROL path and the filtered status values, but the RAW tag stream (TrendChart
 * and historian backfill) is unfiltered, so those dropouts reach the plot.
 *
 * This detects an ISOLATED run (≤ `maxRun` samples) at/near zero whose bracketing
 * samples are both clearly non-zero, and linearly interpolates across it so the
 * line stays smooth. Two properties keep it from ever inventing data:
 *   - A sustained zero (an unused / genuinely-zero channel) or an edge run has no
 *     valid bracket, so it is left untouched.
 *   - A true zero-crossing of a bipolar signal (e.g. +10, 0, −10) interpolates
 *     back to ≈0 — the real value — because the neighbours straddle zero; only a
 *     same-level bracket (475, 0, 475) is lifted back to the signal level.
 *
 * Pure + DOM-free so it is unit-testable and shared by every raw-tag trend (DRY).
 */

export interface BridgeOptions {
  /** |v| ≤ floor counts as "near zero" (a candidate dropout). */
  floor?: number;
  /** Both bracketing samples must have |v| ≥ minNeighbor to bridge. */
  minNeighbor?: number;
  /** Longest run of near-zero samples still treated as a dropout. */
  maxRun?: number;
}

function assertPositive(v: number, name: string): void {
  if (typeof v !== "number" || !Number.isFinite(v) || v < 0) {
    throw new TypeError(`${name} must be a non-negative finite number`);
  }
}

/**
 * Return a copy of `values` with isolated near-zero dropouts interpolated across.
 *
 * Precondition: `values` is an array of numbers; options are non-negative.
 * @throws TypeError on a non-array input or a negative/non-finite option.
 */
export function bridgeIsolatedDropouts(
  values: readonly number[],
  opts: BridgeOptions = {},
): number[] {
  if (!Array.isArray(values)) throw new TypeError("values must be an array");
  const floor = opts.floor ?? 1;
  const minNeighbor = opts.minNeighbor ?? 5;
  const maxRun = opts.maxRun ?? 2;
  assertPositive(floor, "floor");
  assertPositive(minNeighbor, "minNeighbor");
  assertPositive(maxRun, "maxRun");

  const out = values.slice();
  const n = out.length;
  const nearZero = (v: number): boolean => Number.isFinite(v) && Math.abs(v) <= floor;
  const solid = (v: number): boolean => Number.isFinite(v) && Math.abs(v) >= minNeighbor;

  let i = 0;
  while (i < n) {
    if (!nearZero(out[i])) {
      i++;
      continue;
    }
    // Extent of this near-zero run.
    let j = i;
    while (j < n && nearZero(out[j])) j++;
    const runLen = j - i;
    const before = i - 1;
    const after = j;
    if (runLen <= maxRun && before >= 0 && after < n && solid(out[before]) && solid(out[after])) {
      const v0 = out[before];
      const v1 = out[after];
      const span = after - before;
      for (let k = i; k < j; k++) {
        out[k] = v0 + (v1 - v0) * ((k - before) / span);
      }
    }
    i = j;
  }
  return out;
}

/**
 * Apply {@link bridgeIsolatedDropouts} to the `v` field of a timestamped series,
 * returning new point objects with the bridged values (timestamps untouched).
 * Convenience for the time-domain trends, which carry `{ t, v, ... }` points.
 */
export function bridgeTimedSeries<T extends { v: number }>(
  points: readonly T[],
  opts?: BridgeOptions,
): T[] {
  const bridged = bridgeIsolatedDropouts(
    points.map((p) => p.v),
    opts,
  );
  return points.map((p, i) => ({ ...p, v: bridged[i] }));
}
