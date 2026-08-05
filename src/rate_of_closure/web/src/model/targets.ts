/**
 * Landing target regions (epic #4125 H7b) — TS mirror of
 * `shared/python/swing_sim/solver/targets.py` (same geometry, same
 * conventions: carry downrange [m], lateral + right of target [m]).
 *
 * Green = circle at a distance (optional lateral offset) with a
 * radius; fairway = distance band x half-width corridor. Signed
 * distance is exact (negative inside, 0 on the boundary); the solver
 * residual is the distance outside plus a small centering pull.
 */

export type TargetKind = "green" | "fairway";

export interface TargetRegionTs {
  kind: TargetKind;
  distanceM: number;
  radiusM: number;
  lateralM: number;
  bandHalfLengthM: number;
  halfWidthM: number;
}

export const DEFAULT_TARGET: TargetRegionTs = {
  kind: "green",
  distanceM: 230.0,
  radiusM: 10.0,
  lateralM: 0.0,
  bandHalfLengthM: 15.0,
  halfWidthM: 16.0,
};

/** Python-parity centering weight (solver/targets.py CENTERING_WEIGHT). */
export const CENTERING_WEIGHT = 0.05;

/** Exact signed distance [m]: negative inside, 0 on the boundary. */
export function signedDistance(
  region: TargetRegionTs,
  carryM: number,
  lateralM: number,
): number {
  if (region.kind === "green") {
    return (
      Math.hypot(carryM - region.distanceM, lateralM - region.lateralM) -
      region.radiusM
    );
  }
  const dx = Math.abs(carryM - region.distanceM) - region.bandHalfLengthM;
  const dz = Math.abs(lateralM) - region.halfWidthM;
  if (dx <= 0 && dz <= 0) return Math.max(dx, dz);
  return Math.hypot(Math.max(dx, 0), Math.max(dz, 0));
}

/** Whether the landing point is inside (or on) the region. */
export function contains(
  region: TargetRegionTs,
  carryM: number,
  lateralM: number,
): boolean {
  return signedDistance(region, carryM, lateralM) <= 0;
}

/** Solver residual [m]: distance outside (0 inside) + centering pull. */
export function residualM(
  region: TargetRegionTs,
  carryM: number,
  lateralM: number,
): number {
  const outside = Math.max(signedDistance(region, carryM, lateralM), 0);
  const cx = region.distanceM;
  const cz = region.kind === "green" ? region.lateralM : 0;
  return outside + CENTERING_WEIGHT * Math.hypot(carryM - cx, lateralM - cz);
}

/** (held, total) over a landing scatter; non-finite points excluded. */
export function holdStats(
  carriesM: readonly number[],
  lateralsM: readonly number[],
  region: TargetRegionTs,
): { held: number; total: number } {
  let held = 0;
  let total = 0;
  carriesM.forEach((carry, i) => {
    const lateral = lateralsM[i];
    if (!Number.isFinite(carry) || !Number.isFinite(lateral)) return;
    total += 1;
    if (contains(region, carry, lateral)) held += 1;
  });
  return { held, total };
}
