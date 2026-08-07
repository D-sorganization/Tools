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

import {
  createSpatialTarget,
  surfaceCircleTolerance,
  surfaceCorridorTolerance,
  targetPointFromFrame,
  type SpatialTargetTs,
} from "./spatialTarget";

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

/** Lift the unchanged 2D region into an explicit course-surface target. */
export function spatialTargetFromRegion(
  region: TargetRegionTs,
  surfaceElevationM = 0,
  groundSource = "course.surface/default",
  label?: string,
): SpatialTargetTs {
  if (region.kind !== "green" && region.kind !== "fairway") {
    throw new RangeError(`unknown region kind ${String(region.kind)}`);
  }
  const tolerance =
    region.kind === "green"
      ? surfaceCircleTolerance(region.radiusM)
      : surfaceCorridorTolerance(region.bandHalfLengthM, region.halfWidthM);
  return createSpatialTarget({
    label: label ?? `${region.kind[0].toUpperCase()}${region.kind.slice(1)} Target`,
    kind: "landing_area",
    point: targetPointFromFrame(
      [region.distanceM, surfaceElevationM, region.kind === "green" ? region.lateralM : 0],
      "app",
    ),
    tolerance,
    elevationSource: "course_surface",
    groundSource,
  });
}

/** Project a course-surface target back to the stable 2D region API. */
export function spatialTargetToRegion(targetInput: SpatialTargetTs): TargetRegionTs {
  const target = createSpatialTarget(targetInput);
  const [distanceM, , rightM] = target.point.appCoordinatesM;
  if (target.tolerance.kind === "surface_circle") {
    return {
      kind: "green",
      distanceM,
      radiusM: target.tolerance.radiusM,
      lateralM: rightM,
      bandHalfLengthM: 15,
      halfWidthM: 16,
    };
  }
  if (target.tolerance.kind === "surface_corridor") {
    return {
      kind: "fairway",
      distanceM,
      radiusM: 10,
      lateralM: 0,
      bandHalfLengthM: target.tolerance.halfLengthM,
      halfWidthM: target.tolerance.halfWidthM,
    };
  }
  throw new RangeError("only landing_area surface targets have a 2D projection");
}

export {
  boxTolerance,
  createSpatialTarget,
  spatialTargetMiss,
  spatialTargetMissFromFrame,
  sphereTolerance,
  surfaceCircleTolerance,
  surfaceCorridorTolerance,
  targetPointFromFrame,
  targetPointInFrame,
} from "./spatialTarget";
export type {
  AcceptanceGeometryTs,
  BoxToleranceTs,
  ElevationSource,
  SpatialTargetInput,
  SpatialTargetKind,
  SpatialTargetMissTs,
  SpatialTargetTs,
  SphereToleranceTs,
  SurfaceCircleToleranceTs,
  SurfaceCorridorToleranceTs,
  TargetFrame,
  TargetPointTs,
  Vector3,
} from "./spatialTarget";
export {
  LEGACY_GROUND_SOURCE,
  SPATIAL_TARGET_SCHEMA,
  SPATIAL_TARGET_SCHEMA_VERSION,
  spatialTargetFromJson,
  spatialTargetToJson,
} from "./spatialTargetSerialization";
