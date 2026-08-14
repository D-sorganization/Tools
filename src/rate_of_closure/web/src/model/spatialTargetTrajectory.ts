/** Continuous piecewise-linear trajectory assessment for canonical 3D targets. */

import type { FlightPoint } from "./flight";
import {
  spatialTargetMiss,
  type SpatialTargetMissTs,
  type SpatialTargetTs,
  type Vector3,
} from "./spatialTarget";

export interface SpatialTrajectoryAssessment {
  readonly miss: SpatialTargetMissTs;
  readonly actualPointM: Vector3;
  readonly timeS: number;
  readonly segmentIndex: number;
  readonly fraction: number;
}

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value));

function interpolate(left: Vector3, right: Vector3, fraction: number): Vector3 {
  return [
    left[0] + (right[0] - left[0]) * fraction,
    left[1] + (right[1] - left[1]) * fraction,
    left[2] + (right[2] - left[2]) * fraction,
  ];
}

function sphereFraction(left: Vector3, right: Vector3, center: Vector3): number {
  const delta = right.map((value, index) => value - left[index]) as number[];
  const fromCenter = left.map((value, index) => value - center[index]) as number[];
  const denominator = delta.reduce((sum, value) => sum + value * value, 0);
  if (denominator === 0) return 0;
  return clamp01(-delta.reduce((sum, value, index) => sum + value * fromCenter[index], 0) /
    denominator);
}

function boxBreakpoints(
  left: Vector3,
  right: Vector3,
  center: Vector3,
  extents: Vector3,
): number[] {
  const values = [0, 1];
  left.forEach((start, axis) => {
    const delta = right[axis] - start;
    if (delta === 0) return;
    for (const boundary of [center[axis] - extents[axis], center[axis] + extents[axis]]) {
      const fraction = (boundary - start) / delta;
      if (fraction > 0 && fraction < 1) values.push(fraction);
    }
  });
  return [...new Set(values)].sort((first, second) => first - second);
}

function stationaryBoxFraction(
  left: Vector3,
  right: Vector3,
  center: Vector3,
  extents: Vector3,
  lower: number,
  upper: number,
): number | null {
  const midpoint = (lower + upper) / 2;
  let numerator = 0;
  let denominator = 0;
  left.forEach((start, axis) => {
    const delta = right[axis] - start;
    const atMidpoint = start + delta * midpoint;
    const lowerBound = center[axis] - extents[axis];
    const upperBound = center[axis] + extents[axis];
    const boundary = atMidpoint < lowerBound ? lowerBound
      : atMidpoint > upperBound ? upperBound : null;
    if (boundary === null) return;
    numerator += delta * (start - boundary);
    denominator += delta * delta;
  });
  return denominator === 0 ? null : Math.max(lower, Math.min(upper, -numerator / denominator));
}

function boxFractions(
  left: Vector3,
  right: Vector3,
  center: Vector3,
  extents: Vector3,
): number[] {
  const breaks = boxBreakpoints(left, right, center, extents);
  const candidates = [...breaks, sphereFraction(left, right, center)];
  for (let index = 0; index < breaks.length - 1; index += 1) {
    const stationary = stationaryBoxFraction(
      left, right, center, extents, breaks[index], breaks[index + 1],
    );
    if (stationary !== null) candidates.push(stationary);
  }
  return [...new Set(candidates)];
}

function segmentFractions(target: SpatialTargetTs, left: Vector3, right: Vector3): number[] {
  const center = target.point.appCoordinatesM;
  if (target.tolerance.kind === "sphere") return [sphereFraction(left, right, center)];
  if (target.tolerance.kind === "box") {
    return boxFractions(left, right, center, target.tolerance.halfExtentsM);
  }
  throw new RangeError("continuous trajectory assessment requires an aerial volume target");
}

function centerDistanceSquared(target: SpatialTargetTs, point: Vector3): number {
  return point.reduce((sum, value, index) => {
    const delta = value - target.point.appCoordinatesM[index];
    return sum + delta * delta;
  }, 0);
}

function isBetter(
  target: SpatialTargetTs,
  candidate: SpatialTrajectoryAssessment,
  best: SpatialTrajectoryAssessment,
): boolean {
  if (candidate.miss.distanceM !== best.miss.distanceM) {
    return candidate.miss.distanceM < best.miss.distanceM;
  }
  return centerDistanceSquared(target, candidate.actualPointM) <
    centerDistanceSquared(target, best.actualPointM);
}

/** Find the exact closest passage over each linear solver segment and interpolate time. */
export function assessSpatialTargetTrajectory(
  target: SpatialTargetTs,
  points: readonly FlightPoint[],
): SpatialTrajectoryAssessment {
  if (target.kind !== "aerial_waypoint") {
    throw new RangeError("trajectory assessment requires an aerial_waypoint target");
  }
  if (points.length === 0) throw new RangeError("trajectory must contain at least one point");
  let best: SpatialTrajectoryAssessment | null = null;
  const segmentCount = Math.max(1, points.length - 1);
  for (let index = 0; index < segmentCount; index += 1) {
    const left = points[index];
    const right = points[Math.min(index + 1, points.length - 1)];
    for (const fraction of segmentFractions(target, left.position, right.position)) {
      const actualPointM = interpolate(left.position, right.position, fraction);
      const candidate: SpatialTrajectoryAssessment = {
        miss: spatialTargetMiss(target, actualPointM), actualPointM,
        timeS: left.time + (right.time - left.time) * fraction,
        segmentIndex: index, fraction,
      };
      if (best === null || isBetter(target, candidate, best)) best = candidate;
    }
  }
  if (best === null) throw new Error("trajectory assessment failed to produce a candidate");
  return best;
}
