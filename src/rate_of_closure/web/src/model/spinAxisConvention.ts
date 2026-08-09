/** Shared fade/right-positive spin-axis tilt convention. */

export type SpinVector = readonly [number, number, number];

const ZERO_SPIN_TOLERANCE = 1e-12;

/** Project forward gyro spin out before reporting target-frame axis tilt. */
export function spinAxisTiltDeg(spinVector: SpinVector): number | null {
  if (spinVector.some((value) => !Number.isFinite(value))) {
    throw new RangeError("spinVector must contain three finite components");
  }
  if (Math.hypot(...spinVector) <= ZERO_SPIN_TOLERANCE) return null;
  return Math.atan2(-spinVector[1], spinVector[2]) * 180 / Math.PI;
}
