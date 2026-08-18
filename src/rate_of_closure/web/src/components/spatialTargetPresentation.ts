import {
  spatialTargetMiss,
  targetPointInFrame,
  type AcceptanceGeometryTs,
  type SpatialTargetTs,
  type TargetFrame,
} from "../model/spatialTarget";
import type { FlightPoint } from "../model/flight";
import { assessSpatialTargetTrajectory } from "../model/spatialTargetTrajectory";

export interface SpatialTargetDraft {
  label: string;
  kind: SpatialTargetTs["kind"];
  sourceFrame: TargetFrame;
  downrange: string;
  elevation: string;
  right: string;
  toleranceKind: AcceptanceGeometryTs["kind"];
  radius: string;
  halfDownrange: string;
  halfElevation: string;
  halfRight: string;
}

const formatCoordinate = (value: number): string => String(Number(value.toFixed(6)));

export function draftFromSpatialTarget(target: SpatialTargetTs): SpatialTargetDraft {
  const [downrange, second, third] = targetPointInFrame(target.point, target.point.sourceFrame);
  const tolerance = target.tolerance;
  return {
    label: target.label,
    kind: target.kind,
    sourceFrame: target.point.sourceFrame,
    downrange: formatCoordinate(downrange),
    elevation: formatCoordinate(second),
    right: formatCoordinate(third),
    toleranceKind: tolerance.kind,
    radius: tolerance.kind === "sphere" || tolerance.kind === "surface_circle"
      ? formatCoordinate(tolerance.radiusM) : "5",
    halfDownrange: tolerance.kind === "box"
      ? formatCoordinate(tolerance.halfExtentsM[0])
      : tolerance.kind === "surface_corridor" ? formatCoordinate(tolerance.halfLengthM) : "5",
    halfElevation: tolerance.kind === "box"
      ? formatCoordinate(tolerance.halfExtentsM[1]) : "5",
    halfRight: tolerance.kind === "box"
      ? formatCoordinate(tolerance.halfExtentsM[2])
      : tolerance.kind === "surface_corridor" ? formatCoordinate(tolerance.halfWidthM) : "5",
  };
}

export function spatialTargetSummary(target: SpatialTargetTs): string {
  const [downrange, elevation, right] = target.point.appCoordinatesM;
  const geometry = target.tolerance.kind.replace(/_/g, " ");
  return `${target.label}: ${target.kind.replace("_", " ")}, ${downrange.toFixed(1)} m ` +
    `downrange, ${elevation.toFixed(1)} m up, ${right.toFixed(1)} m right; ${geometry}. ` +
    "Canonical app frame: x toward target, y up, z right. SI metres.";
}

/** Assess surface contact at landing or continuous interpolated airborne passage. */
export function spatialTargetAssessment(
  target: SpatialTargetTs,
  points: readonly FlightPoint[],
): string {
  if (points.length === 0) return "Run Flight to evaluate this target.";
  const passage = target.kind === "aerial_waypoint"
    ? assessSpatialTargetTrajectory(target, points)
    : null;
  const finalPoint = points[points.length - 1].position;
  // Flight samples track the ball center and retain launch/tee elevation. A
  // landing-area target instead evaluates the projected course-surface contact;
  // x/z residuals remain signed while the surface-normal residual is defined as 0.
  const landingContact = [
    finalPoint[0], target.point.appCoordinatesM[1], finalPoint[2],
  ] as const;
  const miss = passage?.miss ?? spatialTargetMiss(target, landingContact);
  const sample = target.kind === "landing_area"
    ? "projected course-surface landing contact"
    : "continuous interpolated nearest passage";
  const timing = passage ? ` at ${passage.timeS.toFixed(3)} s` : "";
  if (miss.accepted) return `Target accepted at the ${sample}${timing}.`;
  const [downrange, up, right] = miss.vectorM;
  return `Target missed by ${miss.distanceM.toFixed(2)} m at the ${sample}${timing} ` +
    `(residual: ${downrange.toFixed(2)} m downrange, ${up.toFixed(2)} m up, ` +
    `${right.toFixed(2)} m right).`;
}
