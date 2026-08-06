/** Frame-explicit impact-point kinematics for the web simulation mirror. */

import type { ClubSpec } from "./club";
import { add, cross, dot, norm, scale, sub, type Vec3 } from "./impactPhysics";
import { frame, type ImpactScenario } from "./impact";
import type { SimulationRunTs, SwingSampleTs } from "./simulation";

const RAD_TO_DEG = 180 / Math.PI;
const EPSILON = 1e-12;

export interface ImpactKinematicsTs {
  eventLabel: "Impact" | "Closest Approach";
  eventTimeS: number;
  geometryBasis: string;
  modelLimitations: string;
  referenceAoaDeg: number | null;
  contactAoaDeg: number | null;
  withoutShaftAoaDeg: number | null;
  shaftAoaContributionDeg: number | null;
  shaftRotationRateDps: number;
  shaftVerticalVelocityMps: number;
  faceNormalRateDps: number;
}

const unit = (vector: Vec3, name: string): Vec3 => {
  const magnitude = norm(vector);
  if (!(magnitude > EPSILON)) throw new RangeError(`${name} must be nonzero`);
  return scale(vector, 1 / magnitude);
};

const aoaDeg = (velocity: Vec3): number | null => {
  const horizontal = Math.hypot(velocity[0], velocity[2]);
  return horizontal <= EPSILON
    ? null
    : Math.atan2(velocity[1], horizontal) * RAD_TO_DEG;
};

const eventSample = (run: SimulationRunTs): SwingSampleTs => {
  const time = run.impactTimeS ?? run.impactOutcome.candidateTimeS;
  return run.swing.reduce((best, sample) =>
    Math.abs(sample.t - time) < Math.abs(best.t - time) ? sample : best,
  run.swing[0]);
};

function shaftGeometry(sample: SwingSampleTs, scenario: ImpactScenario) {
  if (sample.joints.length >= 2) {
    const wrist = sample.joints[sample.joints.length - 2];
    return {
      point: wrist,
      axis: unit(sub(wrist, sample.position), "articulated shaft line"),
      basis: "articulated_wrist_to_reference_shaft_line",
      limitations: "The articulated source has no shaft-twist degree of freedom; " +
        "the readout cannot invent torsional head motion.",
    };
  }
  return {
    point: sample.position,
    axis: frame(scenario.lieAngleDeg).shaft,
    basis: "scenario_shaft_line",
    limitations: "The shaft axis is assumed to pass through the tracked head " +
      "reference point. The web mirror does not yet retain full head pose.",
  };
}

export function impactKinematics(
  run: SimulationRunTs,
  scenario: ImpactScenario,
  club: ClubSpec,
): ImpactKinematicsTs {
  if (run.swing.length === 0) throw new RangeError("run must retain swing samples");
  const sample = eventSample(run);
  const shaft = shaftGeometry(sample, scenario);
  const lever: Vec3 = [
    scenario.comToFaceMm / 1000,
    scenario.impactOffsetHighMm / 1000,
    scenario.impactOffsetToeMm / 1000,
  ];
  const contact = add(sample.position, lever);
  const axisVelocity = add(
    sample.velocity,
    cross(sample.angularVelocity, sub(shaft.point, sample.position)),
  );
  const shaftRate = dot(sample.angularVelocity, shaft.axis);
  const shaftOmega = scale(shaft.axis, shaftRate);
  const otherOmega = sub(sample.angularVelocity, shaftOmega);
  const shaftVelocity = cross(shaftOmega, sub(contact, shaft.point));
  const otherVelocity = cross(otherOmega, sub(contact, shaft.point));
  const withoutShaft = add(axisVelocity, otherVelocity);
  const contactVelocity = add(withoutShaft, shaftVelocity);
  const totalAoa = aoaDeg(contactVelocity);
  const noShaftAoa = aoaDeg(withoutShaft);
  const loft = club.loftDeg * Math.PI / 180;
  const faceNormal: Vec3 = [Math.cos(loft), Math.sin(loft), 0];
  return {
    eventLabel: run.impactOutcome.status === "hit" ? "Impact" : "Closest Approach",
    eventTimeS: run.impactTimeS ?? run.impactOutcome.candidateTimeS,
    geometryBasis: shaft.basis,
    modelLimitations: shaft.limitations,
    referenceAoaDeg: aoaDeg(axisVelocity),
    contactAoaDeg: totalAoa,
    withoutShaftAoaDeg: noShaftAoa,
    shaftAoaContributionDeg:
      totalAoa === null || noShaftAoa === null ? null : totalAoa - noShaftAoa,
    shaftRotationRateDps: shaftRate * RAD_TO_DEG,
    shaftVerticalVelocityMps: shaftVelocity[1],
    faceNormalRateDps: norm(cross(sample.angularVelocity, faceNormal)) * RAD_TO_DEG,
  };
}
