/** Frame-explicit exact-event impact kinematics for the web simulation mirror. */

import type { ClubSpec } from "./club";
import { add, cross, dot, norm, scale, sub, type Vec3 } from "./impactPhysics";
import { frame, type ImpactScenario } from "./impact";
import { applyRotation, slerpRotation } from "./rotation";
import type { SimulationRunTs, SwingSampleTs } from "./simulation";

const RAD_TO_DEG = 180 / Math.PI;
const EPSILON = 1e-12;

export interface ImpactVectorTs {
  key: "total" | "axisTranslation" | "shaftRotation" | "otherRotation" | "withoutShaft";
  label: string;
  originM: Vec3;
  vectorMps: Vec3;
  meaning: string;
}

export interface ImpactScrewAxisTs {
  pointM: Vec3;
  directionUnit: Vec3;
  pitchMPerRad: number;
  contactDistanceM: number;
}

export interface ImpactKinematicsTs {
  eventLabel: "Impact" | "Closest Approach";
  eventTimeS: number;
  frameId: "app_frame:x_target,y_up,z_right";
  geometryBasis: string;
  modelLimitations: string;
  referencePointM: Vec3;
  contactPointM: Vec3;
  ballCenterM: Vec3;
  shaftAxisPointM: Vec3;
  shaftAxisUnit: Vec3;
  faceNormalUnit: Vec3;
  leadingEdgeUnit: Vec3;
  arcTangentUnit: Vec3;
  vectors: ImpactVectorTs[];
  screwAxis: ImpactScrewAxisTs | null;
  referenceAoaDeg: number | null;
  contactAoaDeg: number | null;
  withoutShaftAoaDeg: number | null;
  shaftAoaContributionDeg: number | null;
  shaftAoaShapleyDeg: number | null;
  shaftRotationRateDps: number;
  shaftVerticalVelocityMps: number;
  shaftVerticalVelocityShare: number | null;
  faceNormalRateDps: number;
  leadingEdgeRateDps: number;
}

const unit = (vector: Vec3, name: string): Vec3 => {
  const magnitude = norm(vector);
  if (!(magnitude > EPSILON)) throw new RangeError(`${name} must be nonzero`);
  return scale(vector, 1 / magnitude);
};

const lerp = (first: Vec3, second: Vec3, alpha: number): Vec3 =>
  add(scale(first, 1 - alpha), scale(second, alpha));

const aoaDeg = (velocity: Vec3): number | null => {
  const horizontal = Math.hypot(velocity[0], velocity[2]);
  return horizontal <= EPSILON
    ? null
    : Math.atan2(velocity[1], horizontal) * RAD_TO_DEG;
};

export const exactEventSample = (run: SimulationRunTs): SwingSampleTs => {
  if (run.swing.length === 0) throw new RangeError("run must retain swing samples");
  const time = run.impactTimeS ?? run.impactOutcome.candidateTimeS;
  const upper = run.swing.findIndex((sample) => sample.t >= time);
  if (upper <= 0) return run.swing[Math.max(upper, 0)];
  if (upper < 0) return run.swing[run.swing.length - 1];
  const first = run.swing[upper - 1];
  const second = run.swing[upper];
  const alpha = Math.max(0, Math.min(1, (time - first.t) / (second.t - first.t)));
  const jointCount = Math.min(first.joints.length, second.joints.length);
  return {
    t: time,
    position: lerp(first.position, second.position, alpha),
    velocity: lerp(first.velocity, second.velocity, alpha),
    angularVelocity: lerp(first.angularVelocity, second.angularVelocity, alpha),
    rotation: slerpRotation(first.rotation, second.rotation, alpha),
    joints: Array.from({ length: jointCount }, (_, index) =>
      lerp(first.joints[index], second.joints[index], alpha)),
  };
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
    axis: unit(applyRotation(sample.rotation, frame(scenario.lieAngleDeg).shaft), "shaft axis"),
    basis: "scenario_shaft_line",
    limitations: "The shaft axis is assumed to pass through the tracked head " +
      "reference point; flexible-shaft deformation is not included in this rigid-head state.",
  };
}

function screwAxis(sample: SwingSampleTs, contact: Vec3): ImpactScrewAxisTs | null {
  const omegaSquared = dot(sample.angularVelocity, sample.angularVelocity);
  if (omegaSquared <= EPSILON ** 2) return null;
  const velocityAtOrigin = sub(
    sample.velocity,
    cross(sample.angularVelocity, sample.position),
  );
  const point = scale(cross(sample.angularVelocity, velocityAtOrigin), 1 / omegaSquared);
  const direction = unit(sample.angularVelocity, "angular velocity");
  const offset = sub(contact, point);
  const radial = sub(offset, scale(direction, dot(offset, direction)));
  return {
    pointM: point,
    directionUnit: direction,
    pitchMPerRad: dot(sample.angularVelocity, velocityAtOrigin) / omegaSquared,
    contactDistanceM: norm(radial),
  };
}

export function impactKinematics(
  run: SimulationRunTs,
  scenario: ImpactScenario,
  club: ClubSpec,
): ImpactKinematicsTs {
  const sample = exactEventSample(run);
  const shaft = shaftGeometry(sample, scenario);
  const leverLocal: Vec3 = [
    scenario.comToFaceMm / 1000,
    scenario.impactOffsetHighMm / 1000,
    scenario.impactOffsetToeMm / 1000,
  ];
  const contact = add(sample.position, applyRotation(sample.rotation, leverLocal));
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
  const baseAoa = aoaDeg(axisVelocity);
  const shaftOnlyAoa = aoaDeg(add(axisVelocity, shaftVelocity));
  const otherOnlyAoa = aoaDeg(add(axisVelocity, otherVelocity));
  const shaftAoaShapley = [baseAoa, shaftOnlyAoa, otherOnlyAoa, totalAoa]
    .every((value) => value !== null)
    ? 0.5 * (
        (shaftOnlyAoa! - baseAoa!) + (totalAoa! - otherOnlyAoa!)
      )
    : null;
  const loft = club.loftDeg * Math.PI / 180;
  const faceNormal = unit(applyRotation(sample.rotation, [Math.cos(loft), Math.sin(loft), 0]), "face normal");
  const nominalEdge = applyRotation(sample.rotation, [0, 0, 1]);
  const leadingEdge = unit(sub(nominalEdge, scale(faceNormal, dot(nominalEdge, faceNormal))), "leading edge");
  const totalVertical = contactVelocity[1];
  const vectors: ImpactVectorTs[] = [
    { key: "total", label: "Total Contact Velocity", originM: contact, vectorMps: contactVelocity, meaning: "Rigid-body velocity of the declared contact point." },
    { key: "axisTranslation", label: "Shaft-Axis Translation", originM: contact, vectorMps: axisVelocity, meaning: "Velocity at the physical shaft-axis datum." },
    { key: "shaftRotation", label: "Rotation About Shaft", originM: contact, vectorMps: shaftVelocity, meaning: "Contact velocity induced by angular velocity projected onto the shaft." },
    { key: "otherRotation", label: "Other Rotation", originM: contact, vectorMps: otherVelocity, meaning: "Contact velocity induced by angular velocity normal to the shaft." },
    { key: "withoutShaft", label: "Without Shaft Rotation", originM: contact, vectorMps: withoutShaft, meaning: "Counterfactual velocity after removing the shaft component." },
  ];
  return {
    eventLabel: run.impactOutcome.status === "hit" ? "Impact" : "Closest Approach",
    eventTimeS: run.impactTimeS ?? run.impactOutcome.candidateTimeS,
    frameId: "app_frame:x_target,y_up,z_right",
    geometryBasis: shaft.basis,
    modelLimitations: shaft.limitations,
    referencePointM: sample.position,
    contactPointM: contact,
    ballCenterM: run.ballPositionM,
    shaftAxisPointM: shaft.point,
    shaftAxisUnit: shaft.axis,
    faceNormalUnit: faceNormal,
    leadingEdgeUnit: leadingEdge,
    arcTangentUnit: unit(sample.velocity, "arc tangent"),
    vectors,
    screwAxis: screwAxis(sample, contact),
    referenceAoaDeg: aoaDeg(axisVelocity),
    contactAoaDeg: totalAoa,
    withoutShaftAoaDeg: noShaftAoa,
    shaftAoaContributionDeg:
      totalAoa === null || noShaftAoa === null ? null : totalAoa - noShaftAoa,
    shaftAoaShapleyDeg: shaftAoaShapley,
    shaftRotationRateDps: shaftRate * RAD_TO_DEG,
    shaftVerticalVelocityMps: shaftVelocity[1],
    shaftVerticalVelocityShare:
      Math.abs(totalVertical) <= EPSILON ? null : shaftVelocity[1] / totalVertical,
    faceNormalRateDps: norm(cross(sample.angularVelocity, faceNormal)) * RAD_TO_DEG,
    leadingEdgeRateDps: norm(cross(sample.angularVelocity, leadingEdge)) * RAD_TO_DEG,
  };
}
