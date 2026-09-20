/** Ground-relative wedge delivery metrics and linear-velocity waterfall. */

import type { ClubSpec } from "./club";
import type { ImpactScenario } from "./impact";
import { add, cross, dot, norm, sub, type Vec3 } from "./impactPhysics";
import { applyRotation, type Mat3 } from "./rotation";
import type { SimulationRunTs } from "./simulation";
import type { WedgeGroundClearancePayloadTs } from "./wedgeGroundClearance";
import {
  bodyProfile,
  representativeWedgeForClub,
  wedgeContactCandidates,
  wedgeFaceContactPointM,
  type RepresentativeWedge,
} from "./wedgeGroundGeometry";

const APP_FRAME_ID = "app_frame:x_target,y_up,z_right";
const DEFAULT_UP: Vec3 = [0, 1, 0];
const DEFAULT_TARGET: Vec3 = [1, 0, 0];
const DEFAULT_RIGHT: Vec3 = [0, 0, 1];
const MIN_SPEED = 1e-12;

export interface MetricExplainerTs {
  key: string;
  label: string;
  value: number | string | null;
  units: string;
  equation: string;
  frame: string;
  assumptions: string;
  availability: string;
}

export interface LinearWaterfallStepTs {
  name: string;
  label: string;
  downrangeMps: number;
  verticalMps: number;
  lateralMps: number;
  totalSpeedMps: number;
  meaning: string;
}

export interface LinearVelocityWaterfallTs {
  baseAxis: LinearWaterfallStepTs;
  shaftRotation: LinearWaterfallStepTs;
  otherRotation: LinearWaterfallStepTs;
  totalContact: LinearWaterfallStepTs;
  totalAoaDeg: number | null;
  withoutShaftAoaDeg: number | null;
  shaftCounterfactualAoaDeltaDeg: number | null;
  withoutOtherAoaDeg: number | null;
  otherCounterfactualAoaDeltaDeg: number | null;
  shaftShapleyAoaDeg: number | null;
  otherShapleyAoaDeg: number | null;
}

export interface WedgeDeliveryMetricsTs {
  frameId: string;
  totalAoaDeg: number | null;
  withoutShaftAoaDeg: number | null;
  shaftCounterfactualAoaDeltaDeg: number | null;
  shaftShapleyAoaDeg: number | null;
  leDownrangeRateMps: number;
  leVerticalRateMps: number;
  leLateralRateMps: number;
  leTotalSpeedMps: number;
  le3dAngularRateDps: number;
  dynamicLoftDeg: number;
  dynamicLieDeg: number;
  dynamicFaceAngleDeg: number | null;
  deliveredBounceDeg: number;
  pathProjectedEffectiveBounceDeg: number | null;
  bounceUtilizationMarginDeg: number | null;
  lowPointWorldM: Vec3 | null;
  waterfall: LinearVelocityWaterfallTs;
  explainers: MetricExplainerTs[];
}

function aoaDeg(v: Vec3, up: Vec3): number | null {
  const vert = dot(v, up);
  const horiz = sub(v, [up[0] * vert, up[1] * vert, up[2] * vert]);
  const hSpeed = norm(horiz);
  if (hSpeed <= MIN_SPEED) return null;
  return (Math.atan2(vert, hSpeed) * 180) / Math.PI;
}

export function computeLinearVelocityWaterfall(
  referenceVelocityMps: Vec3,
  angularVelocityRadS: Vec3,
  shaftAxisPointM: Vec3,
  shaftAxisUnit: Vec3,
  contactPointM: Vec3,
  groundUpUnit: Vec3 = DEFAULT_UP,
  targetUnit: Vec3 = DEFAULT_TARGET,
  rightUnit: Vec3 = DEFAULT_RIGHT,
): LinearVelocityWaterfallTs {
  const sLen = norm(shaftAxisUnit);
  const sHat: Vec3 = sLen > 0 ? [shaftAxisUnit[0] / sLen, shaftAxisUnit[1] / sLen, shaftAxisUnit[2] / sLen] : [0, 1, 0];

  const vAxis = add(referenceVelocityMps, cross(angularVelocityRadS, shaftAxisPointM));
  const omegaShaftScalar = dot(angularVelocityRadS, sHat);
  const omegaShaft: Vec3 = [sHat[0] * omegaShaftScalar, sHat[1] * omegaShaftScalar, sHat[2] * omegaShaftScalar];
  const omegaOther = sub(angularVelocityRadS, omegaShaft);

  const lever = sub(contactPointM, shaftAxisPointM);
  const vShaft = cross(omegaShaft, lever);
  const vOther = cross(omegaOther, lever);
  const vContact = add(add(vAxis, vShaft), vOther);

  const step = (name: string, label: string, v: Vec3, meaning: string): LinearWaterfallStepTs => ({
    name,
    label,
    downrangeMps: dot(v, targetUnit),
    verticalMps: dot(v, groundUpUnit),
    lateralMps: dot(v, rightUnit),
    totalSpeedMps: norm(v),
    meaning,
  });

  const baseAxis = step("shaft_axis_translation", "Shaft-Axis Translation", vAxis, "Rigid translation of the shaft datum.");
  const shaftRotation = step("shaft_rotation", "Rotation About Shaft", vShaft, "Linear velocity from shaft rotation.");
  const otherRotation = step("other_rotation", "Other Rotation", vOther, "Linear velocity from non-shaft rotation.");
  const totalContact = step("total_contact", "Total Contact Velocity", vContact, "Total rigid-body linear velocity at contact.");

  const totalAoa = aoaDeg(vContact, groundUpUnit);
  const withoutShaftAoa = aoaDeg(sub(vContact, vShaft), groundUpUnit);
  const withoutOtherAoa = aoaDeg(sub(vContact, vOther), groundUpUnit);
  const axisAoa = aoaDeg(vAxis, groundUpUnit);

  const shaftDelta = totalAoa !== null && withoutShaftAoa !== null ? totalAoa - withoutShaftAoa : null;
  const otherDelta = totalAoa !== null && withoutOtherAoa !== null ? totalAoa - withoutOtherAoa : null;

  let shaftShapley: number | null = null;
  let otherShapley: number | null = null;
  if (totalAoa !== null && withoutShaftAoa !== null && withoutOtherAoa !== null && axisAoa !== null) {
    shaftShapley = 0.5 * ((withoutOtherAoa - axisAoa) + (totalAoa - withoutShaftAoa));
    otherShapley = 0.5 * ((withoutShaftAoa - axisAoa) + (totalAoa - withoutOtherAoa));
  }

  return {
    baseAxis,
    shaftRotation,
    otherRotation,
    totalContact,
    totalAoaDeg: totalAoa,
    withoutShaftAoaDeg: withoutShaftAoa,
    shaftCounterfactualAoaDeltaDeg: shaftDelta,
    withoutOtherAoaDeg: withoutOtherAoa,
    otherCounterfactualAoaDeltaDeg: otherDelta,
    shaftShapleyAoaDeg: shaftShapley,
    otherShapleyAoaDeg: otherShapley,
  };
}

export function computeWedgeDeliveryMetrics(
  wedge: RepresentativeWedge,
  pose: { position: Vec3; rotation: Mat3 },
  twist: { velocity: Vec3; angularVelocity: Vec3 },
  shaftAxisUnit?: Vec3,
  shaftAxisPointM?: Vec3,
  contactPointM?: Vec3,
  lowPointWorldM?: Vec3,
): WedgeDeliveryMetricsTs {
  const up = DEFAULT_UP;
  const target = DEFAULT_TARGET;
  const right = DEFAULT_RIGHT;

  let shaftAxis: Vec3;
  if (!shaftAxisUnit) {
    const lieRad = (wedge.lieDeg * Math.PI) / 180;
    const localShaft: Vec3 = [0, Math.sin(lieRad), -Math.cos(lieRad)];
    shaftAxis = applyRotation(pose.rotation, localShaft);
  } else {
    shaftAxis = shaftAxisUnit;
  }
  const shaftPt = shaftAxisPointM ?? pose.position;

  let contactPt: Vec3;
  if (!contactPointM) {
    const localContact = wedgeFaceContactPointM(wedge, 0, 0);
    contactPt = add(pose.position, applyRotation(pose.rotation, localContact));
  } else {
    contactPt = contactPointM;
  }

  const waterfall = computeLinearVelocityWaterfall(
    twist.velocity,
    twist.angularVelocity,
    shaftPt,
    shaftAxis,
    contactPt,
    up,
    target,
    right,
  );

  const profile = bodyProfile(wedge);
  const leLocalPt: Vec3 = [profile[0][0], profile[0][1], 0];
  const leWorldPt = add(pose.position, applyRotation(pose.rotation, leLocalPt));
  const leVelocity = add(twist.velocity, cross(twist.angularVelocity, sub(leWorldPt, pose.position)));

  const leDownrangeRateMps = dot(leVelocity, target);
  const leVerticalRateMps = dot(leVelocity, up);
  const leLateralRateMps = dot(leVelocity, right);
  const leTotalSpeedMps = norm(leVelocity);

  const leLocalTangent: Vec3 = [0, 0, 1];
  const leWorldTangent = applyRotation(pose.rotation, leLocalTangent);
  const leTangentRate = cross(twist.angularVelocity, leWorldTangent);
  const le3dAngularRateDps = (norm(leTangentRate) * 180) / Math.PI;

  const loftRad = (wedge.loftDeg * Math.PI) / 180;
  const localFaceNormal: Vec3 = [Math.cos(loftRad), Math.sin(loftRad), 0];
  const worldFaceNormal = applyRotation(pose.rotation, localFaceNormal);

  const fnVert = dot(worldFaceNormal, up);
  const fnHoriz = sub(worldFaceNormal, [up[0] * fnVert, up[1] * fnVert, up[2] * fnVert]);
  const fnHorizNorm = norm(fnHoriz);
  const dynamicLoftDeg = (Math.atan2(fnVert, fnHorizNorm) * 180) / Math.PI;

  const dynamicFaceAngleDeg =
    fnHorizNorm > MIN_SPEED ? (Math.atan2(dot(fnHoriz, right), dot(fnHoriz, target)) * 180) / Math.PI : null;

  const shaftUp = dot(shaftAxis, up);
  const shaftHoriz = sub(shaftAxis, [up[0] * shaftUp, up[1] * shaftUp, up[2] * shaftUp]);
  const dynamicLieDeg = (Math.atan2(Math.abs(shaftUp), norm(shaftHoriz)) * 180) / Math.PI;

  const bounceRad = (wedge.bounceDeg * Math.PI) / 180;
  const localSole: Vec3 = [-wedge.soleWidthM * Math.cos(bounceRad), wedge.soleWidthM * Math.sin(bounceRad), 0];
  const worldSole = applyRotation(pose.rotation, localSole);
  const soleVert = dot(worldSole, up);
  const soleHoriz = sub(worldSole, [up[0] * soleVert, up[1] * soleVert, up[2] * soleVert]);
  const deliveredBounceDeg = (Math.atan2(soleVert, norm(soleHoriz)) * 180) / Math.PI;

  const vVert = dot(twist.velocity, up);
  const vHoriz = sub(twist.velocity, [up[0] * vVert, up[1] * vVert, up[2] * vVert]);
  const vHorizSpeed = norm(vHoriz);
  let pathProjectedEffectiveBounceDeg: number | null = null;
  let bounceUtilizationMarginDeg: number | null = null;
  if (vHorizSpeed > MIN_SPEED) {
    const pathDir: Vec3 = [vHoriz[0] / vHorizSpeed, vHoriz[1] / vHorizSpeed, vHoriz[2] / vHorizSpeed];
    const trailingAlongPath = dot(soleHoriz, [-pathDir[0], -pathDir[1], -pathDir[2]]);
    pathProjectedEffectiveBounceDeg = (Math.atan2(soleVert, trailingAlongPath) * 180) / Math.PI;
    const refAoa = (Math.atan2(vVert, vHorizSpeed) * 180) / Math.PI;
    bounceUtilizationMarginDeg = pathProjectedEffectiveBounceDeg + refAoa;
  }

  let finalLowPoint: Vec3 | null = lowPointWorldM ?? null;
  if (!finalLowPoint) {
    const candidates = wedgeContactCandidates(wedge);
    const candWorldPts = candidates.map((c) => add(pose.position, applyRotation(pose.rotation, c.localPointM)));
    finalLowPoint = candWorldPts.reduce((lowest, pt) => (dot(pt, up) < dot(lowest, up) ? pt : lowest));
  }

  const lpStr = finalLowPoint ? `(${finalLowPoint[0].toFixed(3)}, ${finalLowPoint[1].toFixed(3)}, ${finalLowPoint[2].toFixed(3)}) m` : null;

  const explainers: MetricExplainerTs[] = [
    { key: "total_aoa", label: "Contact Attack Angle", value: waterfall.totalAoaDeg, units: "°",
      equation: "atan2(v_contact · up, |v_contact,horizontal|)", frame: APP_FRAME_ID,
      assumptions: "Vertical trajectory angle of the contact point in inertial frame.",
      availability: waterfall.totalAoaDeg !== null ? "available" : "undefined" },
    { key: "without_shaft_aoa", label: "AoA Without Shaft Rotation", value: waterfall.withoutShaftAoaDeg, units: "°",
      equation: "AoA(v_contact − v_shaft)", frame: APP_FRAME_ID,
      assumptions: "Counterfactual trajectory angle if angular velocity about the shaft were zero.",
      availability: waterfall.withoutShaftAoaDeg !== null ? "available" : "undefined" },
    { key: "shaft_counterfactual_aoa_delta", label: "Shaft Rotation AoA Delta", value: waterfall.shaftCounterfactualAoaDeltaDeg, units: "°",
      equation: "AoA(v_contact) − AoA(v_contact − v_shaft)", frame: APP_FRAME_ID,
      assumptions: "Nonlinear counterfactual delta; not an additive Euler angle.",
      availability: waterfall.shaftCounterfactualAoaDeltaDeg !== null ? "available" : "undefined" },
    { key: "shaft_shapley_aoa", label: "Shaft-Rotation Shapley AoA", value: waterfall.shaftShapleyAoaDeg, units: "°",
      equation: "mean marginal AoA across both factor evaluation orders", frame: APP_FRAME_ID,
      assumptions: "Order-independent two-factor Shapley attribution of attack angle.",
      availability: waterfall.shaftShapleyAoaDeg !== null ? "available" : "undefined" },
    { key: "le_downrange_rate", label: "LE Downrange Rate", value: leDownrangeRateMps, units: "m/s",
      equation: "(v_ref + ω × r_LE/ref) · target", frame: APP_FRAME_ID,
      assumptions: "Target-line forward speed of the leading-edge center point.", availability: "available" },
    { key: "le_vertical_rate", label: "LE Vertical Rate", value: leVerticalRateMps, units: "m/s",
      equation: "(v_ref + ω × r_LE/ref) · up", frame: APP_FRAME_ID,
      assumptions: "Vertical linear velocity of the leading-edge center point.", availability: "available" },
    { key: "le_lateral_rate", label: "LE Lateral Rate", value: leLateralRateMps, units: "m/s",
      equation: "(v_ref + ω × r_LE/ref) · right", frame: APP_FRAME_ID,
      assumptions: "Lateral cross-target linear speed of the leading-edge center point.", availability: "available" },
    { key: "le_3d_angular_rate", label: "Leading Edge 3D Rate", value: le3dAngularRateDps, units: "°/s",
      equation: "|ω × e_leading|", frame: APP_FRAME_ID,
      assumptions: "Coordinate-free 3D angular rate of leading-edge tangent direction.", availability: "available" },
    { key: "dynamic_loft", label: "Delivered Dynamic Loft", value: dynamicLoftDeg, units: "°",
      equation: "atan2(n_face · up, |n_face,horizontal|)", frame: APP_FRAME_ID,
      assumptions: "Delivered face-normal elevation angle above horizontal.", availability: "available" },
    { key: "dynamic_lie", label: "Delivered Dynamic Lie", value: dynamicLieDeg, units: "°",
      equation: "atan2(|shaft_axis · up|, |shaft_axis,horizontal|)", frame: APP_FRAME_ID,
      assumptions: "Delivered shaft-axis elevation angle above horizontal.", availability: "available" },
    { key: "dynamic_face_angle", label: "Dynamic Face Angle", value: dynamicFaceAngleDeg, units: "°",
      equation: "atan2(n_face · right, n_face · target)", frame: APP_FRAME_ID,
      assumptions: "Horizontal heading of the face normal relative to target line.",
      availability: dynamicFaceAngleDeg !== null ? "available" : "undefined" },
    { key: "delivered_bounce", label: "Delivered Bounce", value: deliveredBounceDeg, units: "°",
      equation: "atan2(world_sole · up, |world_sole,horizontal|)", frame: APP_FRAME_ID,
      assumptions: "Elevation angle of central sole above ground plane.", availability: "available" },
    { key: "low_point", label: "Delivery Low Point", value: lpStr, units: "m",
      equation: "argmin_{candidate} (candidate · up)", frame: APP_FRAME_ID,
      assumptions: "Instantaneous lowest spatial point of the wedge head geometry.",
      availability: finalLowPoint !== null ? "available" : "undefined" },
    { key: "waterfall_axis", label: "Shaft-Axis Translation", value: waterfall.baseAxis.totalSpeedMps, units: "m/s",
      equation: "v_axis = v_ref + ω × (s - r_ref)", frame: APP_FRAME_ID,
      assumptions: "Pure linear translation of the shaft-axis datum.", availability: "available" },
    { key: "waterfall_shaft", label: "Shaft Rotation Velocity", value: waterfall.shaftRotation.totalSpeedMps, units: "m/s",
      equation: "v_shaft = (ω · ŝ) ŝ × (c - s)", frame: APP_FRAME_ID,
      assumptions: "Linear contact velocity induced by shaft rotation.", availability: "available" },
    { key: "waterfall_other", label: "Other Rotation Velocity", value: waterfall.otherRotation.totalSpeedMps, units: "m/s",
      equation: "v_other = (ω - (ω · ŝ) ŝ) × (c - s)", frame: APP_FRAME_ID,
      assumptions: "Linear contact velocity induced by non-shaft rotation.", availability: "available" },
  ];

  return {
    frameId: APP_FRAME_ID,
    totalAoaDeg: waterfall.totalAoaDeg,
    withoutShaftAoaDeg: waterfall.withoutShaftAoaDeg,
    shaftCounterfactualAoaDeltaDeg: waterfall.shaftCounterfactualAoaDeltaDeg,
    shaftShapleyAoaDeg: waterfall.shaftShapleyAoaDeg,
    leDownrangeRateMps,
    leVerticalRateMps,
    leLateralRateMps,
    leTotalSpeedMps,
    le3dAngularRateDps,
    dynamicLoftDeg,
    dynamicLieDeg,
    dynamicFaceAngleDeg,
    deliveredBounceDeg,
    pathProjectedEffectiveBounceDeg,
    bounceUtilizationMarginDeg,
    lowPointWorldM: finalLowPoint,
    waterfall,
    explainers,
  };
}

export function wedgeDeliveryMetricsForRun(
  run: SimulationRunTs,
  _scenario: ImpactScenario,
  club: ClubSpec,
  groundClearance?: WedgeGroundClearancePayloadTs | null,
): WedgeDeliveryMetricsTs | null {
  const wedge = representativeWedgeForClub(club);
  if (!wedge || run.swing.length === 0) return null;

  const eventTime = run.impactTimeS ?? run.swing[0].t;
  let sample = run.swing[0];
  let minDiff = Math.abs(sample.t - eventTime);
  for (let i = 1; i < run.swing.length; i += 1) {
    const diff = Math.abs(run.swing[i].t - eventTime);
    if (diff < minDiff) {
      minDiff = diff;
      sample = run.swing[i];
    }
  }

  const lowPoint = groundClearance?.lowPoint.worldPointM;
  return computeWedgeDeliveryMetrics(
    wedge,
    { position: sample.position, rotation: sample.rotation },
    { velocity: sample.velocity, angularVelocity: sample.angularVelocity },
    undefined,
    undefined,
    undefined,
    lowPoint,
  );
}
