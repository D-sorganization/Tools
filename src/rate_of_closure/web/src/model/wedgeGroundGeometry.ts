/** Canonical modern-wedge datums shared by the web clearance analysis. */

import type { ClubSpec } from "./club";
import type { Vec3 } from "./impactPhysics";

export type WedgeContactFeature =
  | "leading_edge_center" | "leading_edge_heel" | "leading_edge_toe"
  | "primary_sole_center" | "primary_sole_heel" | "primary_sole_toe"
  | "trailing_sole_center" | "trailing_sole_heel" | "trailing_sole_toe";

export interface RepresentativeWedge {
  headId: string;
  handedness: "right" | "left";
  loftDeg: number;
  lieDeg: number;
  bounceDeg: number;
  faceLengthM: number;
  faceHeightM: number;
  soleWidthM: number;
  toplineThicknessM: number;
  leadingEdgeRadiusM: number;
  rearCurveDepthFraction: number;
  faceProgressionM: number;
  targetMassKg: number;
  geometryBasis: string;
  uncertaintyNote: string;
}

export interface WedgeContactCandidate {
  feature: WedgeContactFeature;
  localPointM: Vec3;
}

export function representativeWedgeForClub(club: ClubSpec): RepresentativeWedge | null {
  if (club.clubType !== "Wedge") return null;
  return {
    headId: `rate-${club.name.toLowerCase().replace(/ /g, "-")}`,
    handedness: "right",
    loftDeg: club.loftDeg,
    lieDeg: club.lieDeg,
    bounceDeg: 10,
    faceLengthM: 0.085,
    faceHeightM: 0.052,
    soleWidthM: 0.020,
    toplineThicknessM: 0.0055,
    leadingEdgeRadiusM: 0.0022,
    rearCurveDepthFraction: 0.85,
    faceProgressionM: 0.003,
    targetMassKg: club.headMassKg,
    geometryBasis: "Selected Rate loft, lie, and mass with the generic mid-bounce canonical wedge geometry",
    uncertaintyNote: "Illustrative 10-degree mid-bounce sole; not a measured or manufacturer-specific grind for the selected club.",
  };
}

function bodyProfile(parameters: RepresentativeWedge): [number, number][] {
  const loft = parameters.loftDeg * Math.PI / 180;
  const bounce = parameters.bounceDeg * Math.PI / 180;
  const leading: [number, number] = [parameters.faceProgressionM, parameters.leadingEdgeRadiusM];
  const faceTop: [number, number] = [
    leading[0] - parameters.faceHeightM * Math.sin(loft),
    leading[1] + parameters.faceHeightM * Math.cos(loft),
  ];
  const topBack: [number, number] = [
    faceTop[0] - parameters.toplineThicknessM * Math.cos(loft),
    faceTop[1] - parameters.toplineThicknessM * Math.sin(loft),
  ];
  const trailing: [number, number] = [
    leading[0] - parameters.soleWidthM * Math.cos(bounce),
    leading[1] + parameters.soleWidthM * Math.sin(bounce),
  ];
  const span = topBack[1] - trailing[1];
  return [leading, faceTop, topBack,
    [topBack[0] - 0.12 * parameters.soleWidthM, topBack[1] - 0.18 * span],
    [trailing[0] - parameters.rearCurveDepthFraction * parameters.soleWidthM, trailing[1] + 0.32 * span],
    trailing];
}

export function wedgeContactCandidates(parameters: RepresentativeWedge): WedgeContactCandidate[] {
  const profile = bodyProfile(parameters);
  const leading: [number, number] = [profile[0][0], 0];
  const trailing = profile[profile.length - 1];
  const primary: [number, number] = [
    (leading[0] + trailing[0]) / 2,
    (profile[0][1] + trailing[1]) / 2,
  ];
  const heelSign = parameters.handedness === "right" ? -1 : 1;
  const regions = [
    ["center", 0], ["heel", heelSign * parameters.faceLengthM / 2],
    ["toe", -heelSign * parameters.faceLengthM / 2],
  ] as const;
  const rows = [["leading_edge", leading], ["primary_sole", primary], ["trailing_sole", trailing]] as const;
  return rows.flatMap(([row, point]) => regions.map(([region, z]) => ({
    feature: `${row}_${region}` as WedgeContactFeature,
    localPointM: [point[0], point[1], z],
  })));
}

export function wedgeFaceContactPointM(
  parameters: RepresentativeWedge, toeOffsetM: number, highOffsetM: number,
): Vec3 {
  if (Math.abs(toeOffsetM) > parameters.faceLengthM / 2) throw new RangeError("toe offset exceeds face span");
  if (Math.abs(highOffsetM) > parameters.faceHeightM / 2) throw new RangeError("high offset exceeds face height");
  const loft = parameters.loftDeg * Math.PI / 180;
  const distance = parameters.faceHeightM / 2 + highOffsetM;
  const leading = bodyProfile(parameters)[0];
  const toeSign = parameters.handedness === "right" ? 1 : -1;
  return [
    leading[0] - distance * Math.sin(loft),
    leading[1] + distance * Math.cos(loft),
    toeSign * toeOffsetM,
  ];
}
