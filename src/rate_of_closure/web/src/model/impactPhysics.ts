import { GOLF_BALL_RADIUS_M } from "./ballSetup";
export { GOLF_BALL_RADIUS_M } from "./ballSetup";

export type Vec3 = [number, number, number];

export const GRAVITY_M_S2 = 9.80665;
export const AIR_DENSITY_KG_M3 = 1.225;
export const GOLF_BALL_MASS_KG = 0.04593;
export const GOLF_BALL_MOI_KG_M2 =
  (2 / 5) * GOLF_BALL_MASS_KG * GOLF_BALL_RADIUS_M ** 2;
export const DRIVER_COR = 0.83;
export const DRIVER_MASS_KG = 0.2;
export const DRIVER_MOI_KG_M2 = 4.5e-4;
export const MAX_LIFT_COEFFICIENT = 0.155;
export const MPH_PER_MPS = 1 / 0.44704;
const SPHERE_ROLLING_CAP = 2 / 7;
const FRICTION_COEFFICIENT = 0.4;

export interface ImpactClubProperties {
  headMassKg: number;
  moiAboutShaftKgM2: number;
  coefficientOfRestitution?: number;
}

type ResolvedImpactClubProperties = Required<ImpactClubProperties>;

export const DEFAULT_IMPACT_CLUB: Readonly<ResolvedImpactClubProperties> =
  Object.freeze({
    headMassKg: DRIVER_MASS_KG,
    moiAboutShaftKgM2: DRIVER_MOI_KG_M2,
    coefficientOfRestitution: DRIVER_COR,
  });

export const dot = (a: Vec3, b: Vec3): number =>
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const norm = (a: Vec3): number => Math.hypot(...a);
export const scale = (a: Vec3, factor: number): Vec3 =>
  [a[0] * factor, a[1] * factor, a[2] * factor];
export const add = (a: Vec3, b: Vec3): Vec3 =>
  [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: Vec3, b: Vec3): Vec3 =>
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const toFlightFrame = (v: Vec3): Vec3 => [v[0], -v[2], v[1]];
export const fromFlightFrame = (v: Vec3): Vec3 => [v[0], v[2], -v[1]];

export interface DeliveryInput {
  clubheadSpeedMps: number;
  clubPathDeg: number;
  faceAngleDeg: number;
  attackAngleDeg: number;
  dynamicLoftDeg: number;
  impactOffsetToeMm: number;
  impactOffsetHighMm: number;
  club?: ImpactClubProperties;
}

export interface ImpactOutput {
  ballVelocity: Vec3;
  ballAngularVelocity: Vec3;
}

export interface DeliveryDiagnostics {
  spinLoftDeg: number;
  faceToPathDeg: number;
  spinAxisTiltDeg: number;
}

/** Derive the 3-D D-plane diagnostics used by both simulations and plots. */
export function deliveryDiagnostics(input: DeliveryInput): DeliveryDiagnostics {
  const path = rad(input.clubPathDeg);
  const face = rad(input.faceAngleDeg);
  const attack = rad(input.attackAngleDeg);
  const loft = rad(input.dynamicLoftDeg);
  const velocityDirection: Vec3 = [
    Math.cos(attack) * Math.cos(path),
    Math.sin(attack),
    Math.cos(attack) * Math.sin(path),
  ];
  const normal: Vec3 = [
    Math.cos(loft) * Math.cos(face),
    Math.sin(loft),
    Math.cos(loft) * Math.sin(face),
  ];
  const cosine = Math.max(-1, Math.min(1, dot(velocityDirection, normal)));
  const axisRaw = cross(velocityDirection, normal);
  const axisMagnitude = norm(axisRaw);
  const axis = axisMagnitude > 1e-12
    ? scale(axisRaw, 1 / axisMagnitude)
    : [0, 0, 1] as Vec3;
  return {
    spinLoftDeg: Math.acos(cosine) * 180 / Math.PI,
    faceToPathDeg: input.faceAngleDeg - input.clubPathDeg,
    spinAxisTiltDeg: Math.atan2(-axis[1], Math.hypot(axis[0], axis[2]))
      * 180 / Math.PI,
  };
}

const rad = (degrees: number): number => degrees * Math.PI / 180;

function resolveImpactClub(club?: ImpactClubProperties): ResolvedImpactClubProperties {
  const resolved = {
    headMassKg: club?.headMassKg ?? DEFAULT_IMPACT_CLUB.headMassKg,
    moiAboutShaftKgM2: club?.moiAboutShaftKgM2 ?? DEFAULT_IMPACT_CLUB.moiAboutShaftKgM2,
    coefficientOfRestitution:
      club?.coefficientOfRestitution ?? DEFAULT_IMPACT_CLUB.coefficientOfRestitution,
  };
  if (!Number.isFinite(resolved.headMassKg) || resolved.headMassKg <= 0) {
    throw new RangeError("Club head mass must be a positive finite value.");
  }
  if (!Number.isFinite(resolved.moiAboutShaftKgM2) || resolved.moiAboutShaftKgM2 <= 0) {
    throw new RangeError("Club MOI must be a positive finite value.");
  }
  if (!Number.isFinite(resolved.coefficientOfRestitution) ||
      resolved.coefficientOfRestitution < 0 || resolved.coefficientOfRestitution > 1) {
    throw new RangeError("Club coefficient of restitution must be between 0 and 1.");
  }
  return resolved;
}

/** Scalar-MOI rigid-body COR solve; the ball is initially stationary. */
export function solveImpact(input: DeliveryInput): ImpactOutput {
  const club = resolveImpactClub(input.club);
  const path = rad(input.clubPathDeg);
  const face = rad(input.faceAngleDeg);
  const attack = rad(input.attackAngleDeg);
  const loft = rad(input.dynamicLoftDeg);
  const velocityDirection: Vec3 = [
    Math.cos(attack) * Math.cos(path),
    Math.sin(attack),
    Math.cos(attack) * Math.sin(path),
  ];
  const normal: Vec3 = [
    Math.cos(loft) * Math.cos(face),
    Math.sin(loft),
    Math.cos(loft) * Math.sin(face),
  ];
  const clubVelocity = scale(velocityDirection, input.clubheadSpeedMps);
  const offsetM = Math.hypot(input.impactOffsetToeMm, input.impactOffsetHighMm) / 1000;
  const effectiveClubMass = offsetM > 1e-6
    ? 1 / (1 / club.headMassKg + offsetM ** 2 / club.moiAboutShaftKgM2)
    : club.headMassKg;
  const approach = dot(clubVelocity, normal);
  const effectiveMass = GOLF_BALL_MASS_KG * effectiveClubMass /
    (GOLF_BALL_MASS_KG + effectiveClubMass);
  const impulse = (1 + club.coefficientOfRestitution) * effectiveMass * approach;
  const ballVelocity = scale(normal, impulse / GOLF_BALL_MASS_KG);
  const tangent = sub(clubVelocity, scale(normal, approach));
  const tangentMagnitude = norm(tangent);
  let ballAngularVelocity: Vec3 = [0, 0, 0];
  if (tangentMagnitude > 1e-6) {
    const tangentDirection = scale(tangent, 1 / tangentMagnitude);
    const spinAxis = cross(tangentDirection, normal);
    const frictionImpulse = Math.min(
      FRICTION_COEFFICIENT * impulse,
      GOLF_BALL_MASS_KG * tangentMagnitude * SPHERE_ROLLING_CAP,
    );
    const spinMagnitude = frictionImpulse /
      (GOLF_BALL_MOI_KG_M2 / GOLF_BALL_RADIUS_M);
    ballAngularVelocity = scale(spinAxis, spinMagnitude);
  }
  return { ballVelocity, ballAngularVelocity };
}
