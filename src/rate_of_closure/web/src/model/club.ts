/**
 * Club modeling — TypeScript twin of `rate_of_closure/club/`.
 *
 * Mirrors the Python package field-for-field and number-for-number:
 * the frozen SI `ClubSpec`, the 15-club library (normalized from the
 * imperial/CGS table in UpstreamDrift's MuJoCo `club_configurations.py`
 * — typical published manufacturer specs), the composite head + shaft
 * + grip inertial model, the face-curvature sagitta and normal, and
 * the deterministic parametric head mesh with bulge & roll.
 *
 * Parity-tested in vitest (`club.test.ts`) against the numbers pinned
 * by `tests/rate_of_closure/test_club.py`.
 */

import { triangleNormals, type HeadMesh, type Triangle, type Vec3 } from "./mesh";

export type ClubType =
  | "Driver"
  | "Wood"
  | "Hybrid"
  | "Iron"
  | "Wedge"
  | "Putter";

/** One club's static specification, SI units (see Python docstring). */
export interface ClubSpec {
  name: string;
  clubType: ClubType;
  lengthM: number;
  headMassKg: number;
  loftDeg: number;
  lieDeg: number;
  moiAboutShaftKgM2: number;
  cgDepthM: number;
  cgHeightM: number;
  /** Horizontal (heel-toe) face radius [m]; null = flat. */
  faceBulgeRadiusM: number | null;
  /** Vertical (crown-sole) face radius [m]; null = flat. */
  faceRollRadiusM: number | null;
}

const IN = 0.0254;
const G = 1.0e-3;
const GCM2 = 1.0e-7;
const WOOD_BULGE_M = 0.3;
const WOOD_ROLL_M = 0.28;
const HYBRID_CURVE_M = 0.25;

function spec(
  name: string,
  clubType: ClubType,
  lengthIn: number,
  headG: number,
  loftDeg: number,
  lieDeg: number,
  moiGcm2: number,
  cgDepthMm: number,
  cgHeightMm: number,
  bulgeM: number | null = null,
  rollM: number | null = null,
): ClubSpec {
  return {
    name,
    clubType,
    lengthM: lengthIn * IN,
    headMassKg: headG * G,
    loftDeg,
    lieDeg,
    moiAboutShaftKgM2: moiGcm2 * GCM2,
    cgDepthM: cgDepthMm * 1e-3,
    cgHeightM: cgHeightMm * 1e-3,
    faceBulgeRadiusM: bulgeM,
    faceRollRadiusM: rollM,
  };
}

/** The 15-club library in ladder order (driver first, putter last). */
export const CLUB_LIBRARY: ClubSpec[] = [
  spec("Driver 9.5°", "Driver", 45.5, 200, 9.5, 56, 5200, 25, 28, WOOD_BULGE_M, WOOD_ROLL_M),
  spec("Driver 10.5°", "Driver", 45.5, 200, 10.5, 56, 5200, 25, 28, WOOD_BULGE_M, WOOD_ROLL_M),
  spec("Driver 12°", "Driver", 45.5, 200, 12, 56, 5200, 25, 28, WOOD_BULGE_M, WOOD_ROLL_M),
  spec("3-Wood", "Wood", 43, 210, 15, 57, 4500, 22, 23, WOOD_BULGE_M, WOOD_ROLL_M),
  spec("5-Wood", "Wood", 42, 215, 18, 58, 4300, 20, 22, WOOD_BULGE_M, WOOD_ROLL_M),
  spec("3-Hybrid", "Hybrid", 40.5, 230, 19, 59, 3800, 18, 21, HYBRID_CURVE_M, HYBRID_CURVE_M),
  spec("3-Iron", "Iron", 39, 240, 21, 59.5, 2800, 15, 20),
  spec("5-Iron", "Iron", 38, 245, 27, 61, 2600, 14, 19),
  spec("7-Iron", "Iron", 37, 250, 34, 62.5, 2400, 13, 19),
  spec("9-Iron", "Iron", 36, 255, 41, 64, 2200, 12, 18),
  spec("Pitching Wedge", "Wedge", 35.5, 290, 46, 64, 2100, 11, 17),
  spec("Gap Wedge", "Wedge", 35.25, 295, 52, 64, 2000, 10, 17),
  spec("Sand Wedge", "Wedge", 35, 300, 56, 64, 1900, 10, 16),
  spec("Lob Wedge", "Wedge", 35, 305, 60, 64, 1850, 9, 16),
  spec("Putter", "Putter", 35, 350, 3, 70, 2500, 5, 15),
];

/** Look up a library club by display name. */
export function getClub(name: string): ClubSpec {
  const found = CLUB_LIBRARY.find((c) => c.name === name);
  if (!found) throw new Error(`unknown club ${name}`);
  return found;
}

// ── Inertia (composite head + shaft + grip; see Python inertia.py) ──

export const DEFAULT_SHAFT_MASS_KG = 0.075;
export const DEFAULT_GRIP_MASS_KG = 0.05;
export const GRIP_LENGTH_M = 0.25;
export const SHAFT_TUBE_RADIUS_M = 0.006;
export const GRIP_TUBE_RADIUS_M = 0.011;

export interface ClubInertia {
  totalMassKg: number;
  balancePointM: number;
  moiAboutGripKgM2: number;
  moiAboutShaftKgM2: number;
}

/**
 * Compose head (point mass at L), shaft (uniform rod), and grip
 * (sleeve over the top 0.25 m) into whole-club inertia:
 *
 *   M   = m_h + m_s + m_g
 *   d   = (m_h·L + m_s·L/2 + m_g·l_g/2) / M
 *   I_g = m_h·L² + m_s·L²/3 + m_g·l_g²/3
 *   I_s = I_head + m_s·r_s² + m_g·r_g²
 */
export function clubInertia(
  club: ClubSpec,
  shaftMassKg: number = DEFAULT_SHAFT_MASS_KG,
  gripMassKg: number = DEFAULT_GRIP_MASS_KG,
): ClubInertia {
  if (!(shaftMassKg > 0 && shaftMassKg <= 0.25)) {
    throw new Error("shaftMassKg out of range");
  }
  if (!(gripMassKg > 0 && gripMassKg <= 0.15)) {
    throw new Error("gripMassKg out of range");
  }
  const L = club.lengthM;
  const total = club.headMassKg + shaftMassKg + gripMassKg;
  return {
    totalMassKg: total,
    balancePointM:
      (club.headMassKg * L +
        (shaftMassKg * L) / 2 +
        (gripMassKg * GRIP_LENGTH_M) / 2) /
      total,
    moiAboutGripKgM2:
      club.headMassKg * L * L +
      (shaftMassKg * L * L) / 3 +
      (gripMassKg * GRIP_LENGTH_M * GRIP_LENGTH_M) / 3,
    moiAboutShaftKgM2:
      club.moiAboutShaftKgM2 +
      shaftMassKg * SHAFT_TUBE_RADIUS_M * SHAFT_TUBE_RADIUS_M +
      gripMassKg * GRIP_TUBE_RADIUS_M * GRIP_TUBE_RADIUS_M,
  };
}

// ── Face curvature (bulge & roll; see Python parametric_head.py) ──

function sagitta(radiusM: number | null, offsetM: number): number {
  if (radiusM === null) return 0;
  if (!(Math.abs(offsetM) < radiusM)) {
    throw new Error("offset must be inside the curvature radius");
  }
  return radiusM - Math.sqrt(radiusM * radiusM - offsetM * offsetM);
}

/** Face set-back [m] at an offset from face center (0 when flat). */
export function faceSagitta(
  club: ClubSpec,
  toeM: number,
  highM: number,
): number {
  return (
    sagitta(club.faceBulgeRadiusM, toeM) + sagitta(club.faceRollRadiusM, highM)
  );
}

function loftRotation(loftDeg: number): number[][] {
  const lam = (loftDeg * Math.PI) / 180;
  return [
    [Math.cos(lam), -Math.sin(lam), 0],
    [Math.sin(lam), Math.cos(lam), 0],
    [0, 0, 1],
  ];
}

function apply(m: number[][], v: Vec3): Vec3 {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}

/**
 * Outward unit face normal at an impact offset from face center
 * (millimeters toward toe / above center). Gradient of the pre-loft
 * surface `x = x_face - s_bulge(z) - s_roll(y)` rotated by the loft
 * tilt; `(cos loft, sin loft, 0)` at center or with curvature off.
 */
export function faceNormalAtOffset(
  club: ClubSpec,
  toeMm: number,
  highMm: number,
): Vec3 {
  const toeM = toeMm * 1e-3;
  const highM = highMm * 1e-3;
  let slopeZ = 0;
  if (club.faceBulgeRadiusM !== null) {
    const r = club.faceBulgeRadiusM;
    if (!(Math.abs(toeM) < r)) throw new Error("toe offset inside bulge radius");
    slopeZ = toeM / Math.sqrt(r * r - toeM * toeM);
  }
  let slopeY = 0;
  if (club.faceRollRadiusM !== null) {
    const r = club.faceRollRadiusM;
    if (!(Math.abs(highM) < r)) {
      throw new Error("high offset inside roll radius");
    }
    slopeY = highM / Math.sqrt(r * r - highM * highM);
  }
  const len = Math.hypot(1, slopeY, slopeZ);
  return apply(loftRotation(club.loftDeg), [1 / len, slopeY / len, slopeZ / len]);
}

// ── Parametric head mesh (superellipse loft; see Python geometry.py) ──

export const REFERENCE_HEAD_MASS_KG = 0.2;
const RING_POINTS = 24;
const SUPERELLIPSE_EXPONENT = 4.0;
const BASE_SECTIONS: [number, number, number][] = [
  [0.055, 0.028, 0.058], // face plate
  [0.01, 0.031, 0.062], // crown bulge
  [-0.035, 0.024, 0.048], // rear taper
  [-0.055, 0.01, 0.02], // tail
];
const FACE_FRACTIONS = [1.0, 2.0 / 3.0, 1.0 / 3.0];

function superellipseRing(
  x: number,
  halfHeight: number,
  halfWidth: number,
): Vec3[] {
  const power = 2.0 / SUPERELLIPSE_EXPONENT;
  const ring: Vec3[] = [];
  for (let i = 0; i < RING_POINTS; i += 1) {
    const theta = (2 * Math.PI * i) / RING_POINTS;
    const sy = Math.sin(theta);
    const cz = Math.cos(theta);
    ring.push([
      x,
      halfHeight * Math.sign(sy) * Math.abs(sy) ** power,
      halfWidth * Math.sign(cz) * Math.abs(cz) ** power,
    ]);
  }
  return ring;
}

function loftBand(ringA: Vec3[], ringB: Vec3[]): Triangle[] {
  const triangles: Triangle[] = [];
  for (let i = 0; i < ringA.length; i += 1) {
    const j = (i + 1) % ringA.length;
    triangles.push([ringA[i], ringB[i], ringB[j]]);
    triangles.push([ringA[i], ringB[j], ringA[j]]);
  }
  return triangles;
}

function capFan(center: Vec3, ring: Vec3[], outwardX: boolean): Triangle[] {
  const triangles: Triangle[] = [];
  for (let i = 0; i < ring.length; i += 1) {
    const j = (i + 1) % ring.length;
    triangles.push(
      outwardX ? [center, ring[j], ring[i]] : [center, ring[i], ring[j]],
    );
  }
  return triangles;
}

/**
 * Representative head triangles for a spec, meters — deterministic.
 * Envelope scaled by cbrt(mass / 0.200 kg); face patch honors bulge,
 * roll, and loft exactly as the Python generator does.
 */
export function buildParametricHead(club: ClubSpec): Triangle[] {
  const scale = (club.headMassKg / REFERENCE_HEAD_MASS_KG) ** (1 / 3);
  const sections = BASE_SECTIONS.map(
    ([x, hh, hw]) => [x * scale, hh * scale, hw * scale] as const,
  );
  const rings = sections.map(([x, hh, hw]) => superellipseRing(x, hh, hw));

  const faceX = sections[0][0];
  const center: Vec3 = [faceX, 0, 0];
  const rotation = loftRotation(club.loftDeg);
  const faceRing = (fraction: number): Vec3[] =>
    rings[0].map((v) => {
      const y = v[1] * fraction;
      const z = v[2] * fraction;
      const p: Vec3 = [faceX - faceSagitta(club, z, y), y, z];
      const rotated = apply(rotation, [p[0] - center[0], p[1], p[2]]);
      return [rotated[0] + center[0], rotated[1], rotated[2]];
    });

  const faceRings = FACE_FRACTIONS.map(faceRing);
  const triangles: Triangle[] = [];
  const bodyRings = [faceRings[0], ...rings.slice(1)];
  for (let i = 0; i < bodyRings.length - 1; i += 1) {
    triangles.push(...loftBand(bodyRings[i], bodyRings[i + 1]));
  }
  for (let i = 0; i < faceRings.length - 1; i += 1) {
    triangles.push(...loftBand(faceRings[i], faceRings[i + 1]));
  }
  triangles.push(...capFan(center, faceRings[faceRings.length - 1], true));
  triangles.push(
    ...capFan([sections[sections.length - 1][0], 0, 0], rings[rings.length - 1], false),
  );
  if (triangles.length !== 12 * RING_POINTS) {
    throw new Error("parametric head must be closed");
  }
  return triangles;
}

/** Renderable head mesh for a spec (no normalization — already canonical). */
export function parametricHeadMesh(club: ClubSpec): HeadMesh {
  const triangles = buildParametricHead(club);
  return { triangles, normals: triangleNormals(triangles) };
}
