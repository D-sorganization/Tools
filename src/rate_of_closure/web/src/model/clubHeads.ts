/**
 * Type-specific head profiles — TypeScript twin of
 * `rate_of_closure/club/head_profiles.py` (H1, #4125).
 *
 * One profile per broad club shape, as superellipse loft cross-sections
 * `(x, halfHeight, halfWidth, yCenter)` in the AffineDrift head frame
 * (x face-forward, y up, z toe) at the profile's reference head mass:
 * woods (rounded crown/sole), hybrids (intermediate), iron/wedge blades
 * (thin topline, shallow depth, cavity-back recess on irons), and the
 * mallet and anser-style blade putters (generic forms, no brand
 * geometry). Everything scales by `(headMass / referenceMass)^(1/3)`.
 *
 * Loft is realized as a **leading-edge lean** (#4799): `leanPoint`
 * shears head-frame points about the `y = y_le` leading-edge line, the
 * same affine map the mesh generator applies, so `faceCenterPoint` and
 * `hoselPoint` stay coincident with the generated geometry.
 *
 * Parity-tested against the Python numbers in `heads.test.ts`.
 */

import type { ClubSpec } from "./club";
import type { Vec3 } from "./mesh";

/** Head-shape refinement within a club type (putters). */
export type HeadStyle = "Auto" | "Mallet" | "Blade";

/**
 * Plumber's-neck shaft offset [m] on the blade putter — roughly one
 * shaft diameter ("full-shaft offset" in typical published putter
 * fitting references).
 */
export const PLUMBER_NECK_OFFSET_M = 0.0095;

/**
 * Where the blade (iron/wedge) hosel meets the head, as a fraction of
 * the face slant height above the leading edge (#4799 G2): the hosel
 * enters at the heel, where the face is shorter than at center, so the
 * anchor sits a bit above mid-face rather than at the topline.
 */
export const BLADE_HOSEL_HEIGHT_FRACTION = 0.58;

/** Cross-section row: [x, halfHeight, halfWidth, yCenter], meters. */
export type Section = [number, number, number, number];

export interface HeadProfile {
  referenceMassKg: number;
  /** Loft cross-sections, face (+x) first, tail last. */
  sections: Section[];
  /**
   * Hosel location on/near the envelope (heel side, z < 0). Blades
   * (irons/wedges) use only its z component — their x/y come from the
   * loft-aware leading-edge rule (#4799 G2).
   */
  hoselAnchor: Vec3;
  /** Inward (+x) offset of the tail-cap fan center (cavity recess). */
  rearRecessM: number;
  /**
   * Blade hosel offset behind the leading edge at reference mass
   * (#4799 G2) — a touch of real offset, never onset. Unused by
   * non-blade profiles.
   */
  hoselOffsetM: number;
}

/** Woods & drivers — the historical parametric-head envelope (200 g). */
export const WOOD_PROFILE: HeadProfile = {
  referenceMassKg: 0.2,
  sections: [
    [0.055, 0.028, 0.058, 0.0], // face plate
    [0.01, 0.031, 0.062, 0.0], // crown bulge
    [-0.035, 0.024, 0.048, 0.0], // rear taper
    [-0.055, 0.01, 0.02, 0.0], // tail
  ],
  hoselAnchor: [0.03, 0.03, -0.052], // heel-crown transition
  rearRecessM: 0.0,
  hoselOffsetM: 0.0,
};

/** Hybrids — intermediate: wood silhouette at ~70% depth (230 g). */
export const HYBRID_PROFILE: HeadProfile = {
  referenceMassKg: 0.23,
  sections: [
    [0.038, 0.024, 0.05, 0.0],
    [0.008, 0.026, 0.052, 0.0],
    [-0.022, 0.02, 0.04, 0.0],
    [-0.037, 0.008, 0.016, 0.0],
  ],
  hoselAnchor: [0.022, 0.025, -0.044],
  rearRecessM: 0.0,
  hoselOffsetM: 0.0,
};

/** Irons — blade: thin topline, ~22 mm deep, cavity recess (250 g). */
export const IRON_PROFILE: HeadProfile = {
  referenceMassKg: 0.25,
  sections: [
    [0.011, 0.025, 0.04, 0.0],
    [0.005, 0.024, 0.039, 0.0],
    [-0.005, 0.02, 0.037, -0.002],
    [-0.011, 0.011, 0.032, -0.007],
  ],
  hoselAnchor: [0.008, 0.024, -0.038], // heel side; z only (#4799 G2)
  rearRecessM: 0.006,
  hoselOffsetM: 0.005, // mid-iron offset, typical published range
};

/** Wedges — iron-like, rear mass biased toward the sole (300 g). */
export const WEDGE_PROFILE: HeadProfile = {
  referenceMassKg: 0.3,
  sections: [
    [0.012, 0.026, 0.04, 0.0],
    [0.006, 0.025, 0.039, -0.001],
    [-0.005, 0.021, 0.037, -0.004],
    [-0.012, 0.012, 0.032, -0.009],
  ],
  hoselAnchor: [0.009, 0.025, -0.038], // heel side; z only (#4799 G2)
  rearRecessM: 0.0,
  hoselOffsetM: 0.0035, // wedges carry less offset than irons
};

/** Mallet putter — deep semicircular-plan rounded body (360 g). */
export const MALLET_PROFILE: HeadProfile = {
  referenceMassKg: 0.36,
  sections: [
    [0.02, 0.014, 0.055, 0.0],
    [-0.005, 0.0145, 0.054, 0.0],
    [-0.035, 0.014, 0.047, 0.0],
    [-0.06, 0.012, 0.032, 0.0],
    [-0.08, 0.007, 0.013, 0.0],
  ],
  hoselAnchor: [0.014, 0.014, -0.048],
  rearRecessM: 0.0,
  hoselOffsetM: 0.0,
};

/** Blade putter — anser-style shallow rectangle with a flange back. */
export const BLADE_PUTTER_PROFILE: HeadProfile = {
  referenceMassKg: 0.35,
  sections: [
    [0.012, 0.0125, 0.05, 0.0],
    [0.004, 0.0125, 0.05, 0.0],
    [-0.004, 0.009, 0.048, -0.0035],
    [-0.014, 0.0055, 0.043, -0.007],
  ],
  hoselAnchor: [0.012 - PLUMBER_NECK_OFFSET_M, 0.0125, -0.046],
  rearRecessM: 0.0,
  hoselOffsetM: 0.0,
};

/** The concrete head style for a spec (putter Auto resolves to Blade). */
export function resolvedStyle(club: ClubSpec): HeadStyle {
  const style = club.headStyle ?? "Auto";
  if (style !== "Auto") return style;
  return club.clubType === "Putter" ? "Blade" : "Auto";
}

/** The head profile a spec's type (and putter style) selects. */
export function profileFor(club: ClubSpec): HeadProfile {
  switch (club.clubType) {
    case "Driver":
    case "Wood":
      return WOOD_PROFILE;
    case "Hybrid":
      return HYBRID_PROFILE;
    case "Iron":
      return IRON_PROFILE;
    case "Wedge":
      return WEDGE_PROFILE;
    case "Putter":
      return resolvedStyle(club) === "Mallet"
        ? MALLET_PROFILE
        : BLADE_PUTTER_PROFILE;
  }
}

/** Uniform envelope scale: constant-density mass scaling per type. */
export function massScale(club: ClubSpec): number {
  return (club.headMassKg / profileFor(club).referenceMassKg) ** (1 / 3);
}

/**
 * Leading-edge height `y_le` [m]: the authored face-section bottom,
 * mass-scaled (#4799 G1). The loft lean's fixed line.
 */
export function leadingEdgeHeight(club: ClubSpec): number {
  const profile = profileFor(club);
  const scale = massScale(club);
  const [, hh, , yc] = profile.sections[0];
  return (yc - hh) * scale;
}

/**
 * Leading-edge loft lean of one head-frame point [m] (#4799 G1).
 *
 * The affine map applied to every generated-head vertex:
 *
 *     x' = x - (y - y_le) * sin(loft)
 *     y' = y_le + (y - y_le) * cos(loft)
 *     z' = z
 *
 * The `y = y_le` fiber (leading edge / sole line) is fixed, so lofting
 * a face never throws the leading edge forward of the authored station;
 * the vertical extent compresses by `cos(loft)` — the authored face
 * height becomes slant height, as on a real wedge.
 */
export function leanPoint(club: ClubSpec, point: Vec3): Vec3 {
  for (const value of point) {
    if (!Number.isFinite(value)) throw new Error("point must be finite");
  }
  const lam = (club.loftDeg * Math.PI) / 180;
  const yLe = leadingEdgeHeight(club);
  const dy = point[1] - yLe;
  return [point[0] - dy * Math.sin(lam), yLe + dy * Math.cos(lam), point[2]];
}

/** Face-plate center in the head frame, meters (leaned; #4799 G1). */
export function faceCenterPoint(club: ClubSpec): Vec3 {
  const profile = profileFor(club);
  const scale = massScale(club);
  const [x, , , yc] = profile.sections[0];
  return leanPoint(club, [x * scale, yc * scale, 0]);
}

/**
 * Hosel (shaft attachment) point on the head, meters, head frame.
 *
 * Loft-aware per type (#4799 G2), so the shaft lands even with the
 * leading edge instead of far behind it:
 *
 * - **Irons / wedges** — `x = x_le - offset` (authored offset, never
 *   onset), `y = y_le + f * H * cos(loft)` with
 *   `f = BLADE_HOSEL_HEIGHT_FRACTION` and `H` the authored face
 *   height, z from the authored anchor.
 * - **Woods / hybrids / putters** — the authored anchor under the same
 *   leading-edge lean the mesh gets (heel-crown transition for woods
 *   and hybrids, plumber's-neck set-back on the blade putter).
 *
 * Deterministic per spec; both renderers attach the shaft line here.
 */
export function hoselPoint(club: ClubSpec): Vec3 {
  const profile = profileFor(club);
  const scale = massScale(club);
  let point: Vec3;
  if (club.clubType === "Iron" || club.clubType === "Wedge") {
    const [xLe, hh] = profile.sections[0];
    const height = 2.0 * hh * scale;
    const lam = (club.loftDeg * Math.PI) / 180;
    point = [
      (xLe - profile.hoselOffsetM) * scale,
      leadingEdgeHeight(club) +
        BLADE_HOSEL_HEIGHT_FRACTION * height * Math.cos(lam),
      profile.hoselAnchor[2] * scale,
    ];
  } else {
    const [ax, ay, az] = profile.hoselAnchor;
    point = leanPoint(club, [ax * scale, ay * scale, az * scale]);
  }
  if (!(point[2] < 0)) throw new Error("hosel point must be on the heel side");
  return point;
}
