/**
 * Closed-mesh volume and centroid via the divergence theorem —
 * TypeScript twin of `rate_of_closure/club/volumetrics.py` (H1, #4125).
 *
 * For a watertight, outward-wound triangle mesh:
 *
 *     V   = Σ det(a, b, c) / 6
 *     COG = Σ V_i · (a_i + b_i + c_i) / 4  /  V
 *
 * Exact for polyhedra and origin-independent while the mesh is closed.
 * Watertightness is checked combinatorially (every directed edge once,
 * with its reverse present). Parity-pinned against the Python numbers
 * in `volumetrics.test.ts`.
 */

import { buildParametricHead, type ClubSpec, type Vec3 } from "./club";
import { faceCenterPoint } from "./clubHeads";
import { type Triangle } from "./mesh";

/** Sanity band for generated-head volumes [m³] (see Python twin). */
export const HEAD_VOLUME_BOUNDS_M3: [number, number] = [2.0e-5, 8.0e-4];

function edgeKey(a: Vec3, b: Vec3): string {
  return `${a[0]},${a[1]},${a[2]}|${b[0]},${b[1]},${b[2]}`;
}

/** Whether every directed edge appears once with its reverse present. */
export function isWatertight(triangles: Triangle[]): boolean {
  const edges = new Map<string, number>();
  for (const tri of triangles) {
    for (let i = 0; i < 3; i += 1) {
      const key = edgeKey(tri[i], tri[(i + 1) % 3]);
      edges.set(key, (edges.get(key) ?? 0) + 1);
    }
  }
  for (const tri of triangles) {
    for (let i = 0; i < 3; i += 1) {
      const a = tri[i];
      const b = tri[(i + 1) % 3];
      if (edges.get(edgeKey(a, b)) !== 1) return false;
      if (edges.get(edgeKey(b, a)) !== 1) return false;
    }
  }
  return triangles.length > 0;
}

/** Volume [m³] and centroid [m] of a closed, outward-wound mesh. */
export function meshVolumeCentroid(triangles: Triangle[]): {
  volumeM3: number;
  cog: Vec3;
} {
  if (!isWatertight(triangles)) {
    throw new Error("mesh must be watertight (closed, matched edges)");
  }
  let volume = 0;
  const moment: Vec3 = [0, 0, 0];
  for (const [a, b, c] of triangles) {
    const cross: Vec3 = [
      b[1] * c[2] - b[2] * c[1],
      b[2] * c[0] - b[0] * c[2],
      b[0] * c[1] - b[1] * c[0],
    ];
    const signed = (a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]) / 6;
    volume += signed;
    for (let k = 0; k < 3; k += 1) {
      moment[k] += (signed * (a[k] + b[k] + c[k])) / 4;
    }
  }
  if (!(Number.isFinite(volume) && volume > 0)) {
    throw new Error("volume must be positive (outward winding required)");
  }
  return {
    volumeM3: volume,
    cog: [moment[0] / volume, moment[1] / volume, moment[2] / volume],
  };
}

/** Volumetric COG report for a spec's generated head (see Python). */
export interface CogReport {
  volumeM3: number;
  cog: Vec3;
  /** Centroid distance back from the face's forward extent [m]. */
  cgDepthM: number;
  /** Centroid height above the sole plane (lowest mesh point) [m]. */
  cgHeightM: number;
  specCgDepthM: number;
  specCgHeightM: number;
  faceCenter: Vec3;
}

/** Divergence-theorem COG of the deterministic parametric head. */
export function headCog(club: ClubSpec): CogReport {
  const triangles = buildParametricHead(club);
  const { volumeM3, cog } = meshVolumeCentroid(triangles);
  const [low, high] = HEAD_VOLUME_BOUNDS_M3;
  if (!(low <= volumeM3 && volumeM3 <= high)) {
    throw new Error("head volume outside the sane band");
  }
  let xMax = -Infinity;
  let yMin = Infinity;
  for (const tri of triangles) {
    for (const v of tri) {
      if (v[0] > xMax) xMax = v[0];
      if (v[1] < yMin) yMin = v[1];
    }
  }
  return {
    volumeM3,
    cog,
    cgDepthM: xMax - cog[0],
    cgHeightM: cog[1] - yMin,
    specCgDepthM: club.cgDepthM,
    specCgHeightM: club.cgHeightM,
    faceCenter: faceCenterPoint(club),
  };
}
