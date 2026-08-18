import type { GeneratedHead } from "./clubHeadGeneration";
import {
  HEAD_DEPTH_M,
  loadHeadMeshReport,
  snapshotHeadMesh,
  type HeadMesh,
  type Vec3,
} from "./mesh";

export const IMPORT_NORMALIZATION_REVISION = "roc-stl-display-v1";
const MAX_SOURCE_NAME_CHARS = 64;

export type ClubMeshSourceKind = "procedural" | "generated" | "imported";

export interface ClubMeshSource {
  readonly generation: number;
  readonly kind: ClubMeshSourceKind;
  readonly mesh: HeadMesh | null;
  readonly hosel: Vec3 | null;
  readonly geometricCentroid: Vec3 | null;
  readonly status: string;
  readonly sha256: string | null;
  readonly rawBytes: number | null;
  readonly rawTriangles: number | null;
  readonly retainedTriangles: number | null;
  readonly normalizationRevision: string | null;
}

function requireGeneration(generation: number): void {
  if (!Number.isSafeInteger(generation) || generation < 0) {
    throw new Error("generation must be a nonnegative safe integer");
  }
}

function snapshotPoint(point: Vec3, label: string): Vec3 {
  if (point.length !== 3 || !point.every(Number.isFinite)) {
    throw new Error(`${label} must be a finite 3-vector`);
  }
  return Object.freeze([...point]) as Vec3;
}

export function cleanSourceName(name: string): string {
  const parts = name.replace(/\\/gu, "/").split("/");
  const basename = parts[parts.length - 1] || "mesh.stl";
  const sanitized = basename
    // Unicode general-category control plus explicit bidi isolates/overrides.
    // eslint-disable-next-line no-control-regex
    .replace(/[\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069]/gu, "�");
  return Array.from(sanitized).slice(0, MAX_SOURCE_NAME_CHARS).join("");
}

export function proceduralMeshSource(generation = 0): ClubMeshSource {
  requireGeneration(generation);
  return Object.freeze({
    generation,
    kind: "procedural",
    mesh: null,
    hosel: null,
    geometricCentroid: null,
    status: "Procedural head; fixed 0.110 m face-to-back display envelope",
    sha256: null,
    rawBytes: null, rawTriangles: null, retainedTriangles: null,
    normalizationRevision: null,
  });
}

export function generatedMeshSource(
  generated: GeneratedHead,
  label: string,
  generation: number,
): ClubMeshSource {
  requireGeneration(generation);
  return Object.freeze({
    generation,
    kind: "generated",
    mesh: snapshotHeadMesh(generated.mesh),
    hosel: snapshotPoint(generated.hosel, "hosel"),
    geometricCentroid: snapshotPoint(generated.cog, "geometric centroid"),
    status: `Generated representative ${cleanSourceName(label)}; authored SI geometry`,
    sha256: null,
    rawBytes: null, rawTriangles: null, retainedTriangles: null,
    normalizationRevision: null,
  });
}

async function sha256Hex(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function importedMeshSource(
  fileName: string,
  buffer: ArrayBuffer,
  generation: number,
): Promise<ClubMeshSource> {
  requireGeneration(generation);
  const { mesh, rawTriangleCount } = loadHeadMeshReport(buffer);
  const digest = await sha256Hex(buffer);
  const status = `Imported ${cleanSourceName(fileName)}; ${buffer.byteLength} bytes; `
    + `${rawTriangleCount} raw / ${mesh.triangles.length} retained triangles; `
    + `SHA-256 ${digest.slice(0, 12)}…; `
    + `${IMPORT_NORMALIZATION_REVISION}: unitless axes permuted by stable extent and `
    + `sign adjusted only to preserve handedness, `
    + `0.110 m depth, span ≤0.330 m; no physical registration or mass centroid inferred`;
  return Object.freeze({
    generation,
    kind: "imported",
    mesh,
    hosel: null,
    geometricCentroid: null,
    status,
    sha256: digest,
    rawBytes: buffer.byteLength,
    rawTriangles: rawTriangleCount,
    retainedTriangles: mesh.triangles.length,
    normalizationRevision: IMPORT_NORMALIZATION_REVISION,
  });
}

export function sourceEnvelopeDepth(source: ClubMeshSource): number {
  return source.kind === "generated" ? Number.NaN : HEAD_DEPTH_M;
}
