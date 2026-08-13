/** Deterministic bounded confidence-ellipsoid surface geometry. */

import type { Vec3 } from "./simulation";
import type { DispersionAdequacyTs } from "./variationGeometry";

export const MAX_RENDERED_ELLIPSOIDS = 48;
export const MAX_ELLIPSOID_VERTICES = 2_976;
export const MAX_ELLIPSOID_TRIANGLES = 5_760;
const LONGITUDE_SEGMENTS = 12;
const LATITUDE_SEGMENTS = 6;
const APP_FRAME_ID = "app_frame:x_target,y_up,z_right";

export interface ConfidenceEllipsoidMeshTs {
  coordinateFrame: typeof APP_FRAME_ID;
  interpretation: "gaussian-position-content-region";
  sampleIndices: number[];
  verticesM: Vec3[];
  triangles: Array<[number, number, number]>;
  verticesPerEllipsoid: number;
  trianglesPerEllipsoid: number;
}

export interface ConfidenceEllipsoidGeometryTs {
  centersM: Vec3[];
  principalFrames: Array<[Vec3, Vec3, Vec3]>;
  semiAxisLengthsM: Vec3[];
  adequacy: DispersionAdequacyTs[];
  coordinateFrame: string;
}

export interface EllipsoidMeshBudgetTs {
  longitudeSegments: number;
  latitudeSegments: number;
  maxEllipsoids: number;
  maxVertices: number;
  maxTriangles: number;
}

const DEFAULT_BUDGET: EllipsoidMeshBudgetTs = {
  longitudeSegments: LONGITUDE_SEGMENTS,
  latitudeSegments: LATITUDE_SEGMENTS,
  maxEllipsoids: MAX_RENDERED_ELLIPSOIDS,
  maxVertices: MAX_ELLIPSOID_VERTICES,
  maxTriangles: MAX_ELLIPSOID_TRIANGLES,
};

export function buildConfidenceEllipsoidMesh(
  geometry: ConfidenceEllipsoidGeometryTs,
  budget: EllipsoidMeshBudgetTs = DEFAULT_BUDGET,
): ConfidenceEllipsoidMeshTs {
  validateGeometry(geometry);
  const [unitVertices, unitTriangles] = unitSphere(budget);
  const capacity = Math.min(
    integerBudget(budget.maxEllipsoids, "maxEllipsoids"),
    Math.floor(integerBudget(budget.maxVertices, "maxVertices") / unitVertices.length),
    Math.floor(integerBudget(budget.maxTriangles, "maxTriangles") / unitTriangles.length),
  );
  const eligible = geometry.adequacy.flatMap((state, index) =>
    state === "estimable" ? [index] : []);
  const sampleIndices = decimatedIndices(eligible, capacity);
  const verticesM = sampleIndices.flatMap((sample) => unitVertices.map((unit) =>
    transformVertex(geometry, sample, unit)));
  const triangles = sampleIndices.flatMap((_, meshIndex) => {
    const offset = meshIndex * unitVertices.length;
    return unitTriangles.map(([first, second, third]) =>
      [first + offset, second + offset, third + offset] as [number, number, number]);
  });
  return {
    coordinateFrame: APP_FRAME_ID,
    interpretation: "gaussian-position-content-region",
    sampleIndices,
    verticesM,
    triangles,
    verticesPerEllipsoid: unitVertices.length,
    trianglesPerEllipsoid: unitTriangles.length,
  };
}

function validateGeometry(geometry: ConfidenceEllipsoidGeometryTs): void {
  const samples = geometry.centersM.length;
  if (geometry.coordinateFrame !== APP_FRAME_ID) throw new Error("invalid coordinate frame");
  if (geometry.principalFrames.length !== samples
    || geometry.semiAxisLengthsM.length !== samples
    || geometry.adequacy.length !== samples) throw new Error("geometry length mismatch");
  geometry.adequacy.forEach((state, index) => {
    if (state !== "estimable") return;
    const center = geometry.centersM[index];
    const frame = geometry.principalFrames[index];
    const semiAxes = geometry.semiAxisLengthsM[index];
    if (!finiteVec3(center) || !finiteVec3(semiAxes) || !semiAxes.every((value) => value > 0)) {
      throw new Error("estimable center and semi axes must be finite and positive");
    }
    if (frame.length !== 3 || !frame.every(finiteVec3) || !orthonormal(frame)) {
      throw new Error("estimable principal frame must be finite and orthonormal");
    }
  });
}

function unitSphere(
  budget: EllipsoidMeshBudgetTs,
): [Vec3[], Array<[number, number, number]>] {
  const longitudes = integerBudget(budget.longitudeSegments, "longitudeSegments");
  const latitudes = integerBudget(budget.latitudeSegments, "latitudeSegments");
  if (longitudes < 3 || latitudes < 2) throw new Error("insufficient tessellation");
  const vertices: Vec3[] = [[0, 0, 1]];
  for (let latitude = 1; latitude < latitudes; latitude += 1) {
    const theta = Math.PI * latitude / latitudes;
    for (let longitude = 0; longitude < longitudes; longitude += 1) {
      const phi = 2 * Math.PI * longitude / longitudes;
      vertices.push([
        Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta),
      ]);
    }
  }
  vertices.push([0, 0, -1]);
  const triangles: Array<[number, number, number]> = [];
  for (let longitude = 0; longitude < longitudes; longitude += 1) {
    triangles.push([0, 1 + longitude, 1 + (longitude + 1) % longitudes]);
  }
  for (let latitude = 0; latitude < latitudes - 2; latitude += 1) {
    const first = 1 + latitude * longitudes;
    const second = first + longitudes;
    for (let longitude = 0; longitude < longitudes; longitude += 1) {
      const next = (longitude + 1) % longitudes;
      triangles.push([first + longitude, second + longitude, second + next]);
      triangles.push([first + longitude, second + next, first + next]);
    }
  }
  const south = vertices.length - 1;
  const lastRing = 1 + (latitudes - 2) * longitudes;
  for (let longitude = 0; longitude < longitudes; longitude += 1) {
    triangles.push([south, lastRing + (longitude + 1) % longitudes, lastRing + longitude]);
  }
  return [vertices, triangles];
}

function transformVertex(
  geometry: ConfidenceEllipsoidGeometryTs,
  sample: number,
  unit: Vec3,
): Vec3 {
  const center = geometry.centersM[sample];
  const frame = geometry.principalFrames[sample];
  const scaled = unit.map((value, axis) => value * geometry.semiAxisLengthsM[sample][axis]);
  return [0, 1, 2].map((row) => center[row]
    + frame[0][row] * scaled[0]
    + frame[1][row] * scaled[1]
    + frame[2][row] * scaled[2]) as Vec3;
}

function decimatedIndices(indices: number[], capacity: number): number[] {
  if (capacity <= 0 || indices.length === 0) return [];
  if (indices.length <= capacity) return [...indices];
  if (capacity === 1) return [indices[0]];
  return Array.from({ length: capacity }, (_, index) =>
    indices[Math.floor(index * (indices.length - 1) / (capacity - 1))]);
}

const finiteVec3 = (values: Vec3): boolean => values.length === 3 && values.every(Number.isFinite);

function orthonormal(frame: [Vec3, Vec3, Vec3]): boolean {
  return frame.every((axis, index) => frame.every((other, otherIndex) => {
    const dot = axis.reduce((sum, value, component) => sum + value * other[component], 0);
    return Math.abs(dot - (index === otherIndex ? 1 : 0)) <= 1e-10;
  }));
}

function integerBudget(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) throw new Error(`${name} must be non-negative integer`);
  return value;
}
