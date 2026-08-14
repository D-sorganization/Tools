/**
 * STL clubhead meshes — TypeScript twin of `rate_of_closure/mesh.py`.
 *
 * Parses both binary STL (80-byte header, uint32 count, 50-byte
 * records) and ASCII STL (`solid`/`facet`/`vertex`), then normalizes
 * arbitrary meshes onto the canonical head envelope (AffineDrift
 * frame: x target, y up, z right):
 *
 * 1. Degenerate (zero-area) triangles are dropped.
 * 2. Axes are permuted by bounding-box extent — largest to z (heel-toe
 *    width), middle to x (face-to-back depth), smallest to y (crown
 *    height) — so the face plate ends up facing +x.
 * 3. The bounding box is centered on the origin and scaled uniformly
 *    so the depth (x extent) equals `HEAD_DEPTH_M`.
 *
 * Parity-tested in vitest against the numbers pinned by
 * `tests/rate_of_closure/test_mesh.py`.
 */

export type Vec3 = [number, number, number];
export type Triangle = [Vec3, Vec3, Vec3];

export interface HeadMesh {
  /** Triangles in the canonical head frame (meters, bbox centered). */
  triangles: Triangle[];
  /** Unit normals recomputed from the winding (zero if degenerate). */
  normals: Vec3[];
}

/** Canonical face-to-back depth of the head envelope [m]. */
export const HEAD_DEPTH_M = 0.11;

const BINARY_HEADER_BYTES = 80;
const BINARY_RECORD_BYTES = 50;
const BINARY_PREFIX_BYTES = BINARY_HEADER_BYTES + 4;
const MAX_UINT32 = 0xffff_ffff;
const MIN_AREA = 1e-20;
const ASCII_VERTEX = /vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)/g;

function looksBinary(buffer: ArrayBuffer): boolean {
  if (buffer.byteLength < BINARY_HEADER_BYTES + 4) return false;
  const count = new DataView(buffer).getUint32(BINARY_HEADER_BYTES, true);
  return (
    buffer.byteLength ===
    BINARY_HEADER_BYTES + 4 + count * BINARY_RECORD_BYTES
  );
}

function parseBinary(buffer: ArrayBuffer): Triangle[] {
  const view = new DataView(buffer);
  const count = view.getUint32(BINARY_HEADER_BYTES, true);
  if (count === 0) throw new Error("binary STL declares zero triangles");
  const triangles: Triangle[] = [];
  for (let t = 0; t < count; t += 1) {
    // Skip the 12-byte record normal; normals are recomputed.
    const base = BINARY_HEADER_BYTES + 4 + t * BINARY_RECORD_BYTES + 12;
    const tri: Vec3[] = [];
    for (let v = 0; v < 3; v += 1) {
      tri.push([
        view.getFloat32(base + v * 12, true),
        view.getFloat32(base + v * 12 + 4, true),
        view.getFloat32(base + v * 12 + 8, true),
      ]);
    }
    triangles.push(tri as Triangle);
  }
  return triangles;
}

function parseAscii(buffer: ArrayBuffer): Triangle[] {
  const text = new TextDecoder("utf-8", { fatal: false }).decode(buffer);
  if (!text.trimStart().toLowerCase().startsWith("solid")) {
    throw new Error("not a valid STL: neither binary layout nor ASCII 'solid'");
  }
  const vertices: Vec3[] = [];
  for (const match of text.matchAll(ASCII_VERTEX)) {
    vertices.push([Number(match[1]), Number(match[2]), Number(match[3])]);
  }
  if (vertices.length === 0) {
    throw new Error("ASCII STL contains no vertex lines");
  }
  if (vertices.length % 3 !== 0) {
    throw new Error("ASCII STL vertex count must be a multiple of 3");
  }
  const triangles: Triangle[] = [];
  for (let i = 0; i < vertices.length; i += 3) {
    triangles.push([vertices[i], vertices[i + 1], vertices[i + 2]]);
  }
  return triangles;
}

/** Parse STL bytes (binary or ASCII) into triangles. */
export function parseStl(buffer: ArrayBuffer): Triangle[] {
  if (buffer.byteLength === 0) throw new Error("data must not be empty");
  const triangles = looksBinary(buffer)
    ? parseBinary(buffer)
    : parseAscii(buffer);
  if (triangles.length === 0) throw new Error("STL contained no triangles");
  for (const tri of triangles) {
    for (const v of tri) {
      if (!v.every(Number.isFinite)) {
        throw new Error("STL vertices must be finite");
      }
    }
  }
  return triangles;
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function triangleCross(tri: Triangle): Vec3 {
  return cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]));
}

/** Unit normals from vertex winding; zero vector for degenerates. */
export function triangleNormals(triangles: Triangle[]): Vec3[] {
  return triangles.map((tri) => {
    const c = triangleCross(tri);
    const len = Math.hypot(...c);
    return len > MIN_AREA
      ? ([c[0] / len, c[1] / len, c[2] / len] as Vec3)
      : ([0, 0, 0] as Vec3);
  });
}

function binaryHeader(header: string): Uint8Array {
  const encoded = new Uint8Array(BINARY_HEADER_BYTES);
  let offset = 0;
  for (const character of header) {
    if (offset >= BINARY_HEADER_BYTES) break;
    const codePoint = character.codePointAt(0);
    encoded[offset] =
      codePoint !== undefined && codePoint <= 0x7f ? codePoint : 0x3f;
    offset += 1;
  }
  return encoded;
}

/** Serialize finite triangles as deterministic little-endian binary STL. */
export function writeBinaryStl(
  triangles: Triangle[],
  header = "rate_of_closure",
): ArrayBuffer {
  if (triangles.length === 0) throw new Error("cannot write an empty STL");
  if (triangles.length > MAX_UINT32) throw new Error("too many STL triangles");
  for (const triangle of triangles) {
    for (const vertex of triangle) {
      if (!vertex.every(Number.isFinite)) {
        throw new Error("STL vertices must be finite");
      }
    }
  }
  const normals = triangleNormals(triangles);
  const buffer = new ArrayBuffer(
    BINARY_PREFIX_BYTES + triangles.length * BINARY_RECORD_BYTES,
  );
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  bytes.set(binaryHeader(header));
  view.setUint32(BINARY_HEADER_BYTES, triangles.length, true);
  for (
    let triangleIndex = 0;
    triangleIndex < triangles.length;
    triangleIndex += 1
  ) {
    const recordOffset =
      BINARY_PREFIX_BYTES + triangleIndex * BINARY_RECORD_BYTES;
    const values = [normals[triangleIndex], ...triangles[triangleIndex]];
    for (let vectorIndex = 0; vectorIndex < values.length; vectorIndex += 1) {
      const vectorOffset = recordOffset + vectorIndex * 12;
      const vector = values[vectorIndex];
      for (let axis = 0; axis < 3; axis += 1) {
        view.setFloat32(vectorOffset + axis * 4, vector[axis], true);
      }
    }
  }
  return buffer;
}

function bounds(triangles: Triangle[]): { min: Vec3; max: Vec3 } {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const tri of triangles) {
    for (const v of tri) {
      for (let axis = 0; axis < 3; axis += 1) {
        if (v[axis] < min[axis]) min[axis] = v[axis];
        if (v[axis] > max[axis]) max[axis] = v[axis];
      }
    }
  }
  return { min, max };
}

/** Map arbitrary triangles onto the canonical head envelope. */
export function normalizeHead(
  triangles: Triangle[],
  depthM: number = HEAD_DEPTH_M,
): Triangle[] {
  if (!(depthM > 0)) throw new Error("depthM must be positive");
  const solid = triangles.filter(
    (tri) => Math.hypot(...triangleCross(tri)) > MIN_AREA,
  );
  if (solid.length === 0) {
    throw new Error("mesh has no non-degenerate triangles");
  }
  const { min, max } = bounds(solid);
  const extents: Vec3 = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  if (!extents.every((e) => e > 0)) {
    throw new Error("mesh must have volume on all axes");
  }
  // Stable extent ordering: [smallest, middle, largest] source axes.
  const order = [0, 1, 2].sort((a, b) => extents[a] - extents[b] || a - b);
  // middle -> x (depth), smallest -> y (height), largest -> z (width).
  const permutation = [order[1], order[0], order[2]];
  const permuted = solid.map(
    (tri) =>
      tri.map(
        (v) => [v[permutation[0]], v[permutation[1]], v[permutation[2]]] as Vec3,
      ) as Triangle,
  );
  const box = bounds(permuted);
  const center: Vec3 = [
    (box.max[0] + box.min[0]) / 2,
    (box.max[1] + box.min[1]) / 2,
    (box.max[2] + box.min[2]) / 2,
  ];
  const scale = depthM / extents[order[1]];
  return permuted.map(
    (tri) =>
      tri.map(
        (v) =>
          [
            (v[0] - center[0]) * scale,
            (v[1] - center[1]) * scale,
            (v[2] - center[2]) * scale,
          ] as Vec3,
      ) as Triangle,
  );
}

/** Parse and normalize STL bytes into a renderable head mesh. */
export function loadHeadMesh(
  buffer: ArrayBuffer,
  depthM: number = HEAD_DEPTH_M,
): HeadMesh {
  const triangles = normalizeHead(parseStl(buffer), depthM);
  return { triangles, normals: triangleNormals(triangles) };
}
