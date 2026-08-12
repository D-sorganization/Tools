/**
 * Parity tests for the STL parser and head normalization — pinned
 * against tests/rate_of_closure/test_mesh.py (same fixtures, same
 * expected numbers).
 */

import { describe, expect, it } from "vitest";

import {
  HEAD_DEPTH_M,
  loadHeadMesh,
  normalizeHead,
  parseStl,
  triangleNormals,
  type Triangle,
  type Vec3,
} from "./mesh";

/** Byte-for-byte identical to PARITY_ASCII in test_mesh.py. */
const PARITY_ASCII = `solid parity
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1.5e-1 0 0
      vertex 0 2.5E-1 0
    endloop
  endfacet
endsolid parity
`;

function asciiBuffer(text: string): ArrayBuffer {
  return new TextEncoder().encode(text).buffer as ArrayBuffer;
}

/** Binary STL writer (test-only) matching mesh.py's write_binary_stl. */
function writeBinaryStl(triangles: Triangle[]): ArrayBuffer {
  const buffer = new ArrayBuffer(84 + triangles.length * 50);
  const view = new DataView(buffer);
  view.setUint32(80, triangles.length, true);
  triangles.forEach((tri, t) => {
    const base = 84 + t * 50 + 12; // skip the record normal (zeros)
    tri.forEach((v, i) => {
      view.setFloat32(base + i * 12, v[0], true);
      view.setFloat32(base + i * 12 + 4, v[1], true);
      view.setFloat32(base + i * 12 + 8, v[2], true);
    });
  });
  return buffer;
}

/** Cuboid as 12 triangles — same layout as test_mesh.py. */
function boxTriangles(lo: Vec3, hi: Vec3): Triangle[] {
  const [x0, y0, z0] = lo;
  const [x1, y1, z1] = hi;
  const c: Vec3[] = [
    [x0, y0, z0],
    [x1, y0, z0],
    [x1, y1, z0],
    [x0, y1, z0],
    [x0, y0, z1],
    [x1, y0, z1],
    [x1, y1, z1],
    [x0, y1, z1],
  ];
  const quads: [number, number, number, number][] = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 4, 7, 3],
    [1, 2, 6, 5],
  ];
  const tris: Triangle[] = [];
  for (const [a, b, cc, d] of quads) {
    tris.push([c[a], c[b], c[cc]]);
    tris.push([c[a], c[cc], c[d]]);
  }
  return tris;
}

function flatten(tris: Triangle[]): Vec3[] {
  return tris.flat();
}

function extents(tris: Triangle[]): Vec3 {
  const pts = flatten(tris);
  const out: Vec3 = [0, 0, 0];
  for (let axis = 0; axis < 3; axis += 1) {
    const values = pts.map((p) => p[axis]);
    out[axis] = Math.max(...values) - Math.min(...values);
  }
  return out;
}

describe("STL parser — parity with pytest", () => {
  it("parses the shared ASCII parity fixture to exact vertices", () => {
    const tris = parseStl(asciiBuffer(PARITY_ASCII));
    expect(tris).toEqual([
      [
        [0, 0, 0],
        [0.15, 0, 0],
        [0, 0.25, 0],
      ],
    ]);
  });

  it("round-trips a cuboid through binary STL", () => {
    const tris = boxTriangles([0, 0, 0], [2, 6, 4]);
    const parsed = parseStl(writeBinaryStl(tris));
    expect(parsed.length).toBe(12);
    parsed.forEach((tri, t) =>
      tri.forEach((v, i) =>
        v.forEach((coord, axis) =>
          expect(coord).toBeCloseTo(tris[t][i][axis], 6),
        ),
      ),
    );
  });

  it("rejects empty and garbage input", () => {
    expect(() => parseStl(new ArrayBuffer(0))).toThrow(/empty/);
    expect(() => parseStl(asciiBuffer("not an stl at all"))).toThrow(
      /not a valid STL/,
    );
  });
});

describe("triangle normals", () => {
  it("computes unit normals from winding", () => {
    const normals = triangleNormals([
      [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
      ],
    ]);
    expect(normals).toEqual([[0, 0, 1]]);
  });

  it("gives degenerate triangles a zero normal", () => {
    const normals = triangleNormals([
      [
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
      ],
    ]);
    expect(normals).toEqual([[0, 0, 0]]);
  });
});

describe("head normalization — pinned numbers shared with pytest", () => {
  it("maps the (2, 6, 4) cuboid exactly like the Python rule", () => {
    // largest (y=6) -> z, middle (z=4) -> x, smallest (x=2) -> y;
    // scale = 0.11 / 4 = 0.0275.
    const tris = normalizeHead(boxTriangles([0, 0, 0], [2, 6, 4]));
    const span = extents(tris);
    expect(span[0]).toBeCloseTo(0.11, 12);
    expect(span[1]).toBeCloseTo(0.055, 12);
    expect(span[2]).toBeCloseTo(0.165, 12);
    const corner: Vec3 = [-0.055, -0.0275, -0.0825];
    const hit = flatten(tris).some(
      (p) =>
        Math.abs(p[0] - corner[0]) +
          Math.abs(p[1] - corner[1]) +
          Math.abs(p[2] - corner[2]) <
        1e-12,
    );
    expect(hit).toBe(true);
  });

  it("drops degenerate triangles and keeps the rest", () => {
    const tris = boxTriangles([0, 0, 0], [2, 6, 4]);
    const degenerate: Triangle = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    expect(normalizeHead([...tris, degenerate]).length).toBe(tris.length);
  });

  it("rejects fully degenerate and flat meshes", () => {
    const degenerate: Triangle = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    expect(() => normalizeHead([degenerate])).toThrow(/non-degenerate/);
    const flat: Triangle = [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
    ];
    expect(() => normalizeHead([flat])).toThrow(/volume/);
  });

  it("loadHeadMesh returns unit normals and the canonical depth", () => {
    const mesh = loadHeadMesh(writeBinaryStl(boxTriangles([0, 0, 0], [2, 6, 4])));
    expect(extents(mesh.triangles)[0]).toBeCloseTo(HEAD_DEPTH_M, 12);
    for (const n of mesh.normals) {
      expect(Math.hypot(...n)).toBeCloseTo(1, 9);
    }
  });
});
